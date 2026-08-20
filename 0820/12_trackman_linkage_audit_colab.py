"""공식 두 데이터의 일정 패턴으로 투수 ID를 독립 연결하고 구종 비율로 감사한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def normalized_profiles(frame: pd.DataFrame, id_col: str, key_cols: list[str]):
    grouped = frame.groupby([id_col, *key_cols], observed=True).size().rename("n").reset_index()
    grouped["key"] = grouped[key_cols].astype(str).agg("|".join, axis=1)
    matrix = grouped.pivot_table(index=id_col, columns="key", values="n", fill_value=0.0)
    norm = np.linalg.norm(matrix.to_numpy(float), axis=1, keepdims=True)
    values = matrix.to_numpy(float) / np.where(norm > 0, norm, 1.0)
    hand = frame.groupby(id_col, observed=True).pitcher_hand.first().reindex(matrix.index).astype(str)
    return matrix.index, matrix.columns, values, hand


def align_columns(a_columns, a_values, b_columns, b_values):
    columns = a_columns.union(b_columns)
    a = pd.DataFrame(a_values, columns=a_columns).reindex(columns=columns, fill_value=0.0).to_numpy()
    b = pd.DataFrame(b_values, columns=b_columns).reindex(columns=columns, fill_value=0.0).to_numpy()
    return a, b


def main(train_path: Path, trackman_path: Path, mapping_output: Path, report_output: Path) -> None:
    train_cols = ["pitcher_id", "season", "game_month", "game_dayofweek", "pitcher_hand",
                  "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
                  "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    track_cols = ["pitcher_trackman_id", "season", "game_month", "game_dayofweek",
                  "pitcher_hand", "pitch_type_group"]
    train = pd.read_csv(train_path, usecols=train_cols, low_memory=False)
    track = pd.read_csv(trackman_path, usecols=track_cols, low_memory=False)
    train["pitcher_id"] = train["pitcher_id"].astype(str)
    key_cols = ["season", "game_month", "game_dayofweek"]
    aid, acols, av, ah = normalized_profiles(train, "pitcher_id", key_cols)
    bid, bcols, bv, bh = normalized_profiles(track, "pitcher_trackman_id", key_cols)
    av, bv = align_columns(acols, av, bcols, bv)
    similarity = av @ bv.T
    hand_aliases = {"1": "Left", "2": "Right", "1.0": "Left", "2.0": "Right",
                    "L": "Left", "R": "Right", "LEFT": "Left", "RIGHT": "Right"}
    train_hand = ah.replace(hand_aliases).to_numpy()
    track_hand = bh.replace(hand_aliases).to_numpy()
    similarity[train_hand[:, None] != track_hand[None, :]] = -1.0
    rows, cols = linear_sum_assignment(-similarity)
    score = similarity[rows, cols]
    ordered = np.sort(similarity[rows], axis=1)
    second = ordered[:, -2] if similarity.shape[1] > 1 else np.full(len(rows), -1.0)
    mapping = pd.DataFrame({"pitcher_id": aid[rows].astype(str),
                            "pitcher_trackman_id": bid[cols],
                            "similarity": score, "margin": score - second})

    annual_track = (track.groupby(["pitcher_trackman_id", "season", "pitch_type_group"], observed=True)
                    .size().unstack(fill_value=0))
    for name in ("fastball", "breaking", "offspeed"):
        if name not in annual_track:
            annual_track[name] = 0
    annual_track = annual_track[["fastball", "breaking", "offspeed"]]
    prior = annual_track.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    denominator = prior.sum(axis=1).replace(0.0, np.nan)
    prior_rates = prior.div(denominator, axis=0).reset_index()
    latest_train = (train.sort_values(["pitcher_id", "season", "asof_pitcher_pitchmix_n"])
                    .groupby(["pitcher_id", "season"], observed=True).tail(1))
    audit = latest_train.merge(mapping, on="pitcher_id", how="inner").merge(
        prior_rates, left_on=["pitcher_trackman_id", "season"],
        right_on=["pitcher_trackman_id", "season"], how="inner")
    reliable = audit.similarity.ge(0.80) & audit.margin.ge(0.02) & audit.asof_pitcher_pitchmix_n.ge(100)
    correlations = {}
    for group in ("fastball", "breaking", "offspeed"):
        left = f"asof_pitcher_{group}_rate"
        valid = reliable & audit[left].notna() & audit[group].notna()
        correlations[group] = {"rows": int(valid.sum()),
                               "correlation": float(audit.loc[valid, [left, group]].corr().iloc[0, 1])
                               if valid.sum() >= 3 else None,
                               "mean_absolute_difference": float((audit.loc[valid, left] - audit.loc[valid, group]).abs().mean())
                               if valid.any() else None}
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_output, index=False)
    high_confidence = mapping.similarity.ge(0.80) & mapping.margin.ge(0.02)
    correlation_values = [row["correlation"] for row in correlations.values() if row["correlation"] is not None]
    passed = bool(high_confidence.sum() >= 300 and correlation_values and min(correlation_values) >= 0.35)
    report = {"experiment": "official Trackman pitcher linkage audit",
              "official_data_only": True, "external_mapping_used": False,
              "train_pitchers": int(len(aid)), "trackman_pitchers": int(len(bid)),
              "assigned_pairs": int(len(mapping)), "high_confidence_pairs": int(high_confidence.sum()),
              "high_confidence_rate": float(high_confidence.mean()),
              "similarity_quantiles": {str(q): float(mapping.similarity.quantile(q)) for q in (0.1, 0.5, 0.9)},
              "margin_quantiles": {str(q): float(mapping.margin.quantile(q)) for q in (0.1, 0.5, 0.9)},
              "pitchmix_validation": correlations,
              "decision": "build_prior_season_trackman_features" if passed else "reject_trackman_linkage",
              "gate": "at least 300 pairs with similarity>=0.80 and margin>=0.02; all pitchmix correlations>=0.35"}
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"mapping: {mapping_output}\nreport: {report_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--trackman", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    main(args.train.resolve(), args.trackman.resolve(), args.mapping_output.resolve(), args.report_output.resolve())
