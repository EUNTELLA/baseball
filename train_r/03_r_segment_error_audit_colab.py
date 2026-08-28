"""R-row segment error audit for the own champion OOF files.

This is a diagnostic screen, not a submission builder.  It finds row-local
segments where the R champion has the same residual direction in 2023 and 2024.
The output is used to decide the next strictly defined R correction route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID = "row_id"
TARGET = "control_success"
YEARS = (2023, 2024)
P_BINS = (0.0, 0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 1.0)
ALPHAS = (200.0, 500.0, 1000.0, 3000.0)
SCALES = (0.01, 0.025, 0.05, 0.075, 0.10)


def bss(target, prediction):
    target = np.asarray(target, float)
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate))))


def load_oof(oof_dir: Path, year: int):
    path = oof_dir / f"own_champion_oof_{year}.npz"
    with np.load(path, allow_pickle=True) as data:
        return pd.DataFrame({
            ID: data[ID].astype(str),
            "target": data["target"].astype(float),
            "game_type": data["game_type"].astype(str),
            "pitcher_id": data["pitcher_id"].astype(str),
            "p_champion": data["p_champion"].astype(float),
        })


def first_existing(frame: pd.DataFrame, names):
    for name in names:
        if name in frame.columns:
            return name
    return None


def add_axes(frame: pd.DataFrame):
    result = frame.copy()
    balls = first_existing(result, ("balls_before", "balls", "ball_count"))
    strikes = first_existing(result, ("strikes_before", "strikes", "strike_count"))
    if balls and strikes:
        result["axis_count"] = (
            result[balls].fillna(-1).astype(int).astype(str)
            + "-"
            + result[strikes].fillna(-1).astype(int).astype(str)
        )
    elif "count" in result.columns:
        result["axis_count"] = result["count"].astype(str)
    else:
        result["axis_count"] = "missing"

    pitcher_hand = first_existing(result, ("pitcher_hand", "pitcher_side", "p_throws"))
    batter_hand = first_existing(result, ("batter_hand", "stand", "batter_side"))
    if pitcher_hand and batter_hand:
        result["axis_hand"] = result[pitcher_hand].astype(str) + "_" + result[batter_hand].astype(str)
    elif batter_hand:
        result["axis_hand"] = result[batter_hand].astype(str)
    else:
        result["axis_hand"] = "missing"

    if {"on_1b", "on_2b", "on_3b"}.issubset(result.columns):
        result["axis_base"] = (
            result["on_1b"].notna().astype(int).astype(str)
            + result["on_2b"].notna().astype(int).astype(str)
            + result["on_3b"].notna().astype(int).astype(str)
        )
    elif "base_state" in result.columns:
        result["axis_base"] = result["base_state"].astype(str)
    else:
        runners = first_existing(result, ("runners", "runner_count", "base_runner_count"))
        result["axis_base"] = result[runners].fillna(-1).astype(int).astype(str) if runners else "missing"

    outs = first_existing(result, ("outs_when_up", "outs", "out_count"))
    result["axis_outs"] = result[outs].fillna(-1).astype(int).astype(str) if outs else "missing"

    inning = first_existing(result, ("inning", "inning_no"))
    if inning:
        values = result[inning].fillna(-1).astype(int)
        result["axis_inning_band"] = np.select(
            [values <= 3, values <= 6, values <= 9],
            ["early", "mid", "late"],
            default="extra",
        )
    else:
        result["axis_inning_band"] = "missing"

    score = first_existing(result, ("score_diff", "score_difference", "batting_team_score_diff"))
    if score:
        values = result[score].fillna(0).astype(float)
        result["axis_score_band"] = pd.cut(
            values, [-99, -4, -1, 0, 1, 4, 99],
            labels=["trail_big", "trail", "tie_low", "tie_high", "lead", "lead_big"],
            include_lowest=True,
        ).astype(str)
    else:
        result["axis_score_band"] = "missing"

    result["axis_p_band"] = pd.cut(
        result["p_champion"], P_BINS, labels=False, include_lowest=True
    ).astype(str)
    result["axis_count_hand"] = result["axis_count"] + "|" + result["axis_hand"]
    result["axis_p_count"] = result["axis_p_band"] + "|" + result["axis_count"]
    result["axis_p_hand"] = result["axis_p_band"] + "|" + result["axis_hand"]
    result["axis_p_base"] = result["axis_p_band"] + "|" + result["axis_base"]
    result["axis_p_count_base"] = (
        result["axis_p_band"] + "|" + result["axis_count"] + "|" + result["axis_base"]
    )
    return result


def segment_stats(frame: pd.DataFrame, axis: str):
    residual = frame["target"].to_numpy(float) - frame["p_champion"].to_numpy(float)
    grouped = frame.assign(residual=residual).groupby(axis, observed=True)
    stats = grouped.agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("p_champion", "mean"),
    ).reset_index().rename(columns={axis: "segment"})
    return stats


def stable_segments(frame: pd.DataFrame, axis: str, min_rows: int):
    parts = []
    for year in YEARS:
        one = segment_stats(frame.loc[frame["season"].eq(year)], axis)
        one = one.rename(columns={
            "rows": f"rows_{year}",
            "residual_mean": f"residual_{year}",
            "target_mean": f"target_mean_{year}",
            "prediction_mean": f"prediction_mean_{year}",
        })
        parts.append(one)
    merged = parts[0].merge(parts[1], on="segment", how="inner")
    enough = merged[f"rows_{YEARS[0]}"].ge(min_rows) & merged[f"rows_{YEARS[1]}"].ge(min_rows)
    same_sign = np.sign(merged[f"residual_{YEARS[0]}"]) == np.sign(merged[f"residual_{YEARS[1]}"])
    nonzero = np.sign(merged[f"residual_{YEARS[0]}"]) != 0
    merged = merged.loc[enough & same_sign & nonzero].copy()
    merged["axis"] = axis
    merged["worst_abs_residual"] = np.minimum(
        merged[f"residual_{YEARS[0]}"].abs(), merged[f"residual_{YEARS[1]}"].abs()
    )
    merged["mean_residual"] = (
        merged[f"residual_{YEARS[0]}"] * merged[f"rows_{YEARS[0]}"]
        + merged[f"residual_{YEARS[1]}"] * merged[f"rows_{YEARS[1]}"]
    ) / (merged[f"rows_{YEARS[0]}"] + merged[f"rows_{YEARS[1]}"])
    return merged.sort_values("worst_abs_residual", ascending=False)


def evaluate(frame: pd.DataFrame, stable: pd.DataFrame, axis: str, alpha: float, scale: float):
    values = stable.set_index("segment")
    summaries = []
    for year in YEARS:
        valid = frame.loc[frame["season"].eq(year)].copy()
        joined = valid[[axis]].merge(values, left_on=axis, right_index=True, how="left")
        rows_key = f"rows_{year}"
        residual_key = f"residual_{year}"
        shrink = joined[rows_key].fillna(0).to_numpy(float) / (
            joined[rows_key].fillna(0).to_numpy(float) + alpha
        )
        delta = scale * shrink * joined[residual_key].fillna(0).to_numpy(float)
        candidate = np.clip(valid["p_champion"].to_numpy(float) + delta, 1e-6, 1 - 1e-6)
        baseline = valid["p_champion"].to_numpy(float)
        target = valid["target"].to_numpy(float)
        summaries.append({
            "year": int(year),
            "delta": bss(target, candidate) - bss(target, baseline),
            "mean_prediction_delta": float(candidate.mean() - baseline.mean()),
            "changed_rows": int(np.count_nonzero(np.abs(delta) > 0)),
        })
    return summaries


def main(train_path: Path, oof_dir: Path, output: Path, min_rows: int):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    pieces = []
    for year in YEARS:
        oof = load_oof(oof_dir, year)
        rows = raw.loc[raw["season"].astype(int).eq(year)].copy().reset_index(drop=True)
        if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
            raise ValueError(f"{year} row_id alignment failed")
        merged = pd.concat([rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type"])], axis=1)
        merged["season"] = year
        pieces.append(merged)
    frame = add_axes(pd.concat(pieces, ignore_index=True))
    frame = frame.loc[frame["game_type"].astype(str).eq("R")].copy()

    axes = [
        "axis_p_band", "axis_count", "axis_hand", "axis_base", "axis_outs",
        "axis_inning_band", "axis_score_band", "axis_count_hand",
        "axis_p_count", "axis_p_hand", "axis_p_base", "axis_p_count_base",
    ]
    axis_reports = []
    candidates = []
    for axis in axes:
        stable = stable_segments(frame, axis, min_rows)
        axis_reports.append({
            "axis": axis,
            "stable_segments": int(len(stable)),
            "top_segments": stable.head(12).to_dict("records"),
        })
        if stable.empty:
            continue
        for alpha in ALPHAS:
            for scale in SCALES:
                folds = evaluate(frame, stable, axis, alpha, scale)
                deltas = [fold["delta"] for fold in folds]
                changed = min(fold["changed_rows"] for fold in folds)
                candidates.append({
                    "axis": axis,
                    "alpha": alpha,
                    "scale": scale,
                    "fold_2023_delta": deltas[0],
                    "fold_2024_delta": deltas[1],
                    "worst_delta": min(deltas),
                    "changed_rows_min": int(changed),
                    "passed": bool(min(deltas) > 0 and changed > 0),
                })
    candidates.sort(key=lambda row: (row["passed"], row["worst_delta"]), reverse=True)
    selected = next((row for row in candidates if row["passed"]), None)
    report = {
        "experiment": "R segment residual stability audit",
        "official_train_only": True,
        "test_aggregate_used": False,
        "diagnostic_only": True,
        "baseline": "own champion OOF p_champion, R rows only",
        "years": list(YEARS),
        "min_rows_per_year": min_rows,
        "axis_reports": axis_reports,
        "top_candidates": candidates[:20],
        "selected": selected,
        "decision": "design_strict_r_segment_route" if selected else "keep_current_r_route",
        "note": "Uses validation targets to find stable R error segments; do not package directly.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "top": candidates[:10],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=5000)
    args = parser.parse_args()
    main(args.train.resolve(), args.oof_dir.resolve(), args.output.resolve(), args.min_rows)
