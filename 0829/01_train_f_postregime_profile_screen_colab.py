"""F-row post-regime profile transfer screen.

Estimate small F residual lookup profiles from 2023 OOF and apply them to 2024
OOF.  This is a screen only; deployment must freeze profiles from official
Train years and keep inference row-local.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEGMENT_AUDIT = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
TRAIN_YEAR = 2023
VALID_YEAR = 2024
ID = "row_id"
AXES = ("pitcher_bhand", "batter_team", "pitcher_team_count", "batter_phand")
SHRINKAGES = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
SCALES = (0.10, 0.25, 0.50, 0.75, 1.00)
CAPS = (0.01, 0.02, 0.04, 0.06)


def load_segment_module():
    spec = importlib.util.spec_from_file_location("segment_audit", SEGMENT_AUDIT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SEGMENT_AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate))))


def bootstrap(ids, target, baseline, candidate, seed=824400, repeats=500):
    gain = (np.asarray(target) - np.asarray(baseline)) ** 2 - (
        np.asarray(target) - np.asarray(candidate)
    ) ** 2
    grouped = pd.DataFrame({"id": pd.Series(ids).astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    values = grouped[["gain", "n"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(repeats):
        sample = values[rng.integers(0, len(values), len(values))]
        positive += bool(sample[:, 0].sum() / sample[:, 1].sum() > 0)
    return float(positive / repeats)


def load_oof_frame(train_path: Path, own_oof_dir: Path, history_oof_dir: Path, segment):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    pieces = []
    for year in (TRAIN_YEAR, VALID_YEAR):
        own = segment.load_oof(own_oof_dir, year)
        rows = raw.loc[raw["season"].astype(int).eq(year)].copy().reset_index(drop=True)
        if not np.array_equal(rows[ID].astype(str).to_numpy(), own[ID].astype(str).to_numpy()):
            raise ValueError(f"{year} own OOF alignment failed")
        frame = pd.concat(
            [rows.reset_index(drop=True), own.drop(columns=[ID, "game_type", "pitcher_id"])],
            axis=1,
        )
        frame["season"] = year
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def add_axes(frame: pd.DataFrame):
    result = frame.copy()
    balls = segment_first_existing(result, ("balls_before", "balls", "ball_count"))
    strikes = segment_first_existing(result, ("strikes_before", "strikes", "strike_count"))
    if balls and strikes:
        count = (
            result[balls].fillna(-1).astype(int).astype(str)
            + "-"
            + result[strikes].fillna(-1).astype(int).astype(str)
        )
    elif "count" in result.columns:
        count = result["count"].astype(str)
    else:
        count = pd.Series(["missing"] * len(result), index=result.index)
    pitcher_hand = segment_first_existing(result, ("pitcher_hand", "pitcher_side", "p_throws"))
    batter_hand = segment_first_existing(result, ("batter_hand", "stand", "batter_side"))
    if pitcher_hand and batter_hand:
        result["batter_phand"] = result["batter_id"].astype(str) + "|" + result[pitcher_hand].astype(str)
        hand = result[pitcher_hand].astype(str) + "_" + result[batter_hand].astype(str)
        result["pitcher_bhand"] = result["pitcher_id"].astype(str) + "|" + result[batter_hand].astype(str)
    elif batter_hand:
        hand = result[batter_hand].astype(str)
        result["pitcher_bhand"] = result["pitcher_id"].astype(str) + "|" + result[batter_hand].astype(str)
        result["batter_phand"] = result["batter_id"].astype(str) + "|missing"
    else:
        hand = pd.Series(["missing"] * len(result), index=result.index)
        result["pitcher_bhand"] = result["pitcher_id"].astype(str) + "|missing"
        result["batter_phand"] = result["batter_id"].astype(str) + "|missing"
    result["hand"] = hand
    result["batter_team"] = result["batter_team_id"].astype(str)
    result["pitcher_team_count"] = result["pitcher_team_id"].astype(str) + "|" + count.astype(str)
    return result


def segment_first_existing(frame: pd.DataFrame, names):
    for name in names:
        if name in frame.columns:
            return name
    return None


def fit_profile(train: pd.DataFrame, axis: str, min_rows: int):
    residual = train["target"].to_numpy(float) - train["p_champion"].to_numpy(float)
    grouped = train.assign(residual=residual).groupby(axis, observed=True).agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
    )
    return grouped.loc[grouped["rows"].ge(min_rows)].copy()


def evaluate(valid: pd.DataFrame, profile: pd.DataFrame, axis: str,
             shrinkage: float, scale: float, cap: float):
    joined = valid[[axis]].merge(profile, left_on=axis, right_index=True, how="left")
    rows = joined["rows"].fillna(0).to_numpy(float)
    shrink = rows / (rows + shrinkage)
    delta = scale * shrink * joined["residual_mean"].fillna(0).to_numpy(float)
    delta = np.clip(delta, -cap, cap)
    baseline = valid["p_champion"].to_numpy(float)
    candidate = np.clip(baseline + delta, 1e-6, 1 - 1e-6)
    target = valid["target"].to_numpy(float)
    return {
        "f_delta": bss(target, candidate) - bss(target, baseline),
        "changed_rows": int(np.count_nonzero(np.abs(delta) > 0)),
        "mean_prediction_delta": float(candidate.mean() - baseline.mean()),
        "mean_abs_delta": float(np.mean(np.abs(candidate - baseline))),
        "pitcher_bootstrap_probability": bootstrap(
            valid["pitcher_id"].to_numpy(), target, baseline, candidate
        ),
    }


def main(train_path: Path, own_oof_dir: Path, history_oof_dir: Path,
         output: Path, min_rows: int):
    segment = load_segment_module()
    frame = add_axes(load_oof_frame(train_path, own_oof_dir, history_oof_dir, segment))
    frame = frame.loc[frame["game_type"].astype(str).eq("F")].copy()
    train = frame.loc[frame["season"].eq(TRAIN_YEAR)].copy()
    valid = frame.loc[frame["season"].eq(VALID_YEAR)].copy()
    candidates = []
    profiles = []
    for axis in AXES:
        profile = fit_profile(train, axis, min_rows)
        profiles.append({
            "axis": axis,
            "profile_rows": int(len(profile)),
            "profile_size_min": int(profile["rows"].min()) if len(profile) else 0,
            "profile_size_max": int(profile["rows"].max()) if len(profile) else 0,
            "top_abs_residual": profile.reindex(
                profile["residual_mean"].abs().sort_values(ascending=False).index
            ).head(12).reset_index().rename(columns={axis: "segment"}).to_dict("records"),
        })
        if profile.empty:
            continue
        for shrinkage in SHRINKAGES:
            for scale in SCALES:
                for cap in CAPS:
                    metrics = evaluate(valid, profile, axis, shrinkage, scale, cap)
                    candidates.append({
                        "axis": axis,
                        "shrinkage": shrinkage,
                        "scale": scale,
                        "cap": cap,
                        "fold_2024_f_delta": metrics["f_delta"],
                        "changed_rows": metrics["changed_rows"],
                        "mean_prediction_delta": metrics["mean_prediction_delta"],
                        "mean_abs_delta": metrics["mean_abs_delta"],
                        "pitcher_bootstrap_probability": metrics["pitcher_bootstrap_probability"],
                        "passed": bool(
                            metrics["f_delta"] > 5
                            and metrics["pitcher_bootstrap_probability"] >= 0.80
                            and metrics["changed_rows"] > 0
                            and abs(metrics["mean_prediction_delta"]) <= 0.003
                        ),
                    })
    candidates.sort(key=lambda row: (
        row["passed"], row["fold_2024_f_delta"], row["pitcher_bootstrap_probability"]
    ), reverse=True)
    selected = next((row for row in candidates if row["passed"]), None)
    report = {
        "experiment": "F post-regime profile one-year transfer screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "diagnostic_only": True,
        "train_year": TRAIN_YEAR,
        "valid_year": VALID_YEAR,
        "baseline": "own champion OOF p_champion, F rows only",
        "min_rows": min_rows,
        "profiles": profiles,
        "top_candidates": candidates[:20],
        "selected": selected,
        "decision": "build_f_postregime_profile_route" if selected else "keep_current_f_route",
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
    parser.add_argument("--own-oof-dir", type=Path, required=True)
    parser.add_argument("--history-oof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=300)
    args = parser.parse_args()
    main(args.train.resolve(), args.own_oof_dir.resolve(), args.history_oof_dir.resolve(),
         args.output.resolve(), args.min_rows)
