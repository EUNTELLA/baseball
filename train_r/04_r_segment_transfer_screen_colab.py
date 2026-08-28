"""Strict one-year transfer screen for R-row segment residual correction.

The previous segment audit used 2023 and 2024 targets together only to identify
promising axes.  This script performs the stricter check: estimate each segment's
R residual from 2023 OOF only, then apply it to 2024 OOF rows.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
ID = "row_id"
TARGET = "control_success"
TRAIN_YEAR = 2023
VALID_YEAR = 2024
AXES = ("axis_p_hand", "axis_hand", "axis_p_count", "axis_p_base")
ALPHAS = (50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0)
SCALES = (0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("r_segment_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap(ids, target, baseline, candidate, repeats=500, seed=824300):
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


def make_frame(train_path: Path, oof_dir: Path, audit):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    pieces = []
    for year in (TRAIN_YEAR, VALID_YEAR):
        oof = audit.load_oof(oof_dir, year)
        rows = raw.loc[raw["season"].astype(int).eq(year)].copy().reset_index(drop=True)
        if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
            raise ValueError(f"{year} row_id alignment failed")
        merged = pd.concat([rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type"])], axis=1)
        merged["season"] = year
        pieces.append(merged)
    frame = audit.add_axes(pd.concat(pieces, ignore_index=True))
    return frame.loc[frame["game_type"].astype(str).eq("R")].copy()


def fit_segments(train: pd.DataFrame, axis: str, min_rows: int):
    residual = train["target"].to_numpy(float) - train["p_champion"].to_numpy(float)
    grouped = train.assign(residual=residual).groupby(axis, observed=True)
    stats = grouped.agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("p_champion", "mean"),
    )
    return stats.loc[stats["rows"].ge(min_rows)].copy()


def evaluate(valid: pd.DataFrame, segments: pd.DataFrame, axis: str, alpha: float, scale: float, audit):
    joined = valid[[axis]].merge(segments, left_on=axis, right_index=True, how="left")
    rows = joined["rows"].fillna(0).to_numpy(float)
    shrink = rows / (rows + alpha)
    delta = scale * shrink * joined["residual_mean"].fillna(0).to_numpy(float)
    baseline = valid["p_champion"].to_numpy(float)
    candidate = np.clip(baseline + delta, 1e-6, 1 - 1e-6)
    target = valid["target"].to_numpy(float)
    return {
        "bss_delta": audit.bss(target, candidate) - audit.bss(target, baseline),
        "changed_rows": int(np.count_nonzero(np.abs(delta) > 0)),
        "prediction_mean_delta": float(candidate.mean() - baseline.mean()),
        "pitcher_bootstrap_probability": bootstrap(
            valid["pitcher_id"].to_numpy(), target, baseline, candidate,
            seed=824300 + int(alpha) + int(scale * 1000),
        ),
    }


def main(train_path: Path, oof_dir: Path, output: Path, min_rows: int):
    audit = load_audit_module()
    frame = make_frame(train_path, oof_dir, audit)
    train = frame.loc[frame["season"].eq(TRAIN_YEAR)].copy()
    valid = frame.loc[frame["season"].eq(VALID_YEAR)].copy()
    candidates = []
    segment_reports = []
    for axis in AXES:
        segments = fit_segments(train, axis, min_rows)
        segment_reports.append({
            "axis": axis,
            "segments": int(len(segments)),
            "top_segments": segments.reindex(
                segments["residual_mean"].abs().sort_values(ascending=False).index
            ).head(12).reset_index().rename(columns={axis: "segment"}).to_dict("records"),
        })
        if segments.empty:
            continue
        for alpha in ALPHAS:
            for scale in SCALES:
                metrics = evaluate(valid, segments, axis, alpha, scale, audit)
                candidates.append({
                    "axis": axis,
                    "alpha": alpha,
                    "scale": scale,
                    "fold_2024_delta": metrics["bss_delta"],
                    "changed_rows": metrics["changed_rows"],
                    "prediction_mean_delta": metrics["prediction_mean_delta"],
                    "pitcher_bootstrap_probability": metrics["pitcher_bootstrap_probability"],
                    "passed": bool(
                        metrics["bss_delta"] > 1.0
                        and metrics["pitcher_bootstrap_probability"] >= 0.80
                        and metrics["changed_rows"] > 0
                    ),
                })
    candidates.sort(key=lambda row: (
        row["passed"], row["fold_2024_delta"], row["pitcher_bootstrap_probability"]
    ), reverse=True)
    selected = next((row for row in candidates if row["passed"]), None)
    report = {
        "experiment": "R segment residual one-year transfer screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "diagnostic_only": True,
        "train_year": TRAIN_YEAR,
        "valid_year": VALID_YEAR,
        "baseline": "own champion OOF p_champion, R rows only",
        "min_rows": min_rows,
        "segment_reports": segment_reports,
        "top_candidates": candidates[:20],
        "selected": selected,
        "decision": "build_strict_r_segment_submission_route" if selected else "keep_current_r_route",
        "note": "Validation target is used only for this screen; deployment route must freeze segments from train years only.",
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
