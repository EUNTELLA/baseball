"""Large-lever R residual rebuild screen.

This script is intentionally stronger than small lookup/post-processing tests:
it trains a CatBoostRegressor on R-row residuals of the current own champion
OOF prediction, then audits scale/reliability/cap combinations on later OOF
seasons.  If only 2023/2024 OOF files are available, it performs the strict
2023 -> 2024 transfer screen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
ID = "row_id"
TARGET = "control_success"
SEEDS = (17, 42, 777, 2024)
SCALES = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
ALPHAS = (0.0, 100.0, 300.0, 1000.0, 3000.0)
CAPS = (0.01, 0.02, 0.04, 0.06)
CAT_COLS = (
    "count", "hand", "same_hand", "base_state", "top_bottom",
    "pitcher_team_id", "batter_team_id", "p_band", "gap_band",
    "inning_band", "runner_band", "outs",
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("r_segment_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate))))


def bootstrap(ids, target, baseline, candidate, repeats=500, seed=824600):
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


def available_years(oof_dir: Path) -> list[int]:
    years = []
    for path in sorted(oof_dir.glob("own_champion_oof_*.npz")):
        try:
            years.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return years


def load_frame(train_path: Path, oof_dir: Path, audit) -> pd.DataFrame:
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    pieces = []
    for year in available_years(oof_dir):
        if year < 2022:
            continue
        oof = audit.load_oof(oof_dir, year)
        rows = raw.loc[raw["season"].astype(int).eq(year)].copy().reset_index(drop=True)
        if len(rows) != len(oof):
            raise ValueError(f"{year} row count mismatch: train={len(rows)} oof={len(oof)}")
        if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
            raise ValueError(f"{year} row_id alignment failed")
        merged = pd.concat(
            [rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type", "pitcher_id"])],
            axis=1,
        )
        merged["season"] = year
        pieces.append(merged)
    if not pieces:
        raise FileNotFoundError(f"No own_champion_oof_*.npz files found in {oof_dir}")
    frame = pd.concat(pieces, ignore_index=True)
    return frame.loc[frame["game_type"].astype(str).eq("R")].copy()


def add_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    balls = number(rows, "balls_before", -1).astype(int)
    strikes = number(rows, "strikes_before", -1).astype(int)
    outs = number(rows, "outs_before", 0).astype(int)
    runners = number(rows, "num_runners_on", 0).astype(int)
    inning = number(rows, "inning", 0).astype(int)
    pitcher = number(rows, "asof_pitcher_success_rate", 0.5)
    batter = number(rows, "asof_batter_success_rate", 0.5)
    pitcher_n = number(rows, "asof_pitcher_n", 0).clip(lower=0)
    batter_n = number(rows, "asof_batter_n", 0).clip(lower=0)
    recent1 = number(rows, "asof_pitcher_prev1_game_success_rate", pitcher)
    recent3 = number(rows, "asof_pitcher_prev3_game_success_rate", pitcher)
    recent5 = number(rows, "asof_pitcher_prev5_game_success_rate", pitcher)
    p = rows["p_champion"].astype(float)
    gap = pitcher - batter

    result["anchor"] = p
    result["anchor_logit"] = logit(p)
    result["anchor_center"] = p - 0.5
    result["gap_raw"] = gap
    result["gap_smooth_100"] = (
        (pitcher_n * pitcher + 100 * 0.5) / (pitcher_n + 100)
        - (batter_n * batter + 100 * 0.5) / (batter_n + 100)
    )
    result["gap_smooth_500"] = (
        (pitcher_n * pitcher + 500 * 0.5) / (pitcher_n + 500)
        - (batter_n * batter + 500 * 0.5) / (batter_n + 500)
    )
    result["recent1_gap"] = recent1 - pitcher
    result["recent3_gap"] = recent3 - pitcher
    result["recent5_gap"] = recent5 - pitcher
    result["recent1_vs_batter"] = recent1 - batter
    result["recent3_vs_batter"] = recent3 - batter
    result["recent5_vs_batter"] = recent5 - batter
    result["pitcher_rate"] = pitcher
    result["batter_rate"] = batter
    result["log_pitcher_n"] = np.log1p(pitcher_n)
    result["log_batter_n"] = np.log1p(batter_n)
    result["balls"] = balls
    result["strikes"] = strikes
    result["count_advantage"] = strikes - balls
    result["outs_num"] = outs
    result["runners_num"] = runners
    result["inning"] = inning
    result["score_diff"] = number(rows, "score_diff_pitcher_team", 0)
    result["li"] = number(rows, "li", 1.0)
    result["middle_rate"] = number(rows, "asof_pitcher_middle_rate", 0.0)
    result["reverse_rate"] = number(rows, "asof_pitcher_reverse_rate", 0.0)
    result["count"] = balls.astype(str) + "-" + strikes.astype(str)
    result["hand"] = rows["pitcher_hand"].astype(str) + "_" + rows["batter_hand"].astype(str)
    result["same_hand"] = rows["pitcher_hand"].astype(str).eq(rows["batter_hand"].astype(str)).map(
        {True: "same", False: "opposite"}
    )
    result["base_state"] = rows["base_state"].astype(str)
    result["top_bottom"] = rows["top_bottom"].astype(str)
    result["pitcher_team_id"] = rows["pitcher_team_id"].astype(str)
    result["batter_team_id"] = rows["batter_team_id"].astype(str)
    result["p_band"] = pd.cut(
        p, [0.0, 0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 1.0],
        labels=False, include_lowest=True,
    ).astype(str)
    result["gap_band"] = pd.cut(
        gap, [-2, -0.10, -0.04, 0.0, 0.04, 0.10, 2],
        labels=False, include_lowest=True,
    ).astype(str)
    result["inning_band"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "mid", "late"],
        default="extra",
    )
    result["runner_band"] = runners.clip(lower=0, upper=3).astype(str)
    result["outs"] = outs.clip(lower=0, upper=2).astype(str)
    for column in CAT_COLS:
        result[column] = result[column].astype(str)
    return result


def train_residual(train_x: pd.DataFrame, train_y: np.ndarray,
                   valid_x: pd.DataFrame, task_type: str):
    from catboost import CatBoostRegressor, Pool

    cat_indices = [train_x.columns.get_loc(column) for column in CAT_COLS]
    train_pool = Pool(train_x, train_y, cat_features=cat_indices)
    valid_pool = Pool(valid_x, cat_features=cat_indices)
    predictions, seconds = [], []
    for seed in SEEDS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=900,
            learning_rate=0.025,
            depth=4,
            l2_leaf_reg=500.0,
            random_seed=seed,
            task_type=task_type,
            allow_writing_files=False,
            verbose=False,
        )
        start = time.perf_counter()
        model.fit(train_pool)
        elapsed = float(time.perf_counter() - start)
        seconds.append(elapsed)
        predictions.append(model.predict(valid_pool))
        print(f"residual seed={seed} sec={elapsed:.1f}", flush=True)
    return np.mean(predictions, axis=0), seconds


def evaluate_fold(train_rows: pd.DataFrame, valid_rows: pd.DataFrame,
                  task_type: str, year: int) -> dict:
    train_x = add_features(train_rows)
    valid_x = add_features(valid_rows)
    train_y = train_rows["target"].to_numpy(float) - train_rows["p_champion"].to_numpy(float)
    residual_prediction, seconds = train_residual(train_x, train_y, valid_x, task_type)
    baseline = valid_rows["p_champion"].to_numpy(float)
    target = valid_rows["target"].to_numpy(float)
    pitcher_n = number(valid_rows, "asof_pitcher_n", 0).clip(lower=0).to_numpy(float)
    batter_n = number(valid_rows, "asof_batter_n", 0).clip(lower=0).to_numpy(float)
    candidates = []
    for scale in SCALES:
        for alpha in ALPHAS:
            if alpha <= 0:
                reliability = np.ones(len(valid_rows), dtype=float)
            else:
                pitcher_reliability = pitcher_n / (pitcher_n + alpha)
                batter_reliability = batter_n / (batter_n + alpha)
                reliability = np.sqrt(np.clip(pitcher_reliability * batter_reliability, 0, 1))
            for cap in CAPS:
                delta = np.clip(scale * reliability * residual_prediction, -cap, cap)
                candidate = np.clip(baseline + delta, 1e-6, 1 - 1e-6)
                candidates.append({
                    "scale": float(scale),
                    "alpha": float(alpha),
                    "cap": float(cap),
                    "r_bss_delta": bss(target, candidate) - bss(target, baseline),
                    "prediction_mean_delta": float(candidate.mean() - baseline.mean()),
                    "mean_abs_delta": float(np.mean(np.abs(candidate - baseline))),
                    "max_abs_delta": float(np.max(np.abs(candidate - baseline))),
                    "changed_rows": int(np.count_nonzero(np.abs(delta) > 1e-12)),
                    "pitcher_bootstrap_probability": bootstrap(
                        valid_rows["pitcher_id"].to_numpy(), target, baseline, candidate,
                        seed=824600 + year + int(scale * 1000) + int(alpha) + int(cap * 10000),
                    ),
                })
    for row in candidates:
        row["passed"] = bool(
            row["r_bss_delta"] >= 20.0
            and row["pitcher_bootstrap_probability"] >= 0.90
            and abs(row["prediction_mean_delta"]) <= 0.005
            and row["mean_abs_delta"] <= 0.015
        )
    candidates.sort(key=lambda row: (
        row["passed"], row["r_bss_delta"], row["pitcher_bootstrap_probability"]
    ), reverse=True)
    return {
        "year": int(year),
        "train_years": sorted(int(value) for value in train_rows["season"].unique()),
        "train_rows": int(len(train_rows)),
        "valid_rows": int(len(valid_rows)),
        "residual_seconds": seconds,
        "target_residual_mean": float(train_y.mean()),
        "predicted_residual_mean": float(np.mean(residual_prediction)),
        "predicted_residual_std": float(np.std(residual_prediction)),
        "top_candidates": candidates[:20],
        "selected": next((row for row in candidates if row["passed"]), None),
    }


def summarize(folds: list[dict]) -> list[dict]:
    if len(folds) < 2:
        return []
    summaries = []
    keys = sorted({
        (row["scale"], row["alpha"], row["cap"])
        for fold in folds
        for row in fold["top_candidates"]
    })
    for scale, alpha, cap in keys:
        rows = []
        for fold in folds:
            match = next(
                (
                    row for row in fold["top_candidates"]
                    if row["scale"] == scale and row["alpha"] == alpha and row["cap"] == cap
                ),
                None,
            )
            if match is None:
                rows = []
                break
            rows.append(match)
        if len(rows) != len(folds):
            continue
        deltas = [row["r_bss_delta"] for row in rows]
        boot = min(row["pitcher_bootstrap_probability"] for row in rows)
        summaries.append({
            "scale": scale,
            "alpha": alpha,
            "cap": cap,
            "fold_deltas": deltas,
            "worst_delta": min(deltas),
            "mean_delta": float(np.mean(deltas)),
            "minimum_pitcher_bootstrap_probability": boot,
            "passed": bool(min(deltas) >= 5.0 and boot >= 0.85),
        })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["mean_delta"]), reverse=True)
    return summaries


def main(train_path: Path, oof_dir: Path, output: Path, task_type: str):
    audit = load_audit_module()
    frame = load_frame(train_path, oof_dir, audit)
    years = sorted(int(value) for value in frame["season"].unique())
    folds = []
    for year in years:
        train_rows = frame.loc[frame["season"].lt(year)].copy()
        valid_rows = frame.loc[frame["season"].eq(year)].copy()
        if train_rows.empty or valid_rows.empty:
            continue
        fold = evaluate_fold(train_rows, valid_rows, task_type, year)
        folds.append(fold)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"status": "running", "folds": folds}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"fold={year} complete", flush=True)
    summaries = summarize(folds)
    if summaries:
        selected = next((row for row in summaries if row["passed"]), None)
        decision = "build_r_residual_rebuild_submission_route" if selected else "keep_rstrict_response_route"
    else:
        selected = next((fold["selected"] for fold in folds if fold["selected"]), None)
        decision = (
            "build_latest_year_r_residual_rebuild_probe"
            if selected else "keep_rstrict_response_route"
        )
    report = {
        "experiment": "R residual rebuild transfer screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "diagnostic_only": True,
        "available_oof_years": years,
        "folds": folds,
        "summaries": summaries,
        "selected": selected,
        "decision": decision,
        "gate": "latest R delta>=20 and pitcher bootstrap>=0.90; if multiple folds exist, all-fold worst delta>=5",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "available_oof_years": years,
        "selected": selected,
        "top": summaries[:10] if summaries else [fold["top_candidates"][:5] for fold in folds],
        "decision": decision,
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.oof_dir.resolve(), args.output.resolve(), args.task_type)
