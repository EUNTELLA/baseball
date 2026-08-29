"""R-row gap residual and pocket downshift transfer screen.

This screen uses existing own champion OOF files.  It fits a small residual
model on 2023 R rows, optionally adds a narrow negative pocket correction learned
from the same 2023 rows, and audits transfer to 2024 R rows.
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
TRAIN_YEAR = 2023
VALID_YEAR = 2024
SEEDS = (17, 42, 777)
RESIDUAL_SCALES = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
POCKET_SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
POCKET_CAPS = (0.0025, 0.005, 0.0075, 0.010)
MIN_POCKET_ROWS = (1500, 3000, 5000)
CAT_COLS = ("count", "hand", "base_state", "top_bottom", "pitcher_team_id", "batter_team_id")


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


def sigmoid(values):
    return 1 / (1 + np.exp(-np.asarray(values, dtype=float)))


def bootstrap(ids, target, baseline, candidate, repeats=500, seed=824500):
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
        if len(rows) != len(oof):
            raise ValueError(f"{year} row count mismatch")
        if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
            raise ValueError(f"{year} row_id alignment failed")
        frame = pd.concat(
            [rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type", "pitcher_id"])],
            axis=1,
        )
        frame["season"] = year
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    return result.loc[result["game_type"].astype(str).eq("R")].copy()


def add_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    balls = number(rows, "balls_before", -1).astype(int)
    strikes = number(rows, "strikes_before", -1).astype(int)
    result["count"] = balls.astype(str) + "-" + strikes.astype(str)
    result["hand"] = rows["pitcher_hand"].astype(str) + "_" + rows["batter_hand"].astype(str)
    result["base_state"] = rows["base_state"].astype(str)
    result["top_bottom"] = rows["top_bottom"].astype(str)
    result["pitcher_team_id"] = rows["pitcher_team_id"].astype(str)
    result["batter_team_id"] = rows["batter_team_id"].astype(str)
    pitcher_rate = number(rows, "asof_pitcher_success_rate", 0.5)
    batter_rate = number(rows, "asof_batter_success_rate", 0.5)
    pitcher_n = number(rows, "asof_pitcher_n", 0).clip(lower=0)
    batter_n = number(rows, "asof_batter_n", 0).clip(lower=0)
    pitcher_smooth = (pitcher_n * pitcher_rate + 200 * 0.5) / (pitcher_n + 200)
    batter_smooth = (batter_n * batter_rate + 200 * 0.5) / (batter_n + 200)
    recent1 = number(rows, "asof_pitcher_prev1_game_success_rate", pitcher_rate)
    recent3 = number(rows, "asof_pitcher_prev3_game_success_rate", pitcher_rate)
    recent5 = number(rows, "asof_pitcher_prev5_game_success_rate", pitcher_rate)
    result["anchor_logit"] = logit(rows["p_champion"].to_numpy(float))
    result["p_anchor"] = rows["p_champion"].to_numpy(float)
    result["gap_pitcher_batter"] = pitcher_smooth - batter_smooth
    result["gap_raw"] = pitcher_rate - batter_rate
    result["pitcher_recent1_gap"] = recent1 - pitcher_rate
    result["pitcher_recent3_gap"] = recent3 - pitcher_rate
    result["pitcher_recent5_gap"] = recent5 - pitcher_rate
    result["pitcher_recent1_batter_gap"] = recent1 - batter_smooth
    result["log_pitcher_n"] = np.log1p(pitcher_n)
    result["log_batter_n"] = np.log1p(batter_n)
    result["balls"] = balls
    result["strikes"] = strikes
    result["outs"] = number(rows, "outs_before", 0)
    result["runners"] = number(rows, "num_runners_on", 0)
    result["inning"] = number(rows, "inning", 0)
    result["score_diff"] = number(rows, "score_diff_pitcher_team", 0)
    result["li"] = number(rows, "li", 1.0)
    for column in CAT_COLS:
        result[column] = result[column].astype(str)
    return result


def pocket_key(rows: pd.DataFrame) -> pd.Series:
    balls = number(rows, "balls_before", -1).astype(int).astype(str)
    strikes = number(rows, "strikes_before", -1).astype(int).astype(str)
    hand = rows["pitcher_hand"].astype(str) + "_" + rows["batter_hand"].astype(str)
    base = rows["base_state"].astype(str)
    p_band = pd.cut(
        rows["p_champion"].astype(float),
        [0.0, 0.38, 0.44, 0.50, 0.56, 0.64, 1.0],
        labels=False,
        include_lowest=True,
    ).astype(str)
    gap = number(rows, "asof_pitcher_success_rate", 0.5) - number(rows, "asof_batter_success_rate", 0.5)
    gap_band = pd.cut(
        gap,
        [-2, -0.08, -0.03, 0.03, 0.08, 2],
        labels=False,
        include_lowest=True,
    ).astype(str)
    return balls + "-" + strikes + "|" + hand + "|" + base + "|p" + p_band + "|g" + gap_band


def fit_pockets(train_rows: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    residual = train_rows["target"].to_numpy(float) - train_rows["p_champion"].to_numpy(float)
    table = train_rows.assign(pocket_key=pocket_key(train_rows), residual=residual).groupby(
        "pocket_key", observed=True
    ).agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("p_champion", "mean"),
    )
    table = table.loc[table["rows"].ge(min_rows) & table["residual_mean"].lt(0)].copy()
    table.index = table.index.astype(str)
    return table


def predict_gap_residual(train_x: pd.DataFrame, train_y: np.ndarray,
                         valid_x: pd.DataFrame, task_type: str):
    from catboost import CatBoostRegressor, Pool

    cat_indices = [train_x.columns.get_loc(column) for column in CAT_COLS]
    predictions = []
    timings = []
    for seed in SEEDS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=350,
            learning_rate=0.035,
            depth=3,
            l2_leaf_reg=300.0,
            random_seed=seed,
            task_type=task_type,
            verbose=False,
            allow_writing_files=False,
        )
        start = time.perf_counter()
        model.fit(Pool(train_x, train_y, cat_features=cat_indices))
        timings.append(time.perf_counter() - start)
        predictions.append(model.predict(Pool(valid_x, cat_features=cat_indices)))
        print(f"gap residual seed={seed} sec={timings[-1]:.1f}", flush=True)
    return np.mean(predictions, axis=0), timings


def apply_pockets(valid_rows: pd.DataFrame, pockets: pd.DataFrame,
                  pocket_scale: float, pocket_cap: float) -> np.ndarray:
    if pocket_scale == 0 or pockets.empty:
        return np.zeros(len(valid_rows), dtype=float)
    keys = pocket_key(valid_rows)
    joined = keys.to_frame("pocket_key").merge(pockets, left_on="pocket_key", right_index=True, how="left")
    rows = joined["rows"].fillna(0).to_numpy(float)
    shrink = rows / (rows + 3000.0)
    residual = joined["residual_mean"].fillna(0).to_numpy(float)
    delta = pocket_scale * shrink * residual
    return np.clip(delta, -pocket_cap, 0.0)


def main(train_path: Path, oof_dir: Path, output: Path, task_type: str):
    audit = load_audit_module()
    frame = make_frame(train_path, oof_dir, audit)
    train_rows = frame.loc[frame["season"].eq(TRAIN_YEAR)].copy()
    valid_rows = frame.loc[frame["season"].eq(VALID_YEAR)].copy()
    train_x = add_features(train_rows)
    valid_x = add_features(valid_rows)
    train_y = train_rows["target"].to_numpy(float) - train_rows["p_champion"].to_numpy(float)
    gap_prediction, timings = predict_gap_residual(train_x, train_y, valid_x, task_type)

    baseline = valid_rows["p_champion"].to_numpy(float)
    target = valid_rows["target"].to_numpy(float)
    candidates = []
    pocket_reports = []
    for min_rows in MIN_POCKET_ROWS:
        pockets = fit_pockets(train_rows, min_rows)
        pocket_reports.append({
            "min_rows": min_rows,
            "pockets": int(len(pockets)),
            "top_negative": pockets.sort_values("residual_mean").head(12).reset_index().to_dict("records"),
        })
        for residual_scale in RESIDUAL_SCALES:
            residual_delta = residual_scale * gap_prediction
            for pocket_scale in POCKET_SCALES:
                for pocket_cap in POCKET_CAPS:
                    pocket_delta = apply_pockets(valid_rows, pockets, pocket_scale, pocket_cap)
                    candidate = np.clip(baseline + residual_delta + pocket_delta, 1e-6, 1 - 1e-6)
                    r_delta = audit.bss(target, candidate) - audit.bss(target, baseline)
                    candidates.append({
                        "residual_scale": residual_scale,
                        "pocket_min_rows": min_rows,
                        "pocket_scale": pocket_scale,
                        "pocket_cap": pocket_cap,
                        "fold_2024_r_delta": r_delta,
                        "prediction_mean_delta": float(candidate.mean() - baseline.mean()),
                        "mean_abs_delta": float(np.mean(np.abs(candidate - baseline))),
                        "changed_rows": int(np.count_nonzero(np.abs(candidate - baseline) > 1e-12)),
                        "pocket_changed_rows": int(np.count_nonzero(np.abs(pocket_delta) > 0)),
                        "pitcher_bootstrap_probability": bootstrap(
                            valid_rows["pitcher_id"].to_numpy(), target, baseline, candidate,
                            seed=824500 + int(residual_scale * 10000) + int(pocket_scale * 1000) + min_rows,
                        ),
                    })
    for row in candidates:
        row["passed"] = bool(
            row["fold_2024_r_delta"] > 2.0
            and row["pitcher_bootstrap_probability"] >= 0.80
            and abs(row["prediction_mean_delta"]) <= 0.003
            and row["mean_abs_delta"] <= 0.01
        )
    candidates.sort(key=lambda row: (
        row["passed"], row["fold_2024_r_delta"], row["pitcher_bootstrap_probability"]
    ), reverse=True)
    selected = next((row for row in candidates if row["passed"]), None)
    report = {
        "experiment": "R gap residual plus pocket downshift one-year transfer screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "diagnostic_only": True,
        "train_year": TRAIN_YEAR,
        "valid_year": VALID_YEAR,
        "baseline": "own champion OOF p_champion, R rows only",
        "seeds": list(SEEDS),
        "gap_model_seconds": timings,
        "pocket_reports": pocket_reports,
        "top_candidates": candidates[:30],
        "selected": selected,
        "decision": "build_r_gap_pocket_submission_route" if selected else "keep_rstrict_response_route",
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
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.oof_dir.resolve(), args.output.resolve(), args.task_type)
