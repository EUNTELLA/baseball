"""공식 Train OOF 잔차를 이용한 저용량 행 단위 적응형 게이트를 전방 검증한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ID_COL, TARGET_COL = "row_id", "control_success"
OOF_FOLDS = (2021, 2022, 2023, 2024)
EVAL_FOLDS = (2023, 2024)
SEEDS = (42, 7, 2024)
SCALES = (0.25, 0.50, 0.75, 1.0)
CONFIGS = (
    {"name": "d2_i80_l2_100", "depth": 2, "iterations": 80, "l2": 100.0},
    {"name": "d2_i160_l2_100", "depth": 2, "iterations": 160, "l2": 100.0},
    {"name": "d3_i80_l2_100", "depth": 3, "iterations": 80, "l2": 100.0},
    {"name": "d3_i160_l2_100", "depth": 3, "iterations": 160, "l2": 100.0},
    {"name": "d3_i80_l2_30", "depth": 3, "iterations": 80, "l2": 30.0},
    {"name": "d3_i160_l2_30", "depth": 3, "iterations": 160, "l2": 30.0},
)
TARGET_MODES = ("raw", "season_centered")
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("direct_catboost_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str, fallback: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(fallback)


def gate_features(frame: pd.DataFrame, base_prediction: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    balls = numeric(frame, "balls_before", -1).astype(int).astype(str)
    strikes = numeric(frame, "strikes_before", -1).astype(int).astype(str)
    pitcher_hand = frame["pitcher_hand"].astype(str)
    batter_hand = frame["batter_hand"].astype(str)
    career = numeric(frame, "asof_pitcher_success_rate", 0.5)
    recent1 = numeric(frame, "asof_pitcher_prev1_game_success_rate", 0.5)
    recent3 = numeric(frame, "asof_pitcher_prev3_game_success_rate", 0.5)
    recent5 = numeric(frame, "asof_pitcher_prev5_game_success_rate", 0.5)
    result = pd.DataFrame({
        "game_type": frame["game_type"].astype(str),
        "count": balls + "-" + strikes,
        "hand_match": np.where(pitcher_hand.eq(batter_hand), "same", "opposite"),
        "pitcher_hand": pitcher_hand,
        "batter_hand": batter_hand,
        "inning_band": pd.cut(numeric(frame, "inning"), [-np.inf, 3, 6, np.inf], labels=["early", "middle", "late"]).astype(str),
        "runners_state": np.where(numeric(frame, "num_runners_on").gt(0), "on", "empty"),
        "outs": numeric(frame, "outs_before", -1).astype(int).astype(str),
        "base_prediction": np.asarray(base_prediction, float),
        "li": numeric(frame, "li"),
        "score_diff": numeric(frame, "score_diff_pitcher_team"),
        "log_pitcher_n": np.log1p(numeric(frame, "asof_pitcher_n").clip(lower=0)),
        "career_rate": career,
        "recent1_gap": recent1 - career,
        "recent3_gap": recent3 - career,
        "recent5_gap": recent5 - career,
    }, index=frame.index)
    cats = ["game_type", "count", "hand_match", "pitcher_hand", "batter_hand", "inning_band", "runners_state", "outs"]
    result[cats] = result[cats].astype("string").fillna("__MISSING__").astype(str)
    return result, cats


def gate_params(config: dict, task_type: str, seed: int) -> dict:
    result = {"iterations": config["iterations"], "depth": config["depth"],
              "learning_rate": 0.025, "loss_function": "RMSE",
              "l2_leaf_reg": config["l2"], "random_strength": 0.2,
              "bootstrap_type": "Bernoulli", "subsample": 0.8,
              "random_seed": seed, "allow_writing_files": False, "verbose": 0}
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    oof = np.full(len(frame), np.nan, dtype=float)
    report = {"experiment": "independent low-capacity adaptive residual gate",
              "official_train_only": True, "external_code_or_coefficients_used": False,
              "test_aggregate_used": False, "configs": list(CONFIGS), "base_training": [],
              "fold_results": []}

    for fold in OOF_FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league_rate = float(target[train_mask].mean())
        x = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
        for column in feature_module.CAT_COLS:
            x[column] = x[column].astype(str)
        cat_indices = [x.columns.get_loc(column) for column in feature_module.CAT_COLS]
        train_pool = Pool(x.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool = Pool(x.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        members, iterations, seconds = [], [], []
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostClassifier(**base.classifier_params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            members.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            seconds.append(float(time.perf_counter() - started))
            del model
            gc.collect()
        oof[valid_mask] = np.mean(members, axis=0)
        report["base_training"].append({"fold": fold, "best_iterations": iterations,
                                        "seconds": seconds, "metrics": base.metric(oof[valid_mask], target[valid_mask])})
        print(f"base fold={fold} iterations={iterations}", flush=True)
        write_json(output, report)
        del x, train_pool, valid_pool
        gc.collect()

    gate_x, gate_cats = gate_features(frame, oof)
    residual = target.astype(float) - oof
    centered_residual = residual.copy()
    for year in OOF_FOLDS:
        mask = season == year
        centered_residual[mask] -= float(np.mean(centered_residual[mask]))

    for validation_year in EVAL_FOLDS:
        source = np.isin(season, [year for year in OOF_FOLDS if year < validation_year])
        valid = season == validation_year
        source_weight = np.power(0.55, (validation_year - 1) - season[source])
        train_x, valid_x = gate_x.loc[source], gate_x.loc[valid]
        cat_indices = [train_x.columns.get_loc(column) for column in gate_cats]
        baseline = oof[valid]
        baseline_metrics = base.metric(baseline, target[valid])
        candidates = []
        for mode in TARGET_MODES:
            gate_target = residual if mode == "raw" else centered_residual
            train_pool = Pool(train_x, gate_target[source], cat_features=cat_indices, weight=source_weight)
            valid_pool = Pool(valid_x, cat_features=cat_indices)
            for index, config in enumerate(CONFIGS):
                model = CatBoostRegressor(**gate_params(config, task_type, 820900 + validation_year + index))
                model.fit(train_pool)
                correction = model.predict(valid_pool)
                del model
                gc.collect()
                for scale in SCALES:
                    prediction = np.clip(baseline + scale * correction, 1e-6, 1 - 1e-6)
                    metrics = base.metric(prediction, target[valid])
                    centered_prediction = np.clip(prediction - prediction.mean() + target[valid].mean(), 1e-6, 1 - 1e-6)
                    centered_metrics = base.metric(centered_prediction, target[valid])
                    candidates.append({"mode": mode, "config": config["name"], "scale": scale,
                                       "metrics": metrics, "bss_delta": metrics["bss_score"] - baseline_metrics["bss_score"],
                                       "same_mean_delta": centered_metrics["bss_score"] - base.metric(
                                           np.clip(baseline - baseline.mean() + target[valid].mean(), 1e-6, 1 - 1e-6), target[valid]
                                       )["bss_score"],
                                       "correction_mean": float(correction.mean()), "correction_std": float(correction.std())})
        report["fold_results"].append({"validation_year": validation_year,
                                       "source_years": sorted(np.unique(season[source]).astype(int).tolist()),
                                       "baseline": baseline_metrics, "candidates": candidates})
        write_json(output, report)

    summaries = []
    for mode in TARGET_MODES:
        for config in CONFIGS:
            for scale in SCALES:
                rows = [next(row for row in fold["candidates"] if row["mode"] == mode
                             and row["config"] == config["name"] and row["scale"] == scale)
                        for fold in report["fold_results"]]
                deltas = [float(row["bss_delta"]) for row in rows]
                centered = [float(row["same_mean_delta"]) for row in rows]
                summaries.append({"mode": mode, "config": config["name"], "scale": scale,
                                  "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                                  "same_mean_2023_delta": centered[0], "same_mean_2024_delta": centered[1],
                                  "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
                                  "both_raw_and_centered_positive": bool(min(deltas + centered) > 0)})
    stable = [row for row in summaries if row["both_raw_and_centered_positive"] and row["fold_2024_delta"] >= 5.0]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = sorted(summaries, key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    report["selected"] = selected
    report["decision"] = "continue_adaptive_gate_full_pipeline" if selected else "reject_adaptive_residual_gate"
    report["gate"] = "same setting positive raw/same-mean in 2023/2024 and raw 2024 >= +5"
    write_json(output, report)
    print(json.dumps({"selected": selected, "top": report["summaries"][:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
