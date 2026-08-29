"""공식 as-of 기록으로 만든 동적 투수 기준확률의 CatBoost 잔차 모델을 선별한다."""
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
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 7, 2024)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FEATURE_PATH = ROOT / "common" / "model_features.py"


def load_features_module():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "bss_score": float(100000 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def numeric(frame: pd.DataFrame, column: str, fallback: float) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(fallback).to_numpy(float)


def dynamic_base(frame: pd.DataFrame, league_rate: float) -> np.ndarray:
    n = np.clip(numeric(frame, "asof_pitcher_n", 0.0), 0.0, None)
    career_rate = np.clip(numeric(frame, "asof_pitcher_success_rate", league_rate), 0.0, 1.0)
    career = (n * career_rate + 75.0 * league_rate) / (n + 75.0)
    recent_columns = (
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    )
    recent_members = []
    for column in recent_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        recent_members.append(np.where(np.isfinite(values), values, career))
    recent = 0.50 * recent_members[0] + 0.30 * recent_members[1] + 0.20 * recent_members[2]
    recent_reliability = n / (n + 300.0)
    base = career + 0.15 * recent_reliability * (recent - career)
    return np.clip(base, 1e-5, 1 - 1e-5)


def classifier_params(seed: int, task_type: str) -> dict:
    result = {
        "iterations": 2000, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 1.0, "random_seed": seed, "verbose": 0,
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "early_stopping_rounds": 100, "grow_policy": "SymmetricTree",
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def residual_params(seed: int, task_type: str) -> dict:
    result = {
        "iterations": 600, "depth": 8, "learning_rate": 0.035,
        "l2_leaf_reg": 20.0, "random_strength": 0.35,
        "bootstrap_type": "Bernoulli", "subsample": 0.85,
        "loss_function": "RMSE", "random_seed": seed,
        "allow_writing_files": False, "verbose": 0,
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    feature_module = load_features_module()
    report = {
        "experiment": "official-asof dynamic pitcher baseline residual CatBoost",
        "official_train_only": True, "external_data_used": False,
        "test_aggregate_used": False, "folds": list(FOLDS),
        "seeds": list(SEEDS), "fold_results": [],
    }
    pooled_target, pooled_baseline, pooled_candidate = [], [], []

    for fold in FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league_rate = float(target[train_mask].mean())
        base = dynamic_base(frame, league_rate)
        features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
        features["dynamic_pitcher_base"] = base
        for column in feature_module.CAT_COLS:
            features[column] = features[column].astype(str)
        cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
        train_pool_class = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool_class = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        decay_weight = np.power(0.55, (fold - 1) - season[train_mask])
        residual_target = target.astype(float) - base
        train_pool_residual = Pool(
            features.loc[train_mask], residual_target[train_mask],
            cat_features=cat_indices, weight=decay_weight,
        )
        valid_pool_residual = Pool(features.loc[valid_mask], cat_features=cat_indices)
        classifier_members, residual_members, training = [], [], []
        for seed in SEEDS:
            started = time.perf_counter()
            classifier = CatBoostClassifier(**classifier_params(seed, task_type))
            classifier.fit(train_pool_class, eval_set=valid_pool_class, use_best_model=True)
            classifier_prediction = classifier.predict_proba(valid_pool_class)[:, 1]
            classifier_iteration = max(1, int(classifier.get_best_iteration()) + 1)
            del classifier
            gc.collect()
            residual = CatBoostRegressor(**residual_params(seed, task_type))
            residual.fit(train_pool_residual)
            residual_prediction = np.clip(base[valid_mask] + residual.predict(valid_pool_residual), 1e-6, 1 - 1e-6)
            seconds = float(time.perf_counter() - started)
            classifier_members.append(classifier_prediction)
            residual_members.append(residual_prediction)
            training.append({"seed": seed, "classifier_best_iteration": classifier_iteration, "seconds": seconds})
            print(f"fold={fold} seed={seed} classifier_iter={classifier_iteration} sec={seconds:.1f}", flush=True)
            del residual
            gc.collect()
        baseline_prediction = np.mean(classifier_members, axis=0)
        candidate_prediction = np.mean(residual_members, axis=0)
        y_valid = target[valid_mask]
        baseline_metrics = metric(baseline_prediction, y_valid)
        candidate_metrics = metric(candidate_prediction, y_valid)
        error_correlation = float(np.corrcoef(
            y_valid - baseline_prediction, y_valid - candidate_prediction
        )[0, 1])
        result = {
            "fold": fold, "league_rate": league_rate, "training": training,
            "baseline_direct_classifier": baseline_metrics,
            "candidate_dynamic_base_residual": candidate_metrics,
            "bss_delta": candidate_metrics["bss_score"] - baseline_metrics["bss_score"],
            "error_correlation": error_correlation,
            "base_mean": float(base[valid_mask].mean()),
        }
        report["fold_results"].append(result)
        pooled_target.append(y_valid)
        pooled_baseline.append(baseline_prediction)
        pooled_candidate.append(candidate_prediction)
        write_json(output, report)
        print(f"fold={fold} BSS delta={result['bss_delta']:+.2f} corr={error_correlation:.6f}", flush=True)
        del features, train_pool_class, valid_pool_class, train_pool_residual, valid_pool_residual
        gc.collect()

    pooled_y = np.concatenate(pooled_target)
    pooled_base = np.concatenate(pooled_baseline)
    pooled_new = np.concatenate(pooled_candidate)
    deltas = [float(row["bss_delta"]) for row in report["fold_results"]]
    pooled_base_metrics = metric(pooled_base, pooled_y)
    pooled_candidate_metrics = metric(pooled_new, pooled_y)
    passed = min(deltas) > 0 and deltas[-1] >= 5.0
    report["summary"] = {
        "fold_deltas": deltas, "mean_delta": float(np.mean(deltas)),
        "worst_delta": float(np.min(deltas)),
        "pooled_baseline": pooled_base_metrics,
        "pooled_candidate": pooled_candidate_metrics,
        "pooled_delta": pooled_candidate_metrics["bss_score"] - pooled_base_metrics["bss_score"],
        "decision": "continue_dynamic_baseline_multichannel" if passed else "reject_dynamic_baseline_residual",
        "gate": "positive in 2022/2023/2024 and 2024 >= +5",
    }
    write_json(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
