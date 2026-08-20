"""CatBoost 분류와 Brier 직접 최적화 RMSE 회귀의 시간 순서 비교."""
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
from scipy.optimize import brentq


ID_COL = "row_id"
TARGET_COL = "control_success"
FOLDS = (2022, 2023, 2024)
DEVELOPMENT_FOLDS = (2022, 2023)
CONFIRMATION_FOLD = 2024
SEEDS = (42, 7, 2024)
WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 1)
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_PATH = SCRIPT_DIR.parent / "common" / "model_features.py"


def load_features_module():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "score": float(100000 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def shift_to_mean(prediction: np.ndarray, target_mean: float) -> np.ndarray:
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    logits = np.log(prediction / (1 - prediction))
    objective = lambda shift: float(np.mean(1 / (1 + np.exp(-(logits + shift))))) - target_mean
    shift = brentq(objective, -2, 2)
    return 1 / (1 + np.exp(-(logits + shift)))


def common_params(seed: int, task_type: str) -> dict:
    result = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "random_seed": seed,
        "verbose": 0,
        "early_stopping_rounds": 100,
        "grow_policy": "SymmetricTree",
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def train_fold(
    frame: pd.DataFrame,
    feature_module,
    fold: int,
    task_type: str,
) -> dict:
    target = frame[TARGET_COL].astype(int).to_numpy()
    train_mask = (frame["season"] < fold).to_numpy()
    valid_mask = (frame["season"] == fold).to_numpy()
    global_mean = float(target[train_mask].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)

    classifier_predictions, regressor_predictions = [], []
    classifier_iterations, regressor_iterations = [], []
    classifier_seconds, regressor_seconds = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        classifier = CatBoostClassifier(
            **common_params(seed, task_type), loss_function="Logloss", eval_metric="Logloss"
        )
        classifier.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        classifier_predictions.append(classifier.predict_proba(valid_pool)[:, 1])
        classifier_iterations.append(max(1, int(classifier.get_best_iteration()) + 1))
        classifier_seconds.append(float(time.perf_counter() - started))
        print(
            f"fold={fold} classifier seed={seed} iter={classifier_iterations[-1]} "
            f"sec={classifier_seconds[-1]:.1f}", flush=True,
        )
        del classifier
        gc.collect()

        started = time.perf_counter()
        regressor = CatBoostRegressor(
            **common_params(seed, task_type), loss_function="RMSE", eval_metric="RMSE"
        )
        regressor.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        regressor_predictions.append(np.clip(regressor.predict(valid_pool), 1e-6, 1 - 1e-6))
        regressor_iterations.append(max(1, int(regressor.get_best_iteration()) + 1))
        regressor_seconds.append(float(time.perf_counter() - started))
        print(
            f"fold={fold} regressor seed={seed} iter={regressor_iterations[-1]} "
            f"sec={regressor_seconds[-1]:.1f}", flush=True,
        )
        del regressor
        gc.collect()

    classifier_prediction = np.mean(classifier_predictions, axis=0)
    regressor_prediction = np.mean(regressor_predictions, axis=0)
    y_valid = target[valid_mask]
    weight_results = []
    for weight in WEIGHTS:
        prediction = (1 - weight) * classifier_prediction + weight * regressor_prediction
        weight_results.append({
            "regressor_weight": float(weight),
            "raw": metrics(prediction, y_valid),
            "same_mean": metrics(shift_to_mean(prediction, float(y_valid.mean())), y_valid),
        })
    return {
        "fold": fold,
        "global_mean": global_mean,
        "classifier_best_iterations": classifier_iterations,
        "regressor_best_iterations": regressor_iterations,
        "classifier_seconds": classifier_seconds,
        "regressor_seconds": regressor_seconds,
        "weights": weight_results,
    }


def result_at(fold_result: dict, weight: float) -> dict:
    return next(row for row in fold_result["weights"] if row["regressor_weight"] == weight)


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    feature_module = load_features_module()
    fold_results = []
    for fold in FOLDS:
        fold_results.append(train_fold(frame, feature_module, fold, task_type))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps({"status": "running", "fold_results": fold_results}, ensure_ascii=False, indent=2))
            file.write("\n")

    development = []
    for weight in WEIGHTS:
        deltas = []
        centered_deltas = []
        for fold in DEVELOPMENT_FOLDS:
            fold_result = next(row for row in fold_results if row["fold"] == fold)
            baseline = result_at(fold_result, 0.0)
            candidate = result_at(fold_result, float(weight))
            deltas.append(candidate["raw"]["score"] - baseline["raw"]["score"])
            centered_deltas.append(candidate["same_mean"]["score"] - baseline["same_mean"]["score"])
        development.append({
            "regressor_weight": float(weight),
            "raw_mean_delta": float(np.mean(deltas)),
            "raw_worst_delta": float(np.min(deltas)),
            "same_mean_mean_delta": float(np.mean(centered_deltas)),
            "same_mean_worst_delta": float(np.min(centered_deltas)),
        })
    selected = max(
        development,
        key=lambda row: (row["raw_mean_delta"], row["same_mean_mean_delta"], -row["regressor_weight"]),
    )
    selected_weight = selected["regressor_weight"]
    confirmation_result = next(row for row in fold_results if row["fold"] == CONFIRMATION_FOLD)
    confirmation_baseline = result_at(confirmation_result, 0.0)
    confirmation_candidate = result_at(confirmation_result, selected_weight)
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "regressor_weight": selected_weight,
        "raw_delta": confirmation_candidate["raw"]["score"] - confirmation_baseline["raw"]["score"],
        "same_mean_delta": confirmation_candidate["same_mean"]["score"] - confirmation_baseline["same_mean"]["score"],
        "baseline": confirmation_baseline,
        "candidate": confirmation_candidate,
    }
    passed = (
        selected_weight > 0
        and selected["raw_mean_delta"] >= 3.0
        and selected["raw_worst_delta"] >= -2.0
        and selected["same_mean_mean_delta"] > 0
        and confirmation["raw_delta"] >= 3.0
        and confirmation["same_mean_delta"] > 0
    )
    report = {
        "experiment": "CatBoost Logloss classifier + Brier RMSE regressor blend",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": list(FOLDS),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "confirmation_fold": CONFIRMATION_FOLD,
        "seeds": list(SEEDS),
        "weights": [float(weight) for weight in WEIGHTS],
        "fold_results": fold_results,
        "development_selection": development,
        "selected": selected,
        "confirmation": confirmation,
        "decision": "continue_full_pipeline_validation" if passed else "reject_brier_regression_axis",
        "gate": "dev raw mean>=+3, dev worst>=-2, dev centered>0, 2024 raw>=+3 and centered>0",
    }
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2))
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("selected", "confirmation", "decision")}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
