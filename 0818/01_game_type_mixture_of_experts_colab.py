"""공통 CatBoost와 R/F 경기유형 전문가 모델의 시간 순서 선별."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import brentq


ID_COL = "row_id"
TARGET_COL = "control_success"
TYPE_COL = "game_type"
GAME_TYPES = ("R", "F")
FOLDS = (2021, 2022, 2024)
DEVELOPMENT_FOLDS = (2021, 2022)
CONFIRMATION_FOLD = 2024
SEEDS = (42, 7, 2024)
EXPERT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_TRAIN_ROWS = 10_000
MIN_VALID_ROWS = 2_000
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


def params(seed: int, task_type: str) -> dict:
    result = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": 0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "early_stopping_rounds": 100,
        "grow_policy": "SymmetricTree",
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def train_models(
    features: pd.DataFrame,
    cat_indices: list[int],
    target: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    prediction_mask: np.ndarray,
    task_type: str,
    label: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    prediction_pool = Pool(features.loc[prediction_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        predictions.append(model.predict_proba(prediction_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        seconds.append(float(time.perf_counter() - started))
        print(f"{label} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), iterations, seconds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    feature_module = load_features_module()
    fold_results = []

    for fold in FOLDS:
        print(f"\n===== fold {fold} =====", flush=True)
        train_mask = (frame["season"] < fold).to_numpy()
        valid_mask = (frame["season"] == fold).to_numpy()
        global_mean = float(target[train_mask].mean())
        features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
        for column in feature_module.CAT_COLS:
            features[column] = features[column].astype(str)
        cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]

        global_prediction, global_iterations, global_seconds = train_models(
            features, cat_indices, target, train_mask, valid_mask, valid_mask,
            task_type, f"fold={fold} global",
        )
        routed_expert = global_prediction.copy()
        expert_training = {}
        validation_types = frame.loc[valid_mask, TYPE_COL].astype(str).to_numpy()
        for game_type in GAME_TYPES:
            type_train = train_mask & (frame[TYPE_COL].astype(str).to_numpy() == game_type)
            type_valid = valid_mask & (frame[TYPE_COL].astype(str).to_numpy() == game_type)
            train_rows, valid_rows = int(type_train.sum()), int(type_valid.sum())
            if train_rows < MIN_TRAIN_ROWS or valid_rows < MIN_VALID_ROWS:
                expert_training[game_type] = {
                    "status": "fallback_global",
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                }
                continue
            prediction, iterations, seconds = train_models(
                features, cat_indices, target, type_train, type_valid, type_valid,
                task_type, f"fold={fold} expert={game_type}",
            )
            routed_expert[validation_types == game_type] = prediction
            expert_training[game_type] = {
                "status": "trained",
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "best_iterations": iterations,
                "seconds": seconds,
            }

        y_valid = target[valid_mask]
        candidates = []
        for weight in EXPERT_WEIGHTS:
            prediction = (1 - weight) * global_prediction + weight * routed_expert
            overall_raw = metrics(prediction, y_valid)
            overall_centered = metrics(shift_to_mean(prediction, float(y_valid.mean())), y_valid)
            by_type = {}
            for game_type in GAME_TYPES:
                selected = validation_types == game_type
                by_type[game_type] = {
                    "rows": int(selected.sum()),
                    "raw": metrics(prediction[selected], y_valid[selected]),
                    "global_raw": metrics(global_prediction[selected], y_valid[selected]),
                }
            candidates.append({
                "expert_weight": float(weight),
                "raw": overall_raw,
                "same_mean": overall_centered,
                "by_type": by_type,
            })
        fold_results.append({
            "fold": fold,
            "global_mean": global_mean,
            "global_training": {
                "best_iterations": global_iterations,
                "seconds": global_seconds,
            },
            "expert_training": expert_training,
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "fold_results": fold_results})
        del features, routed_expert
        gc.collect()

    development = []
    for weight in EXPERT_WEIGHTS:
        raw_deltas, centered_deltas, type_deltas = [], [], []
        for fold in DEVELOPMENT_FOLDS:
            fold_result = next(row for row in fold_results if row["fold"] == fold)
            baseline = next(row for row in fold_result["candidates"] if row["expert_weight"] == 0.0)
            candidate = next(row for row in fold_result["candidates"] if row["expert_weight"] == weight)
            raw_deltas.append(candidate["raw"]["score"] - baseline["raw"]["score"])
            centered_deltas.append(candidate["same_mean"]["score"] - baseline["same_mean"]["score"])
            for game_type in GAME_TYPES:
                type_deltas.append(
                    candidate["by_type"][game_type]["raw"]["score"]
                    - candidate["by_type"][game_type]["global_raw"]["score"]
                )
        development.append({
            "expert_weight": float(weight),
            "raw_mean_delta": float(np.mean(raw_deltas)),
            "raw_worst_delta": float(np.min(raw_deltas)),
            "same_mean_mean_delta": float(np.mean(centered_deltas)),
            "type_worst_delta": float(np.min(type_deltas)),
        })
    selected = max(
        development,
        key=lambda row: (row["raw_mean_delta"], row["same_mean_mean_delta"], -row["expert_weight"]),
    )
    confirmation_fold = next(row for row in fold_results if row["fold"] == CONFIRMATION_FOLD)
    confirmation_baseline = next(
        row for row in confirmation_fold["candidates"] if row["expert_weight"] == 0.0
    )
    confirmation_candidate = next(
        row for row in confirmation_fold["candidates"] if row["expert_weight"] == selected["expert_weight"]
    )
    confirmation_type_deltas = {
        game_type: (
            confirmation_candidate["by_type"][game_type]["raw"]["score"]
            - confirmation_candidate["by_type"][game_type]["global_raw"]["score"]
        )
        for game_type in GAME_TYPES
    }
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "expert_weight": selected["expert_weight"],
        "raw_delta": confirmation_candidate["raw"]["score"] - confirmation_baseline["raw"]["score"],
        "same_mean_delta": (
            confirmation_candidate["same_mean"]["score"]
            - confirmation_baseline["same_mean"]["score"]
        ),
        "type_deltas": confirmation_type_deltas,
        "baseline": confirmation_baseline,
        "candidate": confirmation_candidate,
    }
    passed = (
        selected["expert_weight"] > 0
        and selected["raw_mean_delta"] >= 5.0
        and selected["raw_worst_delta"] >= -2.0
        and selected["same_mean_mean_delta"] > 0
        and selected["type_worst_delta"] >= -5.0
        and confirmation["raw_delta"] >= 10.0
        and confirmation["same_mean_delta"] > 0
        and min(confirmation_type_deltas.values()) >= -5.0
    )
    report = {
        "experiment": "CatBoost global + game_type R/F mixture of experts",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": list(FOLDS),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "excluded_transition_fold": 2023,
        "confirmation_fold": CONFIRMATION_FOLD,
        "seeds": list(SEEDS),
        "expert_weights": list(EXPERT_WEIGHTS),
        "fold_results": fold_results,
        "development_selection": development,
        "selected": selected,
        "confirmation": confirmation,
        "decision": "continue_full_pipeline_validation" if passed else "reject_game_type_expert_axis",
        "gate": "dev mean>=+5, worst>=-2, centered>0, type worst>=-5; 2024 raw>=+10, centered>0, type worst>=-5",
    }
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("selected", "confirmation", "decision")}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
