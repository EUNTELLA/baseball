"""성공·MR·wayoff 확률을 단순합 제약으로 결합하는 시간 순서 선별 실험."""
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
FOLDS = (2022, 2023, 2024)
DEVELOPMENT_FOLDS = (2022, 2023)
CONFIRMATION_FOLD = 2024
SEEDS = (42, 7, 2024)
BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR.parent / "0816" / "reference_catboost_best"
FEATURE_PATH = REFERENCE / "common" / "features.py"
LABEL_PATH = REFERENCE / "recovered_labels.csv.gz"


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


def train_target(
    features: pd.DataFrame,
    cat_indices: list[int],
    labels: np.ndarray,
    eligible: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    task_type: str,
    label: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    train_rows = train_mask & eligible
    valid_rows = valid_mask & eligible
    train_pool = Pool(features.loc[train_rows], labels[train_rows], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_rows], labels[valid_rows], cat_features=cat_indices)
    prediction_pool = Pool(features.loc[valid_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        predictions.append(model.predict_proba(prediction_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        seconds.append(float(time.perf_counter() - started))
        print(
            f"{label} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}",
            flush=True,
        )
        del model
        gc.collect()
    return np.mean(predictions, axis=0), iterations, seconds


def candidate_predictions(success: np.ndarray, mr: np.ndarray, wayoff: np.ndarray) -> dict[str, np.ndarray]:
    complement = np.clip(1 - mr - wayoff, 1e-6, 1 - 1e-6)
    denominator = np.clip(success + mr + wayoff, 1e-6, None)
    normalized = np.clip(success / denominator, 1e-6, 1 - 1e-6)
    candidates = {
        "direct_success": success,
        "failure_complement": complement,
        "simplex_normalized": normalized,
    }
    for weight in BLEND_WEIGHTS:
        candidates[f"blend_normalized_{int(weight * 100):02d}"] = (
            (1 - weight) * success + weight * normalized
        )
        candidates[f"blend_complement_{int(weight * 100):02d}"] = (
            (1 - weight) * success + weight * complement
        )
    return candidates


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    recovered = frame[[ID_COL]].merge(pd.read_csv(LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = (
        ((recovered["middle"] == 1) | (recovered["reverse"] == 1))
        .fillna(False).astype(int).to_numpy()
    )
    wayoff_target = ((target == 0) & (mr_target == 0)).astype(int)
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
        predictions, training = {}, {}
        for name, labels, eligible in (
            ("success", target, np.ones(len(frame), dtype=bool)),
            ("mr", mr_target, have),
            ("wayoff", wayoff_target, have),
        ):
            prediction, iterations, seconds = train_target(
                features, cat_indices, labels, eligible, train_mask, valid_mask,
                task_type, f"fold={fold} {name}",
            )
            predictions[name] = prediction
            training[name] = {"best_iterations": iterations, "seconds": seconds}
        y_valid = target[valid_mask]
        evaluated = []
        for name, prediction in candidate_predictions(
            predictions["success"], predictions["mr"], predictions["wayoff"]
        ).items():
            evaluated.append({
                "name": name,
                "raw": metrics(prediction, y_valid),
                "same_mean": metrics(shift_to_mean(prediction, float(y_valid.mean())), y_valid),
            })
        fold_results.append({
            "fold": fold,
            "global_mean": global_mean,
            "training": training,
            "probability_sum": {
                "mean": float((predictions["success"] + predictions["mr"] + predictions["wayoff"]).mean()),
                "min": float((predictions["success"] + predictions["mr"] + predictions["wayoff"]).min()),
                "max": float((predictions["success"] + predictions["mr"] + predictions["wayoff"]).max()),
            },
            "candidates": evaluated,
        })
        write_json(output, {"status": "running", "fold_results": fold_results})
        del features, predictions
        gc.collect()

    candidate_names = [row["name"] for row in fold_results[0]["candidates"]]
    development = []
    for name in candidate_names:
        raw_deltas, centered_deltas = [], []
        for fold in DEVELOPMENT_FOLDS:
            fold_result = next(row for row in fold_results if row["fold"] == fold)
            baseline = next(row for row in fold_result["candidates"] if row["name"] == "direct_success")
            candidate = next(row for row in fold_result["candidates"] if row["name"] == name)
            raw_deltas.append(candidate["raw"]["score"] - baseline["raw"]["score"])
            centered_deltas.append(candidate["same_mean"]["score"] - baseline["same_mean"]["score"])
        development.append({
            "name": name,
            "raw_mean_delta": float(np.mean(raw_deltas)),
            "raw_worst_delta": float(np.min(raw_deltas)),
            "same_mean_mean_delta": float(np.mean(centered_deltas)),
            "same_mean_worst_delta": float(np.min(centered_deltas)),
        })
    selected = max(
        development,
        key=lambda row: (row["raw_mean_delta"], row["same_mean_mean_delta"], row["name"] == "direct_success"),
    )
    confirmation_fold = next(row for row in fold_results if row["fold"] == CONFIRMATION_FOLD)
    confirmation_baseline = next(
        row for row in confirmation_fold["candidates"] if row["name"] == "direct_success"
    )
    confirmation_candidate = next(
        row for row in confirmation_fold["candidates"] if row["name"] == selected["name"]
    )
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "name": selected["name"],
        "raw_delta": confirmation_candidate["raw"]["score"] - confirmation_baseline["raw"]["score"],
        "same_mean_delta": confirmation_candidate["same_mean"]["score"] - confirmation_baseline["same_mean"]["score"],
        "baseline": confirmation_baseline,
        "candidate": confirmation_candidate,
    }
    passed = (
        selected["name"] != "direct_success"
        and selected["raw_mean_delta"] >= 5.0
        and selected["raw_worst_delta"] >= -2.0
        and selected["same_mean_mean_delta"] > 0
        and confirmation["raw_delta"] >= 5.0
        and confirmation["same_mean_delta"] > 0
    )
    report = {
        "experiment": "CatBoost success/MR/wayoff probability simplex screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "confirmation_fold": CONFIRMATION_FOLD,
        "candidates": candidate_names,
        "fold_results": fold_results,
        "development_selection": development,
        "selected": selected,
        "confirmation": confirmation,
        "decision": "continue_full_pipeline_validation" if passed else "reject_failure_simplex_axis",
        "gate": "dev raw mean>=+5, dev worst>=-2, dev centered>0, 2024 raw>=+5 and centered>0",
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
