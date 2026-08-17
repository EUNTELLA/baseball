"""기준/후보 CatBoost 전체 파이프라인의 중첩 시간 순서 로컬 평가."""
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
from scipy.optimize import brentq, minimize


ID_COL = "row_id"
TARGET_COL = "control_success"
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 7, 2024)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
LABEL_PATH = ROOT / "0816" / "reference_catboost_best" / "recovered_labels.csv.gz"
FEATURE_PATH = ROOT / "0816" / "reference_catboost_best" / "common" / "features.py"

CONFIGS = (
    {"name": "baseline_d6_lr05_l2_1", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0},
    {"name": "candidate_d6_lr03_l2_3", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
)
AUX_CONFIG = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0}


def load_features_module():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "score": float(100000 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
        "mean_error": float(prediction.mean() - rate),
    }


def model_params(config: dict, seed: int, iterations: int, task_type: str, early_stop: bool) -> dict:
    params = {
        "iterations": iterations,
        "learning_rate": config["learning_rate"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "random_seed": seed,
        "verbose": 0,
        "eval_metric": "Logloss",
        "grow_policy": "SymmetricTree",
    }
    if early_stop:
        params["early_stopping_rounds"] = 100
    if task_type == "GPU":
        params.update(task_type="GPU", devices="0")
    else:
        params["thread_count"] = -1
    return params


def engineer(frame: pd.DataFrame, feature_module, train_mask: np.ndarray, target: np.ndarray):
    global_mean = float(target[train_mask].mean())
    result = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        result[column] = result[column].astype(str)
    cat_indices = [result.columns.get_loc(column) for column in feature_module.CAT_COLS]
    return result, cat_indices, global_mean


def train_inner_and_predict(
    features: pd.DataFrame,
    cat_indices: list[int],
    labels: np.ndarray,
    eligible: np.ndarray,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    config: dict,
    task_type: str,
    label: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    train_rows = train_mask & eligible
    calibration_rows = calibration_mask & eligible
    train_pool = Pool(features.loc[train_rows], labels[train_rows], cat_features=cat_indices)
    calibration_pool = Pool(features.loc[calibration_rows], labels[calibration_rows], cat_features=cat_indices)
    prediction_pool = Pool(features.loc[calibration_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**model_params(config, seed, 2000, task_type, True))
        model.fit(train_pool, eval_set=calibration_pool, use_best_model=True)
        predictions.append(model.predict_proba(prediction_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        seconds.append(float(time.perf_counter() - started))
        print(f"{label} inner seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), iterations, seconds


def train_outer_and_predict(
    features: pd.DataFrame,
    cat_indices: list[int],
    labels: np.ndarray,
    eligible: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: dict,
    iterations: list[int],
    task_type: str,
    label: str,
) -> tuple[np.ndarray, list[float]]:
    train_rows = train_mask & eligible
    train_pool = Pool(features.loc[train_rows], labels[train_rows], cat_features=cat_indices)
    validation_pool = Pool(features.loc[validation_mask], cat_features=cat_indices)
    predictions, seconds = [], []
    for seed, iteration in zip(SEEDS, iterations):
        started = time.perf_counter()
        model = CatBoostClassifier(**model_params(config, seed, iteration, task_type, False))
        model.fit(train_pool)
        predictions.append(model.predict_proba(validation_pool)[:, 1])
        seconds.append(float(time.perf_counter() - started))
        print(f"{label} outer seed={seed} iter={iteration} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), seconds


def fit_offset(
    success: np.ndarray,
    mr: np.ndarray,
    wayoff: np.ndarray,
    target: np.ndarray,
    have_labels: np.ndarray,
) -> dict[str, float]:
    selected = have_labels
    z = logit(success[selected])
    u = logit(mr[selected])
    v = logit(wayoff[selected])
    y = target[selected]
    mu_mr, mu_wayoff = float(u.mean()), float(v.mean())
    u, v = u - mu_mr, v - mu_wayoff

    def nll(weights):
        prediction = np.clip(sigmoid(z + weights[0] * u + weights[1] * v), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))

    b, c = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    return {"b": float(b), "c": float(c), "mu_mr": mu_mr, "mu_wayoff": mu_wayoff}


def apply_offset(success: np.ndarray, mr: np.ndarray, wayoff: np.ndarray, offset: dict) -> np.ndarray:
    value = (
        logit(success)
        + offset["b"] * (logit(mr) - offset["mu_mr"])
        + offset["c"] * (logit(wayoff) - offset["mu_wayoff"])
    )
    return sigmoid(value)


def select_alpha_and_forecast(frame: pd.DataFrame, forecast_year: int) -> dict:
    rates = frame.loc[frame["season"] < forecast_year].groupby("season")[TARGET_COL].mean().sort_index()
    backtest_years = [year for year in rates.index if year >= 2022 and year - 2 in rates.index]
    grid = np.round(np.linspace(0, 1, 101), 2)
    if backtest_years:
        losses = []
        for alpha in grid:
            errors = [
                (rates.loc[year - 1] + alpha * (rates.loc[year - 1] - rates.loc[year - 2]) - rates.loc[year]) ** 2
                for year in backtest_years
            ]
            losses.append(float(np.mean(errors)))
        alpha = float(grid[int(np.argmin(losses))])
    else:
        alpha = 0.0
    previous = float(rates.loc[forecast_year - 1])
    before_previous = float(rates.loc[forecast_year - 2])
    forecast = previous + alpha * (previous - before_previous)
    return {
        "alpha": alpha,
        "backtest_years": [int(year) for year in backtest_years],
        "previous_rate": previous,
        "forecast": float(forecast),
    }


def fixed_shift(reference_prediction: np.ndarray, target_mean: float) -> float:
    logits = logit(reference_prediction)
    objective = lambda shift: float(sigmoid(logits + shift).mean()) - target_mean
    return float(brentq(objective, -2, 2))


def write_checkpoint(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    recovered = frame[[ID_COL]].merge(pd.read_csv(LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = ((recovered["middle"] == 1) | (recovered["reverse"] == 1)).fillna(False).astype(int).to_numpy()
    wayoff_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = load_features_module()
    report = {
        "experiment": "nested walk-forward full-pipeline baseline vs tuned CatBoost",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "configs": list(CONFIGS),
        "results": [],
    }

    for validation_year in FOLDS:
        calibration_year = validation_year - 1
        inner_train = (frame["season"] < calibration_year).to_numpy()
        calibration = (frame["season"] == calibration_year).to_numpy()
        outer_train = (frame["season"] < validation_year).to_numpy()
        validation = (frame["season"] == validation_year).to_numpy()
        print(f"\n===== fold {validation_year}: calibrate {calibration_year} =====", flush=True)
        inner_x, inner_ci, inner_mean = engineer(frame, feature_module, inner_train, target)
        outer_x, outer_ci, outer_mean = engineer(frame, feature_module, outer_train, target)

        aux = {}
        for name, labels in (("mr", mr_target), ("wayoff", wayoff_target)):
            inner_prediction, best_iterations, inner_seconds = train_inner_and_predict(
                inner_x, inner_ci, labels, have, inner_train, calibration,
                AUX_CONFIG, task_type, f"fold={validation_year} {name}",
            )
            outer_prediction, outer_seconds = train_outer_and_predict(
                outer_x, outer_ci, labels, have, outer_train, validation,
                AUX_CONFIG, best_iterations, task_type, f"fold={validation_year} {name}",
            )
            aux[name] = {
                "inner_prediction": inner_prediction,
                "outer_prediction": outer_prediction,
                "best_iterations": best_iterations,
                "inner_seconds": inner_seconds,
                "outer_seconds": outer_seconds,
            }

        forecast = select_alpha_and_forecast(frame, validation_year)
        fold_result = {
            "validation_year": validation_year,
            "calibration_year": calibration_year,
            "inner_global_mean": inner_mean,
            "outer_global_mean": outer_mean,
            "rate_forecast": forecast,
            "models": [],
        }
        calibration_target = target[calibration]
        validation_target = target[validation]
        calibration_have = have[calibration]

        for config in CONFIGS:
            inner_success, best_iterations, inner_seconds = train_inner_and_predict(
                inner_x, inner_ci, target, np.ones(len(frame), dtype=bool),
                inner_train, calibration, config, task_type,
                f"fold={validation_year} {config['name']} success",
            )
            outer_success, outer_seconds = train_outer_and_predict(
                outer_x, outer_ci, target, np.ones(len(frame), dtype=bool),
                outer_train, validation, config, best_iterations, task_type,
                f"fold={validation_year} {config['name']} success",
            )
            offset = fit_offset(
                inner_success, aux["mr"]["inner_prediction"], aux["wayoff"]["inner_prediction"],
                calibration_target, calibration_have,
            )
            inner_offset = apply_offset(
                inner_success, aux["mr"]["inner_prediction"], aux["wayoff"]["inner_prediction"], offset,
            )
            outer_offset = apply_offset(
                outer_success, aux["mr"]["outer_prediction"], aux["wayoff"]["outer_prediction"], offset,
            )
            shift = fixed_shift(inner_offset, forecast["forecast"])
            outer_final = sigmoid(logit(outer_offset) + shift)
            model_result = {
                "name": config["name"],
                "success_best_iterations": best_iterations,
                "success_inner_seconds": inner_seconds,
                "success_outer_seconds": outer_seconds,
                "offset": offset,
                "shift": shift,
                "calibration_reference_mean_after_offset": float(inner_offset.mean()),
                "raw": metrics(outer_success, validation_target),
                "offset_applied": metrics(outer_offset, validation_target),
                "final": metrics(outer_final, validation_target),
            }
            fold_result["models"].append(model_result)
            print(
                f"fold={validation_year} {config['name']} "
                f"raw={model_result['raw']['score']:.2f} "
                f"offset={model_result['offset_applied']['score']:.2f} "
                f"final={model_result['final']['score']:.2f} shift={shift:+.5f}",
                flush=True,
            )

        report["results"].append(fold_result)
        write_checkpoint(output, report)
        del inner_x, outer_x, aux
        gc.collect()

    comparisons = []
    for fold in report["results"]:
        base, candidate = fold["models"]
        comparisons.append({
            "validation_year": fold["validation_year"],
            "raw_delta": candidate["raw"]["score"] - base["raw"]["score"],
            "offset_delta": candidate["offset_applied"]["score"] - base["offset_applied"]["score"],
            "final_delta": candidate["final"]["score"] - base["final"]["score"],
            "absolute_mean_error_delta": abs(candidate["final"]["mean_error"]) - abs(base["final"]["mean_error"]),
        })
    final_deltas = [row["final_delta"] for row in comparisons]
    mean_error_deltas = [row["absolute_mean_error_delta"] for row in comparisons]
    passed = (
        float(np.mean(final_deltas)) >= 5.0
        and sum(delta > 0 for delta in final_deltas) >= 2
        and min(final_deltas) >= -3.0
        and comparisons[-1]["final_delta"] > 0
        and float(np.mean(mean_error_deltas)) <= 0
    )
    report["comparison"] = comparisons
    report["summary"] = {
        "mean_final_delta": float(np.mean(final_deltas)),
        "worst_final_delta": float(np.min(final_deltas)),
        "improved_folds": int(sum(delta > 0 for delta in final_deltas)),
        "fold_2024_final_delta": float(comparisons[-1]["final_delta"]),
        "mean_absolute_mean_error_delta": float(np.mean(mean_error_deltas)),
        "decision": "build_single_variable_submission" if passed else "keep_997_baseline",
        "gate": "mean>=+5, wins>=2/3, worst>=-3, 2024>0, mean calibration error not worse",
    }
    write_checkpoint(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
