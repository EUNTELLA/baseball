"""공식 train만 사용한 CatBoost 소규모 다중 시즌 하이퍼파라미터 선별."""
from __future__ import annotations

import argparse
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
QUICK_SEED = 42
CONFIRM_SEEDS = (7, 2024)
SCRIPT_DIR = Path(__file__).resolve().parent

CONFIGS = [
    {"name": "d6_lr05_l2_1_baseline", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0},
    {"name": "d5_lr05_l2_1", "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 1.0},
    {"name": "d7_lr05_l2_1", "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 1.0},
    {"name": "d6_lr05_l2_3", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"name": "d6_lr05_l2_5", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
    {"name": "d6_lr05_l2_10", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    {"name": "d6_lr03_l2_3", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
]


def load_features_module():
    path = SCRIPT_DIR.parent / "0816" / "reference_catboost_best" / "common" / "features.py"
    spec = importlib.util.spec_from_file_location("official_features", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bss(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def shift_to_mean(prediction: np.ndarray, target_mean: float) -> np.ndarray:
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    objective = lambda shift: float(np.mean(1 / (1 + np.exp(-(logits + shift))))) - target_mean
    shift = brentq(objective, -2.0, 2.0)
    return 1 / (1 + np.exp(-(logits + shift)))


def alpha_diagnostics(frame: pd.DataFrame) -> dict:
    rates = frame.groupby("season")[TARGET_COL].mean().sort_index()
    grid = np.round(np.linspace(0.0, 1.0, 101), 2)

    def mse(alpha: float, folds: tuple[int, ...]) -> float:
        errors = []
        for year in folds:
            forecast = rates.loc[year - 1] + alpha * (rates.loc[year - 1] - rates.loc[year - 2])
            errors.append((forecast - rates.loc[year]) ** 2)
        return float(np.mean(errors))

    losses = [{"alpha": float(alpha), "mse": mse(float(alpha), FOLDS)} for alpha in grid]
    best = min(losses, key=lambda row: (row["mse"], row["alpha"]))
    leave_one_out = []
    for excluded in FOLDS:
        used = tuple(year for year in FOLDS if year != excluded)
        selected = min(grid, key=lambda alpha: (mse(float(alpha), used), alpha))
        leave_one_out.append({"excluded_fold": excluded, "alpha": float(selected)})
    target_2025 = rates.loc[2024] + best["alpha"] * (rates.loc[2024] - rates.loc[2023])
    return {
        "season_rates": {str(int(year)): float(rate) for year, rate in rates.items()},
        "selection_folds": list(FOLDS),
        "best_alpha": best["alpha"],
        "best_mse": best["mse"],
        "leave_one_fold_out": leave_one_out,
        "target_2025": float(target_2025),
        "grid": losses,
    }


def model_params(config: dict, seed: int, task_type: str) -> dict:
    params = {
        "iterations": 2000,
        "learning_rate": config["learning_rate"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "random_seed": seed,
        "verbose": 0,
        "eval_metric": "Logloss",
        "early_stopping_rounds": 100,
        "grow_policy": "SymmetricTree",
    }
    if task_type == "GPU":
        params.update(task_type="GPU", devices="0")
    else:
        params["thread_count"] = -1
    return params


def train_one(
    frame: pd.DataFrame,
    feature_module,
    config: dict,
    fold: int,
    seed: int,
    task_type: str,
) -> dict:
    train_mask = (frame["season"] < fold).to_numpy()
    valid_mask = (frame["season"] == fold).to_numpy()
    target = frame[TARGET_COL].astype(int).to_numpy()
    global_mean = float(target[train_mask].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    started = time.perf_counter()
    model = CatBoostClassifier(**model_params(config, seed, task_type))
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    prediction = model.predict_proba(valid_pool)[:, 1]
    elapsed = time.perf_counter() - started
    y_valid = target[valid_mask]
    return {
        "fold": fold,
        "seed": seed,
        "best_iteration": int(model.get_best_iteration()),
        "seconds": float(elapsed),
        "raw": bss(prediction, y_valid),
        "same_mean": bss(shift_to_mean(prediction, float(y_valid.mean())), y_valid),
        "prediction": prediction,
        "target": y_valid,
    }


def summarize(name: str, runs: list[dict]) -> dict:
    fold_rows = []
    for fold in FOLDS:
        selected = [run for run in runs if run["fold"] == fold]
        prediction = np.mean([run["prediction"] for run in selected], axis=0)
        target = selected[0]["target"]
        fold_rows.append({
            "fold": fold,
            "seeds": [run["seed"] for run in selected],
            "best_iterations": [run["best_iteration"] for run in selected],
            "seconds": [run["seconds"] for run in selected],
            "raw": bss(prediction, target),
            "same_mean": bss(shift_to_mean(prediction, float(target.mean())), target),
        })
    raw_scores = [row["raw"]["score"] for row in fold_rows]
    centered_scores = [row["same_mean"]["score"] for row in fold_rows]
    return {
        "name": name,
        "folds": fold_rows,
        "raw_mean_score": float(np.mean(raw_scores)),
        "raw_worst_score": float(np.min(raw_scores)),
        "same_mean_mean_score": float(np.mean(centered_scores)),
        "same_mean_worst_score": float(np.min(centered_scores)),
    }


def strip_arrays(runs: list[dict]) -> list[dict]:
    return [{key: value for key, value in run.items() if key not in {"prediction", "target"}} for run in runs]


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    feature_module = load_features_module()
    all_runs: dict[str, list[dict]] = {config["name"]: [] for config in CONFIGS}

    print("--- 1단계: 7개 설정 × 3개 시즌 × 1시드 ---", flush=True)
    for config in CONFIGS:
        for fold in FOLDS:
            run = train_one(frame, feature_module, config, fold, QUICK_SEED, task_type)
            all_runs[config["name"]].append(run)
            print(
                f"{config['name']} fold={fold} iter={run['best_iteration']} "
                f"raw={run['raw']['score']:.2f} centered={run['same_mean']['score']:.2f} "
                f"sec={run['seconds']:.1f}", flush=True,
            )

    quick = [summarize(config["name"], all_runs[config["name"]]) for config in CONFIGS]
    ranked = sorted(
        quick,
        key=lambda row: (row["same_mean_mean_score"], row["same_mean_worst_score"]),
        reverse=True,
    )
    baseline_name = "d6_lr05_l2_1_baseline"
    challenger = next(row["name"] for row in ranked if row["name"] != baseline_name)
    finalists = [baseline_name, challenger]
    print(f"--- 2단계: 기준과 최상위 도전자 3시드 확인: {finalists} ---", flush=True)
    config_by_name = {config["name"]: config for config in CONFIGS}
    for name in finalists:
        for seed in CONFIRM_SEEDS:
            for fold in FOLDS:
                run = train_one(frame, feature_module, config_by_name[name], fold, seed, task_type)
                all_runs[name].append(run)
                print(
                    f"{name} fold={fold} seed={seed} iter={run['best_iteration']} "
                    f"raw={run['raw']['score']:.2f} centered={run['same_mean']['score']:.2f}",
                    flush=True,
                )

    confirmed = [summarize(name, all_runs[name]) for name in finalists]
    baseline = next(row for row in confirmed if row["name"] == baseline_name)
    winner = next(row for row in confirmed if row["name"] == challenger)
    delta_mean = winner["same_mean_mean_score"] - baseline["same_mean_mean_score"]
    delta_worst = winner["same_mean_worst_score"] - baseline["same_mean_worst_score"]
    raw_delta = winner["raw_mean_score"] - baseline["raw_mean_score"]
    passed = delta_mean > 3.0 and delta_worst >= -2.0 and raw_delta >= -10.0
    report = {
        "experiment": "CatBoost FE10 multi-season small hyperparameter screen",
        "official_train_only": True,
        "folds": list(FOLDS),
        "quick_seed": QUICK_SEED,
        "confirm_seeds": [QUICK_SEED, *CONFIRM_SEEDS],
        "configs": CONFIGS,
        "alpha_diagnostics": alpha_diagnostics(frame),
        "quick_results": quick,
        "finalists": finalists,
        "confirmed_results": confirmed,
        "baseline_for_gate": baseline,
        "winner": winner["name"] if passed else baseline_name,
        "challenger_deltas_vs_confirmed_baseline": {
            "same_mean_mean": float(delta_mean),
            "same_mean_worst": float(delta_worst),
            "raw_mean": float(raw_delta),
        },
        "decision": "continue_to_7_seed_build" if passed else "keep_current_model",
        "gate": "same-mean mean > +3, same-mean worst >= -2, raw mean >= -10",
        "individual_runs": {name: strip_arrays(runs) for name, runs in all_runs.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2))
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("winner", "challenger_deltas_vs_confirmed_baseline", "decision")}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
