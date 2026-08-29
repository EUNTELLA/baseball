"""CatBoost d6·FE10 기준 대비 선발/불펜 역할 피처 2개 선별 검증."""
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


SEEDS = [42, 7, 2024, 99, 1, 123, 777]
ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = Path(__file__).resolve().parent
ROLE_PRIOR = 100.0


def load_common():
    path = SCRIPT_DIR / "06_submit012_league_baseline_colab.py"
    spec = importlib.util.spec_from_file_location("common_fe", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_role_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    early = (frame["inning"] <= 3).astype(float)
    global_early_rate = float(early.mean())
    stats = (
        frame.assign(_early=early)
        .groupby("pitcher_id", as_index=False)
        .agg(role_n=("_early", "size"), role_early=("_early", "sum"))
    )
    stats["pitcher_role_score"] = (
        stats["role_early"] + ROLE_PRIOR * global_early_rate
    ) / (stats["role_n"] + ROLE_PRIOR)
    return stats[["pitcher_id", "pitcher_role_score"]], global_early_rate


def add_role_features(
    features: pd.DataFrame,
    raw_frame: pd.DataFrame,
    role_table: pd.DataFrame,
    default_role: float,
) -> pd.DataFrame:
    result = features.copy()
    role_map = role_table.set_index("pitcher_id")["pitcher_role_score"]
    result["pitcher_role_score"] = (
        raw_frame["pitcher_id"].map(role_map).fillna(default_role).to_numpy()
    )
    # 선발 성향 1이면 예상 등판 이닝 1, 불펜 성향 0이면 예상 등판 이닝 8.
    expected_entry_inning = 8.0 - 7.0 * result["pitcher_role_score"]
    result["inning_over_role"] = raw_frame["inning"].to_numpy() - expected_entry_inning
    return result


def logit_shift_to_mean(prediction: np.ndarray, target_mean: float) -> np.ndarray:
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    logits = np.log(prediction / (1 - prediction))
    objective = lambda shift: float(np.mean(1 / (1 + np.exp(-(logits + shift))))) - target_mean
    shift = brentq(objective, -2.0, 2.0)
    return 1 / (1 + np.exp(-(logits + shift)))


def params(task_type: str):
    result = dict(
        iterations=2000, learning_rate=0.05, depth=6,
        verbose=100, eval_metric="Logloss", early_stopping_rounds=100,
        grow_policy="SymmetricTree",
    )
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result.update(thread_count=-1)
    return result


def train_predictions(
    x: pd.DataFrame,
    target: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    cat_cols: list[str],
    task_type: str,
    label: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    cat_indices = [x.columns.get_loc(column) for column in cat_cols]
    train_pool = Pool(x.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(x.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params(task_type), random_seed=seed)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        predictions.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(int(model.get_best_iteration()))
        seconds.append(float(time.perf_counter() - started))
        print(f"{label} seed={seed} iter={iterations[-1]} seconds={seconds[-1]:.1f}", flush=True)
    return np.mean(predictions, axis=0), iterations, seconds


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_common()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    train_mask = (frame["season"] <= 2023).to_numpy()
    valid_mask = (frame["season"] == 2024).to_numpy()
    global_mean = float(target[train_mask].mean())
    raw_features = frame.drop(columns=[ID_COL, TARGET_COL])
    base_x = common.engineer(raw_features, global_mean)
    for column in common.CAT_COLS:
        base_x[column] = base_x[column].astype(str)

    role_table, default_role = build_role_table(raw_features.loc[train_mask])
    role_x = add_role_features(base_x, raw_features, role_table, default_role)

    base_prediction, base_iterations, base_seconds = train_predictions(
        base_x, target, train_mask, valid_mask, common.CAT_COLS, task_type, "base"
    )
    role_prediction, role_iterations, role_seconds = train_predictions(
        role_x, target, train_mask, valid_mask, common.CAT_COLS, task_type, "role"
    )
    y_valid = target[valid_mask]
    actual_mean = float(y_valid.mean())
    base_raw = common.bss(base_prediction, y_valid)
    role_raw = common.bss(role_prediction, y_valid)
    base_centered = common.bss(logit_shift_to_mean(base_prediction, actual_mean), y_valid)
    role_centered = common.bss(logit_shift_to_mean(role_prediction, actual_mean), y_valid)
    raw_delta = role_raw["score"] - base_raw["score"]
    centered_delta = role_centered["score"] - base_centered["score"]
    payload = {
        "experiment": "CatBoost d6 FE10 vs + pitcher role features",
        "role_definition": {
            "pitcher_role_score": "smoothed share of pitches in innings 1-3, training rows only",
            "inning_over_role": "inning - (8 - 7 * pitcher_role_score)",
            "prior_strength": ROLE_PRIOR,
            "default_role": default_role,
        },
        "seeds": SEEDS,
        "base_best_iterations": base_iterations,
        "role_best_iterations": role_iterations,
        "base_seconds": base_seconds,
        "role_seconds": role_seconds,
        "base_raw": base_raw,
        "role_raw": role_raw,
        "raw_score_delta": raw_delta,
        "base_same_mean": base_centered,
        "role_same_mean": role_centered,
        "same_mean_score_delta": centered_delta,
        "decision": "continue_to_build" if centered_delta > 5.0 and raw_delta > 0 else "reject",
        "rule": "continue only when raw delta > 0 and same-mean delta > 5",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
