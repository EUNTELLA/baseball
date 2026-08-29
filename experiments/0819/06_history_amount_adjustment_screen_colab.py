"""투수의 누적 투구 수에 따라 현재 3종 보정의 적용량을 조절한다."""
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


ID_COL = "row_id"
TARGET_COL = "control_success"
OOF_FOLDS = (2021, 2022, 2023, 2024)
EVAL_FOLDS = (2023, 2024)
SEEDS = (42, 7, 2024)
K_VALUES = (0.0, 50.0, 100.0, 300.0, 500.0, 1000.0)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "01_catboost_residual_differential_screen_colab.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("error_adjustment_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def history_factor(history_count: np.ndarray, k: float) -> np.ndarray:
    if k == 0:
        return np.ones(len(history_count), dtype=float)
    count = np.nan_to_num(np.asarray(history_count, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    count = np.maximum(count, 0.0)
    return count / (count + k)


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    history_count = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(dtype=float)
    condition_values = base.contexts(frame)
    feature_module = base.load_feature_module()
    oof_prediction = np.full(len(frame), np.nan, dtype=float)
    training = []

    for fold in OOF_FOLDS:
        train_mask = season < fold
        valid_mask = season == fold
        global_mean = float(target[train_mask].mean())
        features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
        for column in feature_module.CAT_COLS:
            features[column] = features[column].astype(str)
        cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
        train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        predictions, iterations, seconds = [], [], []
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostClassifier(**base.params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            predictions.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            seconds.append(float(time.perf_counter() - started))
            print(f"fold={fold} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
            del model
            gc.collect()
        oof_prediction[valid_mask] = np.mean(predictions, axis=0)
        training.append({
            "fold": fold,
            "best_iterations": iterations,
            "seconds": seconds,
            "baseline": base.metrics(oof_prediction[valid_mask], target[valid_mask]),
        })
        write_json(output, {"status": "running", "training": training})
        del features, train_pool, valid_pool, predictions
        gc.collect()

    residual = target.astype(float) - oof_prediction
    fold_results = []
    for fold in EVAL_FOLDS:
        source = np.isin(season, (fold - 2, fold - 1))
        valid = season == fold
        additions = {}
        for name, shrinkage in base.AXES:
            table = base.differential_table(
                pitcher, condition_values[name], residual, source, shrinkage
            )
            additions[name] = base.apply_table(
                table, pitcher, condition_values[name], valid
            )
        full_adjustment = additions["hand"] + additions["two_strikes"] + additions["runners_on"]
        current_prediction = np.clip(oof_prediction[valid] + full_adjustment, 1e-6, 1 - 1e-6)
        current_metrics = base.metrics(current_prediction, target[valid])
        candidates = []
        for k in K_VALUES:
            factor = history_factor(history_count[valid], k)
            prediction = np.clip(oof_prediction[valid] + factor * full_adjustment, 1e-6, 1 - 1e-6)
            candidate_metrics = base.metrics(prediction, target[valid])
            candidates.append({
                "k": k,
                "factor_mean": float(factor.mean()),
                "factor_min": float(factor.min()),
                "factor_max": float(factor.max()),
                "metrics": candidate_metrics,
                "bss_delta_vs_current": candidate_metrics["bss_score"] - current_metrics["bss_score"],
            })
        fold_results.append({
            "fold": fold,
            "source_seasons": [fold - 2, fold - 1],
            "current_three_adjustments": current_metrics,
            "history_count": {
                "missing": int(np.isnan(history_count[valid]).sum()),
                "median": float(np.nanmedian(history_count[valid])),
                "max": float(np.nanmax(history_count[valid])),
            },
            "candidates": candidates,
        })

    summaries = []
    for k in K_VALUES:
        rows = [next(item for item in fold["candidates"] if item["k"] == k) for fold in fold_results]
        deltas = [float(row["bss_delta_vs_current"]) for row in rows]
        summaries.append({
            "k": k,
            "fold_2023_delta": deltas[0],
            "fold_2024_delta": deltas[1],
            "mean_delta": float(np.mean(deltas)),
            "worst_delta": float(np.min(deltas)),
            "both_positive": bool(min(deltas) > 0),
        })
    stable = [row for row in summaries if row["k"] > 0 and row["both_positive"]]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    passed = selected is not None and selected["fold_2024_delta"] >= 3.0
    report = {
        "experiment": "pitcher-history amount scaling for current three adjustments",
        "official_train_only": True,
        "test_aggregate_used": False,
        "baseline": "current hand/two-strikes/runners adjustments without history scaling",
        "formula": "factor = asof_pitcher_n / (asof_pitcher_n + k); k=0 means factor 1",
        "k_values": list(K_VALUES),
        "training": training,
        "fold_results": fold_results,
        "summaries": summaries,
        "selected": selected,
        "decision": "continue_history_scaling_full_pipeline" if passed else "keep_1029_champion",
        "gate": "increment over current adjustments positive in 2023 and 2024; 2024 >= +3",
    }
    write_json(output, report)
    print(json.dumps({"selected": selected, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
