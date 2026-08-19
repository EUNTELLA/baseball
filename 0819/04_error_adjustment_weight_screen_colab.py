"""투수별 과거 예측 오차 보정 3종의 조합과 적용 강도를 비교한다."""
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
WEIGHTS = (0.5, 0.75, 1.0, 1.25)
SUBSETS = (
    ("hand",),
    ("two_strikes",),
    ("runners_on",),
    ("hand", "two_strikes"),
    ("hand", "runners_on"),
    ("two_strikes", "runners_on"),
    ("hand", "two_strikes", "runners_on"),
)
SCRIPT_DIR = Path(__file__).resolve().parent
SCREEN_PATH = SCRIPT_DIR / "01_catboost_residual_differential_screen_colab.py"


def load_screen_module():
    spec = importlib.util.spec_from_file_location("error_adjustment_base", SCREEN_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SCREEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_screen_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
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
        fold_predictions, iterations, seconds = [], [], []
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostClassifier(**base.params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            fold_predictions.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            seconds.append(float(time.perf_counter() - started))
            print(f"fold={fold} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
            del model
            gc.collect()
        oof_prediction[valid_mask] = np.mean(fold_predictions, axis=0)
        training.append({
            "fold": fold,
            "best_iterations": iterations,
            "seconds": seconds,
            "baseline": base.metrics(oof_prediction[valid_mask], target[valid_mask]),
        })
        write_json(output, {"status": "running", "training": training})
        del features, train_pool, valid_pool, fold_predictions
        gc.collect()

    residual = target.astype(float) - oof_prediction
    fold_results = []
    for fold in EVAL_FOLDS:
        source = np.isin(season, (fold - 2, fold - 1))
        valid = season == fold
        axis_additions = {}
        table_stats = {}
        for name, shrinkage in base.AXES:
            table = base.differential_table(
                pitcher, condition_values[name], residual, source, shrinkage
            )
            axis_additions[name] = base.apply_table(
                table, pitcher, condition_values[name], valid
            )
            table_stats[name] = {
                "pitchers": int(len(table)),
                "median_absolute_value": float(table.abs().median()) if len(table) else 0.0,
            }
        baseline_prediction = oof_prediction[valid]
        baseline_metrics = base.metrics(baseline_prediction, target[valid])
        candidates = []
        for subset in SUBSETS:
            raw_adjustment = sum((axis_additions[name] for name in subset), np.zeros(valid.sum()))
            for weight in WEIGHTS:
                prediction = np.clip(baseline_prediction + weight * raw_adjustment, 1e-6, 1 - 1e-6)
                candidate_metrics = base.metrics(prediction, target[valid])
                candidates.append({
                    "name": "+".join(subset),
                    "axes": list(subset),
                    "weight": weight,
                    "metrics": candidate_metrics,
                    "bss_delta": candidate_metrics["bss_score"] - baseline_metrics["bss_score"],
                })
        fold_results.append({
            "fold": fold,
            "source_seasons": [fold - 2, fold - 1],
            "baseline": baseline_metrics,
            "tables": table_stats,
            "candidates": candidates,
        })

    summaries = []
    for subset in SUBSETS:
        name = "+".join(subset)
        for weight in WEIGHTS:
            deltas = []
            for fold_result in fold_results:
                row = next(
                    item for item in fold_result["candidates"]
                    if item["name"] == name and item["weight"] == weight
                )
                deltas.append(float(row["bss_delta"]))
            summaries.append({
                "name": name,
                "axes": list(subset),
                "weight": weight,
                "fold_2023_delta": deltas[0],
                "fold_2024_delta": deltas[1],
                "mean_delta": float(np.mean(deltas)),
                "worst_delta": float(np.min(deltas)),
                "both_positive": bool(min(deltas) > 0),
            })
    stable = [row for row in summaries if row["both_positive"]]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    passed = selected is not None and selected["fold_2024_delta"] >= 5.0
    report = {
        "experiment": "past-season prediction-error adjustment subset and weight screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "seeds": list(SEEDS),
        "weights": list(WEIGHTS),
        "subsets": [list(item) for item in SUBSETS],
        "training": training,
        "fold_results": fold_results,
        "summaries": summaries,
        "selected": selected,
        "decision": "continue_selected_adjustment_full_pipeline" if passed else "keep_1029_champion",
        "gate": "select only candidates positive in both folds; maximize worst fold; require 2024 >= +5",
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
