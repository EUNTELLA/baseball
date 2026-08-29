"""현재 3종 보정 위에서 game_type=F 행 전용 추가 보정을 선별한다."""
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
WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.25)
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


def f_specific_error(residual: np.ndarray, is_f: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    selected_f = selected & is_f
    selected_r = selected & ~is_f
    if not selected_f.any() or not selected_r.any():
        raise RuntimeError("F 또는 R 학습 행이 없습니다.")
    f_mean = float(residual[selected_f].mean())
    r_mean = float(residual[selected_r].mean())
    return {
        "f_rows": int(selected_f.sum()),
        "r_rows": int(selected_r.sum()),
        "f_mean_error": f_mean,
        "r_mean_error": r_mean,
        "f_relative_error": f_mean - r_mean,
    }


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    is_f = frame["game_type"].astype(str).eq("F").to_numpy()
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
        current_adjustment = additions["hand"] + additions["two_strikes"] + additions["runners_on"]
        current_prediction = np.clip(oof_prediction[valid] + current_adjustment, 1e-6, 1 - 1e-6)
        current_metrics = base.metrics(current_prediction, target[valid])

        error_stats = f_specific_error(residual, is_f, source)
        f_row_value = error_stats["f_relative_error"]
        candidates = []
        for weight in WEIGHTS:
            extra = np.where(is_f[valid], weight * f_row_value, 0.0)
            prediction = np.clip(current_prediction + extra, 1e-6, 1 - 1e-6)
            candidate_metrics = base.metrics(prediction, target[valid])
            candidates.append({
                "weight": weight,
                "applied_f_value": float(weight * f_row_value),
                "metrics": candidate_metrics,
                "bss_delta_vs_current": (
                    candidate_metrics["bss_score"] - current_metrics["bss_score"]
                ),
            })
        fold_results.append({
            "fold": fold,
            "source_seasons": [fold - 2, fold - 1],
            "f_share": float(is_f[valid].mean()),
            "error_stats": error_stats,
            "current_three_adjustments": current_metrics,
            "candidates": candidates,
        })

    summaries = []
    for weight in WEIGHTS:
        rows = [
            next(item for item in fold["candidates"] if item["weight"] == weight)
            for fold in fold_results
        ]
        deltas = [float(row["bss_delta_vs_current"]) for row in rows]
        summaries.append({
            "weight": weight,
            "fold_2023_delta": deltas[0],
            "fold_2024_delta": deltas[1],
            "mean_delta": float(np.mean(deltas)),
            "worst_delta": float(np.min(deltas)),
            "both_positive": bool(min(deltas) > 0),
        })
    stable = [row for row in summaries if row["both_positive"]]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    passed = selected is not None and selected["fold_2024_delta"] >= 3.0
    report = {
        "experiment": "F-row-only past prediction-error adjustment screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "baseline": "current hand/two-strikes/runners adjustments at weight 1.0",
        "weights": list(WEIGHTS),
        "training": training,
        "fold_results": fold_results,
        "summaries": summaries,
        "selected": selected,
        "decision": "continue_f_row_adjustment_full_pipeline" if passed else "keep_1029_champion",
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
