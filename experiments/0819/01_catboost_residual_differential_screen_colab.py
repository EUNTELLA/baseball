"""CatBoost 과거 시즌 예측 오차를 이용한 행 단위 보정 3종을 선별한다.

검증 시즌 S의 보정값은 S-2, S-1을 학습에 포함하지 않은 예측 오차만 사용한다.
평가 데이터에서는 각 행 자체의 투수/손/카운트/주자 값으로 표를 조회한다.
"""
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
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_PATH = SCRIPT_DIR.parent / "common" / "model_features.py"
AXES = (
    ("hand", 1000.0),
    ("two_strikes", 1000.0),
    ("runners_on", 2000.0),
)


def load_feature_module():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(seed: int, task_type: str) -> dict:
    result = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 1.0,
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


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, dtype=float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "bss_score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def contexts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "hand": (frame["pitcher_hand"].astype(str) == frame["batter_hand"].astype(str)).astype(int).to_numpy(),
        "two_strikes": (frame["strikes_before"].fillna(-1).astype(int) == 2).astype(int).to_numpy(),
        "runners_on": (frame["num_runners_on"].fillna(0).astype(float) > 0).astype(int).to_numpy(),
    }


def differential_table(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    selected: np.ndarray,
    shrinkage: float,
) -> pd.Series:
    grouped = (
        pd.DataFrame({
            "pitcher_id": pitcher[selected],
            "context": context[selected],
            "residual": residual[selected],
        })
        .groupby(["pitcher_id", "context"])["residual"]
        .agg(["mean", "size"])
        .unstack()
    )
    required = (("mean", 0), ("mean", 1), ("size", 0), ("size", 1))
    if any(column not in grouped.columns for column in required):
        return pd.Series(dtype=float)
    n0 = grouped[("size", 0)].fillna(0.0)
    n1 = grouped[("size", 1)].fillna(0.0)
    effective_n = n0 * n1 / (n0 + n1).replace(0.0, np.nan)
    difference = grouped[("mean", 1)] - grouped[("mean", 0)]
    return (difference * effective_n / (effective_n + shrinkage)).dropna()


def apply_table(
    table: pd.Series,
    pitcher: np.ndarray,
    context: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    mapped = pd.Series(pitcher[selected]).map(table).fillna(0.0).to_numpy()
    return mapped * np.where(context[selected] == 1, 0.5, -0.5)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    axis_contexts = contexts(frame)
    feature_module = load_feature_module()
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
            model = CatBoostClassifier(**params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            fold_predictions.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            seconds.append(float(time.perf_counter() - started))
            print(
                f"fold={fold} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}",
                flush=True,
            )
            del model
            gc.collect()
        oof_prediction[valid_mask] = np.mean(fold_predictions, axis=0)
        training.append({
            "fold": fold,
            "rows": int(valid_mask.sum()),
            "best_iterations": iterations,
            "seconds": seconds,
            "baseline": metrics(oof_prediction[valid_mask], target[valid_mask]),
        })
        write_json(output, {"status": "running", "training": training})
        del features, train_pool, valid_pool, fold_predictions
        gc.collect()

    residual = target.astype(float) - oof_prediction
    evaluations = []
    candidate_order = ("baseline", "hand", "hand_two_strikes", "all_three")
    for fold in EVAL_FOLDS:
        valid = season == fold
        source = np.isin(season, (fold - 2, fold - 1))
        if np.isnan(residual[source]).any():
            raise RuntimeError(f"fold={fold} 표 원천에 OOF 예측 누락")
        additions, tables = {}, {}
        for name, shrinkage in AXES:
            table = differential_table(
                pitcher, axis_contexts[name], residual, source, shrinkage
            )
            additions[name] = apply_table(table, pitcher, axis_contexts[name], valid)
            tables[name] = {
                "shrinkage": shrinkage,
                "pitchers": int(len(table)),
                "median_absolute_difference": float(table.abs().median()) if len(table) else 0.0,
            }
        base = oof_prediction[valid]
        predictions = {
            "baseline": base,
            "hand": base + additions["hand"],
            "hand_two_strikes": base + additions["hand"] + additions["two_strikes"],
            "all_three": base + additions["hand"] + additions["two_strikes"] + additions["runners_on"],
        }
        rows = []
        baseline_metrics = metrics(predictions["baseline"], target[valid])
        for name in candidate_order:
            candidate_metrics = metrics(predictions[name], target[valid])
            rows.append({
                "name": name,
                "metrics": candidate_metrics,
                "bss_delta": candidate_metrics["bss_score"] - baseline_metrics["bss_score"],
            })
        evaluations.append({
            "fold": fold,
            "source_seasons": [fold - 2, fold - 1],
            "tables": tables,
            "candidates": rows,
        })
        final_row = rows[-1]
        print(
            f"fold={fold} all_three BSS delta={final_row['bss_delta']:+.2f}",
            flush=True,
        )

    final_deltas = []
    for evaluation in evaluations:
        row = next(item for item in evaluation["candidates"] if item["name"] == "all_three")
        final_deltas.append(float(row["bss_delta"]))
    confirmation = final_deltas[-1]
    passed = min(final_deltas) > 0.0 and confirmation >= 5.0
    report = {
        "experiment": "CatBoost past-season prediction-error adjustment screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "baseline": "CatBoost d6 lr0.05 l2=1 success model, 3-seed OOF",
        "note": "구조 선별 단계이며 MR/wayoff offset과 train-trend shift는 아직 미적용",
        "seeds": list(SEEDS),
        "oof_folds": list(OOF_FOLDS),
        "evaluation_folds": list(EVAL_FOLDS),
        "axes": {name: shrinkage for name, shrinkage in AXES},
        "training": training,
        "evaluations": evaluations,
        "all_three_bss_deltas": final_deltas,
        "decision": "continue_full_997_pipeline" if passed else "reject_residual_differential_axis",
        "gate": "all_three BSS delta positive in 2023 and 2024, and 2024 delta >= +5",
    }
    write_json(output, report)
    print(json.dumps({
        "all_three_bss_deltas": final_deltas,
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
