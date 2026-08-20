"""기존 offset과 실패 여집합 대체안을 동일 shift 격자에서 비교."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
FOLDS = (2022, 2023, 2024)
DEVELOPMENT_FOLDS = (2022, 2023)
CONFIRMATION_FOLD = 2024
SIMPLEX_WEIGHT = 0.10
DEPLOYMENT_SHIFT = -0.03842671927234861
SHIFTS = (
    -0.058,
    DEPLOYMENT_SHIFT - 0.01,
    DEPLOYMENT_SHIFT,
    DEPLOYMENT_SHIFT + 0.01,
    -0.018,
    0.0,
)
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "03_catboost_full_pipeline_walkforward_colab.py"
MODEL_CONFIG = {
    "name": "d6_lr05_l2_3",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
}


def load_common():
    spec = importlib.util.spec_from_file_location("walkforward_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simplex_blend(success: np.ndarray, mr: np.ndarray, wayoff: np.ndarray) -> np.ndarray:
    complement = np.clip(1 - mr - wayoff, 1e-6, 1 - 1e-6)
    return (1 - SIMPLEX_WEIGHT) * success + SIMPLEX_WEIGHT * complement


def apply_shift(common, prediction: np.ndarray, shift: float) -> np.ndarray:
    return common.sigmoid(common.logit(prediction) + shift)


def write_json(common, output: Path, payload: dict) -> None:
    common.write_checkpoint(output, payload)


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_common()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    failure_module = common.load_module("failure_labels", common.FAILURE_LABEL_PATH)
    recovered = failure_module.recover_failure_labels(frame)
    have = recovered["middle"].notna().to_numpy()
    mr_target = (
        ((recovered["middle"] == 1) | (recovered["reverse"] == 1))
        .fillna(False).astype(int).to_numpy()
    )
    wayoff_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    report = {
        "experiment": "failure complement 10% replacement under shared fixed shifts",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": list(FOLDS),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "confirmation_fold": CONFIRMATION_FOLD,
        "seeds": list(common.SEEDS),
        "simplex_weight": SIMPLEX_WEIGHT,
        "deployment_shift": DEPLOYMENT_SHIFT,
        "shared_shifts": list(SHIFTS),
        "model_config": MODEL_CONFIG,
        "results": [],
    }

    for validation_year in FOLDS:
        calibration_year = validation_year - 1
        inner_train = (frame["season"] < calibration_year).to_numpy()
        calibration = (frame["season"] == calibration_year).to_numpy()
        outer_train = (frame["season"] < validation_year).to_numpy()
        validation = (frame["season"] == validation_year).to_numpy()
        print(f"\n===== fold {validation_year}: calibrate offset on {calibration_year} =====", flush=True)
        inner_x, inner_ci, inner_mean = common.engineer(frame, feature_module, inner_train, target)
        outer_x, outer_ci, outer_mean = common.engineer(frame, feature_module, outer_train, target)

        predictions, training = {}, {}
        for name, labels, eligible in (
            ("success", target, np.ones(len(frame), dtype=bool)),
            ("mr", mr_target, have),
            ("wayoff", wayoff_target, have),
        ):
            inner_prediction, best_iterations, inner_seconds = common.train_inner_and_predict(
                inner_x, inner_ci, labels, eligible, inner_train, calibration,
                MODEL_CONFIG, task_type, f"fold={validation_year} {name}",
            )
            outer_prediction, outer_seconds = common.train_outer_and_predict(
                outer_x, outer_ci, labels, eligible, outer_train, validation,
                MODEL_CONFIG, best_iterations, task_type, f"fold={validation_year} {name}",
            )
            predictions[name] = {"inner": inner_prediction, "outer": outer_prediction}
            training[name] = {
                "best_iterations": best_iterations,
                "inner_seconds": inner_seconds,
                "outer_seconds": outer_seconds,
            }

        calibration_target = target[calibration]
        validation_target = target[validation]
        baseline_offset = common.fit_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["wayoff"]["inner"], calibration_target, have[calibration],
        )
        baseline_outer = common.apply_offset(
            predictions["success"]["outer"], predictions["mr"]["outer"],
            predictions["wayoff"]["outer"], baseline_offset,
        )
        candidate_outer = simplex_blend(
            predictions["success"]["outer"], predictions["mr"]["outer"],
            predictions["wayoff"]["outer"],
        )
        shift_results = []
        for shift in SHIFTS:
            baseline_shifted = apply_shift(common, baseline_outer, shift)
            candidate_shifted = apply_shift(common, candidate_outer, shift)
            baseline_metrics = common.metrics(baseline_shifted, validation_target)
            candidate_metrics = common.metrics(candidate_shifted, validation_target)
            shift_results.append({
                "shift": float(shift),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "score_delta": candidate_metrics["score"] - baseline_metrics["score"],
                "absolute_mean_error_delta": (
                    abs(candidate_metrics["mean_error"]) - abs(baseline_metrics["mean_error"])
                ),
            })
        primary = next(row for row in shift_results if row["shift"] == DEPLOYMENT_SHIFT)
        print(
            f"fold={validation_year} deployment_shift={DEPLOYMENT_SHIFT:+.8f} "
            f"baseline={primary['baseline']['score']:.2f} "
            f"candidate={primary['candidate']['score']:.2f} "
            f"delta={primary['score_delta']:+.2f}", flush=True,
        )
        report["results"].append({
            "validation_year": validation_year,
            "calibration_year": calibration_year,
            "inner_global_mean": inner_mean,
            "outer_global_mean": outer_mean,
            "training": training,
            "baseline_offset": baseline_offset,
            "pre_shift": {
                "baseline": common.metrics(baseline_outer, validation_target),
                "candidate": common.metrics(candidate_outer, validation_target),
            },
            "shared_shift_results": shift_results,
        })
        write_json(common, output, report)
        del inner_x, outer_x, predictions
        gc.collect()

    primary_rows = []
    for fold in report["results"]:
        row = next(item for item in fold["shared_shift_results"] if item["shift"] == DEPLOYMENT_SHIFT)
        primary_rows.append({"validation_year": fold["validation_year"], **row})
    development_rows = [row for row in primary_rows if row["validation_year"] in DEVELOPMENT_FOLDS]
    confirmation = next(row for row in primary_rows if row["validation_year"] == CONFIRMATION_FOLD)
    neighbor_shifts = (DEPLOYMENT_SHIFT - 0.01, DEPLOYMENT_SHIFT + 0.01)
    confirmation_fold = next(row for row in report["results"] if row["validation_year"] == CONFIRMATION_FOLD)
    neighbor_rows = [
        row for row in confirmation_fold["shared_shift_results"] if row["shift"] in neighbor_shifts
    ]
    summary = {
        "development_mean_delta": float(np.mean([row["score_delta"] for row in development_rows])),
        "development_worst_delta": float(np.min([row["score_delta"] for row in development_rows])),
        "development_mean_absolute_mean_error_delta": float(np.mean([
            row["absolute_mean_error_delta"] for row in development_rows
        ])),
        "confirmation_delta": float(confirmation["score_delta"]),
        "confirmation_absolute_mean_error_delta": float(confirmation["absolute_mean_error_delta"]),
        "confirmation_neighbor_deltas": {
            str(row["shift"]): float(row["score_delta"]) for row in neighbor_rows
        },
        "confirmation_neighbor_worst_delta": float(np.min([row["score_delta"] for row in neighbor_rows])),
    }
    passed = (
        summary["development_mean_delta"] >= 5.0
        and summary["development_worst_delta"] >= -2.0
        and summary["confirmation_delta"] >= 5.0
        and summary["confirmation_neighbor_worst_delta"] > 0
        and summary["confirmation_absolute_mean_error_delta"] <= 0.001
    )
    summary["decision"] = "build_single_variable_zip" if passed else "keep_997_baseline"
    summary["gate"] = "dev mean>=+5, worst>=-2, 2024>=+5, 2024 +/-0.01 both positive, 2024 mean error delta<=0.001"
    report["primary_shift_results"] = primary_rows
    report["summary"] = summary
    write_json(common, output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
