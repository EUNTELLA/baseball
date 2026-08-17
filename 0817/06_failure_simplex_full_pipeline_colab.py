"""실패 여집합 10% 혼합을 기존 offset·shift 파이프라인과 중첩 비교."""
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
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "03_catboost_full_pipeline_walkforward_colab.py"

SUCCESS_CONFIG = {
    "name": "d6_lr05_l2_3",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
}
AUX_CONFIG = {
    "name": "aux_d6_lr05_l2_3",
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


def final_prediction(common, prediction: np.ndarray, reference: np.ndarray, forecast: float):
    shift = common.fixed_shift(reference, forecast)
    return common.sigmoid(common.logit(prediction) + shift), shift


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_common()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    recovered = frame[[ID_COL]].merge(pd.read_csv(common.LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = (
        ((recovered["middle"] == 1) | (recovered["reverse"] == 1))
        .fillna(False).astype(int).to_numpy()
    )
    wayoff_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    report = {
        "experiment": "failure complement 10% blend full-pipeline nested walk-forward",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": list(FOLDS),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "confirmation_fold": CONFIRMATION_FOLD,
        "seeds": list(common.SEEDS),
        "simplex_weight": SIMPLEX_WEIGHT,
        "success_config": SUCCESS_CONFIG,
        "aux_config": AUX_CONFIG,
        "results": [],
    }

    for validation_year in FOLDS:
        calibration_year = validation_year - 1
        inner_train = (frame["season"] < calibration_year).to_numpy()
        calibration = (frame["season"] == calibration_year).to_numpy()
        outer_train = (frame["season"] < validation_year).to_numpy()
        validation = (frame["season"] == validation_year).to_numpy()
        print(f"\n===== fold {validation_year}: calibrate {calibration_year} =====", flush=True)
        inner_x, inner_ci, inner_mean = common.engineer(frame, feature_module, inner_train, target)
        outer_x, outer_ci, outer_mean = common.engineer(frame, feature_module, outer_train, target)

        predictions = {}
        training = {}
        for name, labels, eligible, config in (
            ("success", target, np.ones(len(frame), dtype=bool), SUCCESS_CONFIG),
            ("mr", mr_target, have, AUX_CONFIG),
            ("wayoff", wayoff_target, have, AUX_CONFIG),
        ):
            inner_prediction, best_iterations, inner_seconds = common.train_inner_and_predict(
                inner_x, inner_ci, labels, eligible, inner_train, calibration,
                config, task_type, f"fold={validation_year} {name}",
            )
            outer_prediction, outer_seconds = common.train_outer_and_predict(
                outer_x, outer_ci, labels, eligible, outer_train, validation,
                config, best_iterations, task_type, f"fold={validation_year} {name}",
            )
            predictions[name] = {"inner": inner_prediction, "outer": outer_prediction}
            training[name] = {
                "best_iterations": best_iterations,
                "inner_seconds": inner_seconds,
                "outer_seconds": outer_seconds,
            }

        calibration_target = target[calibration]
        validation_target = target[validation]
        calibration_have = have[calibration]
        forecast = common.select_alpha_and_forecast(frame, validation_year)

        baseline_offset = common.fit_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["wayoff"]["inner"], calibration_target, calibration_have,
        )
        baseline_inner = common.apply_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["wayoff"]["inner"], baseline_offset,
        )
        baseline_outer = common.apply_offset(
            predictions["success"]["outer"], predictions["mr"]["outer"],
            predictions["wayoff"]["outer"], baseline_offset,
        )
        baseline_final, baseline_shift = final_prediction(
            common, baseline_outer, baseline_inner, forecast["forecast"]
        )

        simplex_inner = simplex_blend(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["wayoff"]["inner"],
        )
        simplex_outer = simplex_blend(
            predictions["success"]["outer"], predictions["mr"]["outer"],
            predictions["wayoff"]["outer"],
        )
        replace_final, replace_shift = final_prediction(
            common, simplex_outer, simplex_inner, forecast["forecast"]
        )

        augmented_offset = common.fit_offset(
            simplex_inner, predictions["mr"]["inner"], predictions["wayoff"]["inner"],
            calibration_target, calibration_have,
        )
        augmented_inner = common.apply_offset(
            simplex_inner, predictions["mr"]["inner"], predictions["wayoff"]["inner"], augmented_offset,
        )
        augmented_outer = common.apply_offset(
            simplex_outer, predictions["mr"]["outer"], predictions["wayoff"]["outer"], augmented_offset,
        )
        augmented_final, augmented_shift = final_prediction(
            common, augmented_outer, augmented_inner, forecast["forecast"]
        )

        strategies = [
            {
                "name": "baseline_offset",
                "offset": baseline_offset,
                "shift": baseline_shift,
                "pre_shift": common.metrics(baseline_outer, validation_target),
                "final": common.metrics(baseline_final, validation_target),
            },
            {
                "name": "simplex_replace_offset",
                "offset": None,
                "shift": replace_shift,
                "pre_shift": common.metrics(simplex_outer, validation_target),
                "final": common.metrics(replace_final, validation_target),
            },
            {
                "name": "simplex_plus_refit_offset",
                "offset": augmented_offset,
                "shift": augmented_shift,
                "pre_shift": common.metrics(augmented_outer, validation_target),
                "final": common.metrics(augmented_final, validation_target),
            },
        ]
        for strategy in strategies:
            print(
                f"fold={validation_year} {strategy['name']} "
                f"pre={strategy['pre_shift']['score']:.2f} "
                f"final={strategy['final']['score']:.2f} shift={strategy['shift']:+.5f}",
                flush=True,
            )
        report["results"].append({
            "validation_year": validation_year,
            "calibration_year": calibration_year,
            "inner_global_mean": inner_mean,
            "outer_global_mean": outer_mean,
            "rate_forecast": forecast,
            "training": training,
            "strategies": strategies,
        })
        common.write_checkpoint(output, report)
        del inner_x, outer_x, predictions
        gc.collect()

    candidate_names = ("simplex_replace_offset", "simplex_plus_refit_offset")
    development = []
    for name in candidate_names:
        deltas = []
        mean_error_deltas = []
        for fold in report["results"]:
            if fold["validation_year"] not in DEVELOPMENT_FOLDS:
                continue
            baseline = next(row for row in fold["strategies"] if row["name"] == "baseline_offset")
            candidate = next(row for row in fold["strategies"] if row["name"] == name)
            deltas.append(candidate["final"]["score"] - baseline["final"]["score"])
            mean_error_deltas.append(
                abs(candidate["final"]["mean_error"]) - abs(baseline["final"]["mean_error"])
            )
        development.append({
            "name": name,
            "mean_final_delta": float(np.mean(deltas)),
            "worst_final_delta": float(np.min(deltas)),
            "mean_absolute_mean_error_delta": float(np.mean(mean_error_deltas)),
        })
    selected = max(development, key=lambda row: (row["mean_final_delta"], -row["mean_absolute_mean_error_delta"]))
    confirmation_fold = next(row for row in report["results"] if row["validation_year"] == CONFIRMATION_FOLD)
    confirmation_baseline = next(
        row for row in confirmation_fold["strategies"] if row["name"] == "baseline_offset"
    )
    confirmation_candidate = next(
        row for row in confirmation_fold["strategies"] if row["name"] == selected["name"]
    )
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "name": selected["name"],
        "final_delta": confirmation_candidate["final"]["score"] - confirmation_baseline["final"]["score"],
        "absolute_mean_error_delta": (
            abs(confirmation_candidate["final"]["mean_error"])
            - abs(confirmation_baseline["final"]["mean_error"])
        ),
        "baseline": confirmation_baseline,
        "candidate": confirmation_candidate,
    }
    passed = (
        selected["mean_final_delta"] >= 5.0
        and selected["worst_final_delta"] >= -2.0
        and selected["mean_absolute_mean_error_delta"] <= 0
        and confirmation["final_delta"] >= 5.0
        and confirmation["absolute_mean_error_delta"] <= 0
    )
    report["development_selection"] = development
    report["selected"] = selected
    report["confirmation"] = confirmation
    report["decision"] = "build_single_submission" if passed else "keep_997_baseline"
    report["gate"] = "dev mean>=+5, dev worst>=-2, dev mean error not worse, 2024>=+5 and mean error not worse"
    common.write_checkpoint(output, report)
    print(json.dumps({key: report[key] for key in ("selected", "confirmation", "decision")}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
