"""검증 JSON을 사용해 보조 채널 6시드 + Train shift 탐색 제출 ZIP을 만든다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SUCCESS_SEEDS = (42, 7, 2024, 99, 1, 123, 777)
AUX_SEEDS = (42, 7, 2024, 99, 1, 123)
SELECTED_NAME = "aux6_train_recomputed_shift"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
COMMON_PATH = ROOT / "0817" / "03_catboost_full_pipeline_walkforward_colab.py"
BASE_BUILDER_PATH = ROOT / "0819" / "03_build_residual_differential_submission_colab.py"
FAILURE_LABEL_PATH = ROOT / "common" / "failure_labels.py"
BASE_ZIP = ROOT / "0819" / "results" / "submit_catboost_residual_differential.zip"
OUTPUT_ZIP = SCRIPT_DIR / "results" / "submit_catboost_aux6_recomputed_shift_probe.zip"
BUILD_DIR = SCRIPT_DIR / "results" / "build_aux6_recomputed_shift_probe"
BASE_CONFIG = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_validation(common, features, cat_indices, labels, eligible, train_mask,
                     valid_mask, seeds, fixed_iterations, task_type, label):
    train_pool = Pool(features.loc[train_mask & eligible], labels[train_mask & eligible], cat_features=cat_indices)
    valid_prediction_pool = Pool(features.loc[valid_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    if len(seeds) != len(fixed_iterations):
        raise ValueError(f"{label} seed/iteration 길이 불일치")
    for seed, iteration in zip(seeds, fixed_iterations):
        started = time.perf_counter()
        model = CatBoostClassifier(**common.model_params(BASE_CONFIG, seed, int(iteration), task_type, False))
        model.fit(train_pool)
        predictions.append(model.predict_proba(valid_prediction_pool)[:, 1])
        iterations.append(int(iteration))
        seconds.append(float(time.perf_counter() - started))
        print(f"validation {label} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(predictions, axis=0), iterations, seconds


def main(train_path: Path, test_path: Path, sample_path: Path,
         validation_json: Path, report_path: Path, task_type: str) -> None:
    common = load_module("full_pipeline_common", COMMON_PATH)
    base_builder = load_module("residual_submission_builder", BASE_BUILDER_PATH)
    validation_report = json.loads(validation_json.read_text(encoding="utf-8"))
    summary = next(row for row in validation_report["summaries"] if row["name"] == SELECTED_NAME)
    if summary["both_positive"]:
        purpose = "gate-passed candidate"
    else:
        purpose = "one-off leaderboard probe requested despite local gate failure"

    # Colab 새 세션에서도 현재 1029 구조와 보정표를 먼저 재생성한다.
    base_report_path = SCRIPT_DIR / "results" / "aux6_base_build.json"
    base_builder.main(train_path, test_path, sample_path, base_report_path)
    if not BASE_ZIP.exists():
        raise FileNotFoundError(BASE_ZIP)

    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    success_iterations_from_report = next(
        row["best_iterations"] for row in base_report["training"] if row["fold"] == 2024
    )
    validation_fold = next(
        row for row in validation_report["fold_results"] if row["validation_year"] == 2024
    )
    auxiliary_iterations_from_report = {
        name: validation_fold["training"][name]["best_iterations"]
        for name in ("mr", "large_miss")
    }

    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    train_mask, valid_mask = season < 2024, season == 2024
    failure_module = load_module("failure_labels", FAILURE_LABEL_PATH)
    labels = failure_module.recover_failure_labels(frame)
    have = labels["middle"].notna().to_numpy()
    mr_target = ((labels["middle"].eq(1) | labels["reverse"].eq(1)).fillna(False).astype(int).to_numpy())
    large_miss_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    validation_features, validation_ci, _ = common.engineer(frame, feature_module, train_mask, target)

    success_prediction, success_iterations, success_seconds = train_validation(
        common, validation_features, validation_ci, target,
        np.ones(len(frame), dtype=bool), train_mask, valid_mask,
        SUCCESS_SEEDS, success_iterations_from_report, task_type, "success",
    )
    auxiliary_predictions, auxiliary_iterations, auxiliary_seconds = {}, {}, {}
    for name, aux_target in (("mr", mr_target), ("large_miss", large_miss_target)):
        prediction, iterations, seconds = train_validation(
            common, validation_features, validation_ci, aux_target, have,
            train_mask, valid_mask, AUX_SEEDS,
            auxiliary_iterations_from_report[name], task_type, name,
        )
        auxiliary_predictions[name] = prediction
        auxiliary_iterations[name] = iterations
        auxiliary_seconds[name] = seconds

    selected_valid = have[valid_mask]
    offset = common.fit_offset(
        success_prediction, auxiliary_predictions["mr"],
        auxiliary_predictions["large_miss"], target[valid_mask], selected_valid,
    )
    combined = common.apply_offset(
        success_prediction, auxiliary_predictions["mr"],
        auxiliary_predictions["large_miss"], offset,
    )
    forecast = common.select_alpha_and_forecast(frame, 2025)
    shift = common.fixed_shift(combined, forecast["forecast"])
    print(f"production offset={offset} shift={shift:+.10f}", flush=True)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(BUILD_DIR)
    model_dir = BUILD_DIR / "model"
    for name in ("mr", "wayoff"):
        for old in model_dir.glob(f"{name}_*.cbm"):
            old.unlink()

    full_mask = np.ones(len(frame), dtype=bool)
    full_features, full_ci, _ = common.engineer(frame, feature_module, full_mask, target)
    full_training_seconds = {}
    for model_name, target_name, aux_target in (
        ("mr", "mr", mr_target), ("wayoff", "large_miss", large_miss_target)
    ):
        full_pool = Pool(full_features.loc[have], aux_target[have], cat_features=full_ci)
        full_training_seconds[target_name] = []
        for seed, iteration in zip(AUX_SEEDS, auxiliary_iterations[target_name]):
            started = time.perf_counter()
            model = CatBoostClassifier(**common.model_params(BASE_CONFIG, seed, iteration, task_type, False))
            model.fit(full_pool)
            model.save_model(model_dir / f"{model_name}_{seed}.cbm")
            elapsed = float(time.perf_counter() - started)
            full_training_seconds[target_name].append(elapsed)
            print(f"full {model_name}_{seed}.cbm iter={iteration} sec={elapsed:.1f}", flush=True)
            del model
            gc.collect()

    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["offset"] = {"seeds": list(AUX_SEEDS), **offset}
    meta["logit_shift"] = shift
    meta["auxiliary_ensemble"] = {
        "seeds": list(AUX_SEEDS),
        "selection": SELECTED_NAME,
        "validation_summary": summary,
        "purpose": purpose,
    }
    meta["shift_provenance"] = {
        "data": "official train.csv and 2024 predictions from models trained through 2023",
        "target_year": 2025,
        "forecast": forecast,
        "reference_prediction_mean_after_offset": float(combined.mean()),
        "logit_shift": shift,
        "test_aggregate_used": False,
        "external_data_used": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_DIR))
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    verify_dir = SCRIPT_DIR / "results" / "verify_aux6_recomputed_shift"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        archive.extractall(verify_dir)
    (verify_dir / "data").mkdir()
    shutil.copy2(test_path, verify_dir / "data" / "test.csv")
    shutil.copy2(sample_path, verify_dir / "data" / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=verify_dir,
        capture_output=True, text=True, check=True, timeout=600,
    )
    submission = pd.read_csv(verify_dir / "output" / "submission.csv")
    if submission[TARGET_COL].isna().any() or not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("샘플 추론 결과의 결측 또는 확률 범위 오류")
    report = {
        "model": "1029 residual-differential model plus 6-seed auxiliary ensemble and train-only shift",
        "purpose": purpose, "official_train_only": True,
        "external_data_used": False, "test_aggregate_used": False,
        "validation_json": str(validation_json), "validation_summary": summary,
        "success_validation_iterations": success_iterations,
        "success_validation_seconds": success_seconds,
        "auxiliary_iterations": auxiliary_iterations,
        "auxiliary_validation_seconds": auxiliary_seconds,
        "auxiliary_full_seconds": full_training_seconds,
        "offset": offset, "forecast": forecast, "logit_shift": shift,
        "output_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "zip_test_error": bad_member, "members": members,
        "sample_rows": int(len(submission)),
        "sample_missing": int(submission[TARGET_COL].isna().sum()),
        "sample_min": float(submission[TARGET_COL].min()),
        "sample_max": float(submission[TARGET_COL].max()),
        "sample_mean": float(submission[TARGET_COL].mean()),
        "sample_stdout": completed.stdout.strip(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.test.resolve(), args.sample.resolve(),
         args.validation_json.resolve(), args.report.resolve(), args.task_type)
