"""선별된 CatBoost 설정을 7시드 학습하고 train-only offset/shift ZIP을 생성."""
from __future__ import annotations

import argparse
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
from scipy.optimize import brentq, minimize


SEEDS = (42, 7, 2024, 99, 1, 123, 777)
AUX_SEEDS = (42, 7, 2024)
ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_ZIP = ROOT / "0816" / "results" / "submit_catboost_train_trend_shift.zip"
AUX_DIR = ROOT / "0816" / "derived" / "auxpred"
FEATURE_PATH = ROOT / "common" / "model_features.py"
FAILURE_LABEL_PATH = ROOT / "common" / "failure_labels.py"
OUTPUT_NAME = "submit_catboost_lr03_l2_3_train_only"
TARGET_2025 = 0.477793531589047


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_features_module():
    path = FEATURE_PATH
    spec = importlib.util.spec_from_file_location("official_features", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(task_type: str, seed: int, iterations: int, early_stop: bool) -> dict:
    result = {
        "iterations": iterations,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": 100,
        "eval_metric": "Logloss",
        "grow_policy": "SymmetricTree",
    }
    if early_stop:
        result["early_stopping_rounds"] = 100
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "score": float(100000 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def load_aux_predictions() -> tuple[np.ndarray, np.ndarray]:
    mr = np.mean([np.load(AUX_DIR / f"mr_2024_{seed}.npy") for seed in AUX_SEEDS], axis=0)
    wayoff = np.mean([np.load(AUX_DIR / f"wayoff_2024_{seed}.npy") for seed in AUX_SEEDS], axis=0)
    return mr, wayoff


def fit_offset(success: np.ndarray, mr: np.ndarray, wayoff: np.ndarray, target: np.ndarray):
    z, u, v = logit(success), logit(mr), logit(wayoff)
    mu_mr, mu_wayoff = float(u.mean()), float(v.mean())
    u_centered, v_centered = u - mu_mr, v - mu_wayoff

    def nll(weights):
        prediction = np.clip(sigmoid(z + weights[0] * u_centered + weights[1] * v_centered), 1e-9, 1 - 1e-9)
        return float(-np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)))

    b, c = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    combined = sigmoid(z + b * u_centered + c * v_centered)
    return combined, {"b": float(b), "c": float(c), "mu_mr": mu_mr, "mu_wayoff": mu_wayoff}


def shift_for_target(prediction: np.ndarray, target_mean: float) -> float:
    logits = logit(prediction)
    objective = lambda shift: float(sigmoid(logits + shift).mean()) - target_mean
    return float(brentq(objective, -2.0, 2.0))


def validate_package(package_dir: Path, test_path: Path | None, sample_path: Path | None) -> dict:
    if test_path is None or sample_path is None:
        return {"sample_validation": "skipped"}
    data_dir = package_dir / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(test_path, data_dir / "test.csv")
    shutil.copy2(sample_path, data_dir / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=package_dir, check=True,
        capture_output=True, text=True,
    )
    output = pd.read_csv(package_dir / "output" / "submission.csv")
    prediction = output[TARGET_COL]
    if prediction.isna().any() or not prediction.between(0, 1).all():
        raise ValueError("샘플 추론 결과가 올바르지 않습니다.")
    return {
        "sample_validation": "passed",
        "sample_rows": int(len(output)),
        "sample_min": float(prediction.min()),
        "sample_max": float(prediction.max()),
        "sample_stdout": completed.stdout.strip(),
    }


def main(
    train_path: Path,
    output_dir: Path,
    work_dir: Path,
    task_type: str,
    test_path: Path | None,
    sample_path: Path | None,
) -> None:
    features_module = load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    train_mask = (frame["season"] <= 2023).to_numpy()
    valid_mask = (frame["season"] == 2024).to_numpy()
    global_mean = float(target[train_mask].mean())
    features = features_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in features_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in features_module.CAT_COLS]
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)

    validation_predictions, best_iterations, validation_seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params(task_type, seed, 2000, True))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        validation_predictions.append(model.predict_proba(valid_pool)[:, 1])
        best_iterations.append(max(1, int(model.get_best_iteration()) + 1))
        validation_seconds.append(float(time.perf_counter() - started))
        print(f"validation seed={seed} iter={best_iterations[-1]} sec={validation_seconds[-1]:.1f}", flush=True)

    success_all_valid = np.mean(validation_predictions, axis=0)
    failure_module = load_module("failure_labels", FAILURE_LABEL_PATH)
    labels = failure_module.recover_failure_labels(frame)
    have_labels = frame[[ID_COL]].merge(labels, on=ID_COL, how="left")["middle"].notna().to_numpy()
    offset_mask = valid_mask & have_labels
    success = success_all_valid[have_labels[valid_mask]]
    offset_target = target[offset_mask]
    mr, wayoff = load_aux_predictions()
    if not (len(success) == len(mr) == len(wayoff) == len(offset_target)):
        raise ValueError(f"offset 길이 불일치: {len(success)}, {len(mr)}, {len(wayoff)}, {len(offset_target)}")
    combined, offset = fit_offset(success, mr, wayoff, offset_target)
    fixed_shift = shift_for_target(combined, TARGET_2025)
    shifted = sigmoid(logit(combined) + fixed_shift)
    print(f"offset={offset} shift={fixed_shift:+.10f} mean={combined.mean():.6f}->{shifted.mean():.6f}", flush=True)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    package_dir = work_dir / "package"
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(package_dir)
    for cache_dir in package_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir)
    for bytecode in package_dir.rglob("*.pyc"):
        bytecode.unlink()
    model_dir = package_dir / "model"
    for old_model in model_dir.glob("model_*.cbm"):
        old_model.unlink()

    full_pool = Pool(features, target, cat_features=cat_indices)
    tags = []
    full_seconds = []
    for seed, iterations in zip(SEEDS, best_iterations):
        tag = f"Sym_{seed}"
        tags.append(tag)
        started = time.perf_counter()
        model = CatBoostClassifier(**params(task_type, seed, iterations, False))
        model.fit(full_pool)
        model.save_model(model_dir / f"model_{tag}.cbm")
        full_seconds.append(float(time.perf_counter() - started))
        print(f"full model_{tag}.cbm iter={iterations} sec={full_seconds[-1]:.1f}", flush=True)

    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "seeds": tags,
        "global_mean": global_mean,
        "offset": {"seeds": list(AUX_SEEDS), **offset},
        "logit_shift": fixed_shift,
        "success_model": {
            "algorithm": "CatBoostClassifier",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "selection": "official train walk-forward folds 2022, 2023, 2024",
        },
        "shift_provenance": {
            "data": "official train.csv and train-derived OOF predictions only",
            "reference": "2019-2023 trained models predicting labeled 2024 rows",
            "reference_prediction_mean_after_offset": float(combined.mean()),
            "target_2025": TARGET_2025,
            "logit_shift": fixed_shift,
            "test_aggregate_used": False,
            "external_data_used": False,
        },
    })
    with meta_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(meta, ensure_ascii=False))

    verification = validate_package(package_dir, test_path, sample_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / f"{OUTPUT_NAME}.zip"
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file() and "data" not in path.relative_to(package_dir).parts and "output" not in path.relative_to(package_dir).parts:
                archive.write(path, path.relative_to(package_dir))
    with zipfile.ZipFile(output_zip) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    report = {
        "experiment": "CatBoost d6 lr0.03 l2=3 7-seed + refit MR/wayoff offset + train-only trend shift",
        "official_train_only": True,
        "external_data_used": False,
        "test_aggregate_used": False,
        "best_iterations": best_iterations,
        "validation_seconds": validation_seconds,
        "full_seconds": full_seconds,
        "validation_success": score(success_all_valid, target[valid_mask]),
        "offset_fit_rows": int(len(offset_target)),
        "offset": offset,
        "offset_validation_before": score(success, offset_target),
        "offset_validation_after": score(combined, offset_target),
        "target_2025": TARGET_2025,
        "train_only_reference_mean": float(combined.mean()),
        "logit_shift": fixed_shift,
        "shifted_reference_mean": float(shifted.mean()),
        "zip": output_zip.name,
        "zip_mib": output_zip.stat().st_size / 1024**2,
        "zip_test_error": bad_member,
        "members": members,
        **verification,
    }
    report_path = output_dir / f"{OUTPUT_NAME}.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2))
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/0817_tuned_build"))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--test", type=Path)
    parser.add_argument("--sample", type=Path)
    args = parser.parse_args()
    main(
        args.train.resolve(), args.output_dir.resolve(), args.work_dir.resolve(), args.task_type,
        args.test.resolve() if args.test else None,
        args.sample.resolve() if args.sample else None,
    )
