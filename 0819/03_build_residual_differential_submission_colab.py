"""기존 Public 997 ZIP에 과거 시즌 예측 오차 기반 보정표를 추가한다."""
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


ID_COL = "row_id"
TARGET_COL = "control_success"
SOURCE_FOLDS = (2023, 2024)
SEEDS = (42, 7, 2024, 99, 1, 123, 777)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FEATURE_PATH = ROOT / "0816" / "reference_catboost_best" / "common" / "features.py"
BASE_ZIP = ROOT / "0816" / "results" / "submit_catboost_train_trend_shift.zip"
OUTPUT_ZIP = SCRIPT_DIR / "results" / "submit_catboost_residual_differential.zip"
BUILD_DIR = SCRIPT_DIR / "results" / "build_residual_differential"
TABLE_FILE = "residual_differential.json"
AXES = (("hand", 1000.0), ("two_strikes", 1000.0), ("runners_on", 2000.0))


def load_features_module():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contexts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "hand": (frame["pitcher_hand"].astype(str) == frame["batter_hand"].astype(str)).astype(int).to_numpy(),
        "two_strikes": (frame["strikes_before"].fillna(-1).astype(int) == 2).astype(int).to_numpy(),
        "runners_on": (frame["num_runners_on"].fillna(0).astype(float) > 0).astype(int).to_numpy(),
    }


def params(seed: int) -> dict:
    return {
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
        "task_type": "GPU",
        "devices": "0",
    }


def differential_table(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    selected: np.ndarray,
    shrinkage: float,
) -> pd.Series:
    grouped = (
        pd.DataFrame({"pitcher": pitcher[selected], "context": context[selected], "residual": residual[selected]})
        .groupby(["pitcher", "context"])["residual"]
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


def inference_block() -> str:
    return '''
    # Train의 2023·2024를 학습하지 않은 예측 오차로 만든 투수별 보정표.
    # test에서는 현재 행의 값만 조회하며 다른 test 행을 집계하지 않는다.
    differential_path = os.path.join(BASE, "model", "residual_differential.json")
    if os.path.exists(differential_path):
        differential = json.load(open(differential_path, encoding="utf-8"))
        pitcher_key = pd.to_numeric(test["pitcher_id"], errors="coerce").astype("Int64").astype(str)
        same_hand = test["pitcher_hand"].astype(str) == test["batter_hand"].astype(str)
        two_strikes = pd.to_numeric(test["strikes_before"], errors="coerce").fillna(-1).astype(int) == 2
        runners_on = pd.to_numeric(test["num_runners_on"], errors="coerce").fillna(0) > 0
        correction = np.zeros(len(test), dtype=float)
        for name, context in (("hand", same_hand), ("two_strikes", two_strikes), ("runners_on", runners_on)):
            values = pitcher_key.map(differential["tables"][name]).fillna(0.0).to_numpy(dtype=float)
            correction += values * np.where(np.asarray(context), 0.5, -0.5)
        p = np.clip(p + correction, 1e-6, 1 - 1e-6)
'''


def main(train_path: Path, test_path: Path, sample_path: Path, report_path: Path) -> None:
    if not BASE_ZIP.exists():
        raise FileNotFoundError(BASE_ZIP)
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").astype(int).to_numpy()
    axis_contexts = contexts(frame)
    feature_module = load_features_module()
    oof = np.full(len(frame), np.nan, dtype=float)
    training = []

    for fold in SOURCE_FOLDS:
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
            model = CatBoostClassifier(**params(seed))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            predictions.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            seconds.append(float(time.perf_counter() - started))
            print(f"source fold={fold} seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
            del model
            gc.collect()
        oof[valid_mask] = np.mean(predictions, axis=0)
        training.append({"fold": fold, "best_iterations": iterations, "seconds": seconds})
        del features, train_pool, valid_pool, predictions
        gc.collect()

    selected = np.isin(season, SOURCE_FOLDS)
    residual = target.astype(float) - oof
    if np.isnan(residual[selected]).any():
        raise RuntimeError("OOF 잔차 누락")
    tables = {}
    table_stats = {}
    for name, shrinkage in AXES:
        table = differential_table(pitcher, axis_contexts[name], residual, selected, shrinkage)
        tables[name] = {str(int(key)): float(value) for key, value in table.items()}
        table_stats[name] = {
            "shrinkage": shrinkage,
            "pitchers": int(len(table)),
            "median_absolute_difference": float(table.abs().median()) if len(table) else 0.0,
        }
        print(f"table={name} pitchers={len(table)} median|d|={table_stats[name]['median_absolute_difference']:.6f}")

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(BUILD_DIR)
    table_payload = {
        "source": "official train; predictions made without fitting the predicted season",
        "seeds": list(SEEDS),
        "axes": table_stats,
        "tables": tables,
        "test_aggregate_used": False,
    }
    (BUILD_DIR / "model" / TABLE_FILE).write_text(
        json.dumps(table_payload, ensure_ascii=False), encoding="utf-8"
    )
    script_path = BUILD_DIR / "script.py"
    script = script_path.read_text(encoding="utf-8")
    anchor = "    off = meta.get(\"offset\")\n"
    if anchor not in script:
        raise RuntimeError("기준 추론 코드의 offset 위치를 찾지 못했습니다.")
    script_path.write_text(script.replace(anchor, inference_block() + "\n" + anchor), encoding="utf-8")

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

    verify_dir = SCRIPT_DIR / "results" / "verify_residual_differential"
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
        "model": "Public 997 CatBoost + OOF residual differential hand/2S/runners",
        "official_train_only": True,
        "test_aggregate_used": False,
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "output_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "source_folds": list(SOURCE_FOLDS),
        "training": training,
        "tables": table_stats,
        "members": members,
        "zip_test_error": bad_member,
        "sample_rows": int(len(submission)),
        "sample_missing": int(submission[TARGET_COL].isna().sum()),
        "sample_min": float(submission[TARGET_COL].min()),
        "sample_max": float(submission[TARGET_COL].max()),
        "sample_mean": float(submission[TARGET_COL].mean()),
        "sample_stdout": completed.stdout.strip(),
    }
    write_target = report_path.resolve()
    write_target.parent.mkdir(parents=True, exist_ok=True)
    write_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.train.resolve(), args.test.resolve(), args.sample.resolve(), args.report.resolve())
