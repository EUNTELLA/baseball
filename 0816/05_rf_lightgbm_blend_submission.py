from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "open" / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUILD_DIR = RESULTS_DIR / "build" / "submit_rf85_lgb15_affine"
ZIP_PATH = RESULTS_DIR / "submit_rf85_lgb15_affine.zip"
TARGET_COL = "control_success"
ID_COL = "row_id"
RF_WEIGHT = 0.85
LGB_WEIGHT = 0.15
SLOPE = 1.1518727059948277
INTERCEPT = -0.0888594826718308
LGB_ITERATIONS = 84


def load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def inference_script() -> str:
    return f'''from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = {ID_COL!r}
TARGET_COL = {TARGET_COL!r}
RF_WEIGHT = {RF_WEIGHT!r}
LGB_WEIGHT = {LGB_WEIGHT!r}
SLOPE = {SLOPE!r}
INTERCEPT = {INTERCEPT!r}
CAT_COLS = {['game_month', 'game_dayofweek', 'top_bottom', 'game_type', 'base_state', 'pitcher_id', 'batter_id', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id']!r}


def main():
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(root / "data" / "sample_submission.csv", encoding="utf-8-sig")
    features = [col for col in test.columns if col != ID_COL]
    x_rf = test[features]
    x_lgb = x_rf.copy()
    for col in CAT_COLS:
        x_lgb[col] = x_lgb[col].astype("category")
    rf = joblib.load(root / "model" / "random_forest.pkl")
    lgb = joblib.load(root / "model" / "lightgbm.pkl")
    raw = RF_WEIGHT * rf.predict_proba(x_rf)[:, 1] + LGB_WEIGHT * lgb.predict_proba(x_lgb)[:, 1]
    prediction = np.clip(INTERCEPT + SLOPE * raw, 0.0, 1.0)
    prediction_by_id = pd.Series(prediction, index=test[ID_COL].astype(str))
    submission = sample[[ID_COL]].copy()
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_by_id)
    if submission[TARGET_COL].isna().any():
        raise ValueError("sample_submission과 test의 row_id가 일치하지 않습니다.")
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    rf_module = load_module("rf_submission", ROOT / "0814" / "01_weighted_recent_rf_submission.py")
    lgb_module = load_module("lgb_validation", Path(__file__).resolve().parent / "03_lightgbm_time_validation.py")
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    recent = train[train["season"].isin([2023, 2024])].copy()
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    weights = np.where(recent["season"].to_numpy() == 2024, 2.0, 1.0)

    started = time.perf_counter()
    rf = rf_module.build_model(features)
    rf.fit(recent[features], recent[TARGET_COL], clf__sample_weight=weights)
    lgb = lightgbm.LGBMClassifier(
        objective="binary", n_estimators=LGB_ITERATIONS, learning_rate=0.03,
        num_leaves=15, min_child_samples=500, max_depth=-1, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=10.0,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb.fit(
        lgb_module.prepare(recent, features), recent[TARGET_COL],
        sample_weight=weights, categorical_feature=lgb_module.CAT_COLS,
    )
    train_seconds = time.perf_counter() - started

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    (BUILD_DIR / "model").mkdir(parents=True)
    joblib.dump(rf, BUILD_DIR / "model" / "random_forest.pkl", compress=3)
    joblib.dump(lgb, BUILD_DIR / "model" / "lightgbm.pkl", compress=3)
    (BUILD_DIR / "script.py").write_text(inference_script(), encoding="utf-8")
    (BUILD_DIR / "requirements.txt").write_text(
        "numpy\npandas\nscikit-learn\njoblib\nlightgbm==4.7.0\n", encoding="utf-8"
    )
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_DIR))

    validation = RESULTS_DIR / "validation_submit_rf85_lgb15_affine"
    if validation.exists():
        shutil.rmtree(validation)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = sorted(archive.namelist())
        archive.extractall(validation)
    (validation / "data").mkdir()
    shutil.copy2(DATA_DIR / "test.csv", validation / "data" / "test.csv")
    shutil.copy2(DATA_DIR / "sample_submission.csv", validation / "data" / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=validation, capture_output=True,
        text=True, check=True, timeout=600,
    )
    output = pd.read_csv(validation / "output" / "submission.csv")
    if not output[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("예측 확률 범위가 잘못되었습니다.")

    report = {
        "model": "RandomForest 85% + LightGBM 15% + affine",
        "train_rows": int(len(recent)), "train_seconds": train_seconds,
        "calibration": {"slope": SLOPE, "intercept": INTERCEPT},
        "zip": str(ZIP_PATH), "zip_mib": ZIP_PATH.stat().st_size / 1024**2,
        "members": members, "sample_rows": int(len(output)),
        "sample_stdout": completed.stdout.strip(),
    }
    (RESULTS_DIR / "submit_rf85_lgb15_affine_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
