from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "open" / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUILD_DIR = RESULTS_DIR / "build"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
TARGET_COL = "control_success"
ID_COL = "row_id"
DEFAULT_OUTPUT_STEM = "submit_rf_recent_weighted_15"
DEFAULT_RECENT_WEIGHT = 1.5
CALIBRATION_OFFSET = -0.009683759059887942
DEFAULT_CALIBRATION_SLOPE = 1.0
DEFAULT_CALIBRATION_INTERCEPT = CALIBRATION_OFFSET
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def build_model(features: list[str]) -> Pipeline:
    numeric_cols = [col for col in features if col not in CAT_COLS]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_COLS,
            ),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ]
    )
    return Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=200,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_inference_script(calibration_slope: float, calibration_intercept: float) -> str:
    return f'''from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = {ID_COL!r}
TARGET_COL = {TARGET_COL!r}
CALIBRATION_SLOPE = {calibration_slope!r}
CALIBRATION_INTERCEPT = {calibration_intercept!r}


def main():
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(root / "data" / "sample_submission.csv", encoding="utf-8-sig")
    model = joblib.load(root / "model" / "weighted_recent_rf.pkl")
    prediction = np.clip(
        CALIBRATION_INTERCEPT
        + CALIBRATION_SLOPE
        * model.predict_proba(test.drop(columns=[ID_COL]))[:, 1],
        0.0,
        1.0,
    )
    prediction_by_id = pd.Series(prediction, index=test[ID_COL].astype(str))
    submission = sample[[ID_COL]].copy()
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_by_id)
    if submission[TARGET_COL].isna().any():
        raise ValueError("sample_submission과 test의 row_id가 일치하지 않습니다.")
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    print("Saved:", output_dir / "submission.csv", "rows=", len(submission))


if __name__ == "__main__":
    main()
'''


def main(
    recent_weight: float,
    output_stem: str,
    calibration_slope: float,
    calibration_intercept: float,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    year_weights = {2023: 1.0, 2024: recent_weight}
    recent = train[train["season"].isin(year_weights)].copy()
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    sample_weight = recent["season"].map(year_weights).to_numpy(dtype=float)

    model = build_model(features)
    started = time.perf_counter()
    model.fit(
        recent[features],
        recent[TARGET_COL],
        clf__sample_weight=sample_weight,
    )
    train_seconds = time.perf_counter() - started

    package_dir = BUILD_DIR / output_stem
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "model").mkdir(parents=True)
    joblib.dump(model, package_dir / "model" / "weighted_recent_rf.pkl", compress=3)
    (package_dir / "script.py").write_text(
        make_inference_script(calibration_slope, calibration_intercept),
        encoding="utf-8",
    )
    (package_dir / "requirements.txt").write_text(
        f"numpy=={np.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"joblib=={joblib.__version__}\n",
        encoding="utf-8",
    )

    zip_path = RESULTS_DIR / f"{output_stem}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir))

    validation_dir = RESULTS_DIR / f"validation_{output_stem}"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    validation_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(archive.namelist())
        archive.extractall(validation_dir)
    expected = ["model/weighted_recent_rf.pkl", "requirements.txt", "script.py"]
    if members != expected:
        raise ValueError(f"ZIP 구조 불일치: {members}")

    data_dir = validation_dir / "data"
    data_dir.mkdir()
    shutil.copy2(TEST_PATH, data_dir / "test.csv")
    shutil.copy2(SAMPLE_PATH, data_dir / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=validation_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    output = pd.read_csv(validation_dir / "output" / "submission.csv")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    if len(output) != len(test):
        raise ValueError("샘플 추론 행 수가 test와 다릅니다.")
    if not output[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("예측 확률 범위가 잘못되었습니다.")

    x = test.drop(columns=[ID_COL])
    batch_prediction = model.predict_proba(x)[:, 1]
    single_prediction = np.array(
        [model.predict_proba(x.iloc[[index]])[0, 1] for index in range(len(x))]
    )
    independence_max_difference = float(
        np.max(np.abs(batch_prediction - single_prediction))
    )
    if independence_max_difference > 1e-12:
        raise ValueError(
            f"행 독립성 검사 실패: 최대 차이={independence_max_difference}"
        )

    report = {
        "model": "weighted recent-window RandomForest",
        "year_weights": year_weights,
        "calibration": {
            "formula": "intercept + slope * probability",
            "slope": calibration_slope,
            "intercept": calibration_intercept,
        },
        "train_rows": int(len(recent)),
        "effective_weight_sum": float(sample_weight.sum()),
        "train_seconds": train_seconds,
        "zip": str(zip_path),
        "zip_mib": zip_path.stat().st_size / 1024**2,
        "members": members,
        "row_independence_max_difference": independence_max_difference,
        "sample_stdout": completed.stdout.strip(),
    }
    (RESULTS_DIR / f"{output_stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-weight", type=float, default=DEFAULT_RECENT_WEIGHT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--calibration-slope", type=float, default=DEFAULT_CALIBRATION_SLOPE
    )
    parser.add_argument(
        "--calibration-intercept", type=float, default=DEFAULT_CALIBRATION_INTERCEPT
    )
    args = parser.parse_args()
    main(
        args.recent_weight,
        args.output_stem,
        args.calibration_slope,
        args.calibration_intercept,
    )
