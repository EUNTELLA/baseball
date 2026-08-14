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
BUILD_DIR = RESULTS_DIR / "build_recent_rf"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
TARGET_COL = "control_success"
ID_COL = "row_id"
DEFAULT_TRAIN_YEARS = (2023, 2024)
DEFAULT_OUTPUT_STEM = "submit_rf_recent_calibrated"

# 2023 학습 -> 2024 검증에서 측정한 평균 잔차 E[y - p].
CALIBRATION_OFFSET = -0.009683759059887942
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


def inference_script() -> str:
    return f'''from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = {ID_COL!r}
TARGET_COL = {TARGET_COL!r}
CALIBRATION_OFFSET = {CALIBRATION_OFFSET!r}


def main():
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(root / "data" / "sample_submission.csv", encoding="utf-8-sig")
    model = joblib.load(root / "model" / "recent_rf.pkl")
    features = test.drop(columns=[ID_COL])
    prediction = np.clip(
        model.predict_proba(features)[:, 1] + CALIBRATION_OFFSET, 0.0, 1.0
    )
    prediction_by_id = pd.Series(prediction, index=test[ID_COL].astype(str))
    submission = sample[[ID_COL]].copy()
    submission[TARGET_COL] = submission[ID_COL].astype(str).map(prediction_by_id)
    if submission[TARGET_COL].isna().any():
        raise ValueError("sample_submission과 test의 row_id가 일치하지 않습니다.")
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    print("rows=", len(submission), "mean=", float(submission[TARGET_COL].mean()))


if __name__ == "__main__":
    main()
'''


def main(train_years: tuple[int, ...], output_stem: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    recent = train[train["season"].isin(train_years)].copy()
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    model = build_model(features)
    started = time.perf_counter()
    model.fit(recent[features], recent[TARGET_COL])
    train_seconds = time.perf_counter() - started

    package_dir = BUILD_DIR / output_stem
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "model").mkdir(parents=True)
    joblib.dump(model, package_dir / "model" / "recent_rf.pkl", compress=3)
    (package_dir / "script.py").write_text(inference_script(), encoding="utf-8")
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
    expected = ["model/recent_rf.pkl", "requirements.txt", "script.py"]
    if members != expected:
        raise ValueError(f"ZIP 구조 불일치: {members}")
    shutil.copytree(DATA_DIR, validation_dir / "data")
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=validation_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    output = pd.read_csv(validation_dir / "output" / "submission.csv")
    if len(output) != len(pd.read_csv(TEST_PATH)):
        raise ValueError("샘플 추론 행 수가 test와 다릅니다.")
    if not output[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("예측 확률 범위가 잘못되었습니다.")

    report = {
        "model": "recent-window RandomForest",
        "train_years": list(train_years),
        "calibration_offset": CALIBRATION_OFFSET,
        "calibration_source_2023_to_2024": {
            "raw_score": 560.1049224554399,
            "oracle_centered_score": 597.6439894250562,
        },
        "train_rows": int(len(recent)),
        "train_seconds": train_seconds,
        "zip": str(zip_path),
        "zip_mib": zip_path.stat().st_size / 1024**2,
        "members": members,
        "sample_stdout": completed.stdout.strip(),
    }
    (RESULTS_DIR / f"{output_stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRAIN_YEARS),
        help="최종 모델 학습에 사용할 시즌 목록",
    )
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    main(tuple(args.train_years), args.output_stem)
