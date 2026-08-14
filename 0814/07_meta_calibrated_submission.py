from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATA_DIR = ROOT / "open" / "data"
BASE_MODEL_CANDIDATES = (
    RESULTS_DIR / "build" / "submit_rf_recent_weighted_20" / "model" / "weighted_recent_rf.pkl",
    RESULTS_DIR / "validation_submit_rf_recent_weighted_20" / "model" / "weighted_recent_rf.pkl",
    RESULTS_DIR / "build" / "submit_rf_recent_weighted_20_affine" / "model" / "weighted_recent_rf.pkl",
)
META_MODEL_PATH = RESULTS_DIR / "06_meta_calibrator.pkl"
OUTPUT_STEM = "submit_rf_recent_weighted_20_meta_ridge"
ID_COL = "row_id"
TARGET_COL = "control_success"


def make_inference_script() -> str:
    return f'''from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = {ID_COL!r}
TARGET_COL = {TARGET_COL!r}


def predict_rows(test, base_model, meta_bundle):
    base_prediction = base_model.predict_proba(test.drop(columns=[ID_COL]))[:, 1]
    meta_frame = pd.DataFrame({{
        "base_prediction": base_prediction,
        "base_prediction_sq": base_prediction ** 2,
    }})
    feature_columns = meta_bundle["feature_columns"]
    meta_frame = pd.concat(
        [meta_frame, test[feature_columns].reset_index(drop=True)], axis=1
    )
    return np.clip(meta_bundle["model"].predict(meta_frame), 0.0, 1.0)


def main():
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(root / "data" / "sample_submission.csv", encoding="utf-8-sig")
    base_model = joblib.load(root / "model" / "weighted_recent_rf.pkl")
    meta_bundle = joblib.load(root / "model" / "meta_calibrator.pkl")
    prediction = predict_rows(test, base_model, meta_bundle)
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


def main() -> None:
    base_model_path = next(
        (path for path in BASE_MODEL_CANDIDATES if path.exists()), None
    )
    if base_model_path is None:
        raise FileNotFoundError(f"기본 RF 모델을 찾지 못했습니다: {BASE_MODEL_CANDIDATES}")
    if not META_MODEL_PATH.exists():
        raise FileNotFoundError(f"메타 보정기를 찾지 못했습니다: {META_MODEL_PATH}")

    build_dir = RESULTS_DIR / "build" / OUTPUT_STEM
    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "model").mkdir(parents=True)
    shutil.copy2(base_model_path, build_dir / "model" / "weighted_recent_rf.pkl")
    shutil.copy2(META_MODEL_PATH, build_dir / "model" / "meta_calibrator.pkl")
    (build_dir / "script.py").write_text(make_inference_script(), encoding="utf-8")
    (build_dir / "requirements.txt").write_text(
        f"numpy=={np.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"joblib=={joblib.__version__}\n",
        encoding="utf-8",
    )

    zip_path = RESULTS_DIR / f"{OUTPUT_STEM}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir))

    validation_dir = RESULTS_DIR / f"validation_{OUTPUT_STEM}"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    validation_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(archive.namelist())
        archive.extractall(validation_dir)
    expected = [
        "model/meta_calibrator.pkl",
        "model/weighted_recent_rf.pkl",
        "requirements.txt",
        "script.py",
    ]
    if members != expected:
        raise ValueError(f"ZIP 구조 불일치: {members}")

    validation_data = validation_dir / "data"
    validation_data.mkdir()
    shutil.copy2(DATA_DIR / "test.csv", validation_data / "test.csv")
    shutil.copy2(
        DATA_DIR / "sample_submission.csv",
        validation_data / "sample_submission.csv",
    )
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=validation_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    output = pd.read_csv(validation_dir / "output" / "submission.csv")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    if len(output) != len(test) or not output[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("출력 형식 또는 확률 범위가 잘못됐습니다.")

    base_model = joblib.load(validation_dir / "model" / "weighted_recent_rf.pkl")
    meta_bundle = joblib.load(validation_dir / "model" / "meta_calibrator.pkl")

    def predict(frame: pd.DataFrame) -> np.ndarray:
        base_prediction = base_model.predict_proba(frame.drop(columns=[ID_COL]))[:, 1]
        meta_frame = pd.DataFrame(
            {
                "base_prediction": base_prediction,
                "base_prediction_sq": base_prediction**2,
            }
        )
        meta_frame = pd.concat(
            [
                meta_frame,
                frame[meta_bundle["feature_columns"]].reset_index(drop=True),
            ],
            axis=1,
        )
        return np.clip(meta_bundle["model"].predict(meta_frame), 0.0, 1.0)

    batch = predict(test)
    single = np.array([predict(test.iloc[[index]])[0] for index in range(len(test))])
    independence_max_difference = float(np.max(np.abs(batch - single)))
    if independence_max_difference > 1e-12:
        raise ValueError(f"행 독립성 검사 실패: {independence_max_difference}")

    report = {
        "model": "weighted RF + row-wise Ridge meta calibration",
        "base_model": str(base_model_path),
        "meta_feature_columns": meta_bundle["feature_columns"],
        "local_cv_score": 610.9370987611884,
        "zip": str(zip_path),
        "zip_mib": zip_path.stat().st_size / 1024**2,
        "members": members,
        "row_independence_max_difference": independence_max_difference,
        "sample_stdout": completed.stdout.strip(),
    }
    (RESULTS_DIR / "07_meta_submission_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
