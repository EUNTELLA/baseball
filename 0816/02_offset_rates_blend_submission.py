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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "0814" / "results"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = ROOT / "open" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
OOF_PATH = SOURCE_RESULTS / "03_seed_ensemble_oof.npz"
BASE_MODEL_CANDIDATES = (
    SOURCE_RESULTS / "build" / "submit_rf_recent_weighted_20" / "model" / "weighted_recent_rf.pkl",
    SOURCE_RESULTS / "validation_submit_rf_recent_weighted_20" / "model" / "weighted_recent_rf.pkl",
    SOURCE_RESULTS / "build" / "submit_rf_recent_weighted_20_affine" / "model" / "weighted_recent_rf.pkl",
)
OUTPUT_STEM = "submit_rf_weighted20_offset70_rates30"
ID_COL = "row_id"
TARGET_COL = "control_success"
OFFSET_WEIGHT = 0.7
RATES_WEIGHT = 0.3


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=100.0)),
        ]
    )


def make_inference_script(offset: float) -> str:
    return f'''from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = {ID_COL!r}
TARGET_COL = {TARGET_COL!r}
OFFSET = {offset!r}
OFFSET_WEIGHT = {OFFSET_WEIGHT!r}
RATES_WEIGHT = {RATES_WEIGHT!r}


def predict_rows(test, base_model, rates_bundle):
    base_prediction = base_model.predict_proba(test.drop(columns=[ID_COL]))[:, 1]
    rate_columns = rates_bundle["feature_columns"]
    meta_frame = pd.concat(
        [
            pd.DataFrame({{
                "base_prediction": base_prediction,
                "base_prediction_sq": base_prediction ** 2,
            }}),
            test[rate_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    rates_prediction = rates_bundle["model"].predict(meta_frame)
    return np.clip(
        OFFSET_WEIGHT * (base_prediction + OFFSET)
        + RATES_WEIGHT * rates_prediction,
        0.0,
        1.0,
    )


def main():
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(root / "data" / "sample_submission.csv", encoding="utf-8-sig")
    base_model = joblib.load(root / "model" / "weighted_recent_rf.pkl")
    rates_bundle = joblib.load(root / "model" / "rates_calibrator.pkl")
    prediction = predict_rows(test, base_model, rates_bundle)
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_model_path = next((path for path in BASE_MODEL_CANDIDATES if path.exists()), None)
    if base_model_path is None:
        raise FileNotFoundError("가중치 2.0 기본 RF 모델을 찾지 못했습니다.")
    data = np.load(OOF_PATH)
    y = data["y"].astype(float)
    base_prediction = data["predictions"][0].astype(float)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    frame = train[train["season"] == 2024].reset_index(drop=True)
    rate_columns = [
        col
        for col in frame.columns
        if col.startswith("asof_") and col.endswith("_rate")
    ]
    meta_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "base_prediction": base_prediction,
                    "base_prediction_sq": base_prediction**2,
                }
            ),
            frame[rate_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    rates_model = make_ridge().fit(meta_frame, y)
    offset = float(np.mean(y - base_prediction))
    rates_bundle = {"model": rates_model, "feature_columns": rate_columns}

    build_dir = RESULTS_DIR / "build" / OUTPUT_STEM
    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "model").mkdir(parents=True)
    shutil.copy2(base_model_path, build_dir / "model" / "weighted_recent_rf.pkl")
    joblib.dump(rates_bundle, build_dir / "model" / "rates_calibrator.pkl", compress=3)
    (build_dir / "script.py").write_text(make_inference_script(offset), encoding="utf-8")
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
    validation_data = validation_dir / "data"
    validation_data.mkdir()
    shutil.copy2(DATA_DIR / "test.csv", validation_data / "test.csv")
    shutil.copy2(DATA_DIR / "sample_submission.csv", validation_data / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=validation_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    loaded_base = joblib.load(validation_dir / "model" / "weighted_recent_rf.pkl")
    loaded_rates = joblib.load(validation_dir / "model" / "rates_calibrator.pkl")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")

    def predict(frame_to_score: pd.DataFrame) -> np.ndarray:
        p = loaded_base.predict_proba(frame_to_score.drop(columns=[ID_COL]))[:, 1]
        x = pd.concat(
            [
                pd.DataFrame({"base_prediction": p, "base_prediction_sq": p**2}),
                frame_to_score[loaded_rates["feature_columns"]].reset_index(drop=True),
            ],
            axis=1,
        )
        q = loaded_rates["model"].predict(x)
        return np.clip(OFFSET_WEIGHT * (p + offset) + RATES_WEIGHT * q, 0.0, 1.0)

    batch = predict(test)
    single = np.array([predict(test.iloc[[index]])[0] for index in range(len(test))])
    independence_max_difference = float(np.max(np.abs(batch - single)))
    if independence_max_difference > 1e-12:
        raise ValueError(f"행 독립성 검사 실패: {independence_max_difference}")
    report = {
        "model": "weighted RF + 0.7 offset + 0.3 rates Ridge",
        "offset": offset,
        "strict_score": 593.1747942272358,
        "internal_cv_score": 603.3593171951624,
        "zip": str(zip_path),
        "zip_mib": zip_path.stat().st_size / 1024**2,
        "members": members,
        "row_independence_max_difference": independence_max_difference,
        "sample_stdout": completed.stdout.strip(),
    }
    (RESULTS_DIR / "02_offset_rates_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
