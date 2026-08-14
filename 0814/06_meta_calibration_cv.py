from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
OOF_PATH = Path(__file__).resolve().parent / "results" / "03_seed_ensemble_oof.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET_COL = "control_success"
ID_COL = "row_id"
N_SPLITS = 5


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    rate = float(y.mean())
    brier = float(np.mean((prediction - y) ** 2))
    return {
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
        "prediction_mean": float(prediction.mean()),
    }


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def main() -> None:
    data = np.load(OOF_PATH)
    y = data["y"].astype(float)
    base_prediction = data["predictions"][0].astype(float)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    valid = train[train["season"] == 2024].reset_index(drop=True)
    if len(valid) != len(y):
        raise ValueError("2024 OOF와 학습 데이터 행 수가 다릅니다.")

    rate_cols = [
        col
        for col in valid.columns
        if col.startswith("asof_") and col.endswith("_rate")
    ]
    numeric_cols = [
        col
        for col in valid.select_dtypes(include=[np.number]).columns
        if col
        not in {
            TARGET_COL,
            "season",
            "pitcher_id",
            "batter_id",
            "pitcher_team_id",
            "batter_team_id",
        }
    ]
    feature_sets = {
        "base_quadratic": [],
        "base_plus_rates": rate_cols,
        "base_plus_all_numeric": numeric_cols,
    }
    alphas = (10.0, 100.0, 1000.0)
    folds = list(KFold(n_splits=N_SPLITS, shuffle=False).split(valid))
    rows = []
    best_prediction = None
    best_key = None
    best_brier = float("inf")

    for feature_name, columns in feature_sets.items():
        base_frame = pd.DataFrame(
            {"base_prediction": base_prediction, "base_prediction_sq": base_prediction**2}
        )
        x = pd.concat([base_frame, valid[columns].reset_index(drop=True)], axis=1)
        for alpha in alphas:
            cv_prediction = np.zeros(len(y), dtype=float)
            for train_index, valid_index in folds:
                model = make_pipeline(alpha)
                model.fit(x.iloc[train_index], y[train_index])
                cv_prediction[valid_index] = model.predict(x.iloc[valid_index])
            result = metrics(y, cv_prediction)
            row = {
                "feature_set": feature_name,
                "alpha": alpha,
                "feature_count": int(x.shape[1]),
                **result,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["brier"] < best_brier:
                best_brier = result["brier"]
                best_prediction = cv_prediction.copy()
                best_key = (feature_name, alpha, columns)

    result_frame = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    feature_name, alpha, columns = best_key
    final_x = pd.concat(
        [
            pd.DataFrame(
                {
                    "base_prediction": base_prediction,
                    "base_prediction_sq": base_prediction**2,
                }
            ),
            valid[columns].reset_index(drop=True),
        ],
        axis=1,
    )
    final_model = make_pipeline(alpha).fit(final_x, y)
    import joblib

    joblib.dump(
        {"model": final_model, "feature_columns": list(columns)},
        RESULTS_DIR / "06_meta_calibrator.pkl",
        compress=3,
    )
    np.savez_compressed(
        RESULTS_DIR / "06_meta_calibration_oof.npz",
        y=y.astype(np.int8),
        prediction=best_prediction.astype(np.float32),
    )
    payload = {
        "cv": "5 contiguous folds within 2024 OOF",
        "best": result_frame.iloc[0].to_dict(),
        "all_results": result_frame.to_dict(orient="records"),
        "saved_calibrator": str(RESULTS_DIR / "06_meta_calibrator.pkl"),
    }
    result_frame.to_csv(RESULTS_DIR / "06_meta_calibration_cv.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "06_meta_calibration_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 순위 ===")
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
