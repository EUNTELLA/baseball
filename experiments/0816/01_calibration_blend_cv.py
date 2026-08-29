from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
SOURCE_RESULTS = ROOT / "0814" / "results"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OOF_2024_PATH = SOURCE_RESULTS / "03_seed_ensemble_oof.npz"
OOF_2023_PATH = SOURCE_RESULTS / "08_base_oof_2023.npz"
TARGET_COL = "control_success"
ID_COL = "row_id"
BLEND_WEIGHTS = np.arange(0.0, 1.01, 0.1)


def make_ridge(alpha: float = 100.0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    rate = float(y.mean())
    brier = float(np.mean((prediction - y) ** 2))
    return {
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def meta_frame(frame: pd.DataFrame, prediction: np.ndarray, rate_cols: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "base_prediction": prediction,
                    "base_prediction_sq": prediction**2,
                }
            ),
            frame[rate_cols].reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    frame_2023 = train[train["season"] == 2023].reset_index(drop=True)
    frame_2024 = train[train["season"] == 2024].reset_index(drop=True)
    data_2023 = np.load(OOF_2023_PATH)
    data_2024 = np.load(OOF_2024_PATH)
    y_2023 = data_2023["y"].astype(float)
    p_2023 = data_2023["prediction"].astype(float)
    y_2024 = data_2024["y"].astype(float)
    p_2024 = data_2024["predictions"][0].astype(float)
    rate_cols = [
        col
        for col in frame_2024.columns
        if col.startswith("asof_") and col.endswith("_rate")
    ]

    # 2024 내부 연속 5-fold: 후보 탐색용 안정성 지표
    affine_cv = np.zeros(len(y_2024), dtype=float)
    offset_cv = np.zeros(len(y_2024), dtype=float)
    rates_cv = np.zeros(len(y_2024), dtype=float)
    x_2024 = meta_frame(frame_2024, p_2024, rate_cols)
    for train_index, valid_index in KFold(n_splits=5, shuffle=False).split(frame_2024):
        affine = LinearRegression().fit(p_2024[train_index, None], y_2024[train_index])
        affine_cv[valid_index] = affine.predict(p_2024[valid_index, None])
        fold_offset = float(np.mean(y_2024[train_index] - p_2024[train_index]))
        offset_cv[valid_index] = p_2024[valid_index] + fold_offset
        rates = make_ridge().fit(x_2024.iloc[train_index], y_2024[train_index])
        rates_cv[valid_index] = rates.predict(x_2024.iloc[valid_index])

    # 엄격 Walk-forward: 보정기는 2023에서만 학습하고 2024에 한 번 적용
    affine_strict_model = LinearRegression().fit(p_2023[:, None], y_2023)
    affine_strict = affine_strict_model.predict(p_2024[:, None])
    strict_offset = float(np.mean(y_2023 - p_2023))
    offset_strict = p_2024 + strict_offset
    x_2023 = meta_frame(frame_2023, p_2023, rate_cols)
    rates_strict_model = make_ridge().fit(x_2023, y_2023)
    rates_strict = rates_strict_model.predict(x_2024)

    rows = []
    anchors = {
        "affine_rates": (affine_cv, affine_strict),
        "offset_rates": (offset_cv, offset_strict),
    }
    for family, (anchor_cv, anchor_strict) in anchors.items():
        for rates_weight in BLEND_WEIGHTS:
            anchor_weight = 1.0 - rates_weight
            cv_prediction = anchor_weight * anchor_cv + rates_weight * rates_cv
            strict_prediction = anchor_weight * anchor_strict + rates_weight * rates_strict
            cv_result = metrics(y_2024, cv_prediction)
            strict_result = metrics(y_2024, strict_prediction)
            rows.append(
                {
                    "family": family,
                    "rates_weight": float(rates_weight),
                    "anchor_weight": float(anchor_weight),
                    "cv_brier": cv_result["brier"],
                    "cv_score": cv_result["score"],
                    "strict_brier": strict_result["brier"],
                    "strict_score": strict_result["score"],
                }
            )

    result = pd.DataFrame(rows)
    offset_baseline = result[
        (result["family"] == "offset_rates") & (result["rates_weight"] == 0.0)
    ].iloc[0]
    eligible = result[
        (result["cv_brier"] < offset_baseline["cv_brier"])
        & (result["strict_brier"] < offset_baseline["strict_brier"])
    ].sort_values(["strict_brier", "cv_brier"])
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    payload = {
        "selection_rule": "must improve both 2024 contiguous CV and 2023->2024 strict Brier over offset",
        "offset_baseline": offset_baseline.to_dict(),
        "selected": selected,
        "all_results": result.to_dict(orient="records"),
    }
    result.to_csv(RESULTS_DIR / "01_calibration_blend_cv.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "01_calibration_blend_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
