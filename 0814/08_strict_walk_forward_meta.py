from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OOF_2024_PATH = RESULTS_DIR / "03_seed_ensemble_oof.npz"
TARGET_COL = "control_success"
ID_COL = "row_id"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
RECENT_WEIGHT = 2.0
SEED = 42
ALPHAS = (10.0, 100.0, 1000.0)


def build_base_model(features: list[str]) -> Pipeline:
    numeric_cols = [col for col in features if col not in CAT_COLS]
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            CAT_COLS,
                        ),
                        ("num", SimpleImputer(strategy="median"), numeric_cols),
                    ]
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=200,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_ridge(alpha: float) -> Pipeline:
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
        "prediction_mean": float(prediction.mean()),
        "target_rate": rate,
    }


def meta_frame(frame: pd.DataFrame, prediction: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "base_prediction": prediction,
                    "base_prediction_sq": prediction**2,
                }
            ),
            frame[columns].reset_index(drop=True),
        ],
        axis=1,
    )


def choose_alpha(x: pd.DataFrame, y: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    folds = KFold(n_splits=5, shuffle=False)
    rows = []
    for alpha in ALPHAS:
        prediction = np.zeros(len(y), dtype=float)
        for train_index, valid_index in folds.split(x):
            model = make_ridge(alpha).fit(x.iloc[train_index], y[train_index])
            prediction[valid_index] = model.predict(x.iloc[valid_index])
        result = metrics(y, prediction)
        rows.append({"alpha": alpha, **result})
    best = min(rows, key=lambda row: row["brier"])
    return float(best["alpha"]), rows


def main() -> None:
    if not OOF_2024_PATH.exists():
        raise FileNotFoundError("먼저 03_seed_ensemble_cv.py를 실행하세요.")
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]

    base_train_mask = train["season"].isin((2021, 2022))
    valid_2023_mask = train["season"] == 2023
    base_model = build_base_model(features)
    base_weight = np.where(
        train.loc[base_train_mask, "season"].to_numpy() == 2022,
        RECENT_WEIGHT,
        1.0,
    )
    started = time.perf_counter()
    base_model.fit(
        train.loc[base_train_mask, features],
        train.loc[base_train_mask, TARGET_COL],
        clf__sample_weight=base_weight,
    )
    prediction_2023 = base_model.predict_proba(
        train.loc[valid_2023_mask, features]
    )[:, 1]
    base_seconds = time.perf_counter() - started
    y_2023 = train.loc[valid_2023_mask, TARGET_COL].to_numpy(dtype=float)
    frame_2023 = train.loc[valid_2023_mask].reset_index(drop=True)

    oof_2024 = np.load(OOF_2024_PATH)
    prediction_2024 = oof_2024["predictions"][0].astype(float)
    y_2024 = oof_2024["y"].astype(float)
    frame_2024 = train[train["season"] == 2024].reset_index(drop=True)
    if len(frame_2024) != len(y_2024):
        raise ValueError("2024 OOF 행 정렬이 일치하지 않습니다.")

    numeric_cols = [
        col
        for col in frame_2023.select_dtypes(include=[np.number]).columns
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
    rate_cols = [
        col
        for col in numeric_cols
        if col.startswith("asof_") and col.endswith("_rate")
    ]

    results = []
    saved_models: dict[str, object] = {}

    offset = float(np.mean(y_2023 - prediction_2023))
    results.append(
        {
            "method": "offset",
            "parameters": {"offset": offset},
            **metrics(y_2024, prediction_2024 + offset),
        }
    )

    affine = LinearRegression().fit(prediction_2023[:, None], y_2023)
    results.append(
        {
            "method": "affine",
            "parameters": {
                "intercept": float(affine.intercept_),
                "slope": float(affine.coef_[0]),
            },
            **metrics(y_2024, affine.predict(prediction_2024[:, None])),
        }
    )

    feature_sets = {
        "quadratic_ridge": [],
        "rates_ridge": rate_cols,
        "all_numeric_ridge": numeric_cols,
    }
    alpha_search = {}
    for method, columns in feature_sets.items():
        x_2023 = meta_frame(frame_2023, prediction_2023, columns)
        x_2024 = meta_frame(frame_2024, prediction_2024, columns)
        alpha, search_rows = choose_alpha(x_2023, y_2023)
        model = make_ridge(alpha).fit(x_2023, y_2023)
        valid_prediction = model.predict(x_2024)
        results.append(
            {
                "method": method,
                "parameters": {"alpha": alpha, "feature_count": int(x_2023.shape[1])},
                **metrics(y_2024, valid_prediction),
            }
        )
        alpha_search[method] = search_rows
        saved_models[method] = {"model": model, "feature_columns": columns}

    result_frame = pd.DataFrame(
        [
            {
                "method": row["method"],
                "brier": row["brier"],
                "score": row["score"],
                "prediction_mean": row["prediction_mean"],
                "target_rate": row["target_rate"],
            }
            for row in results
        ]
    ).sort_values("brier")
    best_method = str(result_frame.iloc[0]["method"])
    if best_method in saved_models:
        joblib.dump(
            saved_models[best_method],
            RESULTS_DIR / "08_strict_best_meta_model.pkl",
            compress=3,
        )
    np.savez_compressed(
        RESULTS_DIR / "08_base_oof_2023.npz",
        y=y_2023.astype(np.int8),
        prediction=prediction_2023.astype(np.float32),
    )
    payload = {
        "protocol": "train base on 2021-2022 -> predict 2023; fit calibration on 2023 -> evaluate once on 2024",
        "base_train_seconds": base_seconds,
        "alpha_selection": "5 contiguous folds within 2023 only",
        "results": results,
        "alpha_search_2023": alpha_search,
        "best_method": best_method,
    }
    result_frame.to_csv(RESULTS_DIR / "08_strict_walk_forward.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "08_strict_walk_forward.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result_frame.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
