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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET_COL = "control_success"
ID_COL = "row_id"
EXCLUDED = {"game_type"}
CAT_COLS = ["top_bottom", "base_state"]
BASELINE_STRICT_SCORE = 573.250788213675


def build_model(features: list[str]) -> Pipeline:
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
                    random_state=42,
                ),
            ),
        ]
    )


def fold(
    train: pd.DataFrame,
    features: list[str],
    train_years: tuple[int, int],
    valid_year: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float]:
    train_mask = train["season"].isin(train_years)
    valid_mask = train["season"] == valid_year
    weight = np.where(
        train.loc[train_mask, "season"].to_numpy() == max(train_years), 2.0, 1.0
    )
    model = build_model(features)
    started = time.perf_counter()
    model.fit(
        train.loc[train_mask, features],
        train.loc[train_mask, TARGET_COL],
        clf__sample_weight=weight,
    )
    prediction = model.predict_proba(train.loc[valid_mask, features])[:, 1]
    y = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    return model, y, prediction, time.perf_counter() - started


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [
        col
        for col in train.columns
        if col not in (ID_COL, TARGET_COL) and col not in EXCLUDED
    ]
    _, y_2023, prediction_2023, seconds_2023 = fold(
        train, features, (2021, 2022), 2023
    )
    offset_2023 = float(np.mean(y_2023 - prediction_2023))
    model_2024, y_2024, prediction_2024, seconds_2024 = fold(
        train, features, (2022, 2023), 2024
    )
    calibrated = np.clip(prediction_2024 + offset_2023, 0.0, 1.0)
    rate = float(y_2024.mean())
    brier = float(np.mean((calibrated - y_2024) ** 2))
    score = float(100000.0 * (1.0 - brier / (rate * (1.0 - rate))))
    offset_for_2025 = float(np.mean(y_2024 - prediction_2024))
    payload = {
        "feature_set": "drop_game_type",
        "n_estimators": 100,
        "recent_weight": 2.0,
        "offset_learned_on_2023": offset_2023,
        "strict_2024_brier": brier,
        "strict_2024_score": score,
        "baseline_all_features_score": BASELINE_STRICT_SCORE,
        "improvement": score - BASELINE_STRICT_SCORE,
        "offset_for_2025": offset_for_2025,
        "seconds_2023": seconds_2023,
        "seconds_2024": seconds_2024,
        "confirmed": score > BASELINE_STRICT_SCORE,
    }
    np.savez_compressed(
        RESULTS_DIR / "11_drop_game_type_oof.npz",
        y_2024=y_2024.astype(np.int8),
        prediction_2024=prediction_2024.astype(np.float32),
        prediction_2023=prediction_2023.astype(np.float32),
        y_2023=y_2023.astype(np.int8),
    )
    joblib.dump(model_2024, RESULTS_DIR / "11_drop_game_type_2024_fold.pkl", compress=3)
    (RESULTS_DIR / "11_confirm_drop_game_type.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
