from __future__ import annotations

import json
import time
from pathlib import Path

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
RECENT_WEIGHT = 2.0
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ENTITY_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]
FEATURE_SETS = {
    "all_features": set(),
    "drop_game_type": {"game_type"},
    "drop_entities": set(ENTITY_COLS),
    "drop_game_type_entities": {"game_type", *ENTITY_COLS},
}


def build_model(features: list[str]) -> Pipeline:
    cat_cols = [col for col in CAT_COLS if col in features]
    numeric_cols = [col for col in features if col not in cat_cols]
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            cat_cols,
                        ),
                        ("num", SimpleImputer(strategy="median"), numeric_cols),
                    ]
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=12,
                    min_samples_leaf=200,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
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
    }


def fit_predict(
    train: pd.DataFrame,
    features: list[str],
    train_years: tuple[int, int],
    valid_year: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_mask = train["season"].isin(train_years)
    valid_mask = train["season"] == valid_year
    sample_weight = np.where(
        train.loc[train_mask, "season"].to_numpy() == max(train_years),
        RECENT_WEIGHT,
        1.0,
    )
    model = build_model(features)
    started = time.perf_counter()
    model.fit(
        train.loc[train_mask, features],
        train.loc[train_mask, TARGET_COL],
        clf__sample_weight=sample_weight,
    )
    prediction = model.predict_proba(train.loc[valid_mask, features])[:, 1]
    seconds = time.perf_counter() - started
    y = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    return y, prediction, seconds


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    all_features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    rows = []

    for name, excluded in FEATURE_SETS.items():
        features = [col for col in all_features if col not in excluded]
        y_2023, prediction_2023, seconds_2023 = fit_predict(
            train, features, (2021, 2022), 2023
        )
        learned_offset = float(np.mean(y_2023 - prediction_2023))
        y_2024, prediction_2024, seconds_2024 = fit_predict(
            train, features, (2022, 2023), 2024
        )
        strict = metrics(y_2024, prediction_2024 + learned_offset)
        row = {
            "feature_set": name,
            "excluded": sorted(excluded),
            "feature_count": len(features),
            "offset_learned_on_2023": learned_offset,
            "seconds_2023": seconds_2023,
            "seconds_2024": seconds_2024,
            **strict,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    result = pd.DataFrame(
        [
            {
                "feature_set": row["feature_set"],
                "feature_count": row["feature_count"],
                "offset_learned_on_2023": row["offset_learned_on_2023"],
                "brier": row["brier"],
                "score": row["score"],
                "prediction_mean": row["prediction_mean"],
                "seconds": row["seconds_2023"] + row["seconds_2024"],
            }
            for row in rows
        ]
    ).sort_values("brier")
    payload = {
        "protocol": "2021-2022 -> 2023 offset; 2022-2023 -> 2024 evaluation",
        "recent_weight": RECENT_WEIGHT,
        "n_estimators": 80,
        "results": rows,
        "best_feature_set": str(result.iloc[0]["feature_set"]),
    }
    result.to_csv(RESULTS_DIR / "10_strict_feature_ablation.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "10_strict_feature_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 엄격 피처 순위 ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
