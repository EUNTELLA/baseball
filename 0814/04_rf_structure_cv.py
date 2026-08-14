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
FIXED_OFFSET = -0.009683759059887942
BASELINE_SCORE_100_TREES = 587.379125068177
CAT_COLS = ["top_bottom", "game_type", "base_state"]
CONFIGS = (
    {"name": "depth10_leaf200_sqrt", "max_depth": 10, "min_samples_leaf": 200, "max_features": "sqrt"},
    {"name": "depth14_leaf200_sqrt", "max_depth": 14, "min_samples_leaf": 200, "max_features": "sqrt"},
    {"name": "depth12_leaf100_sqrt", "max_depth": 12, "min_samples_leaf": 100, "max_features": "sqrt"},
    {"name": "depth12_leaf200_f30", "max_depth": 12, "min_samples_leaf": 200, "max_features": 0.3},
)


def build_model(features: list[str], config: dict[str, object]) -> Pipeline:
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
                    n_estimators=80,
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    max_features=config["max_features"],
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def score(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    prediction = np.clip(prediction + FIXED_OFFSET, 0.0, 1.0)
    rate = float(y.mean())
    brier = float(np.mean((prediction - y) ** 2))
    return brier, float(100000.0 * (1.0 - brier / (rate * (1.0 - rate))))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    train_mask = train["season"].isin((2022, 2023))
    valid_mask = train["season"] == 2024
    x_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, TARGET_COL]
    x_valid = train.loc[valid_mask, features]
    y_valid = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    sample_weight = np.where(train.loc[train_mask, "season"].to_numpy() == 2023, 2.0, 1.0)
    rows = []

    for config in CONFIGS:
        model = build_model(features, config)
        started = time.perf_counter()
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        prediction = model.predict_proba(x_valid)[:, 1]
        seconds = time.perf_counter() - started
        brier, competition_score = score(y_valid, prediction)
        row = {
            **config,
            "n_estimators": 80,
            "seconds": seconds,
            "brier": brier,
            "score": competition_score,
            "prediction_mean": float(prediction.mean()),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    result = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    result.to_csv(RESULTS_DIR / "04_rf_structure_cv.csv", index=False, encoding="utf-8-sig")
    best = result.iloc[0].to_dict()
    payload = {
        "train_years": [2022, 2023],
        "valid_year": 2024,
        "recent_weight": 2.0,
        "fixed_offset": FIXED_OFFSET,
        "baseline_score_100_trees": BASELINE_SCORE_100_TREES,
        "best": best,
        "all_results": result.to_dict(orient="records"),
        "should_package": best["score"] > BASELINE_SCORE_100_TREES,
    }
    (RESULTS_DIR / "04_rf_structure_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 순위 ===")
    print(result[["name", "score", "brier", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
