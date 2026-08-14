from __future__ import annotations

import argparse
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
TRAIN_YEARS = (2022, 2023)
VALID_YEAR = 2024
RECENT_YEAR = 2023
RECENT_WEIGHTS = (1.0, 1.25, 1.5, 1.75, 2.0)
FIXED_OFFSET = -0.009683759059887942
CAT_COLS = ["top_bottom", "game_type", "base_state"]


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
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
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


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    rate = float(y_true.mean())
    brier = float(np.mean((prediction - y_true) ** 2))
    reference = rate * (1.0 - rate)
    skill = 1.0 - brier / reference
    return {
        "target_rate": rate,
        "prediction_mean": float(prediction.mean()),
        "brier": brier,
        "score": float(100000.0 * skill),
    }


def main(recent_weights: tuple[float, ...], output_prefix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    train_mask = train["season"].isin(TRAIN_YEARS)
    valid_mask = train["season"] == VALID_YEAR
    x_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, TARGET_COL]
    x_valid = train.loc[valid_mask, features]
    y_valid = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    train_season = train.loc[train_mask, "season"].to_numpy()
    rows = []

    for recent_weight in recent_weights:
        sample_weight = np.where(train_season == RECENT_YEAR, recent_weight, 1.0)
        model = build_model(features)
        started = time.perf_counter()
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        prediction = model.predict_proba(x_valid)[:, 1]
        seconds = time.perf_counter() - started
        oracle_offset = float(np.mean(y_valid - prediction))
        raw = metrics(y_valid, prediction)
        fixed = metrics(y_valid, prediction + FIXED_OFFSET)
        oracle = metrics(y_valid, prediction + oracle_offset)
        row = {
            "recent_weight": recent_weight,
            "seconds": seconds,
            "oracle_offset": oracle_offset,
            "raw_brier": raw["brier"],
            "raw_score": raw["score"],
            "fixed_brier": fixed["brier"],
            "fixed_score": fixed["score"],
            "oracle_brier": oracle["brier"],
            "oracle_score": oracle["score"],
            "prediction_mean": raw["prediction_mean"],
            "target_rate": raw["target_rate"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    result = pd.DataFrame(rows).sort_values(
        ["fixed_brier", "oracle_brier"], ascending=True
    )
    result.to_csv(
        RESULTS_DIR / f"{output_prefix}.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "train_years": list(TRAIN_YEARS),
        "valid_year": VALID_YEAR,
        "recent_year": RECENT_YEAR,
        "fixed_offset": FIXED_OFFSET,
        "selection_rule": "minimum fixed-offset Brier on 2024 validation",
        "best": result.iloc[0].to_dict(),
        "all_results": result.to_dict(orient="records"),
    }
    (RESULTS_DIR / f"{output_prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 순위 ===")
    print(
        result[
            [
                "recent_weight",
                "fixed_score",
                "oracle_score",
                "oracle_offset",
                "seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(RECENT_WEIGHTS),
    )
    parser.add_argument("--output-prefix", default="02_weight_grid_cv")
    args = parser.parse_args()
    main(tuple(args.weights), args.output_prefix)
