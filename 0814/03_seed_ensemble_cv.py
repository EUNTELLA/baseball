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
TRAIN_YEARS = (2022, 2023)
VALID_YEAR = 2024
RECENT_YEAR = 2023
RECENT_WEIGHT = 2.0
SEEDS = (42, 2026, 814)
FIXED_OFFSET = -0.009683759059887942
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def build_model(features: list[str], seed: int) -> Pipeline:
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
                    random_state=seed,
                ),
            ),
        ]
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    rate = float(y_true.mean())
    brier = float(np.mean((prediction - y_true) ** 2))
    reference = rate * (1.0 - rate)
    return {
        "prediction_mean": float(prediction.mean()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    train_mask = train["season"].isin(TRAIN_YEARS)
    valid_mask = train["season"] == VALID_YEAR
    x_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, TARGET_COL]
    x_valid = train.loc[valid_mask, features]
    y_valid = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    season = train.loc[train_mask, "season"].to_numpy()
    sample_weight = np.where(season == RECENT_YEAR, RECENT_WEIGHT, 1.0)
    predictions = []
    rows = []

    for seed in SEEDS:
        model = build_model(features, seed)
        started = time.perf_counter()
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        prediction = model.predict_proba(x_valid)[:, 1]
        seconds = time.perf_counter() - started
        predictions.append(prediction)
        single = metrics(y_valid, prediction + FIXED_OFFSET)
        ensemble = metrics(
            y_valid, np.mean(predictions, axis=0) + FIXED_OFFSET
        )
        row = {
            "seed": seed,
            "seconds": seconds,
            "single_brier": single["brier"],
            "single_score": single["score"],
            "ensemble_size": len(predictions),
            "ensemble_brier": ensemble["brier"],
            "ensemble_score": ensemble["score"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    prediction_matrix = np.vstack(predictions).astype(np.float32)
    np.savez_compressed(
        RESULTS_DIR / "03_seed_ensemble_oof.npz",
        y=y_valid.astype(np.int8),
        predictions=prediction_matrix,
        seeds=np.asarray(SEEDS, dtype=np.int32),
    )
    final_prediction = prediction_matrix.mean(axis=0, dtype=np.float64)
    final_metrics = metrics(y_valid, final_prediction + FIXED_OFFSET)
    baseline_score = 587.379125068177
    payload = {
        "train_years": list(TRAIN_YEARS),
        "valid_year": VALID_YEAR,
        "recent_weight": RECENT_WEIGHT,
        "seeds": list(SEEDS),
        "fixed_offset": FIXED_OFFSET,
        "single_seed_42_baseline_score": baseline_score,
        "runs": rows,
        "final_ensemble": final_metrics,
        "improvement_over_seed_42": final_metrics["score"] - baseline_score,
        "should_package": final_metrics["score"] > baseline_score,
    }
    (RESULTS_DIR / "03_seed_ensemble_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 최종 결과 ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
