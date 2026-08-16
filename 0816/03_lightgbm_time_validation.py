from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET_COL = "control_success"
ID_COL = "row_id"
RECENT_WEIGHT = 2.0
CAT_COLS = [
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
CONFIGS = {
    "leaves15": {"num_leaves": 15, "min_child_samples": 500},
    "leaves31": {"num_leaves": 31, "min_child_samples": 500},
    "leaves31_leaf1000": {"num_leaves": 31, "min_child_samples": 1000},
}


def prepare(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for col in CAT_COLS:
        result[col] = result[col].astype("category")
    return result


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    rate = float(np.mean(y))
    brier = float(np.mean((prediction - y) ** 2))
    return {
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
        "prediction_mean": float(np.mean(prediction)),
    }


def fit_predict(
    frame: pd.DataFrame,
    features: list[str],
    train_years: tuple[int, int],
    valid_year: int,
    config: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    train_mask = frame["season"].isin(train_years)
    valid_mask = frame["season"] == valid_year
    train_frame = frame.loc[train_mask]
    valid_frame = frame.loc[valid_mask]
    weights = np.where(
        train_frame["season"].to_numpy() == max(train_years), RECENT_WEIGHT, 1.0
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **config,
    )
    started = time.perf_counter()
    model.fit(
        prepare(train_frame, features),
        train_frame[TARGET_COL],
        sample_weight=weights,
        categorical_feature=CAT_COLS,
        eval_set=[(prepare(valid_frame, features), valid_frame[TARGET_COL])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    prediction = model.predict_proba(prepare(valid_frame, features))[:, 1]
    return (
        valid_frame[TARGET_COL].to_numpy(dtype=float),
        prediction,
        int(model.best_iteration_),
        time.perf_counter() - started,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in frame.columns if col not in (ID_COL, TARGET_COL)]
    rows = []
    best_payload = None

    for name, config in CONFIGS.items():
        y_2023, p_2023, iteration_2023, seconds_2023 = fit_predict(
            frame, features, (2021, 2022), 2023, config
        )
        y_2024, p_2024, iteration_2024, seconds_2024 = fit_predict(
            frame, features, (2022, 2023), 2024, config
        )
        affine = LinearRegression().fit(p_2023[:, None], y_2023)
        strict_prediction = affine.predict(p_2024[:, None])
        oracle_affine = LinearRegression().fit(p_2024[:, None], y_2024)
        oracle_prediction = oracle_affine.predict(p_2024[:, None])
        row = {
            "name": name,
            "config": config,
            "best_iteration_2023": iteration_2023,
            "best_iteration_2024": iteration_2024,
            "seconds_2023": seconds_2023,
            "seconds_2024": seconds_2024,
            "raw_2024": metrics(y_2024, p_2024),
            "strict_affine_2024": metrics(y_2024, strict_prediction),
            "oracle_affine_2024": metrics(y_2024, oracle_prediction),
            "strict_affine": {
                "slope": float(affine.coef_[0]),
                "intercept": float(affine.intercept_),
            },
            "oracle_affine": {
                "slope": float(oracle_affine.coef_[0]),
                "intercept": float(oracle_affine.intercept_),
            },
        }
        rows.append(row)
        if best_payload is None or (
            row["strict_affine_2024"]["brier"], row["oracle_affine_2024"]["brier"]
        ) < (
            best_payload["strict_affine_2024"]["brier"],
            best_payload["oracle_affine_2024"]["brier"],
        ):
            best_payload = row
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    payload = {
        "selection_rule": "strict affine Brier first, 2024 oracle affine Brier second",
        "rf_reference": {
            "public_score": 761.7509255482,
            "oracle_affine_2024_score": 598.21,
            "strict_affine_2024_score": 171.71,
        },
        "best": best_payload,
        "runs": rows,
    }
    (RESULTS_DIR / "03_lightgbm_time_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 최종 비교 ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
