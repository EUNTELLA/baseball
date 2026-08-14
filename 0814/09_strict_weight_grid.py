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
CAT_COLS = ["top_bottom", "game_type", "base_state"]
WEIGHTS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


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


def load_2024_grid() -> pd.DataFrame:
    paths = [
        RESULTS_DIR / "02_weight_grid_cv.csv",
        RESULTS_DIR / "02_weight_upper_grid_cv.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if len(frames) != len(paths):
        raise FileNotFoundError("먼저 02_weight_grid_cv.py의 기본/상단 탐색을 실행하세요.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("recent_weight")


def adjusted_metrics(
    raw_brier: float,
    prediction_mean: float,
    target_rate: float,
    offset: float,
) -> tuple[float, float]:
    # clip에 걸리는 현재 확률이 사실상 없으므로 Brier의 상수 이동 항등식을 사용한다.
    brier = raw_brier + 2.0 * offset * (prediction_mean - target_rate) + offset**2
    reference = target_rate * (1.0 - target_rate)
    score = 100000.0 * (1.0 - brier / reference)
    return float(brier), float(score)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_2024 = load_2024_grid().set_index("recent_weight")
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    features = [col for col in train.columns if col not in (ID_COL, TARGET_COL)]
    train_mask = train["season"].isin((2021, 2022))
    valid_mask = train["season"] == 2023
    x_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, TARGET_COL]
    x_valid = train.loc[valid_mask, features]
    y_valid = train.loc[valid_mask, TARGET_COL].to_numpy(dtype=float)
    season = train.loc[train_mask, "season"].to_numpy()
    rows = []

    for recent_weight in WEIGHTS:
        model = build_model(features)
        sample_weight = np.where(season == 2022, recent_weight, 1.0)
        started = time.perf_counter()
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        prediction_2023 = model.predict_proba(x_valid)[:, 1]
        seconds = time.perf_counter() - started
        learned_offset = float(np.mean(y_valid - prediction_2023))
        source = grid_2024.loc[recent_weight]
        strict_brier, strict_score = adjusted_metrics(
            float(source["raw_brier"]),
            float(source["prediction_mean"]),
            float(source["target_rate"]),
            learned_offset,
        )
        row = {
            "recent_weight": recent_weight,
            "offset_learned_on_2023": learned_offset,
            "prediction_mean_2023": float(prediction_2023.mean()),
            "target_rate_2023": float(y_valid.mean()),
            "strict_brier_2024": strict_brier,
            "strict_score_2024": strict_score,
            "seconds": seconds,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    result = pd.DataFrame(rows).sort_values("strict_brier_2024").reset_index(drop=True)
    payload = {
        "protocol": "fit RF on 2021-2022; learn offset on 2023; apply to matching 2024 OOF",
        "selection_rule": "minimum strict 2024 Brier",
        "best": result.iloc[0].to_dict(),
        "all_results": result.to_dict(orient="records"),
    }
    result.to_csv(RESULTS_DIR / "09_strict_weight_grid.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "09_strict_weight_grid.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 엄격 순위 ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
