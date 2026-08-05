from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "open" / "data" / "train.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ID_COL = "row_id"
TARGET_COL = "control_success"
VALID_YEARS = (2022, 2023, 2024)

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
CATBOOST_CAT_COLS = [
    "season",
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


def load_train() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    features = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    return df, features


def prepare_catboost_frame(
    frame: pd.DataFrame, cat_cols: list[str]
) -> pd.DataFrame:
    result = frame.copy()
    for col in cat_cols:
        if col in result:
            result[col] = result[col].astype("string").fillna("__MISSING__")
    return result


def brier_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    rate = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    reference = rate * (1.0 - rate)
    skill = 1.0 - brier / reference if reference > 0 else 0.0
    return {
        "n": int(len(y)),
        "target_rate": rate,
        "prediction_mean": float(p.mean()),
        "brier": brier,
        "brier_skill": skill,
        "competition_score": max(0.0, 100000.0 * skill),
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: n={metrics['n']:,}, "
        f"Brier={metrics['brier']:.8f}, "
        f"Skill={metrics['brier_skill']:.6f}, "
        f"Score={metrics['competition_score']:.2f}, "
        f"actual={metrics['target_rate']:.5f}, "
        f"pred={metrics['prediction_mean']:.5f}"
    )


class Timer:
    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.seconds = time.perf_counter() - self.started_at

