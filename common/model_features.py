"""공식 입력 한 행만 사용해 CatBoost 공통 피처를 생성한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
CAT_COLS = BASE_CAT_COLS + ["count_state"]


def _number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def engineer(frame: pd.DataFrame, league_rate: float, smoothing: float = 30.0) -> pd.DataFrame:
    """학습·평가 행 사이의 집계를 만들지 않는 결정적 변환이다."""
    result = frame.copy()
    for entity in ("pitcher", "batter"):
        count = _number(result, f"asof_{entity}_n").clip(lower=0)
        rate = pd.to_numeric(result[f"asof_{entity}_success_rate"], errors="coerce").fillna(league_rate)
        result[f"smoothed_{entity}_success_rate"] = (
            count * rate + smoothing * league_rate
        ) / (count + smoothing)

    result["platoon_advantage"] = result["pitcher_hand"].eq(result["batter_hand"]).astype("int8")
    balls = _number(result, "balls_before").astype(int)
    strikes = _number(result, "strikes_before").astype(int)
    result["count_advantage"] = strikes - balls
    result["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    career = pd.to_numeric(result["asof_pitcher_success_rate"], errors="coerce")
    recent1 = pd.to_numeric(result["asof_pitcher_prev1_game_success_rate"], errors="coerce")
    recent5 = pd.to_numeric(result["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    result["recent_control_momentum"] = recent1 - career
    result["form_trend_5_1"] = recent1 - recent5
    result["is_home"] = result["top_bottom"].astype(str).eq("T").astype("int8")
    result["pitcher_win_expectancy"] = np.where(
        result["is_home"].eq(1),
        _number(result, "home_win_expectancy"),
        _number(result, "away_win_expectancy"),
    )
    result["is_coldstart_pitcher"] = result["asof_pitcher_n"].isna().astype("int8")
    return result


def prepare(frame: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    categorical = set(categorical_columns)
    for column in feature_columns:
        if column in categorical:
            result[column] = result[column].astype(str)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result
