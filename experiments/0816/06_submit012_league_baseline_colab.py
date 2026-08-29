"""CatBoost d6·FE10 성공모델 대비 league-rate baseline 선별 검증.

Colab 사용 예:
  !python baseball/experiments/0816/06_submit012_league_baseline_colab.py \
      --train /content/drive/MyDrive/LG_Aimers/train.csv

2024 정답 평균은 모델 baseline 산출에 사용하지 않는다. 2019~2023으로 학습하고,
2021~2023 시즌 평균의 선형 추세로 2024 baseline을 외삽한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


SEEDS = [42, 7, 2024]
PARAMS = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    thread_count=-1,
    verbose=0,
    eval_metric="Logloss",
    early_stopping_rounds=100,
    grow_policy="SymmetricTree",
)
ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id",
]
CAT_COLS = BASE_CAT_COLS + ["count_state"]


def engineer(frame: pd.DataFrame, global_mean: float, smoothing: int = 30):
    """CatBoost d6·FE10 기준 모델과 동일한 행 단위 피처 생성."""
    result = frame.copy()
    for who in ("pitcher", "batter"):
        count = result[f"asof_{who}_n"].fillna(0)
        rate = result[f"asof_{who}_success_rate"].fillna(global_mean)
        result[f"smoothed_{who}_success_rate"] = (
            count * rate + smoothing * global_mean
        ) / (count + smoothing)
    result["platoon_advantage"] = (
        result["pitcher_hand"] == result["batter_hand"]
    ).astype(int)
    result["count_advantage"] = result["strikes_before"] - result["balls_before"]
    result["count_state"] = (
        result["balls_before"].astype(str) + "-" + result["strikes_before"].astype(str)
    )
    result["recent_control_momentum"] = (
        result["asof_pitcher_prev1_game_success_rate"]
        - result["asof_pitcher_success_rate"]
    )
    result["form_trend_5_1"] = (
        result["asof_pitcher_prev1_game_success_rate"]
        - result["asof_pitcher_prev5_game_success_rate"]
    )
    result["is_home"] = (result["top_bottom"] == "T").astype(int)
    result["pitcher_win_expectancy"] = np.where(
        result["is_home"] == 1,
        result["home_win_expectancy"], result["away_win_expectancy"],
    )
    result["is_coldstart_pitcher"] = result["asof_pitcher_n"].isna().astype(int)
    return result


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, dtype=float), 0, 1)
    target = np.asarray(target, dtype=float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "score": float(100000 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
    }


def forecast_rate(rates: dict[int, float], target_year: int) -> float:
    years = np.asarray(sorted(rates)[-3:], dtype=float)
    values = np.asarray([rates[int(year)] for year in years], dtype=float)
    slope, intercept = np.polyfit(years, values, 1)
    return float(np.clip(intercept + slope * target_year, 0.35, 0.65))


def main(train_path: Path, output: Path) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    train_mask = (frame["season"] <= 2023).to_numpy()
    valid_mask = (frame["season"] == 2024).to_numpy()
    global_mean = float(target[train_mask].mean())

    x = engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in CAT_COLS:
        x[column] = x[column].astype(str)
    cat_indices = [x.columns.get_loc(column) for column in CAT_COLS]

    rates = (
        frame.loc[train_mask]
        .groupby("season")[TARGET_COL]
        .mean()
        .to_dict()
    )
    forecast_2024 = forecast_rate(rates, 2024)
    train_baseline = logit(
        frame.loc[train_mask, "season"].map(rates).to_numpy()
    )[:, None]
    valid_baseline = np.full((int(valid_mask.sum()), 1), logit(forecast_2024))
    train_pool = Pool(
        x.loc[train_mask], target[train_mask],
        cat_features=cat_indices, baseline=train_baseline,
    )
    valid_pool = Pool(
        x.loc[valid_mask], target[valid_mask],
        cat_features=cat_indices, baseline=valid_baseline,
    )

    predictions = []
    iterations = []
    seconds = []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**PARAMS, random_seed=seed)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        predictions.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(int(model.get_best_iteration()))
        seconds.append(float(time.perf_counter() - started))
        print(
            f"seed={seed} iter={iterations[-1]} "
            f"score={bss(predictions[-1], target[valid_mask])['score']:.2f}",
            flush=True,
        )

    # 비교군도 현재 train.csv와 같은 행, 같은 CatBoost 환경에서 재학습한다.
    # 저장된 기준 모델 OOF와 open.zip의 데이터 버전이 다를 수 있기 때문이다.
    standard_train_pool = Pool(
        x.loc[train_mask], target[train_mask], cat_features=cat_indices
    )
    standard_valid_pool = Pool(
        x.loc[valid_mask], target[valid_mask], cat_features=cat_indices
    )
    standard_predictions = []
    standard_iterations = []
    print("--- 동일 데이터·환경 CatBoost d6·FE10 비교군 재학습 ---", flush=True)
    for seed in SEEDS:
        model = CatBoostClassifier(**PARAMS, random_seed=seed)
        model.fit(
            standard_train_pool,
            eval_set=standard_valid_pool,
            use_best_model=True,
        )
        standard_predictions.append(model.predict_proba(standard_valid_pool)[:, 1])
        standard_iterations.append(int(model.get_best_iteration()))
        print(
            f"standard seed={seed} iter={standard_iterations[-1]} "
            f"score={bss(standard_predictions[-1], target[valid_mask])['score']:.2f}",
            flush=True,
        )
    standard = np.mean(standard_predictions, axis=0)
    league_baseline = np.mean(predictions, axis=0)
    standard_metrics = bss(standard, target[valid_mask])
    candidate_metrics = bss(league_baseline, target[valid_mask])
    score_delta = candidate_metrics["score"] - standard_metrics["score"]
    payload = {
        "experiment": "CatBoost d6 FE10 structure vs league-rate baseline screening",
        "season_rates": {str(key): float(value) for key, value in rates.items()},
        "forecast_2024_rate": forecast_2024,
        "actual_2024_rate_for_reporting_only": float(target[valid_mask].mean()),
        "seeds": SEEDS,
        "best_iterations": iterations,
        "standard_best_iterations": standard_iterations,
        "seconds": seconds,
        "catboost_d6_fe10_success_model_reference": standard_metrics,
        "league_baseline_candidate": candidate_metrics,
        "score_delta": float(score_delta),
        "decision": "continue_to_7_seed_build" if score_delta > 3.0 else "reject",
        "rule": "continue only when 3-seed 2024 score improves by more than 3.0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("0816_submit012_league_baseline_result.json"),
    )
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve())
