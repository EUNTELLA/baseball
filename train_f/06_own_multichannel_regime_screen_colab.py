"""자체 성공·실패 채널로 Futures 다중채널 residual regime을 검증한다."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


YEARS = (2023, 2024)
SEEDS = (17, 42, 777)
RECENT_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
SCALES = (0.01, 0.025, 0.05, 0.075)
SHIFT_DELTA = -0.0416386466 - (-0.03842671927234861)
CAT_COLS = (
    "count", "hand", "base_state", "top_bottom",
    "pitcher_team_id", "batter_team_id",
)


def load_asset(directory: Path, year: int):
    asset = np.load(directory / f"components_{year}.npz", allow_pickle=True)
    return {name: asset[name] for name in asset.files}


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.asarray(value, float)))


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def features(rows: pd.DataFrame, asset: dict[str, np.ndarray]):
    success = np.clip(asset["success"].astype(float), 1e-6, 1 - 1e-6)
    middle_reverse = np.clip(asset["mr"].astype(float), 1e-6, 1 - 1e-6)
    large_miss = np.clip(asset["wayoff"].astype(float), 1e-6, 1 - 1e-6)
    anchor = np.clip(asset["anchor"].astype(float), 1e-6, 1 - 1e-6)
    failure_complement = np.clip(1 - middle_reverse - large_miss, 1e-6, 1 - 1e-6)
    result = pd.DataFrame({
        "anchor_logit": logit(anchor),
        "success_logit": logit(success),
        "middle_reverse_logit": logit(middle_reverse),
        "large_miss_logit": logit(large_miss),
        "failure_complement_logit": logit(failure_complement),
        "success_minus_anchor": success - anchor,
        "failure_minus_anchor": failure_complement - anchor,
        "failure_mass": middle_reverse + large_miss,
        "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
        "hand": rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str),
        "base_state": rows["base_state"].astype(str),
        "top_bottom": rows["top_bottom"].astype(str),
        "pitcher_team_id": rows["pitcher_team_id"].astype(str),
        "batter_team_id": rows["batter_team_id"].astype(str),
        "inning": pd.to_numeric(rows["inning"], errors="coerce"),
        "outs": pd.to_numeric(rows["outs_before"], errors="coerce"),
        "runners": pd.to_numeric(rows["num_runners_on"], errors="coerce"),
        "score_diff": pd.to_numeric(rows["score_diff_pitcher_team"], errors="coerce"),
        "li": pd.to_numeric(rows["li"], errors="coerce"),
        "pitcher_n": np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0)),
        "pitcher_rate": pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce"),
        "recent1": pd.to_numeric(rows["asof_pitcher_prev1_game_success_rate"], errors="coerce"),
        "recent3": pd.to_numeric(rows["asof_pitcher_prev3_game_success_rate"], errors="coerce"),
        "recent5": pd.to_numeric(rows["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
    })
    for column in CAT_COLS:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def params(seed, task_type):
    result = {
        "iterations": 400, "depth": 3, "learning_rate": 0.025,
        "loss_function": "RMSE", "l2_leaf_reg": 100,
        "random_strength": 0.2, "bootstrap_type": "Bernoulli",
        "subsample": 0.8, "random_seed": seed,
        "allow_writing_files": False, "verbose": False,
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def bootstrap(ids, base, candidate, target, seed, count=500):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"id": ids.astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, rows = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(count):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / rows[sample].sum() > 0)
    return float(positive / count)


def fit_channel(x_train, y_train, x_valid, weights, task_type, label):
    train_pool = Pool(x_train, y_train, weight=weights, cat_features=list(CAT_COLS))
    valid_pool = Pool(x_valid, cat_features=list(CAT_COLS))
    members, seconds = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostRegressor(**params(seed, task_type))
        model.fit(train_pool)
        members.append(model.predict(valid_pool))
        seconds.append(float(time.perf_counter() - started))
        print(f"{label} seed={seed} sec={seconds[-1]:.1f}", flush=True)
    return np.mean(members, axis=0), seconds


def main(component_dir: Path, train_path: Path, output: Path, task_type: str):
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    rows_by_year = {
        year: frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    assets = {year: load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    feature_by_year = {}
    f_masks = {}
    for year in (2022, 2023, 2024):
        rows, asset = rows_by_year[year], assets[year]
        if not np.array_equal(rows["row_id"].astype(str), asset["row_id"].astype(str)):
            raise ValueError(f"{year} row_id 정렬 불일치")
        f_masks[year] = rows["game_type"].astype(str).eq("F").to_numpy()
        feature_by_year[year] = features(rows, asset).loc[f_masks[year]].reset_index(drop=True)

    folds = []
    for valid_year in YEARS:
        train_years = tuple(year for year in (2022, 2023) if year < valid_year)
        recent_year = valid_year - 1
        x_all = pd.concat([feature_by_year[year] for year in train_years], ignore_index=True)
        base_all = np.concatenate([
            sigmoid(logit(assets[year]["anchor"][f_masks[year]]) + SHIFT_DELTA)
            for year in train_years
        ])
        y_all = np.concatenate([assets[year]["target"][f_masks[year]] for year in train_years])
        season_all = np.concatenate([
            np.full(int(f_masks[year].sum()), year, dtype=int) for year in train_years
        ])
        residual_all = y_all - base_all
        temporal_weights = np.power(0.55, valid_year - 1 - season_all)
        x_recent = feature_by_year[recent_year]
        base_recent = sigmoid(logit(assets[recent_year]["anchor"][f_masks[recent_year]]) + SHIFT_DELTA)
        y_recent = assets[recent_year]["target"][f_masks[recent_year]]
        residual_recent = y_recent - base_recent
        x_valid = feature_by_year[valid_year]
        valid_rows = rows_by_year[valid_year].loc[f_masks[valid_year]].reset_index(drop=True)
        y_valid = assets[valid_year]["target"][f_masks[valid_year]]
        base_valid = sigmoid(logit(assets[valid_year]["anchor"][f_masks[valid_year]]) + SHIFT_DELTA)

        all_correction, all_seconds = fit_channel(
            x_all, residual_all, x_valid, temporal_weights, task_type, f"fold={valid_year} all"
        )
        recent_correction, recent_seconds = fit_channel(
            x_recent, residual_recent, x_valid, np.ones(len(x_recent)),
            task_type, f"fold={valid_year} recent",
        )
        base_score = bss(base_valid, y_valid)
        candidates = []
        for recent_weight in RECENT_WEIGHTS:
            correction = (
                (1 - recent_weight) * all_correction + recent_weight * recent_correction
            )
            for scale in SCALES:
                prediction = np.clip(base_valid + scale * correction, 1e-6, 1 - 1e-6)
                candidates.append({
                    "recent_weight": recent_weight, "scale": scale,
                    "bss_delta": bss(prediction, y_valid) - base_score,
                    "absolute_mean_error_delta": (
                        abs(float(prediction.mean()) - float(y_valid.mean()))
                        - abs(float(base_valid.mean()) - float(y_valid.mean()))
                    ),
                    "pitcher_bootstrap_probability": bootstrap(
                        valid_rows["pitcher_id"].to_numpy(), base_valid, prediction,
                        y_valid, 826000 + valid_year + int(recent_weight * 100) + int(scale * 1000),
                    ),
                })
        folds.append({
            "year": valid_year, "train_years": list(train_years),
            "recent_year": recent_year, "all_seconds": all_seconds,
            "recent_seconds": recent_seconds, "base_bss": base_score,
            "candidates": candidates,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summaries = []
    for recent_weight in RECENT_WEIGHTS:
        for scale in SCALES:
            rows = [next(
                candidate for candidate in fold["candidates"]
                if candidate["recent_weight"] == recent_weight and candidate["scale"] == scale
            ) for fold in folds]
            deltas = [float(row["bss_delta"]) for row in rows]
            ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 0
            probability = min(float(row["pitcher_bootstrap_probability"]) for row in rows)
            summaries.append({
                "recent_weight": recent_weight, "scale": scale,
                "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                "worst_delta": min(deltas), "magnitude_ratio": ratio,
                "minimum_pitcher_bootstrap_probability": probability,
                "passed": bool(min(deltas) >= 1 and ratio >= 0.25 and probability >= 0.80),
            })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]),
                   reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "own Futures multichannel residual regime",
        "official_train_only": True, "test_aggregate_used": False,
        "channels": ["all_history", "previous_season"],
        "auxiliary_probabilities": ["success", "middle_reverse", "large_miss", "failure_complement"],
        "seeds": list(SEEDS), "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "build_own_f_regime_submission" if passed else "keep_current_champion",
        "gate": "2023/2024 delta>=+1; magnitude ratio>=0.25; pitcher bootstrap>=0.80",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "top": summaries[:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.component_dir.resolve(), args.train.resolve(), args.output.resolve(), args.task_type)
