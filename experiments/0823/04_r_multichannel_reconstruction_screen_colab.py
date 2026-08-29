"""현재 챔피언의 R residual 일부를 자체 문맥·이력 채널로 교체한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_MODULE = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
PAIRS = ((2022, 2023), (2023, 2024))
SEEDS = (17, 42, 777)
MIXES = (0.10, 0.25, 0.50, 0.75, 1.0)
FAILURE_BLEND = 0.20

CONTEXT_CAT = (
    "game_type", "top_bottom", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "count_state",
)
CONTEXT_NUM = (
    "inning", "balls_before", "strikes_before", "outs_before", "num_runners_on",
    "score_diff_pitcher_team", "li", "home_win_expectancy", "away_win_expectancy",
)
HISTORY_CAT = ("pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "game_type")
HISTORY_NUM = (
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(frame, categorical, numeric):
    data = pd.DataFrame(index=frame.index)
    for column in categorical:
        if column == "count_state":
            data[column] = (
                pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
                + "-" + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
            )
        else:
            data[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    for column in numeric:
        data[column] = pd.to_numeric(frame[column], errors="coerce")
    return data


def train_channel(name, x, cat_cols, frame, calibration_year, validation_year, target, task_type,
                  depth, l2, iterations):
    train_mask = frame["season"].astype(int).eq(calibration_year) & frame["game_type"].astype(str).eq("R")
    valid_mask = frame["season"].astype(int).eq(validation_year) & frame["game_type"].astype(str).eq("R")
    train_pool = Pool(x.loc[train_mask], target, cat_features=list(cat_cols))
    valid_pool = Pool(x.loc[valid_mask], cat_features=list(cat_cols))
    members, seconds = [], []
    for seed in SEEDS:
        model = CatBoostRegressor(
            iterations=iterations, depth=depth, learning_rate=0.025, loss_function="RMSE",
            l2_leaf_reg=l2, random_strength=0.25, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=seed, task_type=task_type,
            devices="0" if task_type == "GPU" else None, thread_count=6,
            allow_writing_files=False, verbose=False,
        )
        started = time.perf_counter()
        model.fit(train_pool)
        members.append(model.predict(valid_pool))
        seconds.append(float(time.perf_counter() - started))
        print(f"fold={validation_year} channel={name} seed={seed} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(members, axis=0), seconds


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(component_dir, train_path, output, task_type):
    module = load_module(CHAMPION_MODULE, "champion_validation")
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    feature_module = module.load_features()
    league_rate = float(frame.loc[frame["season"].astype(int).lt(2022), "control_success"].mean())
    full_features = feature_module.engineer(frame.drop(columns=["row_id", "control_success"]), league_rate)
    for column in feature_module.CAT_COLS:
        full_features[column] = full_features[column].astype(str)
    full_cat_indices = [full_features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    context = prepare(frame, CONTEXT_CAT, CONTEXT_NUM)
    history = prepare(frame, HISTORY_CAT, HISTORY_NUM)
    assets = {year: module.load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    folds = []
    for calibration_year, validation_year in PAIRS:
        calibration, valid = assets[calibration_year], assets[validation_year]
        full, valid_rows, full_seconds = module.train_correction(
            frame, full_features, full_cat_indices, calibration_year, validation_year,
            calibration, valid, task_type,
        )
        calibration_rows = frame.loc[frame["season"].astype(int).eq(calibration_year)].reset_index(drop=True)
        calibration_r = calibration_rows["game_type"].astype(str).eq("R").to_numpy()
        residual = calibration["target"].astype(float) - calibration["anchor"].astype(float)
        residual_r = residual[calibration_r]
        context_prediction, context_seconds = train_channel(
            "context", context, CONTEXT_CAT, frame, calibration_year, validation_year,
            residual_r, task_type, depth=3, l2=100, iterations=500,
        )
        history_prediction, history_seconds = train_channel(
            "history", history, HISTORY_CAT, frame, calibration_year, validation_year,
            residual_r, task_type, depth=5, l2=100, iterations=700,
        )
        alternate = 0.5 * context_prediction + 0.5 * history_prediction
        target = valid["target"].astype(float)
        anchor = valid["anchor"].astype(float)
        valid_r = valid_rows["game_type"].astype(str).eq("R").to_numpy()
        alignment_shift = module.shift_to_mean(
            calibration["failure_complement"].astype(float),
            float(calibration["anchor"].astype(float).mean()),
        )
        aligned_failure = module.sigmoid(
            module.logit(valid["failure_complement"].astype(float)) + alignment_shift
        )
        reconstructed = anchor.copy()
        reconstructed[valid_r] = (
            (1 - FAILURE_BLEND) * reconstructed[valid_r] + FAILURE_BLEND * aligned_failure[valid_r]
        )
        reconstructed = module.sigmoid(module.logit(reconstructed) + module.VERIFIED_SHIFT_DELTA)
        champion = reconstructed.copy()
        champion[valid_r] = np.clip(
            champion[valid_r] + module.R_SCALE * full, 1e-6, 1 - 1e-6
        )
        champion_score = module.bss(champion, target)
        candidates = []
        for mix in MIXES:
            correction = (1 - mix) * full + mix * alternate
            candidate = reconstructed.copy()
            candidate[valid_r] = np.clip(
                candidate[valid_r] + module.R_SCALE * correction, 1e-6, 1 - 1e-6
            )
            candidates.append({
                "mix": mix, "bss_delta": module.bss(candidate, target) - champion_score,
                "pitcher_bootstrap_probability": module.bootstrap(
                    valid_rows["pitcher_id"].to_numpy(), champion, candidate, target,
                    823400 + validation_year + int(mix * 1000),
                ),
                "absolute_mean_error_delta": (
                    abs(float(candidate.mean()) - float(target.mean()))
                    - abs(float(champion.mean()) - float(target.mean()))
                ),
            })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "champion_bss": champion_score,
            "error_correlations": {
                "full_context": float(np.corrcoef(full, context_prediction)[0, 1]),
                "full_history": float(np.corrcoef(full, history_prediction)[0, 1]),
                "context_history": float(np.corrcoef(context_prediction, history_prediction)[0, 1]),
            },
            "training_seconds": {
                "full": full_seconds, "context": context_seconds, "history": history_seconds,
            },
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} complete", flush=True)
    summaries = []
    for mix in MIXES:
        rows = [next(row for row in fold["candidates"] if row["mix"] == mix) for fold in folds]
        deltas = [float(row["bss_delta"]) for row in rows]
        probabilities = [float(row["pitcher_bootstrap_probability"]) for row in rows]
        ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 0.0
        summaries.append({
            "mix": mix, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "worst_delta": min(deltas), "magnitude_ratio": ratio,
            "minimum_pitcher_bootstrap_probability": min(probabilities),
            "passed": bool(min(deltas) >= 1 and ratio >= 0.20 and min(probabilities) >= 0.80),
        })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "self-trained R multichannel residual reconstruction",
        "official_train_only": True, "test_aggregate_used": False,
        "fixed_f_prediction": True, "fixed_failure_complement_blend": FAILURE_BLEND,
        "fixed_r_scale": module.R_SCALE, "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_self_trained_r_multichannel" if passed else "keep_current_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.20, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries,
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
