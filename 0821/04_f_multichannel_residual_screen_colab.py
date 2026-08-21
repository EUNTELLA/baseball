"""F 행 장기·최근·강한 최근가중 잔차 채널을 시간 전방으로 선별한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
BASE_SEEDS = (42, 7, 2024)
RESIDUAL_SEEDS = (17, 42, 777)
OOF_YEARS = (2021, 2022, 2023, 2024)
VALID_YEARS = (2023, 2024)
SCALES = (0.25, 0.5, 0.75, 1.0)
ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
BASE_PATH = ROOT / "0820" / "04_dynamic_pitcher_baseline_residual_screen_colab.py"

CHANNELS = {
    "long_memory": {"latest_only": False, "decay": 0.55, "iterations": 180},
    "recent_only": {"latest_only": True, "decay": None, "iterations": 240},
    "fast_decay": {"latest_only": False, "decay": 0.30, "iterations": 210},
}
MIXES = {
    "long_only": (1.0, 0.0, 0.0),
    "recent_only": (0.0, 1.0, 0.0),
    "fast_only": (0.0, 0.0, 1.0),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "stable_recent": (0.50, 0.25, 0.25),
    "recent_focus": (0.25, 0.50, 0.25),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "bss_score": float(1e5 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()), "target_mean": rate,
    }


def prepare_features(frame: pd.DataFrame, target: np.ndarray, season: np.ndarray,
                     before_year: int, feature_module):
    global_mean = float(target[season < before_year].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    return features, cat_indices


def build_base_oof(frame, target, season, feature_module, base_module, task_type, output):
    prediction = np.full(len(frame), np.nan, dtype=float)
    training = []
    for year in OOF_YEARS:
        train_mask, valid_mask = season < year, season == year
        features, cat_indices = prepare_features(frame, target, season, year, feature_module)
        train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        members, iterations = [], []
        for seed in BASE_SEEDS:
            started = time.perf_counter()
            model = CatBoostClassifier(**base_module.classifier_params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            members.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            print(f"base year={year} seed={seed} iter={iterations[-1]} sec={time.perf_counter()-started:.1f}", flush=True)
            del model
            gc.collect()
        prediction[valid_mask] = np.mean(members, axis=0)
        training.append({"year": year, "best_iterations": iterations})
        write_json(output, {"status": "base_oof", "base_training": training})
        del features, train_pool, valid_pool, members
        gc.collect()
    return prediction, training


def fit_channel(features, cat_indices, residual, season, is_f, valid_year,
                config, task_type):
    available = np.isfinite(residual) & is_f & (season < valid_year)
    if config["latest_only"]:
        train_mask = available & (season == valid_year - 1)
    else:
        train_mask = available
    valid_mask = season == valid_year
    weights = None
    if config["decay"] is not None:
        weights = np.power(float(config["decay"]), (valid_year - 1) - season[train_mask])
    train_pool = Pool(features.loc[train_mask], residual[train_mask], cat_features=cat_indices,
                      weight=weights)
    valid_pool = Pool(features.loc[valid_mask], cat_features=cat_indices)
    members, seconds = [], []
    for seed in RESIDUAL_SEEDS:
        started = time.perf_counter()
        model = CatBoostRegressor(
            iterations=config["iterations"], depth=7, learning_rate=0.035,
            loss_function="RMSE", l2_leaf_reg=20, random_strength=0.35,
            bootstrap_type="Bernoulli", subsample=0.85, one_hot_max_size=16,
            random_seed=seed, task_type=task_type,
            devices="0" if task_type == "GPU" else None,
            thread_count=6, allow_writing_files=False, verbose=False,
        )
        model.fit(train_pool)
        members.append(model.predict(valid_pool))
        seconds.append(float(time.perf_counter() - started))
        del model
        gc.collect()
    return np.mean(members, axis=0), int(train_mask.sum()), seconds


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    is_f = frame["game_type"].astype(str).eq("F").to_numpy()
    feature_module = load_module("f_multichannel_features", FEATURE_PATH)
    base_module = load_module("f_multichannel_base", BASE_PATH)
    base_oof, base_training = build_base_oof(
        frame, target, season, feature_module, base_module, task_type, output
    )
    residual = target.astype(float) - base_oof
    fold_results = []

    for valid_year in VALID_YEARS:
        valid_mask = season == valid_year
        valid_f = is_f[valid_mask]
        features, cat_indices = prepare_features(frame, target, season, valid_year, feature_module)
        channel_prediction, channel_training = {}, {}
        for name, config in CHANNELS.items():
            correction, rows, seconds = fit_channel(
                features, cat_indices, residual, season, is_f, valid_year, config, task_type
            )
            correction[~valid_f] = 0.0
            channel_prediction[name] = correction
            channel_training[name] = {"rows": rows, "seconds": seconds}
            print(f"fold={valid_year} channel={name} rows={rows} sec={sum(seconds):.1f}", flush=True)

        baseline = base_oof[valid_mask]
        y_valid = target[valid_mask]
        baseline_metrics = metric(baseline, y_valid)
        candidates = []
        ordered = tuple(CHANNELS)
        for mix_name, weights in MIXES.items():
            combined = sum(weight * channel_prediction[name]
                           for name, weight in zip(ordered, weights))
            for scale in SCALES:
                prediction = np.clip(baseline + scale * combined, 1e-6, 1 - 1e-6)
                result = metric(prediction, y_valid)
                candidates.append({
                    "mix": mix_name, "weights": list(weights), "scale": scale,
                    "bss_delta": result["bss_score"] - baseline_metrics["bss_score"],
                    "absolute_mean_error_delta": abs(result["prediction_mean"] - result["target_mean"])
                    - abs(baseline_metrics["prediction_mean"] - baseline_metrics["target_mean"]),
                })
        fold_results.append({
            "fold": valid_year, "f_rows": int(valid_f.sum()),
            "baseline": baseline_metrics, "channel_training": channel_training,
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "base_training": base_training,
                            "fold_results": fold_results})
        del features
        gc.collect()

    summaries = []
    for mix_name in MIXES:
        for scale in SCALES:
            rows = [next(row for row in fold["candidates"]
                         if row["mix"] == mix_name and row["scale"] == scale)
                    for fold in fold_results]
            deltas = [float(row["bss_delta"]) for row in rows]
            summaries.append({
                "mix": mix_name, "weights": list(MIXES[mix_name]), "scale": scale,
                "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
                "both_positive": bool(min(deltas) > 0),
            })
    summaries.sort(key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 3.0]
    selected = stable[0] if stable else None
    report = {
        "experiment": "F-row multichannel residual transfer screen",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale_fixed": 0.05, "r_rows_modified_in_this_experiment": False,
        "channels": CHANNELS, "mixes": {key: list(value) for key, value in MIXES.items()},
        "base_training": base_training, "fold_results": fold_results,
        "summaries": summaries, "selected": selected,
        "decision": "continue_f_auxiliary_transition_validation" if selected else "keep_r_scale0050_champion",
        "gate": "same mix/scale positive in 2023 and 2024; 2024 >= +3",
    }
    write_json(output, report)
    print(json.dumps({"selected": selected, "top": summaries[:10], "decision": report["decision"]},
                     ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
