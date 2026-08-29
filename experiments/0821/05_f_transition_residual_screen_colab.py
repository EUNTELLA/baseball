"""이전 리그 유형에서 현재 F행으로의 전환 잔차를 시간 전방 검증한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024)
PAIRS = ((2022, 2023), (2023, 2024))
DEPTHS = (3, 5)
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
CAT_COLS = ("game_type", "prior_type", "transition", "count", "hand", "team_type")
ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
BASE_PATH = ROOT / "0820" / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dominant_prior_type(history: pd.DataFrame, target_year: int) -> pd.Series:
    past = history.loc[history["season"].astype(int) < target_year]
    counts = (past.groupby(["pitcher_id", "season", "game_type"], observed=True)
              .size().rename("rows").reset_index())
    if counts.empty:
        return pd.Series(dtype=str)
    dominant = counts.sort_values("rows").groupby(["pitcher_id", "season"]).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id").tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def transition_features(rows: pd.DataFrame, base_prediction: np.ndarray,
                        history: pd.DataFrame, target_year: int) -> pd.DataFrame:
    prior = dominant_prior_type(history, target_year)
    pitcher = rows["pitcher_id"].astype(str)
    previous = pitcher.map(prior).fillna("NEW").astype(str)
    current = rows["game_type"].astype(str)
    numeric = lambda name: pd.to_numeric(rows[name], errors="coerce")
    features = pd.DataFrame({
        "game_type": current,
        "prior_type": previous,
        "transition": previous + ">" + current,
        "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
        "hand": rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str),
        "team_type": rows["pitcher_team_id"].astype(str) + "|" + current,
        "base_prediction": np.asarray(base_prediction, float),
        "log_pitcher_n": np.log1p(numeric("asof_pitcher_n").fillna(0).clip(lower=0)),
        "career": numeric("asof_pitcher_success_rate"),
        "recent1": numeric("asof_pitcher_prev1_game_success_rate"),
        "recent3": numeric("asof_pitcher_prev3_game_success_rate"),
        "recent5": numeric("asof_pitcher_prev5_game_success_rate"),
        "middle": numeric("asof_pitcher_middle_rate"),
        "reverse": numeric("asof_pitcher_reverse_rate"),
        "li": numeric("li"),
        "inning": numeric("inning"),
        "runners": numeric("num_runners_on"),
    })
    for column in CAT_COLS:
        features[column] = features[column].astype("string").fillna("__MISSING__").astype(str)
    return features


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


def base_prediction(frame, target, season, year, feature_module, base_module, task_type):
    train_mask, valid_mask = season < year, season == year
    global_mean = float(target[train_mask].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    members, iterations = [], []
    for seed in SEEDS:
        model = CatBoostClassifier(**base_module.classifier_params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        members.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        del model
        gc.collect()
    return np.mean(members, axis=0), iterations


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    feature_module = load_module("f_transition_features", FEATURE_PATH)
    base_module = load_module("f_transition_base", BASE_PATH)
    predictions, training = {}, []
    for year in (2022, 2023, 2024):
        prediction, iterations = base_prediction(
            frame, target, season, year, feature_module, base_module, task_type
        )
        predictions[year] = prediction
        training.append({"year": year, "best_iterations": iterations})
        print(f"base year={year} iterations={iterations}", flush=True)

    fold_results = []
    for source_year, valid_year in PAIRS:
        source_mask, valid_mask = season == source_year, season == valid_year
        source_rows = frame.loc[source_mask].reset_index(drop=True)
        valid_rows = frame.loc[valid_mask].reset_index(drop=True)
        source_x = transition_features(source_rows, predictions[source_year], frame, source_year)
        valid_x = transition_features(valid_rows, predictions[valid_year], frame, valid_year)
        source_target = target[source_mask] - predictions[source_year]
        valid_target = target[valid_mask]
        valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        baseline = predictions[valid_year]
        baseline_score = metric(baseline, valid_target)["bss_score"]
        cat_indices = [source_x.columns.get_loc(column) for column in CAT_COLS]
        train_pool = Pool(source_x, source_target, cat_features=cat_indices)
        valid_pool = Pool(valid_x, cat_features=cat_indices)
        candidates = []
        for depth in DEPTHS:
            members = []
            for seed in SEEDS:
                model = CatBoostRegressor(
                    iterations=250, depth=depth, learning_rate=0.025,
                    loss_function="RMSE", l2_leaf_reg=100, random_strength=0.2,
                    bootstrap_type="Bernoulli", subsample=0.8,
                    random_seed=seed, task_type=task_type,
                    devices="0" if task_type == "GPU" else None,
                    thread_count=6, allow_writing_files=False, verbose=False,
                )
                model.fit(train_pool)
                members.append(model.predict(valid_pool))
                del model
                gc.collect()
            correction = np.mean(members, axis=0)
            correction[~valid_f] = 0.0
            for scale in SCALES:
                candidate = np.clip(baseline + scale * correction, 1e-6, 1 - 1e-6)
                result = metric(candidate, valid_target)
                candidates.append({
                    "depth": depth, "scale": scale,
                    "bss_delta": result["bss_score"] - baseline_score,
                    "absolute_mean_error_delta": abs(result["prediction_mean"] - result["target_mean"])
                    - abs(float(baseline.mean()) - float(valid_target.mean())),
                })
        fold_results.append({
            "source_year": source_year, "valid_year": valid_year,
            "source_rows": int(source_mask.sum()), "valid_f_rows": int(valid_f.sum()),
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "training": training,
                            "fold_results": fold_results})

    summaries = []
    for depth in DEPTHS:
        for scale in SCALES:
            rows = [next(row for row in fold["candidates"]
                         if row["depth"] == depth and row["scale"] == scale)
                    for fold in fold_results]
            deltas = [float(row["bss_delta"]) for row in rows]
            summaries.append({
                "depth": depth, "scale": scale,
                "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
                "both_positive": bool(min(deltas) > 0),
            })
    summaries.sort(key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 2.0]
    selected = stable[0] if stable else None
    report = {
        "experiment": "F-row prior-type transition residual screen",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale_fixed": 0.05, "r_rows_modified_in_this_experiment": False,
        "training": training, "fold_results": fold_results,
        "summaries": summaries, "selected": selected,
        "decision": "continue_f_transition_full_pipeline" if selected else "keep_r_scale0050_champion",
        "gate": "same depth/scale positive in 2023 and 2024; 2024 >= +2",
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
