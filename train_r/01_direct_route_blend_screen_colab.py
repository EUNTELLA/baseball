"""자체 R 챔피언에 R 전용 6시드 직접확률 경로를 혼합한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
BASELINE_PATH = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
YEARS = (2023, 2024)
SEEDS = (42, 7, 2024, 99, 1, 123)
STRENGTHS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
MINIMUM_ITERATIONS = 128


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(seed, iterations, task_type, early_stopping=False):
    result = {
        "iterations": iterations, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "loss_function": "Logloss", "eval_metric": "Logloss",
        "random_seed": seed, "grow_policy": "SymmetricTree",
        "allow_writing_files": False, "verbose": False,
    }
    if early_stopping:
        result["early_stopping_rounds"] = 100
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.asarray(value, float)))


def bss(target, prediction):
    target = np.asarray(target, float)
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate))))


def optimal_shift(prediction, target):
    values = logit(prediction)
    target = np.asarray(target, float)
    low, high = -0.25, 0.25
    for _ in range(80):
        middle = (low + high) / 2
        if float(np.mean(sigmoid(values + middle) - target)) < 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def bootstrap(ids, target, baseline, candidate, seed, repeats=500):
    gain = (np.asarray(target) - np.asarray(baseline)) ** 2 - (
        np.asarray(target) - np.asarray(candidate)
    ) ** 2
    grouped = pd.DataFrame({"id": pd.Series(ids).astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    values = grouped[["gain", "n"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(repeats):
        sample = values[rng.integers(0, len(values), len(values))]
        positive += bool(sample[:, 0].sum() / sample[:, 1].sum() > 0)
    return float(positive / repeats)


def train_direct(frame, target, season, features, cat_indices, year, task_type):
    is_r = frame["game_type"].astype(str).eq("R").to_numpy()
    inner_train = (season < year - 1) & is_r
    inner_valid = (season == year - 1) & is_r
    outer_train = (season < year) & is_r
    outer_valid = (season == year) & is_r
    inner_train_pool = Pool(features.loc[inner_train], target[inner_train], cat_features=cat_indices)
    inner_valid_pool = Pool(features.loc[inner_valid], target[inner_valid], cat_features=cat_indices)
    outer_train_pool = Pool(features.loc[outer_train], target[outer_train], cat_features=cat_indices)
    outer_valid_pool = Pool(features.loc[outer_valid], cat_features=cat_indices)
    inner_members, outer_members, iterations, seconds = [], [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        selector = CatBoostClassifier(**params(seed, 2000, task_type, True))
        selector.fit(inner_train_pool, eval_set=inner_valid_pool, use_best_model=True)
        selected = max(1, int(selector.get_best_iteration()) + 1)
        fixed = max(MINIMUM_ITERATIONS, selected)
        inner_members.append(selector.predict_proba(inner_valid_pool)[:, 1])
        del selector
        gc.collect()
        model = CatBoostClassifier(**params(seed, fixed, task_type, False))
        model.fit(outer_train_pool)
        outer_members.append(model.predict_proba(outer_valid_pool)[:, 1])
        iterations.append(fixed)
        seconds.append(float(time.perf_counter() - started))
        print(f"fold={year} R direct seed={seed} selected={selected} fixed={fixed} "
              f"sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    inner_prediction = np.mean(inner_members, axis=0)
    outer_prediction = np.mean(outer_members, axis=0)
    shift = optimal_shift(inner_prediction, target[inner_valid])
    return outer_prediction, shift, iterations, seconds


def reconstruct_baseline(frame, features, cat_indices, assets, year, task_type, baseline_module):
    calibration_year = year - 1
    correction, valid_rows, seconds = baseline_module.train_correction(
        frame, features, cat_indices, calibration_year, year,
        assets[calibration_year], assets[year], task_type,
    )
    valid = assets[year]
    calibration = assets[calibration_year]
    valid_r = valid_rows["game_type"].astype(str).eq("R").to_numpy()
    prediction = baseline_module.sigmoid(
        baseline_module.logit(valid["anchor"].astype(float))
        + baseline_module.VERIFIED_SHIFT_DELTA
    )
    alignment_shift = baseline_module.shift_to_mean(
        calibration["failure_complement"].astype(float),
        float(calibration["anchor"].astype(float).mean()),
    )
    aligned = baseline_module.sigmoid(
        baseline_module.logit(valid["failure_complement"].astype(float)) + alignment_shift
    )
    mixed = valid["anchor"].astype(float).copy()
    mixed[valid_r] = 0.8 * mixed[valid_r] + 0.2 * aligned[valid_r]
    prediction = baseline_module.sigmoid(
        baseline_module.logit(mixed) + baseline_module.VERIFIED_SHIFT_DELTA
    )
    prediction[valid_r] = np.clip(
        prediction[valid_r] + baseline_module.R_SCALE * correction, 1e-6, 1 - 1e-6
    )
    return prediction, valid_rows, valid_r, seconds


def main(component_dir: Path, train_path: Path, output: Path, task_type: str):
    feature_module = load_module(FEATURE_PATH, "r_direct_features")
    baseline_module = load_module(BASELINE_PATH, "r_baseline")
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame["control_success"].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    prior = float(target[season < 2022].mean())
    features = feature_module.engineer(frame.drop(columns=["row_id", "control_success"]), prior)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    assets = {year: baseline_module.load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    folds = []
    for year in YEARS:
        direct, shift, iterations, direct_seconds = train_direct(
            frame, target, season, features, cat_indices, year, task_type
        )
        baseline, valid_rows, valid_r, baseline_seconds = reconstruct_baseline(
            frame, features, cat_indices, assets, year, task_type, baseline_module
        )
        y = assets[year]["target"].astype(float)
        y_r = y[valid_r]
        base_r = baseline[valid_r]
        routes = {
            "raw": np.clip(direct, 1e-6, 1 - 1e-6),
            "prior_calibrated": sigmoid(logit(direct) + shift),
        }
        candidates = []
        for mode, route in routes.items():
            for space in ("probability", "logit"):
                for strength in STRENGTHS:
                    if space == "probability":
                        candidate_r = (1 - strength) * base_r + strength * route
                    else:
                        candidate_r = sigmoid((1 - strength) * logit(base_r) + strength * logit(route))
                    candidate_r = np.clip(candidate_r, 1e-6, 1 - 1e-6)
                    candidate = baseline.copy()
                    candidate[valid_r] = candidate_r
                    candidates.append({
                        "mode": mode, "space": space, "strength": strength,
                        "overall_bss_delta": bss(y, candidate) - bss(y, baseline),
                        "r_bss_delta": bss(y_r, candidate_r) - bss(y_r, base_r),
                        "r_pitcher_bootstrap_probability": bootstrap(
                            valid_rows.loc[valid_r, "pitcher_id"].to_numpy(), y_r,
                            base_r, candidate_r, 824100 + year + int(strength * 1000),
                        ),
                        "r_prediction_mean_delta": float(candidate_r.mean() - base_r.mean()),
                    })
        folds.append({
            "year": year, "direct_iterations": iterations,
            "direct_seconds": direct_seconds, "baseline_seconds": baseline_seconds,
            "calibration_shift": float(shift), "r_rows": int(valid_r.sum()),
            "candidates": candidates,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"fold={year} complete", flush=True)
    summaries = []
    for mode in ("raw", "prior_calibrated"):
        for space in ("probability", "logit"):
            for strength in STRENGTHS:
                rows = [next(row for row in fold["candidates"] if row["mode"] == mode
                             and row["space"] == space and row["strength"] == strength)
                        for fold in folds]
                deltas = [row["overall_bss_delta"] for row in rows]
                r_deltas = [row["r_bss_delta"] for row in rows]
                probability = min(row["r_pitcher_bootstrap_probability"] for row in rows)
                ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 1.0
                passed = bool(min(deltas) > 0 and min(r_deltas) > 0 and probability >= 0.80)
                summaries.append({
                    "mode": mode, "space": space, "strength": strength,
                    "fold_2023_overall_delta": deltas[0],
                    "fold_2024_overall_delta": deltas[1],
                    "fold_2023_r_delta": r_deltas[0], "fold_2024_r_delta": r_deltas[1],
                    "worst_overall_delta": min(deltas), "magnitude_ratio": ratio,
                    "minimum_r_pitcher_bootstrap_probability": probability,
                    "passed": passed,
                })
    summaries.sort(key=lambda row: (row["passed"], row["worst_overall_delta"],
                                     row["magnitude_ratio"]), reverse=True)
    selected = next((row for row in summaries if row["passed"]), None)
    report = {
        "experiment": "R-only six-seed direct route blend screen",
        "official_train_only": True, "test_aggregate_used": False,
        "baseline": "verified shift + R residual 0.075 + failure complement 0.20",
        "seeds": list(SEEDS), "minimum_iterations": MINIMUM_ITERATIONS,
        "folds": folds, "summaries": summaries, "selected": selected,
        "decision": "continue_r_direct_route" if selected else "keep_rchampion_fgeneral6",
        "gate": "2023/2024 overall and R delta positive; minimum R pitcher bootstrap>=0.80",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "top": summaries[:10],
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
