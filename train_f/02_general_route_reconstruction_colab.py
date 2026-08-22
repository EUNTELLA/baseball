"""전체 리그 공통 모델을 nested-forward로 학습하고 F행 하드 라우팅을 검증한다."""
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
YEARS = (2023, 2024)
DEFAULT_SEEDS = (42, 7, 2024)
SHIFT_DELTA = -0.0416386466 - (-0.03842671927234861)


def load_features():
    spec = importlib.util.spec_from_file_location("general_route_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_component(directory: Path, year: int):
    asset = np.load(directory / f"components_{year}.npz", allow_pickle=True)
    return {name: asset[name] for name in asset.files}


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def optimal_shift(prediction, target):
    values = logit(prediction)
    target = np.asarray(target, float)
    low, high = -0.25, 0.25
    for _ in range(80):
        middle = (low + high) / 2
        gradient = float(np.mean(sigmoid(values + middle) - target))
        if gradient < 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def params(seed, iterations, task_type, early_stopping=False):
    result = {
        "iterations": iterations,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 1.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": seed,
        "grow_policy": "SymmetricTree",
        "allow_writing_files": False,
        "verbose": False,
    }
    if early_stopping:
        result["early_stopping_rounds"] = 100
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


def build_features(frame, target, season, before_year, module):
    prior = float(target[season < before_year].mean())
    values = module.engineer(frame.drop(columns=["row_id", "control_success"]), prior)
    for column in module.CAT_COLS:
        values[column] = values[column].astype(str)
    indices = [values.columns.get_loc(column) for column in module.CAT_COLS]
    return values, indices


def train_fold(frame, target, season, year, seeds, task_type, feature_module,
               minimum_iterations):
    inner_train = season < year - 1
    inner_valid = season == year - 1
    outer_train = season < year
    outer_valid = season == year
    inner_features, inner_cat = build_features(
        frame, target, season, year - 1, feature_module
    )
    outer_features, outer_cat = build_features(
        frame, target, season, year, feature_module
    )
    inner_train_pool = Pool(
        inner_features.loc[inner_train], target[inner_train], cat_features=inner_cat
    )
    inner_valid_pool = Pool(
        inner_features.loc[inner_valid], target[inner_valid], cat_features=inner_cat
    )
    outer_train_pool = Pool(
        outer_features.loc[outer_train], target[outer_train], cat_features=outer_cat
    )
    outer_valid_pool = Pool(outer_features.loc[outer_valid], cat_features=outer_cat)
    outer_members, inner_members, iterations, seconds = [], [], [], []
    for seed in seeds:
        started = time.perf_counter()
        selector = CatBoostClassifier(**params(seed, 2000, task_type, early_stopping=True))
        selector.fit(inner_train_pool, eval_set=inner_valid_pool, use_best_model=True)
        selected_iteration = max(1, int(selector.get_best_iteration()) + 1)
        iteration = max(minimum_iterations, selected_iteration)
        inner_members.append(selector.predict_proba(inner_valid_pool)[:, 1])
        del selector
        gc.collect()
        model = CatBoostClassifier(**params(seed, iteration, task_type, early_stopping=False))
        model.fit(outer_train_pool)
        outer_members.append(model.predict_proba(outer_valid_pool)[:, 1])
        iterations.append(iteration)
        seconds.append(float(time.perf_counter() - started))
        print(
            f"year={year} seed={seed} selected_iter={selected_iteration} "
            f"fixed_iter={iteration} sec={seconds[-1]:.1f}", flush=True,
        )
        del model
        gc.collect()
    inner_prediction = np.mean(inner_members, axis=0)
    outer_prediction = np.mean(outer_members, axis=0)
    shift = optimal_shift(inner_prediction, target[inner_valid])
    del inner_features, outer_features
    del inner_train_pool, inner_valid_pool, outer_train_pool, outer_valid_pool
    gc.collect()
    return outer_prediction, shift, iterations, seconds


def main(train_path: Path, component_dir: Path, output: Path,
         task_type: str, seeds, minimum_iterations: int):
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame["control_success"].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    module = load_features()
    folds = []
    for year in YEARS:
        prediction, calibration_shift, iterations, seconds = train_fold(
            frame, target, season, year, seeds, task_type, module, minimum_iterations
        )
        valid_rows = frame.loc[season == year].reset_index(drop=True)
        component = load_component(component_dir, year)
        if not np.array_equal(valid_rows["row_id"].astype(str), component["row_id"].astype(str)):
            raise ValueError(f"{year} 구성요소 정렬 불일치")
        y_valid = component["target"].astype(float)
        anchor = component["anchor"].astype(float)
        champion = sigmoid(logit(anchor) + SHIFT_DELTA)
        f_mask = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        base_all = bss(champion, y_valid)
        base_f = bss(champion[f_mask], y_valid[f_mask])
        candidates = []
        for mode, route in (
            ("raw", prediction),
            ("prior_calibrated", sigmoid(logit(prediction) + calibration_shift)),
        ):
            candidate = champion.copy()
            candidate[f_mask] = np.clip(route[f_mask], 1e-6, 1 - 1e-6)
            candidates.append({
                "mode": mode,
                "overall_bss_delta": bss(candidate, y_valid) - base_all,
                "f_bss_delta": bss(candidate[f_mask], y_valid[f_mask]) - base_f,
                "f_pitcher_bootstrap_probability": bootstrap(
                    valid_rows.loc[f_mask, "pitcher_id"].to_numpy(), champion[f_mask],
                    candidate[f_mask], y_valid[f_mask], 824000 + year,
                ),
                "r_max_absolute_delta": 0.0,
                "f_prediction_mean_delta": float(candidate[f_mask].mean() - champion[f_mask].mean()),
            })
        folds.append({
            "year": year,
            "fixed_iterations": iterations,
            "seconds": seconds,
            "prior_year_calibration_shift": calibration_shift,
            "candidates": candidates,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds}, ensure_ascii=False,
                                     indent=2) + "\n", encoding="utf-8")
        print(f"year={year} complete", flush=True)
    summaries = []
    for mode in ("raw", "prior_calibrated"):
        rows = [next(row for row in fold["candidates"] if row["mode"] == mode) for fold in folds]
        summaries.append({
            "mode": mode,
            "fold_2023_overall_delta": rows[0]["overall_bss_delta"],
            "fold_2024_overall_delta": rows[1]["overall_bss_delta"],
            "fold_2023_f_delta": rows[0]["f_bss_delta"],
            "fold_2024_f_delta": rows[1]["f_bss_delta"],
            "minimum_f_pitcher_bootstrap_probability": min(
                row["f_pitcher_bootstrap_probability"] for row in rows
            ),
            "passed": bool(
                min(row["overall_bss_delta"] for row in rows) >= 5
                and min(row["f_bss_delta"] for row in rows) > 0
                and min(row["f_pitcher_bootstrap_probability"] for row in rows) >= 0.80
            ),
        })
    summaries.sort(key=lambda row: (row["passed"], min(
        row["fold_2023_overall_delta"], row["fold_2024_overall_delta"]
    )), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "nested-forward general route reconstruction for Futures rows",
        "official_train_only": True,
        "test_aggregate_used": False,
        "target_year_early_stopping_used": False,
        "seeds": list(seeds),
        "minimum_iterations": minimum_iterations,
        "folds": folds,
        "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_six_seed_general_route" if passed else "keep_current_champion",
        "gate": "2023/2024 overall delta>=+5, F delta positive, F bootstrap>=0.80",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "summaries": summaries,
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--seeds", default="42,7,2024")
    parser.add_argument("--minimum-iterations", type=int, default=1)
    args = parser.parse_args()
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    main(args.train.resolve(), args.component_dir.resolve(), args.output.resolve(),
         args.task_type, selected_seeds, args.minimum_iterations)
