"""R 직접확률의 시즌 전이를 시간가중 학습으로 다시 검증한다."""
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
BASE_PATH = ROOT / "train_r" / "01_direct_route_blend_screen_colab.py"
DECAYS = (0.30, 0.55, 0.75)
STRENGTHS = (0.01, 0.025, 0.05, 0.075, 0.10)
SEEDS = (42, 7, 2024)
YEARS = (2023, 2024)


def load_base():
    spec = importlib.util.spec_from_file_location("r_direct_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(frame, target, season, features, cat_indices, year, decay, task_type, base):
    is_r = frame["game_type"].astype(str).eq("R").to_numpy()
    inner_train = (season < year - 1) & is_r
    inner_valid = (season == year - 1) & is_r
    outer_train = (season < year) & is_r
    outer_valid = (season == year) & is_r
    inner_weight = np.power(decay, (year - 2) - season[inner_train])
    outer_weight = np.power(decay, (year - 1) - season[outer_train])
    inner_train_pool = Pool(features.loc[inner_train], target[inner_train],
                            weight=inner_weight, cat_features=cat_indices)
    inner_valid_pool = Pool(features.loc[inner_valid], target[inner_valid], cat_features=cat_indices)
    outer_train_pool = Pool(features.loc[outer_train], target[outer_train],
                            weight=outer_weight, cat_features=cat_indices)
    outer_valid_pool = Pool(features.loc[outer_valid], cat_features=cat_indices)
    inner_members, outer_members, iterations, seconds = [], [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        selector = CatBoostClassifier(**base.params(seed, 2000, task_type, True))
        selector.fit(inner_train_pool, eval_set=inner_valid_pool, use_best_model=True)
        selected = max(1, int(selector.get_best_iteration()) + 1)
        fixed = max(128, selected)
        inner_members.append(selector.predict_proba(inner_valid_pool)[:, 1])
        del selector
        gc.collect()
        model = CatBoostClassifier(**base.params(seed, fixed, task_type, False))
        model.fit(outer_train_pool)
        outer_members.append(model.predict_proba(outer_valid_pool)[:, 1])
        iterations.append(fixed)
        seconds.append(float(time.perf_counter() - started))
        print(f"fold={year} decay={decay:.2f} seed={seed} selected={selected} "
              f"fixed={fixed} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    inner_prediction = np.mean(inner_members, axis=0)
    return (np.mean(outer_members, axis=0),
            base.optimal_shift(inner_prediction, target[inner_valid]), iterations, seconds)


def main(component_dir: Path, train_path: Path, output: Path, task_type: str):
    base = load_base()
    feature_module = base.load_module(base.FEATURE_PATH, "r_recent_features")
    baseline_module = base.load_module(base.BASELINE_PATH, "r_recent_baseline")
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
        baseline, valid_rows, valid_r, baseline_seconds = base.reconstruct_baseline(
            frame, features, cat_indices, assets, year, task_type, baseline_module
        )
        y = assets[year]["target"].astype(float)
        y_r, base_r = y[valid_r], baseline[valid_r]
        candidates = []
        training = []
        for decay in DECAYS:
            raw, shift, iterations, seconds = train(
                frame, target, season, features, cat_indices, year, decay, task_type, base
            )
            training.append({"decay": decay, "iterations": iterations, "seconds": seconds,
                             "calibration_shift": float(shift)})
            for mode, route in (("raw", raw), ("prior_calibrated", base.sigmoid(base.logit(raw) + shift))):
                for strength in STRENGTHS:
                    candidate_r = base.sigmoid(
                        (1 - strength) * base.logit(base_r) + strength * base.logit(route)
                    )
                    candidate = baseline.copy()
                    candidate[valid_r] = candidate_r
                    candidates.append({
                        "decay": decay, "mode": mode, "strength": strength,
                        "overall_bss_delta": base.bss(y, candidate) - base.bss(y, baseline),
                        "r_bss_delta": base.bss(y_r, candidate_r) - base.bss(y_r, base_r),
                        "r_pitcher_bootstrap_probability": base.bootstrap(
                            valid_rows.loc[valid_r, "pitcher_id"].to_numpy(), y_r, base_r,
                            candidate_r, 824200 + year + int(decay * 100) + int(strength * 1000),
                        ),
                    })
        folds.append({"year": year, "baseline_seconds": baseline_seconds,
                      "training": training, "candidates": candidates})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"fold={year} complete", flush=True)
    summaries = []
    for decay in DECAYS:
        for mode in ("raw", "prior_calibrated"):
            for strength in STRENGTHS:
                rows = [next(row for row in fold["candidates"] if row["decay"] == decay
                             and row["mode"] == mode and row["strength"] == strength)
                        for fold in folds]
                deltas = [row["overall_bss_delta"] for row in rows]
                r_deltas = [row["r_bss_delta"] for row in rows]
                probability = min(row["r_pitcher_bootstrap_probability"] for row in rows)
                summaries.append({
                    "decay": decay, "mode": mode, "strength": strength,
                    "fold_2023_overall_delta": deltas[0], "fold_2024_overall_delta": deltas[1],
                    "fold_2023_r_delta": r_deltas[0], "fold_2024_r_delta": r_deltas[1],
                    "worst_overall_delta": min(deltas),
                    "minimum_r_pitcher_bootstrap_probability": probability,
                    "passed": bool(min(deltas) > 0 and min(r_deltas) > 0 and probability >= 0.80),
                })
    summaries.sort(key=lambda row: (row["passed"], row["worst_overall_delta"]), reverse=True)
    selected = next((row for row in summaries if row["passed"]), None)
    report = {
        "experiment": "R recent-weighted direct route screen",
        "official_train_only": True, "test_aggregate_used": False,
        "folds": folds, "summaries": summaries, "selected": selected,
        "decision": "continue_r_recent_route" if selected else "keep_rchampion_fgeneral6",
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
