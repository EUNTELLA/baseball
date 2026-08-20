"""Trackman 피처를 고신뢰·충분한 이력 투수로 제한하고 혼합 강도를 비교한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ID_COL, TARGET_COL = "row_id", "control_success"
FOLDS, SEEDS = (2022, 2023, 2024), (42, 7, 2024)
BLENDS = (0.25, 0.50, 0.75, 1.0)
RELIABILITY = (
    {"name": "sim90_n200", "similarity": 0.90, "margin": 0.02, "prior_n": 200.0},
    {"name": "sim95_n300", "similarity": 0.95, "margin": 0.02, "prior_n": 300.0},
    {"name": "sim95_margin05_n300", "similarity": 0.95, "margin": 0.05, "prior_n": 300.0},
)
SCRIPT_DIR = Path(__file__).resolve().parent
TRACKMAN_PATH = SCRIPT_DIR / "13_trackman_prior_feature_screen_colab.py"
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


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


def train_prediction(data, target, train_mask, valid_mask, cat_cols, base, task_type):
    cat_indices = [data.columns.get_loc(column) for column in cat_cols]
    train_pool = Pool(data.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(data.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    members, iterations = [], []
    for seed in SEEDS:
        model = CatBoostClassifier(**base.classifier_params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        members.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        del model
        gc.collect()
    return np.mean(members, axis=0), iterations


def main(train_path: Path, trackman_file: Path, mapping_path: Path, output: Path, task_type: str) -> None:
    tm = load_module("trackman_prior_screen", TRACKMAN_PATH)
    base = load_module("direct_catboost_base", BASE_PATH)
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    assets = tm.build_prior_assets(frame, trackman_file, mapping_path)
    merge_key = pd.DataFrame({"pitcher_id": frame["pitcher_id"].astype(str), "season": season,
                              "_row_order": np.arange(len(frame))})
    attached = (merge_key.merge(assets, on=["pitcher_id", "season"], how="left", sort=False)
                .sort_values("_row_order").reset_index(drop=True))
    tm_columns = [column for column in attached.columns if column.startswith("tm_")]
    report = {"experiment": "Trackman reliability filtering and prediction blend screen",
              "official_data_only": True, "test_aggregate_used": False,
              "reliability_configs": list(RELIABILITY), "fold_results": []}

    for fold in FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league = float(target[train_mask].mean())
        basic = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league)
        for column in feature_module.CAT_COLS:
            basic[column] = basic[column].astype(str)
        direct, direct_iterations = train_prediction(
            basic, target, train_mask, valid_mask, feature_module.CAT_COLS, base, task_type)
        baseline_metrics = base.metric(direct, target[valid_mask])
        candidates = []
        for config in RELIABILITY:
            reliable = (attached["tm_mapping_similarity"].ge(config["similarity"])
                        & attached["tm_mapping_margin"].ge(config["margin"])
                        & attached["tm_prior_n"].ge(config["prior_n"]))
            augmented = basic.copy()
            for column in tm_columns:
                values = pd.to_numeric(attached[column], errors="coerce").where(reliable)
                augmented[column] = values.to_numpy()
            trackman_prediction, iterations = train_prediction(
                augmented, target, train_mask, valid_mask, feature_module.CAT_COLS, base, task_type)
            for blend in BLENDS:
                prediction = np.clip((1.0 - blend) * direct + blend * trackman_prediction, 1e-6, 1 - 1e-6)
                metrics = base.metric(prediction, target[valid_mask])
                candidates.append({"config": config["name"], "blend": blend,
                                   "reliable_train_coverage": float(reliable[train_mask].mean()),
                                   "reliable_valid_coverage": float(reliable[valid_mask].mean()),
                                   "best_iterations": iterations, "metrics": metrics,
                                   "bss_delta": metrics["bss_score"] - baseline_metrics["bss_score"]})
            del augmented
            gc.collect()
        report["fold_results"].append({"fold": fold, "baseline": baseline_metrics,
                                       "direct_best_iterations": direct_iterations, "candidates": candidates})
        print(f"fold={fold} reliability candidates={len(candidates)}", flush=True)
        write_json(output, report)
        del basic
        gc.collect()

    summaries = []
    for config in RELIABILITY:
        for blend in BLENDS:
            rows = [next(row for row in fold["candidates"] if row["config"] == config["name"]
                         and row["blend"] == blend) for fold in report["fold_results"]]
            deltas = [float(row["bss_delta"]) for row in rows]
            summaries.append({"config": config["name"], "blend": blend,
                              "fold_2022_delta": deltas[0], "fold_2023_delta": deltas[1],
                              "fold_2024_delta": deltas[2], "mean_delta": float(np.mean(deltas)),
                              "worst_delta": float(np.min(deltas)), "all_positive": bool(min(deltas) > 0)})
    stable = [row for row in summaries if row["all_positive"] and row["fold_2024_delta"] >= 3.0]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = sorted(summaries, key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    report["selected"] = selected
    report["decision"] = "continue_reliable_trackman_full_pipeline" if selected else "reject_reliable_trackman_axis"
    report["gate"] = "same reliability/blend positive in all three folds and 2024 >= +3"
    write_json(output, report)
    print(json.dumps({"selected": selected, "top": report["summaries"][:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--trackman", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.trackman.resolve(), args.mapping.resolve(), args.output.resolve(), args.task_type)
