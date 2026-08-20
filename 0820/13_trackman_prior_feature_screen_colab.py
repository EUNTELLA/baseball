"""고신뢰 공식 매핑의 이전 시즌 Trackman 물리 피처를 CatBoost에서 검증한다."""
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
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 7, 2024)
METRICS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed")
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("direct_catboost_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prior_assets(frame: pd.DataFrame, trackman_path: Path, mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path, dtype={"pitcher_id": str})
    mapping = mapping[mapping.similarity.ge(0.80) & mapping.margin.ge(0.02)].copy()
    usecols = ["pitcher_trackman_id", "season", "pitch_type_group", *METRICS]
    track = pd.read_csv(trackman_path, usecols=usecols, low_memory=False)
    track = track.merge(mapping[["pitcher_id", "pitcher_trackman_id", "similarity", "margin"]],
                        on="pitcher_trackman_id", how="inner")
    parts = {"tm_pitch_n": np.ones(len(track), dtype=float)}
    for metric in METRICS:
        value = pd.to_numeric(track[metric], errors="coerce")
        parts[f"{metric}_sum"] = value.fillna(0.0)
        parts[f"{metric}_sq"] = value.fillna(0.0).pow(2)
        parts[f"{metric}_n"] = value.notna().astype(float)
    for group in PITCH_GROUPS:
        parts[f"group_{group}_n"] = track["pitch_type_group"].astype(str).eq(group).astype(float)
    annual = pd.DataFrame(parts)
    annual["pitcher_id"] = track["pitcher_id"].astype(str).to_numpy()
    annual["season"] = pd.to_numeric(track["season"], errors="coerce").fillna(-1).astype(int).to_numpy()
    annual = annual.groupby(["pitcher_id", "season"], observed=True).sum().reset_index()
    confidence = mapping.set_index("pitcher_id")[["similarity", "margin"]]
    rows = []
    for target_season in sorted(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).unique()):
        history = annual[annual.season.lt(target_season)].groupby("pitcher_id", observed=True).sum(numeric_only=True)
        if history.empty:
            continue
        out = pd.DataFrame(index=history.index)
        out["season"] = target_season
        out["tm_prior_n"] = history["tm_pitch_n"]
        for metric in METRICS:
            n = history[f"{metric}_n"].replace(0.0, np.nan)
            mean = history[f"{metric}_sum"] / n
            out[f"tm_{metric}_mean"] = mean
            out[f"tm_{metric}_std"] = np.sqrt(np.clip(history[f"{metric}_sq"] / n - mean.pow(2), 0.0, None))
        total = sum(history[f"group_{group}_n"] for group in PITCH_GROUPS).replace(0.0, np.nan)
        for group in PITCH_GROUPS:
            out[f"tm_{group}_rate"] = history[f"group_{group}_n"] / total
        out = out.join(confidence.rename(columns={"similarity": "tm_mapping_similarity",
                                                   "margin": "tm_mapping_margin"}))
        rows.append(out.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["pitcher_id", "season"])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, trackman_path: Path, mapping_path: Path, output: Path, task_type: str) -> None:
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    assets = build_prior_assets(frame, trackman_path, mapping_path)
    merge_key = pd.DataFrame({"pitcher_id": frame["pitcher_id"].astype(str), "season": season,
                              "_row_order": np.arange(len(frame))})
    attached = (merge_key.merge(assets, on=["pitcher_id", "season"], how="left", sort=False)
                .sort_values("_row_order").reset_index(drop=True))
    tm_columns = [column for column in attached.columns if column.startswith("tm_")]
    coverage = {str(year): float(attached.loc[season == year, "tm_prior_n"].notna().mean()) for year in FOLDS}
    report = {"experiment": "official prior-season Trackman feature CatBoost screen",
              "official_data_only": True, "external_mapping_used": False,
              "test_aggregate_used": False, "trackman_columns": tm_columns,
              "coverage": coverage, "fold_results": []}

    for fold in FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league = float(target[train_mask].mean())
        basic = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league)
        augmented = basic.copy()
        for column in tm_columns:
            augmented[column] = pd.to_numeric(attached[column], errors="coerce").to_numpy()
        for data in (basic, augmented):
            for column in feature_module.CAT_COLS:
                data[column] = data[column].astype(str)
        predictions = {}
        training = {}
        for name, data in (("baseline", basic), ("trackman", augmented)):
            cat_indices = [data.columns.get_loc(column) for column in feature_module.CAT_COLS]
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
            predictions[name] = np.mean(members, axis=0)
            training[name] = iterations
        baseline_metrics = base.metric(predictions["baseline"], target[valid_mask])
        candidate_metrics = base.metric(predictions["trackman"], target[valid_mask])
        row = {"fold": fold, "training": training, "baseline": baseline_metrics,
               "candidate": candidate_metrics,
               "bss_delta": candidate_metrics["bss_score"] - baseline_metrics["bss_score"],
               "error_correlation": float(np.corrcoef(target[valid_mask] - predictions["baseline"],
                                                       target[valid_mask] - predictions["trackman"])[0, 1])}
        report["fold_results"].append(row)
        print(f"fold={fold} Trackman BSS delta={row['bss_delta']:+.2f}", flush=True)
        write_json(output, report)
        del basic, augmented
        gc.collect()
    deltas = [float(row["bss_delta"]) for row in report["fold_results"]]
    passed = min(deltas) > 0.0 and deltas[-1] >= 5.0
    report["summary"] = {"fold_deltas": deltas, "mean_delta": float(np.mean(deltas)),
                         "worst_delta": float(np.min(deltas)),
                         "decision": "continue_trackman_full_pipeline" if passed else "reject_trackman_features",
                         "gate": "positive in 2022/2023/2024 and 2024 >= +5"}
    write_json(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
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
