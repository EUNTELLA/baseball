"""이전 시즌 Trackman 물리 변화량으로 다음 시즌 잔차를 전방 검증한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "07_count_hand_leverage_differential_screen_colab.py"
METRICS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed")
GROUPS = ("fastball", "breaking", "offspeed", "other")
PAIRS = ((2022, 2023), (2023, 2024))
ALPHAS = (100.0, 1000.0, 10000.0, 100000.0)
SCALES = (0.05, 0.10, 0.20, 0.30, 0.50)


def load_base():
    spec = importlib.util.spec_from_file_location("forward_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_changes(trackman_path: Path, mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path, dtype={"pitcher_id": str})
    mapping = mapping[mapping.similarity.ge(0.90) & mapping.margin.ge(0.02)].copy()
    usecols = ["pitcher_trackman_id", "season", "pitch_type_group", *METRICS]
    track = pd.read_csv(trackman_path, usecols=usecols, low_memory=False)
    track = track.merge(mapping[["pitcher_id", "pitcher_trackman_id", "similarity", "margin"]],
                        on="pitcher_trackman_id", how="inner")
    track["season"] = pd.to_numeric(track["season"], errors="coerce").fillna(-1).astype(int)
    aggregations = {metric: ["mean", "count"] for metric in METRICS}
    annual = track.groupby(["pitcher_id", "season"], observed=True).agg(aggregations)
    annual.columns = [f"{metric}_{stat}" for metric, stat in annual.columns]
    group_counts = (track.groupby(["pitcher_id", "season", "pitch_type_group"], observed=True)
                    .size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0))
    group_rates = group_counts.div(group_counts.sum(axis=1).replace(0, np.nan), axis=0)
    group_rates.columns = [f"group_{column}_rate" for column in group_rates.columns]
    annual = annual.join(group_rates).reset_index()

    rows = []
    for target_year in range(2021, 2025):
        last = annual[annual.season.eq(target_year - 1)].set_index("pitcher_id")
        older = annual[annual.season.lt(target_year - 1)].groupby("pitcher_id", observed=True)
        older_mean = older[[f"{metric}_mean" for metric in METRICS]
                           + [f"group_{group}_rate" for group in GROUPS]].mean()
        out = pd.DataFrame(index=last.index)
        out["season"] = target_year
        out["tm_last_n"] = last[f"{METRICS[0]}_count"]
        for metric in METRICS:
            out[f"change_{metric}"] = last[f"{metric}_mean"] - older_mean[f"{metric}_mean"]
        for group in GROUPS:
            column = f"group_{group}_rate"
            out[f"change_{column}"] = last[column] - older_mean[column]
        out = out.join(mapping.set_index("pitcher_id")[["similarity", "margin"]])
        rows.append(out.reset_index())
    return pd.concat(rows, ignore_index=True)


def attach(frame: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    key = pd.DataFrame({"pitcher_id": frame.pitcher_id.astype(str),
                        "season": frame.season.astype(int), "_order": np.arange(len(frame))})
    return (key.merge(assets, on=["pitcher_id", "season"], how="left", sort=False)
            .sort_values("_order").reset_index(drop=True))


def standardized(train: pd.DataFrame, valid: pd.DataFrame):
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, 1).fillna(1)
    return ((train - mean) / std).fillna(0), ((valid - mean) / std).fillna(0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, trackman_path: Path, mapping_path: Path,
         output: Path, task_type: str) -> None:
    base = load_base()
    direct = base.load_base()
    feature_module = direct.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame.control_success.astype(int).to_numpy()
    season = frame.season.astype(int).to_numpy()
    values = attach(frame, physical_changes(trackman_path, mapping_path))
    change_columns = [column for column in values if column.startswith("change_")]
    reliable = values.tm_last_n.ge(200) & values[change_columns].notna().any(axis=1)
    predictions = {}
    training = []
    for year in (2022, 2023, 2024):
        prediction, iterations = base.train_prediction(
            frame, target, season, year, year, direct, feature_module, task_type)
        predictions[year] = prediction
        training.append({"year": year, "best_iterations": iterations})
        print(f"base year={year} iterations={iterations}", flush=True)

    report = {"experiment": "prior-season Trackman physical-change residual screen",
              "official_data_only": True, "test_aggregate_used": False,
              "reliability": "similarity>=0.90, margin>=0.02, last-season pitches>=200",
              "change_columns": change_columns, "base_training": training,
              "pair_results": [], "candidates": []}
    for source_year, valid_year in PAIRS:
        source_mask = (season == source_year) & reliable.to_numpy()
        valid_mask = season == valid_year
        valid_reliable = valid_mask & reliable.to_numpy()
        x_source, x_valid = standardized(values.loc[source_mask, change_columns],
                                         values.loc[valid_mask, change_columns])
        residual = target[source_mask] - predictions[source_year][reliable[season == source_year].to_numpy()]
        baseline = predictions[valid_year]
        base_score = base.metric(baseline, target[valid_mask])["bss_score"]
        pair = {"source_year": source_year, "valid_year": valid_year,
                "source_rows": int(source_mask.sum()), "valid_coverage": float(valid_reliable.sum() / valid_mask.sum())}
        report["pair_results"].append(pair)
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=False).fit(x_source, residual)
            correction = model.predict(x_valid)
            correction[~reliable[valid_mask].to_numpy()] = 0.0
            for scale in SCALES:
                candidate = np.clip(baseline + scale * correction, 1e-6, 1 - 1e-6)
                metrics = base.metric(candidate, target[valid_mask])
                report["candidates"].append({"valid_year": valid_year, "alpha": alpha, "scale": scale,
                    "bss_delta": metrics["bss_score"] - base_score,
                    "absolute_mean_error_delta": abs(metrics["prediction_mean"] - metrics["target_mean"])
                    - abs(float(baseline.mean()) - float(target[valid_mask].mean()))})
        print(f"pair={source_year}->{valid_year} rows={source_mask.sum()} coverage={pair['valid_coverage']:.3f}", flush=True)
        write_json(output, report)
        gc.collect()

    table = pd.DataFrame(report["candidates"])
    summaries = []
    for (alpha, scale), group in table.groupby(["alpha", "scale"]):
        indexed = group.set_index("valid_year")
        deltas = [float(indexed.loc[year, "bss_delta"]) for year in (2023, 2024)]
        summaries.append({"alpha": float(alpha), "scale": float(scale),
                          "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                          "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
                          "both_positive": bool(min(deltas) > 0)})
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 3.0]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = sorted(summaries, key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    report["selected"] = selected
    report["decision"] = "continue_physical_change_full_pipeline" if selected else "reject_physical_change_axis"
    report["gate"] = "same alpha/scale positive in 2023 and 2024, with 2024 >= +3"
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
    main(args.train.resolve(), args.trackman.resolve(), args.mapping.resolve(),
         args.output.resolve(), args.task_type)
