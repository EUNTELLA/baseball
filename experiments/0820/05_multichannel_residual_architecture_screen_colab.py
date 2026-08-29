"""서로 다른 시간가중치·복잡도의 잔차 채널 3개를 3연도에서 선별한다."""
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
FOLDS = (2022, 2023, 2024)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"

CHANNELS = (
    {"name": "compact_slow", "depth": 6, "iterations": 350, "decay": 0.75},
    {"name": "expanded_slow", "depth": 8, "iterations": 450, "decay": 0.55},
    {"name": "expanded_recent", "depth": 8, "iterations": 450, "decay": 0.30},
)


def load_base():
    spec = importlib.util.spec_from_file_location("dynamic_base_screen", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(channel: dict, task_type: str) -> dict:
    result = {
        "iterations": channel["iterations"], "depth": channel["depth"],
        "learning_rate": 0.035, "loss_function": "RMSE",
        "l2_leaf_reg": 20.0, "random_strength": 0.35,
        "bootstrap_type": "Bernoulli", "subsample": 0.85,
        "random_seed": 820500 + channel["depth"] + int(channel["decay"] * 100),
        "allow_writing_files": False, "verbose": 0,
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def compact_columns(frame: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    preferred = [
        "season", "inning", "top_bottom", "game_type", "balls_before",
        "strikes_before", "outs_before", "score_diff_pitcher_team",
        "num_runners_on", "li", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "dynamic_pitcher_base",
    ]
    columns = [column for column in preferred if column in frame.columns]
    selected_cats = [column for column in cat_cols if column in columns]
    return frame[columns].copy(), selected_cats


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base_module = load_base()
    feature_module = base_module.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    report = {
        "experiment": "independent three-channel residual architecture screen",
        "official_train_only": True, "external_data_used": False,
        "test_aggregate_used": False, "channels": list(CHANNELS),
        "fold_results": [],
    }
    pooled = {"target": [], "direct": [], **{c["name"]: [] for c in CHANNELS}, "equal_blend": []}

    for fold in FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league_rate = float(target[train_mask].mean())
        dynamic = base_module.dynamic_base(frame, league_rate)
        expanded = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
        expanded["dynamic_pitcher_base"] = dynamic
        for column in feature_module.CAT_COLS:
            expanded[column] = expanded[column].astype(str)
        expanded_cats = list(feature_module.CAT_COLS)
        compact, compact_cats = compact_columns(expanded, expanded_cats)
        for column in compact_cats:
            compact[column] = compact[column].astype(str)

        direct_ci = [expanded.columns.get_loc(c) for c in expanded_cats]
        direct_train = Pool(expanded.loc[train_mask], target[train_mask], cat_features=direct_ci)
        direct_valid = Pool(expanded.loc[valid_mask], target[valid_mask], cat_features=direct_ci)
        direct_members, direct_iters = [], []
        for seed in (42, 7, 2024):
            model = CatBoostClassifier(**base_module.classifier_params(seed, task_type))
            model.fit(direct_train, eval_set=direct_valid, use_best_model=True)
            direct_members.append(model.predict_proba(direct_valid)[:, 1])
            direct_iters.append(max(1, int(model.get_best_iteration()) + 1))
            del model
            gc.collect()
        direct = np.mean(direct_members, axis=0)
        residual_target = target.astype(float) - dynamic
        channel_predictions, channel_rows = {}, []
        for channel in CHANNELS:
            started = time.perf_counter()
            x, cats = (compact, compact_cats) if channel["name"] == "compact_slow" else (expanded, expanded_cats)
            ci = [x.columns.get_loc(c) for c in cats]
            weights = np.power(channel["decay"], (fold - 1) - season[train_mask])
            train_pool = Pool(x.loc[train_mask], residual_target[train_mask], cat_features=ci, weight=weights)
            valid_pool = Pool(x.loc[valid_mask], cat_features=ci)
            model = CatBoostRegressor(**params(channel, task_type))
            model.fit(train_pool)
            prediction = np.clip(dynamic[valid_mask] + model.predict(valid_pool), 1e-6, 1 - 1e-6)
            channel_predictions[channel["name"]] = prediction
            channel_rows.append({
                "name": channel["name"], "metrics": base_module.metric(prediction, target[valid_mask]),
                "seconds": float(time.perf_counter() - started),
                "error_correlation_vs_direct": float(np.corrcoef(target[valid_mask] - direct, target[valid_mask] - prediction)[0, 1]),
            })
            print(f"fold={fold} channel={channel['name']} sec={channel_rows[-1]['seconds']:.1f}", flush=True)
            del model, train_pool, valid_pool
            gc.collect()
        equal = np.mean(list(channel_predictions.values()), axis=0)
        direct_metrics = base_module.metric(direct, target[valid_mask])
        candidates = []
        for name, prediction in {**channel_predictions, "equal_blend": equal}.items():
            metrics = base_module.metric(prediction, target[valid_mask])
            candidates.append({"name": name, "metrics": metrics, "bss_delta_vs_direct": metrics["bss_score"] - direct_metrics["bss_score"]})
            pooled[name].append(prediction)
        pooled["target"].append(target[valid_mask])
        pooled["direct"].append(direct)
        report["fold_results"].append({
            "fold": fold, "direct_best_iterations": direct_iters,
            "direct": direct_metrics, "channel_diagnostics": channel_rows,
            "candidates": candidates,
        })
        write_json(output, report)

    candidate_names = [c["name"] for c in report["fold_results"][0]["candidates"]]
    summaries = []
    pooled_y = np.concatenate(pooled["target"])
    pooled_direct = np.concatenate(pooled["direct"])
    pooled_direct_score = base_module.metric(pooled_direct, pooled_y)["bss_score"]
    for name in candidate_names:
        deltas = [next(c["bss_delta_vs_direct"] for c in fold["candidates"] if c["name"] == name) for fold in report["fold_results"]]
        pooled_prediction = np.concatenate(pooled[name])
        summaries.append({
            "name": name, "fold_deltas": deltas, "mean_delta": float(np.mean(deltas)),
            "worst_delta": float(np.min(deltas)),
            "pooled_delta": base_module.metric(pooled_prediction, pooled_y)["bss_score"] - pooled_direct_score,
            "all_positive": bool(min(deltas) > 0),
        })
    stable = [row for row in summaries if row["all_positive"] and row["fold_deltas"][-1] >= 5]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["pooled_delta"])) if stable else None
    report["summaries"], report["selected"] = summaries, selected
    report["decision"] = "continue_selected_channels_to_seed_ensemble" if selected else "reject_multichannel_residual_architecture"
    write_json(output, report)
    print(json.dumps({"summaries": summaries, "selected": selected, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
