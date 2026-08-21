"""F 행의 조건별 과거 예측 오차가 다음 시즌으로 전이되는지 선별한다."""
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
SEEDS = (42, 7, 2024)
PAIRS = ((2022, 2023), (2023, 2024))
SHRINKAGES = (500.0, 1500.0, 3000.0)
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "0820" / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("f_condition_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_keys(frame: pd.DataFrame) -> pd.DataFrame:
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    count = balls + "-" + strikes
    pitcher_hand = frame["pitcher_hand"].astype(str)
    batter_hand = frame["batter_hand"].astype(str)
    hand = pd.Series(np.where(pitcher_hand.eq(batter_hand), "same", "opposite"), index=frame.index)
    leverage = pd.cut(
        pd.to_numeric(frame["li"], errors="coerce"),
        [-np.inf, 0.7, 1.5, np.inf], labels=["low", "medium", "high"],
    ).astype(str)
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(-1).astype(int).astype(str)
    inning = pd.cut(
        pd.to_numeric(frame["inning"], errors="coerce"),
        [-np.inf, 3, 6, np.inf], labels=["early", "middle", "late"],
    ).astype(str)
    return pd.DataFrame({
        "count_hand": count + "|" + hand,
        "count_leverage": count + "|" + leverage,
        "hand_runners": hand + "|" + runners,
        "count_inning": count + "|" + inning,
        "count_hand_runners": count + "|" + hand + "|" + runners,
    }, index=frame.index)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "bss_score": float(1e5 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def train_prediction(frame, target, season, year, base, feature_module, task_type):
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
        model = CatBoostClassifier(**base.classifier_params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        members.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        del model
        gc.collect()
    return np.mean(members, axis=0), iterations


def correction(source_key, source_residual, target_key, shrinkage):
    table = pd.DataFrame({"key": source_key.to_numpy(), "residual": source_residual})
    grouped = table.groupby("key").residual.agg(["sum", "count"])
    grouped["value"] = grouped["sum"] / (grouped["count"] + shrinkage)
    return target_key.map(grouped["value"]).fillna(0.0).to_numpy(float), int(len(grouped))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    is_f = frame["game_type"].astype(str).eq("F").to_numpy()
    keys = condition_keys(frame)
    predictions, training = {}, []

    for year in (2022, 2023, 2024):
        prediction, iterations = train_prediction(
            frame, target, season, year, base, feature_module, task_type
        )
        predictions[year] = prediction
        training.append({"year": year, "best_iterations": iterations})
        print(f"base year={year} iterations={iterations}", flush=True)

    candidate_rows = []
    for source_year, valid_year in PAIRS:
        source_all = season == source_year
        valid_all = season == valid_year
        source_f = source_all & is_f
        valid_f_local = is_f[valid_all]
        source_residual = target[source_f] - predictions[source_year][is_f[source_all]]
        baseline = predictions[valid_year]
        valid_target = target[valid_all]
        base_score = metrics(baseline, valid_target)["bss_score"]
        for axis in keys.columns:
            for shrinkage in SHRINKAGES:
                values, groups = correction(
                    keys.loc[source_f, axis], source_residual,
                    keys.loc[valid_all, axis], shrinkage,
                )
                values[~valid_f_local] = 0.0
                for scale in SCALES:
                    candidate = np.clip(baseline + scale * values, 1e-6, 1 - 1e-6)
                    result = metrics(candidate, valid_target)
                    candidate_rows.append({
                        "source_year": source_year, "valid_year": valid_year,
                        "axis": axis, "shrinkage": shrinkage, "scale": scale,
                        "source_f_rows": int(source_f.sum()), "valid_f_rows": int(valid_f_local.sum()),
                        "groups": groups, "bss_delta": result["bss_score"] - base_score,
                        "absolute_mean_error_delta": abs(result["prediction_mean"] - result["target_mean"])
                        - abs(float(baseline.mean()) - float(valid_target.mean())),
                    })
        write_json(output, {"status": "running", "training": training, "candidates": candidate_rows})

    candidates = pd.DataFrame(candidate_rows)
    summaries = []
    for (axis, shrinkage, scale), group in candidates.groupby(["axis", "shrinkage", "scale"]):
        values = group.set_index("valid_year").bss_delta
        deltas = [float(values.loc[year]) for year in (2023, 2024)]
        summaries.append({
            "axis": axis, "shrinkage": float(shrinkage), "scale": float(scale),
            "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
            "both_positive": bool(min(deltas) > 0),
        })
    summaries.sort(key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 2.0]
    selected = stable[0] if stable else None
    report = {
        "experiment": "F-row condition error transfer screen",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale_fixed": 0.05, "r_rows_modified_in_this_experiment": False,
        "training": training, "candidates": candidate_rows, "summaries": summaries,
        "selected": selected,
        "decision": "continue_f_condition_full_pipeline" if selected else "keep_r_scale0050_champion",
        "gate": "2023/2024 positive and 2024 delta >= +2",
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
