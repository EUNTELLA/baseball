"""직전 시즌의 카운트·손 조합·LI 잔차 차등을 다음 시즌에 검증한다."""
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
SCALES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75)
SHRINKAGES = (500.0, 1500.0, 3000.0)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("dynamic_base_screen", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keys(frame: pd.DataFrame) -> pd.DataFrame:
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    count = balls + "-" + strikes
    ph = frame["pitcher_hand"].astype(str)
    bh = frame["batter_hand"].astype(str)
    hand = pd.Series(np.where(ph.eq(bh), "same", "opposite"), index=frame.index)
    li = pd.to_numeric(frame["li"], errors="coerce")
    leverage = pd.cut(li, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "medium", "high"]).astype(str)
    return pd.DataFrame({
        "count": count,
        "count_hand": count + "|" + hand,
        "count_leverage": count + "|" + leverage,
        "count_hand_leverage": count + "|" + hand + "|" + leverage,
    }, index=frame.index)


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {"brier": brier, "bss_score": float(1e5 * (1 - brier / (rate * (1 - rate)))),
            "prediction_mean": float(prediction.mean()), "target_mean": rate}


def train_prediction(frame, target, season, train_before, predict_year, base, feature_module, task_type):
    train_mask, valid_mask = season < train_before, season == predict_year
    league_rate = float(target[train_mask].mean())
    x = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
    for column in feature_module.CAT_COLS:
        x[column] = x[column].astype(str)
    cat_indices = [x.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(x.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(x.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    members, iterations = [], []
    for seed in SEEDS:
        model = CatBoostClassifier(**base.classifier_params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        members.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        del model
        gc.collect()
    return np.mean(members, axis=0), iterations


def differential(source_key, source_residual, target_key, shrinkage):
    table = pd.DataFrame({"key": source_key.to_numpy(), "residual": source_residual}).groupby("key").residual.agg(["sum", "count"])
    global_mean = float(np.mean(source_residual))
    table["difference"] = (table["sum"] - table["count"] * global_mean) / (table["count"] + shrinkage)
    return target_key.map(table["difference"]).fillna(0.0).to_numpy(float), int(len(table))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    condition_keys = keys(frame)
    report = {"experiment": "count-hand-leverage residual differential screen",
              "official_train_only": True, "external_predictions_used": False,
              "test_aggregate_used": False, "pairs": [], "candidates": []}

    predictions = {}
    for year in (2022, 2023, 2024):
        prediction, iterations = train_prediction(frame, target, season, year, year, base, feature_module, task_type)
        predictions[year] = prediction
        report.setdefault("base_training", []).append({"year": year, "best_iterations": iterations})
        print(f"base year={year} iterations={iterations}", flush=True)

    corrections = {}
    for source_year, valid_year in PAIRS:
        source_mask, valid_mask = season == source_year, season == valid_year
        source_residual = target[source_mask] - predictions[source_year]
        report["pairs"].append({"source_year": source_year, "valid_year": valid_year,
                                "source_global_residual_mean": float(source_residual.mean())})
        for axis in condition_keys.columns:
            for shrinkage in SHRINKAGES:
                correction, groups = differential(condition_keys.loc[source_mask, axis], source_residual,
                                                  condition_keys.loc[valid_mask, axis], shrinkage)
                corrections[(source_year, valid_year, axis, shrinkage)] = correction
                baseline = predictions[valid_year]
                y_valid = target[valid_mask]
                base_score = metric(baseline, y_valid)["bss_score"]
                for scale in SCALES:
                    candidate = np.clip(baseline + scale * correction, 1e-6, 1 - 1e-6)
                    candidate_metrics = metric(candidate, y_valid)
                    report["candidates"].append({
                        "source_year": source_year, "valid_year": valid_year,
                        "axis": axis, "shrinkage": shrinkage, "scale": scale, "groups": groups,
                        "bss_delta": candidate_metrics["bss_score"] - base_score,
                        "absolute_mean_error_delta": abs(candidate_metrics["prediction_mean"] - candidate_metrics["target_mean"])
                        - abs(float(baseline.mean()) - float(y_valid.mean())),
                    })
        write_json(output, report)

    candidates = pd.DataFrame(report["candidates"])
    summaries = []
    for (axis, shrinkage, scale), group in candidates.groupby(["axis", "shrinkage", "scale"]):
        values = group.set_index("valid_year").bss_delta
        if not all(year in values.index for year in (2023, 2024)):
            continue
        deltas = [float(values.loc[year]) for year in (2023, 2024)]
        summaries.append({"axis": axis, "shrinkage": float(shrinkage), "scale": float(scale),
                          "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                          "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
                          "both_positive": bool(min(deltas) > 0)})
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 2.0]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = sorted(summaries, key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    report["selected"] = selected
    report["decision"] = "continue_differential_full_pipeline" if selected else "reject_count_hand_leverage_differential"
    write_json(output, report)
    print(json.dumps({"selected": selected, "top": report["summaries"][:10], "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
