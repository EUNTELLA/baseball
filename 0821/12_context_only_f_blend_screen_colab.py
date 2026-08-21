"""선수 이력을 제외한 경기 문맥 모델을 F행에 저강도로 혼합해 검증한다."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024)
PAIRS = ((2022, 2023), (2023, 2024))
BLENDS = (0.025, 0.05, 0.10, 0.15, 0.20)
BOOTSTRAPS = 500
CAT_COLS = (
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "count_state", "inning_band",
)
NUM_COLS = (
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "score_diff_home", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
)


def context_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    for column in CAT_COLS[:-2]:
        result[column] = rows[column].astype("string").fillna("__MISSING__").astype(str)
    balls = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(-1).astype(int)
    result["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    result["inning_band"] = pd.cut(
        pd.to_numeric(rows["inning"], errors="coerce"),
        [-np.inf, 3, 6, np.inf], labels=("early", "middle", "late"),
    ).astype("string").fillna("__MISSING__").astype(str)
    for column in NUM_COLS:
        result[column] = pd.to_numeric(rows[column], errors="coerce")
    return result


def aligned_anchor(frame: pd.DataFrame, year: int, anchor_dir: Path):
    rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
    asset = np.load(anchor_dir / f"anchor_{year}.npz", allow_pickle=True)
    if len(rows) != len(asset["row_id"]):
        raise ValueError(f"{year} anchor 행 수 불일치")
    if not np.array_equal(rows[ID_COL].astype(str).to_numpy(), asset["row_id"].astype(str)):
        raise ValueError(f"{year} anchor row_id 순서 불일치")
    target = rows[TARGET_COL].astype(int).to_numpy()
    return rows, target, asset["prediction"].astype(float)


def params(seed: int, task_type: str, iterations: int = 1200) -> dict:
    result = dict(
        iterations=iterations, depth=5, learning_rate=0.04, loss_function="Logloss",
        eval_metric="Logloss", l2_leaf_reg=10, random_strength=0.5,
        bootstrap_type="Bernoulli", subsample=0.8, random_seed=seed,
        allow_writing_files=False, verbose=False,
    )
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def train_pair(frame, target, season, calibration_year, validation_year, task_type):
    inner_train = season < calibration_year
    calibration = season == calibration_year
    outer_train = season < validation_year
    validation = season == validation_year
    x = context_features(frame.drop(columns=[ID_COL, TARGET_COL]))
    cat_indices = [x.columns.get_loc(column) for column in CAT_COLS]
    inner_train_pool = Pool(x.loc[inner_train], target[inner_train], cat_features=cat_indices)
    calibration_pool = Pool(x.loc[calibration], target[calibration], cat_features=cat_indices)
    outer_train_pool = Pool(x.loc[outer_train], target[outer_train], cat_features=cat_indices)
    validation_pool = Pool(x.loc[validation], cat_features=cat_indices)
    calibration_members, validation_members, iterations = [], [], []
    for seed in SEEDS:
        inner = CatBoostClassifier(**params(seed, task_type))
        inner.fit(inner_train_pool, eval_set=calibration_pool, use_best_model=True,
                  early_stopping_rounds=100)
        best = max(1, int(inner.get_best_iteration()) + 1)
        calibration_members.append(inner.predict_proba(calibration_pool)[:, 1])
        del inner
        gc.collect()
        outer = CatBoostClassifier(**params(seed, task_type, best))
        outer.fit(outer_train_pool)
        validation_members.append(outer.predict_proba(validation_pool)[:, 1])
        iterations.append(best)
        del outer
        gc.collect()
    return np.mean(calibration_members, axis=0), np.mean(validation_members, axis=0), iterations


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def mean_shift(prediction: np.ndarray, target_mean: float) -> float:
    values = logit(prediction)
    low, high = -1.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if float(sigmoid(values + middle).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def bss(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    rate = float(np.mean(target))
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def pitcher_bootstrap(rows, base, candidate, target, seed):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"pitcher": rows["pitcher_id"].astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("pitcher", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, counts = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / counts[sample].sum() > 0)
    return float(positive / BOOTSTRAPS)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, anchor_dir: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    anchors = {year: aligned_anchor(frame, year, anchor_dir) for year in (2022, 2023, 2024)}
    folds = []
    for calibration_year, validation_year in PAIRS:
        calibration_rows, _, calibration_anchor = anchors[calibration_year]
        valid_rows, valid_target, valid_anchor = anchors[validation_year]
        inner_prediction, outer_prediction, iterations = train_pair(
            frame, target, season, calibration_year, validation_year, task_type
        )
        calibration_f = calibration_rows["game_type"].astype(str).eq("F").to_numpy()
        valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        shift = mean_shift(inner_prediction[calibration_f], float(calibration_anchor[calibration_f].mean()))
        alternate = sigmoid(logit(outer_prediction) + shift)
        baseline_score = bss(valid_anchor, valid_target)
        error_correlation = float(np.corrcoef(
            valid_target[valid_f] - valid_anchor[valid_f],
            valid_target[valid_f] - alternate[valid_f],
        )[0, 1])
        candidates = []
        for blend in BLENDS:
            prediction = valid_anchor.copy()
            prediction[valid_f] = ((1 - blend) * valid_anchor[valid_f]
                                   + blend * alternate[valid_f])
            candidates.append({
                "blend": blend,
                "bss_delta": bss(prediction, valid_target) - baseline_score,
                "pitcher_bootstrap_probability": pitcher_bootstrap(
                    valid_rows, valid_anchor, prediction, valid_target,
                    seed=821200 + validation_year + int(blend * 1000),
                ),
                "absolute_mean_error_delta": (
                    abs(float(prediction.mean()) - float(valid_target.mean()))
                    - abs(float(valid_anchor.mean()) - float(valid_target.mean()))
                ),
            })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "best_iterations": iterations, "train_derived_logit_shift": shift,
            "f_error_correlation": error_correlation, "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} iter={iterations} corr={error_correlation:.6f}", flush=True)

    summaries = []
    for blend in BLENDS:
        rows = [next(item for item in fold["candidates"] if item["blend"] == blend) for fold in folds]
        deltas = [float(item["bss_delta"]) for item in rows]
        probabilities = [float(item["pitcher_bootstrap_probability"]) for item in rows]
        ratio = min(abs(value) for value in deltas) / max(abs(value) for value in deltas) if max(map(abs, deltas)) else 0.0
        passed = min(deltas) >= 1.0 and ratio >= 0.25 and min(probabilities) >= 0.80
        summaries.append({
            "blend": blend, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "worst_delta": min(deltas), "magnitude_ratio": ratio,
            "minimum_pitcher_bootstrap_probability": min(probabilities), "passed": bool(passed),
        })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "context-only low-correlation F blend screen",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale_fixed": 0.075, "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_context_f_full_pipeline" if passed else "keep_r_scale0075_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "fold_error_correlations":
                      [fold["f_error_correlation"] for fold in folds], "top": summaries,
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.anchor_dir.resolve(), args.output.resolve(), args.task_type)
