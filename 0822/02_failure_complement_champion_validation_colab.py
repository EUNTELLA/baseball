"""현재 챔피언 위에서 R 실패확률 여집합 혼합의 추가 이득을 확인한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
PAIRS = ((2022, 2023), (2023, 2024))
SEEDS = (17, 42, 777)
BLENDS = (0.05, 0.10, 0.15, 0.20)
R_SCALE = 0.075
VERIFIED_SHIFT_DELTA = -0.0416386466 - (-0.03842671927234861)
BOOTSTRAPS = 500


def load_features():
    spec = importlib.util.spec_from_file_location("model_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_asset(directory: Path, year: int) -> dict[str, np.ndarray]:
    asset = np.load(directory / f"components_{year}.npz", allow_pickle=True)
    result = {name: asset[name] for name in asset.files}
    result["failure_complement"] = np.clip(
        1.0 - result["mr"].astype(float) - result["wayoff"].astype(float), 1e-6, 1 - 1e-6
    )
    return result


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def shift_to_mean(prediction: np.ndarray, target_mean: float) -> float:
    values = logit(prediction)
    low, high = -2.0, 2.0
    for _ in range(80):
        middle = (low + high) / 2
        if float(sigmoid(values + middle).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def bootstrap(ids, base, candidate, target, seed):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"id": ids.astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, counts = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / counts[sample].sum() > 0)
    return float(positive / BOOTSTRAPS)


def train_correction(frame, features, cat_indices, calibration_year, validation_year,
                     calibration, valid, task_type):
    train_mask = frame["season"].astype(int).eq(calibration_year).to_numpy()
    valid_mask = frame["season"].astype(int).eq(validation_year).to_numpy()
    train_rows = frame.loc[train_mask].reset_index(drop=True)
    valid_rows = frame.loc[valid_mask].reset_index(drop=True)
    if not np.array_equal(train_rows["row_id"].astype(str), calibration["row_id"].astype(str)):
        raise ValueError(f"{calibration_year} 구성요소 정렬 불일치")
    if not np.array_equal(valid_rows["row_id"].astype(str), valid["row_id"].astype(str)):
        raise ValueError(f"{validation_year} 구성요소 정렬 불일치")
    train_r = train_mask & frame["game_type"].astype(str).eq("R").to_numpy()
    valid_r = valid_mask & frame["game_type"].astype(str).eq("R").to_numpy()
    residual = calibration["target"].astype(float) - calibration["anchor"].astype(float)
    train_pool = Pool(
        features.loc[train_r], residual[train_rows["game_type"].astype(str).eq("R").to_numpy()],
        cat_features=cat_indices,
    )
    valid_pool = Pool(features.loc[valid_r], cat_features=cat_indices)
    members, seconds = [], []
    for seed in SEEDS:
        model = CatBoostRegressor(
            iterations=1200, depth=7, learning_rate=0.025, loss_function="RMSE",
            l2_leaf_reg=20, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=seed, task_type=task_type,
            devices="0" if task_type == "GPU" else None, thread_count=6,
            allow_writing_files=False, verbose=False,
        )
        started = time.perf_counter()
        model.fit(train_pool)
        seconds.append(float(time.perf_counter() - started))
        members.append(model.predict(valid_pool))
        print(f"fold={validation_year} R seed={seed} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    return np.mean(members, axis=0), valid_rows, seconds


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(component_dir: Path, train_path: Path, output: Path, task_type: str):
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    module = load_features()
    league_rate = float(frame.loc[frame["season"].astype(int).lt(2022), "control_success"].mean())
    features = module.engineer(frame.drop(columns=["row_id", "control_success"]), league_rate)
    for column in module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in module.CAT_COLS]
    assets = {year: load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    folds = []
    for calibration_year, validation_year in PAIRS:
        calibration, valid = assets[calibration_year], assets[validation_year]
        correction, valid_rows, seconds = train_correction(
            frame, features, cat_indices, calibration_year, validation_year,
            calibration, valid, task_type,
        )
        target = valid["target"].astype(float)
        anchor = valid["anchor"].astype(float)
        valid_r = valid_rows["game_type"].astype(str).eq("R").to_numpy()
        champion = anchor.copy()
        champion[valid_r] = np.clip(champion[valid_r] + R_SCALE * correction, 1e-6, 1 - 1e-6)
        champion = sigmoid(logit(champion) + VERIFIED_SHIFT_DELTA)
        champion_score = bss(champion, target)
        alignment_shift = shift_to_mean(
            calibration["failure_complement"].astype(float),
            float(calibration["anchor"].astype(float).mean()),
        )
        aligned = sigmoid(logit(valid["failure_complement"].astype(float)) + alignment_shift)
        candidates = []
        for blend in BLENDS:
            candidate = anchor.copy()
            candidate[valid_r] = (1 - blend) * candidate[valid_r] + blend * aligned[valid_r]
            candidate[valid_r] = np.clip(candidate[valid_r] + R_SCALE * correction, 1e-6, 1 - 1e-6)
            candidate = sigmoid(logit(candidate) + VERIFIED_SHIFT_DELTA)
            candidates.append({
                "blend": blend,
                "bss_delta": bss(candidate, target) - champion_score,
                "pitcher_bootstrap_probability": bootstrap(
                    valid_rows["pitcher_id"].to_numpy(), champion, candidate, target,
                    822200 + validation_year + int(blend * 1000),
                ),
                "absolute_mean_error_delta": (
                    abs(float(candidate.mean()) - float(target.mean()))
                    - abs(float(champion.mean()) - float(target.mean()))
                ),
            })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "champion_score": champion_score, "alignment_shift": alignment_shift,
            "r_training_seconds": seconds, "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} complete", flush=True)
    summaries = []
    for blend in BLENDS:
        rows = [next(row for row in fold["candidates"] if row["blend"] == blend) for fold in folds]
        deltas = [float(row["bss_delta"]) for row in rows]
        probabilities = [float(row["pitcher_bootstrap_probability"]) for row in rows]
        ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 0.0
        summaries.append({
            "blend": blend, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "worst_delta": min(deltas), "magnitude_ratio": ratio,
            "minimum_pitcher_bootstrap_probability": min(probabilities),
            "passed": bool(min(deltas) >= 1 and ratio >= 0.25 and min(probabilities) >= 0.80),
        })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "failure complement R blend over R0075 verified-shift champion",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale": R_SCALE, "verified_shift_delta": VERIFIED_SHIFT_DELTA,
        "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "build_reconstructed_anchor_submission" if passed else "keep_current_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries,
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
