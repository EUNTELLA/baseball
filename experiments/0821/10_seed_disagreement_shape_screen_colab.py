"""시드별 예측 불일치로 만든 행 단위 불확실성 보정을 시간 전방 검증한다."""
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
SEEDS = (42, 7, 2024, 99, 1, 123, 777)
PAIRS = ((2022, 2023), (2023, 2024))
SCALES = (0.25, 0.50, 0.75, 1.00)
RIDGES = (1e2, 1e3, 1e4)
BOOTSTRAPS = 500
ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
BASE_PATH = ROOT / "0820" / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_anchor(frame: pd.DataFrame, year: int, anchor_dir: Path):
    rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
    asset = np.load(anchor_dir / f"anchor_{year}.npz", allow_pickle=True)
    if len(rows) != len(asset["row_id"]):
        raise ValueError(f"{year} anchor 행 수 불일치")
    if not np.array_equal(rows[ID_COL].astype(str).to_numpy(), asset["row_id"].astype(str)):
        raise ValueError(f"{year} anchor row_id 순서 불일치")
    target = rows[TARGET_COL].astype(int).to_numpy()
    return rows, target, asset["prediction"].astype(float)


def seed_predictions(frame, target, season, year, feature_module, base_module, task_type):
    train_mask, valid_mask = season < year, season == year
    league_rate = float(target[train_mask].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    members, iterations = [], []
    for seed in SEEDS:
        model = CatBoostClassifier(**base_module.classifier_params(seed, task_type))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        members.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        del model
        gc.collect()
    matrix = np.column_stack(members)
    return matrix, iterations


def uncertainty_features(rows: pd.DataFrame, anchor: np.ndarray, members: np.ndarray) -> np.ndarray:
    std = members.std(axis=1)
    spread = members.max(axis=1) - members.min(axis=1)
    mean_gap = members.mean(axis=1) - anchor
    confidence = np.abs(anchor - 0.5)
    league_f = rows["game_type"].astype(str).eq("F").to_numpy(float)
    return np.column_stack([
        std,
        spread,
        mean_gap,
        std * (anchor - 0.5),
        std * confidence,
        spread * (anchor - 0.5),
        league_f * std,
        league_f * std * (anchor - 0.5),
    ])


def standardize(source: np.ndarray, valid: np.ndarray):
    mean = source.mean(axis=0)
    std = source.std(axis=0)
    std[std < 1e-12] = 1.0
    return (source - mean) / std, (valid - mean) / std


def ridge_correction(source_x: np.ndarray, source_residual: np.ndarray,
                     valid_x: np.ndarray, alpha: float) -> np.ndarray:
    gram = source_x.T @ source_x
    coefficient = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), source_x.T @ source_residual)
    return valid_x @ coefficient


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
    feature_module = load_module("seed_disagreement_features", FEATURE_PATH)
    base_module = load_module("seed_disagreement_base", BASE_PATH)
    anchors = {year: aligned_anchor(frame, year, anchor_dir) for year in (2022, 2023, 2024)}
    member_predictions, training = {}, []
    for year in (2022, 2023, 2024):
        matrix, iterations = seed_predictions(
            frame, target, season, year, feature_module, base_module, task_type
        )
        member_predictions[year] = matrix
        training.append({"year": year, "best_iterations": iterations})
        print(f"year={year} seed models complete iterations={iterations}", flush=True)

    fold_results = []
    for source_year, valid_year in PAIRS:
        source_rows, source_target, source_anchor = anchors[source_year]
        valid_rows, valid_target, valid_anchor = anchors[valid_year]
        source_x = uncertainty_features(source_rows, source_anchor, member_predictions[source_year])
        valid_x = uncertainty_features(valid_rows, valid_anchor, member_predictions[valid_year])
        source_x, valid_x = standardize(source_x, valid_x)
        source_residual = source_target - source_anchor
        baseline_score = bss(valid_anchor, valid_target)
        candidates = []
        for alpha in RIDGES:
            raw_correction = ridge_correction(source_x, source_residual, valid_x, alpha)
            # 검증 행 평균을 사용하지 않고 원천 예측 보정 평균만 제거한다.
            source_correction = ridge_correction(source_x, source_residual, source_x, alpha)
            correction = raw_correction - float(source_correction.mean())
            for scale in SCALES:
                applied = scale * correction
                candidate = np.clip(valid_anchor + applied, 1e-6, 1 - 1e-6)
                candidates.append({
                    "alpha": alpha, "scale": scale,
                    "bss_delta": bss(candidate, valid_target) - baseline_score,
                    "pitcher_bootstrap_probability": pitcher_bootstrap(
                        valid_rows, valid_anchor, candidate, valid_target,
                        seed=821100 + valid_year + int(np.log10(alpha)) * 100 + int(scale * 100),
                    ),
                    "correction_mean": float(applied.mean()),
                    "correction_std": float(applied.std()),
                    "residual_correlation": float(np.corrcoef(applied, valid_target - valid_anchor)[0, 1]),
                })
        fold_results.append({"source_year": source_year, "valid_year": valid_year,
                             "candidates": candidates})
        write_json(output, {"status": "running", "training": training, "fold_results": fold_results})
        print(f"fold={valid_year} complete", flush=True)

    summaries = []
    for alpha in RIDGES:
        for scale in SCALES:
            rows = [next(item for item in fold["candidates"]
                         if item["alpha"] == alpha and item["scale"] == scale)
                    for fold in fold_results]
            deltas = [float(item["bss_delta"]) for item in rows]
            probabilities = [float(item["pitcher_bootstrap_probability"]) for item in rows]
            ratio = min(abs(value) for value in deltas) / max(abs(value) for value in deltas) if max(map(abs, deltas)) else 0.0
            passed = min(deltas) >= 1.0 and ratio >= 0.25 and min(probabilities) >= 0.80
            summaries.append({
                "alpha": alpha, "scale": scale,
                "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                "worst_delta": min(deltas), "magnitude_ratio": ratio,
                "minimum_pitcher_bootstrap_probability": min(probabilities),
                "passed": bool(passed),
            })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "seed disagreement row uncertainty shape screen",
        "official_train_only": True, "test_aggregate_used": False,
        "training": training, "fold_results": fold_results, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_uncertainty_full_pipeline" if passed else "keep_r_scale0050_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries[:10],
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
