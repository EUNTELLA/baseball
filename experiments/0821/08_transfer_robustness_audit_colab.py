"""후보 보정의 시즌 전이 안정성과 수준·형태 기여를 엄격하게 점검한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024)
PAIRS = ((2022, 2023), (2023, 2024))
SCALES = (0.025, 0.05, 0.075, 0.10)
BOOTSTRAPS = 500
ROOT = Path(__file__).resolve().parents[1]
TRANSITION_PATH = ROOT / "0821" / "05_f_transition_residual_screen_colab.py"


def load_transition_module():
    spec = importlib.util.spec_from_file_location("transition_robustness_source", TRANSITION_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(TRANSITION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_anchor(frame: pd.DataFrame, year: int, anchor_dir: Path):
    rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
    asset = np.load(anchor_dir / f"anchor_{year}.npz", allow_pickle=True)
    ids = rows[ID_COL].astype(str).to_numpy()
    if len(rows) != len(asset["row_id"]) or not np.array_equal(ids, asset["row_id"].astype(str)):
        raise ValueError(f"{year} anchor row_id 정렬 불일치")
    target = rows[TARGET_COL].astype(int).to_numpy()
    if not np.array_equal(target.astype(np.int8), asset["target"].astype(np.int8)):
        raise ValueError(f"{year} anchor 정답 불일치")
    return rows, target, asset["prediction"].astype(float)


def score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    denominator = float(target.mean() * (1.0 - target.mean()))
    return float(1e5 * (1.0 - np.mean((prediction - target) ** 2) / denominator))


def gain_parts(base: np.ndarray, correction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """클리핑 전 Brier 개선을 수준과 중심화 형태 항으로 분해한다."""
    residual = np.asarray(target, float) - np.asarray(base, float)
    correction = np.asarray(correction, float)
    level = float(correction.mean())
    shape = correction - level
    total = float(np.mean(2.0 * residual * correction - correction ** 2))
    level_part = float(np.mean(2.0 * residual * level - level ** 2))
    shape_part = float(np.mean(2.0 * residual * shape - shape ** 2))
    interaction = total - level_part - shape_part
    return {
        "brier_gain": total,
        "level_brier_gain": level_part,
        "shape_brier_gain": shape_part,
        "level_shape_interaction": interaction,
        "correction_mean": level,
        "correction_std": float(correction.std()),
        "residual_correlation": float(np.corrcoef(correction, residual)[0, 1]),
        "base_prediction_correlation": float(np.corrcoef(correction, base)[0, 1]),
    }


def pitcher_bootstrap(rows: pd.DataFrame, base: np.ndarray, candidate: np.ndarray,
                      target: np.ndarray, seed: int) -> dict[str, float]:
    """같은 투수의 반복 투구를 한 묶음으로 재표본화한다."""
    loss_delta = (np.asarray(base, float) - target) ** 2 - (np.asarray(candidate, float) - target) ** 2
    grouped = pd.DataFrame({
        "pitcher": rows["pitcher_id"].astype(str).to_numpy(),
        "gain_sum": loss_delta,
        "rows": 1,
    }).groupby("pitcher", observed=True).agg({"gain_sum": "sum", "rows": "sum"})
    sums = grouped["gain_sum"].to_numpy(float)
    counts = grouped["rows"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        values[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "clusters": int(len(grouped)),
        "probability_positive": float(np.mean(values > 0.0)),
        "brier_gain_p05": float(np.quantile(values, 0.05)),
        "brier_gain_p50": float(np.quantile(values, 0.50)),
        "brier_gain_p95": float(np.quantile(values, 0.95)),
    }


def train_correction(transition, frame, source_rows, source_target, source_prediction,
                     valid_rows, valid_prediction, source_year, valid_year, task_type):
    source_x = transition.transition_features(source_rows, source_prediction, frame, source_year)
    valid_x = transition.transition_features(valid_rows, valid_prediction, frame, valid_year)
    cat_indices = [source_x.columns.get_loc(column) for column in transition.CAT_COLS]
    source_pool = Pool(source_x, source_target - source_prediction, cat_features=cat_indices)
    source_predict_pool = Pool(source_x, cat_features=cat_indices)
    valid_pool = Pool(valid_x, cat_features=cat_indices)
    source_members, valid_members = [], []
    for seed in SEEDS:
        params = dict(
            iterations=250, depth=3, learning_rate=0.025, loss_function="RMSE",
            l2_leaf_reg=100, random_strength=0.2, bootstrap_type="Bernoulli",
            subsample=0.8, random_seed=seed, thread_count=6,
            allow_writing_files=False, verbose=False,
        )
        if task_type == "GPU":
            params.update(task_type="GPU", devices="0")
        model = CatBoostRegressor(**params)
        model.fit(source_pool)
        source_members.append(model.predict(source_predict_pool))
        valid_members.append(model.predict(valid_pool))
        del model
        gc.collect()
    return np.mean(source_members, axis=0), np.mean(valid_members, axis=0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, anchor_dir: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    transition = load_transition_module()
    folds = []
    for source_year, valid_year in PAIRS:
        source_rows, source_target, source_prediction = aligned_anchor(frame, source_year, anchor_dir)
        valid_rows, valid_target, valid_prediction = aligned_anchor(frame, valid_year, anchor_dir)
        source_correction, valid_correction = train_correction(
            transition, frame, source_rows, source_target, source_prediction,
            valid_rows, valid_prediction, source_year, valid_year, task_type,
        )
        source_f = source_rows["game_type"].astype(str).eq("F").to_numpy()
        valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        source_level = float(source_correction[source_f].mean())
        raw = np.where(valid_f, valid_correction, 0.0)
        centered = np.where(valid_f, valid_correction - source_level, 0.0)
        candidates = []
        baseline_score = score(valid_prediction, valid_target)
        for scale in SCALES:
            modes = {"raw": raw, "train_centered": centered}
            mode_results = {}
            for mode, correction in modes.items():
                applied = scale * correction
                candidate = np.clip(valid_prediction + applied, 1e-6, 1 - 1e-6)
                mode_results[mode] = {
                    "bss_delta": score(candidate, valid_target) - baseline_score,
                    "absolute_mean_error_delta": (
                        abs(float(candidate.mean()) - float(valid_target.mean()))
                        - abs(float(valid_prediction.mean()) - float(valid_target.mean()))
                    ),
                    "attribution": gain_parts(valid_prediction, applied, valid_target),
                    "pitcher_bootstrap": pitcher_bootstrap(
                        valid_rows, valid_prediction, candidate, valid_target,
                        seed=820000 + valid_year * 10 + int(scale * 1000) + (mode == "raw"),
                    ),
                }
            candidates.append({"scale": scale, "modes": mode_results})
        folds.append({
            "source_year": source_year, "valid_year": valid_year,
            "valid_rows": int(len(valid_rows)), "valid_f_rows": int(valid_f.sum()),
            "train_derived_f_correction_mean": source_level,
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={valid_year} complete", flush=True)

    summaries = []
    for scale in SCALES:
        for mode in ("raw", "train_centered"):
            selected = [next(row for row in fold["candidates"] if row["scale"] == scale)["modes"][mode]
                        for fold in folds]
            deltas = [float(row["bss_delta"]) for row in selected]
            probabilities = [float(row["pitcher_bootstrap"]["probability_positive"]) for row in selected]
            shape_gains = [float(row["attribution"]["shape_brier_gain"]) for row in selected]
            magnitude_ratio = (min(abs(value) for value in deltas) / max(abs(value) for value in deltas)
                               if max(abs(value) for value in deltas) > 0 else 0.0)
            passed = (
                min(deltas) >= 1.0
                and magnitude_ratio >= 0.25
                and min(probabilities) >= 0.80
                and min(shape_gains) > 0.0
            )
            summaries.append({
                "scale": scale, "mode": mode,
                "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                "worst_delta": min(deltas), "magnitude_ratio": magnitude_ratio,
                "minimum_pitcher_bootstrap_probability": min(probabilities),
                "shape_gain_positive_both_folds": bool(min(shape_gains) > 0.0),
                "passed": bool(passed),
            })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "candidate correction transfer robustness audit",
        "official_train_only": True,
        "test_aggregate_used": False,
        "anchor": "complete time-forward prediction",
        "candidate_family": "F prior-type transition residual",
        "folds": folds,
        "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "eligible_for_submission_review" if passed else "keep_r_scale0050_champion",
        "gate": {
            "minimum_bss_delta_each_fold": 1.0,
            "minimum_fold_magnitude_ratio": 0.25,
            "minimum_pitcher_bootstrap_probability": 0.80,
            "shape_brier_gain_positive_each_fold": True,
        },
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries[:8],
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
