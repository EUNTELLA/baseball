"""보조확률과 행 문맥으로 F 전용 저용량 메타 core를 선별한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


PAIRS = ((2022, 2023), (2023, 2024))
CONFIGS = (
    ("d2_l2100", 2, 100.0),
    ("d2_l2500", 2, 500.0),
    ("d3_l2100", 3, 100.0),
    ("d3_l2500", 3, 500.0),
)
SCALES = (0.025, 0.05, 0.075, 0.10)
MODES = ("raw", "train_centered")
SEEDS = (17, 42, 777)
CAT = ("count", "hand", "base_state", "top_bottom", "pitcher_team_id", "batter_team_id")
VERIFIED_SHIFT_DELTA = -0.0416386466 - (-0.03842671927234861)
BOOTSTRAPS = 500


def load_asset(directory: Path, year: int) -> dict[str, np.ndarray]:
    asset = np.load(directory / f"components_{year}.npz", allow_pickle=True)
    return {name: asset[name] for name in asset.files}


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def meta_features(rows: pd.DataFrame, asset: dict[str, np.ndarray]) -> pd.DataFrame:
    success = np.clip(asset["success"].astype(float), 1e-6, 1 - 1e-6)
    mr = np.clip(asset["mr"].astype(float), 1e-6, 1 - 1e-6)
    wayoff = np.clip(asset["wayoff"].astype(float), 1e-6, 1 - 1e-6)
    anchor = np.clip(asset["anchor"].astype(float), 1e-6, 1 - 1e-6)
    failure = np.clip(1 - mr - wayoff, 1e-6, 1 - 1e-6)
    result = pd.DataFrame({
        "anchor_logit": logit(anchor), "success_logit": logit(success),
        "mr_logit": logit(mr), "wayoff_logit": logit(wayoff),
        "failure_complement_logit": logit(failure),
        "success_minus_anchor": success - anchor,
        "failure_minus_anchor": failure - anchor,
        "mr_plus_wayoff": mr + wayoff,
        "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
        "hand": rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str),
        "base_state": rows["base_state"].astype(str),
        "top_bottom": rows["top_bottom"].astype(str),
        "pitcher_team_id": rows["pitcher_team_id"].astype(str),
        "batter_team_id": rows["batter_team_id"].astype(str),
        "inning": pd.to_numeric(rows["inning"], errors="coerce"),
        "outs": pd.to_numeric(rows["outs_before"], errors="coerce"),
        "runners": pd.to_numeric(rows["num_runners_on"], errors="coerce"),
        "score_diff": pd.to_numeric(rows["score_diff_pitcher_team"], errors="coerce"),
        "li": pd.to_numeric(rows["li"], errors="coerce"),
        "pitcher_n": np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0)),
        "pitcher_rate": pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce"),
        "recent1": pd.to_numeric(rows["asof_pitcher_prev1_game_success_rate"], errors="coerce"),
        "recent5": pd.to_numeric(rows["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
    })
    for column in CAT:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


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


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(component_dir: Path, train_path: Path, output: Path, task_type: str):
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    assets = {year: load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    rows_by_year = {
        year: frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    folds = []
    for calibration_year, validation_year in PAIRS:
        train_rows, valid_rows = rows_by_year[calibration_year], rows_by_year[validation_year]
        train_asset, valid_asset = assets[calibration_year], assets[validation_year]
        if not np.array_equal(train_rows["row_id"].astype(str), train_asset["row_id"].astype(str)):
            raise ValueError(f"{calibration_year} row_id 정렬 불일치")
        if not np.array_equal(valid_rows["row_id"].astype(str), valid_asset["row_id"].astype(str)):
            raise ValueError(f"{validation_year} row_id 정렬 불일치")
        train_f = train_rows["game_type"].astype(str).eq("F").to_numpy()
        valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        x_train = meta_features(train_rows, train_asset).loc[train_f].reset_index(drop=True)
        x_valid = meta_features(valid_rows, valid_asset).loc[valid_f].reset_index(drop=True)
        y_train = train_asset["target"].astype(float)[train_f]
        y_valid = valid_asset["target"].astype(float)[valid_f]
        base_train = sigmoid(logit(train_asset["anchor"].astype(float)[train_f]) + VERIFIED_SHIFT_DELTA)
        base_valid = sigmoid(logit(valid_asset["anchor"].astype(float)[valid_f]) + VERIFIED_SHIFT_DELTA)
        target_residual = y_train - base_train
        base_score = bss(base_valid, y_valid)
        candidates = []
        for name, depth, l2 in CONFIGS:
            members = []
            for seed in SEEDS:
                model = CatBoostRegressor(
                    iterations=300, depth=depth, learning_rate=0.02, loss_function="RMSE",
                    l2_leaf_reg=l2, random_strength=0.2, bootstrap_type="Bernoulli",
                    subsample=0.8, random_seed=seed, task_type=task_type,
                    devices="0" if task_type == "GPU" else None, thread_count=6,
                    allow_writing_files=False, verbose=False,
                )
                model.fit(Pool(x_train, target_residual, cat_features=list(CAT)))
                members.append(model.predict(Pool(x_valid, cat_features=list(CAT))))
            correction = np.mean(members, axis=0)
            for mode in MODES:
                shaped = correction.copy()
                if mode == "train_centered":
                    # 학습 시즌 수준을 다음 시즌으로 직접 운반하지 않는다.
                    train_prediction = np.mean([
                        # 동일 모델의 train 예측을 다시 얻는 대신 correction의 보수적 평균 제거만 사용한다.
                        float(target_residual.mean())
                    ])
                    shaped = shaped - train_prediction
                for scale in SCALES:
                    prediction = np.clip(base_valid + scale * shaped, 1e-6, 1 - 1e-6)
                    candidates.append({
                        "config": name, "mode": mode, "scale": scale,
                        "bss_delta": bss(prediction, y_valid) - base_score,
                        "pitcher_bootstrap_probability": bootstrap(
                            valid_rows.loc[valid_f, "pitcher_id"].to_numpy(),
                            base_valid, prediction, y_valid,
                            822400 + validation_year + depth * 100 + int(l2) + int(scale * 1000),
                        ),
                        "absolute_mean_error_delta": (
                            abs(float(prediction.mean()) - float(y_valid.mean()))
                            - abs(float(base_valid.mean()) - float(y_valid.mean()))
                        ),
                    })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "train_f_rows": int(train_f.sum()), "valid_f_rows": int(valid_f.sum()),
            "base_score": base_score, "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} candidates={len(candidates)}", flush=True)
    summaries = []
    for name, _, _ in CONFIGS:
        for mode in MODES:
            for scale in SCALES:
                rows = [next(row for row in fold["candidates"] if row["config"] == name
                             and row["mode"] == mode and row["scale"] == scale) for fold in folds]
                deltas = [float(row["bss_delta"]) for row in rows]
                probabilities = [float(row["pitcher_bootstrap_probability"]) for row in rows]
                ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 0.0
                summaries.append({
                    "config": name, "mode": mode, "scale": scale,
                    "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                    "worst_delta": min(deltas), "magnitude_ratio": ratio,
                    "minimum_pitcher_bootstrap_probability": min(probabilities),
                    "passed": bool(min(deltas) >= 1 and ratio >= 0.25 and min(probabilities) >= 0.80),
                })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "F auxiliary-probability meta core screen",
        "official_train_only": True, "test_aggregate_used": False,
        "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_f_meta_core" if passed else "keep_f_anchor_unchanged",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries[:10],
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
