"""Strict F OOF를 행 독립 F 전용 student 모델로 재구성한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
YEARS = (2022, 2023, 2024)
SEEDS = (42, 7, 2024)
ID, TARGET = "row_id", "control_success"


def load_features():
    spec = importlib.util.spec_from_file_location("strict_student_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick(asset, *names):
    for name in names:
        if name in asset.files:
            return asset[name]
    raise KeyError(f"필요한 키 없음: {names}; 현재 키={asset.files}")


def load_oof(directory: Path, year: int):
    paths = list(directory.rglob(f"strict_f_regime075_oof_{year}.npz"))
    if not paths:
        paths = list(directory.rglob(f"*oof*{year}*.npz"))
    if not paths:
        raise FileNotFoundError(f"{year} OOF 없음: {directory}")
    asset = np.load(sorted(paths)[0], allow_pickle=True)
    return {
        "row_id": pick(asset, "row_id", "row_ids", "id").astype(str),
        "target": pick(asset, "target", "y", TARGET).astype(float),
        "teacher": pick(asset, "p_f_stack").astype(float),
        "shared": pick(asset, "p_shared_stack", "p_shared_adaptive").astype(float),
    }


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


def sigmoid(value):
    value = np.clip(np.asarray(value, float), -20, 20)
    return 1 / (1 + np.exp(-value))


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def params(seed, task_type, iterations):
    result = {
        "iterations": iterations, "depth": 7, "learning_rate": 0.04,
        "loss_function": "RMSE", "l2_leaf_reg": 20,
        "random_strength": 0.25, "bootstrap_type": "Bernoulli",
        "subsample": 0.85, "random_seed": seed,
        "allow_writing_files": False, "verbose": False,
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def main(train_path: Path, oof_dir: Path, output: Path, task_type: str, iterations: int):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    module = load_features()
    assets = {year: load_oof(oof_dir, year) for year in YEARS}
    rows, features = {}, {}
    global_prior = float(raw[TARGET].mean())
    for year in YEARS:
        frame = raw.loc[raw["season"].astype(int).eq(year)].reset_index(drop=True)
        asset = assets[year]
        if not np.array_equal(frame[ID].astype(str).to_numpy(), asset["row_id"]):
            raise ValueError(f"{year} row_id 정렬 불일치")
        if not np.array_equal(frame[TARGET].to_numpy(float), asset["target"]):
            raise ValueError(f"{year} target 정렬 불일치")
        f_mask = frame["game_type"].astype(str).eq("F").to_numpy()
        rows[year] = frame.loc[f_mask].reset_index(drop=True)
        engineered = module.engineer(
            frame.drop(columns=[ID, TARGET]).loc[f_mask].reset_index(drop=True), global_prior
        )
        for column in module.CAT_COLS:
            engineered[column] = engineered[column].astype(str)
        features[year] = engineered

    cat_indices = [features[2022].columns.get_loc(c) for c in module.CAT_COLS]
    folds = []
    for valid_year, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
        x_train = pd.concat([features[year] for year in train_years], ignore_index=True)
        teacher_train = np.concatenate([
            assets[year]["teacher"][raw.loc[raw["season"].astype(int).eq(year), "game_type"]
                .astype(str).eq("F").to_numpy()]
            for year in train_years
        ])
        valid_full = raw.loc[raw["season"].astype(int).eq(valid_year)].reset_index(drop=True)
        valid_mask = valid_full["game_type"].astype(str).eq("F").to_numpy()
        teacher_valid = assets[valid_year]["teacher"][valid_mask]
        shared_valid = assets[valid_year]["shared"][valid_mask]
        target_valid = assets[valid_year]["target"][valid_mask]
        train_pool = Pool(x_train, logit(teacher_train), cat_features=cat_indices)
        valid_pool = Pool(features[valid_year], cat_features=cat_indices)
        members, timings = [], []
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostRegressor(**params(seed, task_type, iterations))
            model.fit(train_pool)
            members.append(sigmoid(model.predict(valid_pool)))
            timings.append(float(time.perf_counter() - started))
            print(f"fold={valid_year} seed={seed} sec={timings[-1]:.1f}", flush=True)
        student = np.mean(members, axis=0)
        teacher_score = bss(teacher_valid, target_valid)
        shared_score = bss(shared_valid, target_valid)
        student_score = bss(student, target_valid)
        folds.append({
            "year": valid_year, "train_years": list(train_years),
            "train_f_rows": int(len(x_train)), "valid_f_rows": int(valid_mask.sum()),
            "teacher_bss": teacher_score, "shared_bss": shared_score,
            "student_bss": student_score,
            "teacher_delta_vs_shared": teacher_score - shared_score,
            "student_delta_vs_shared": student_score - shared_score,
            "student_gap_vs_teacher": student_score - teacher_score,
            "teacher_probability_rmse": float(np.sqrt(np.mean((student - teacher_valid) ** 2))),
            "teacher_probability_correlation": float(np.corrcoef(student, teacher_valid)[0, 1]),
            "seconds": timings,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    passed = bool(
        min(row["student_delta_vs_shared"] for row in folds) > 0
        and min(row["teacher_probability_correlation"] for row in folds) >= 0.90
        and max(row["teacher_probability_rmse"] for row in folds) <= 0.04
    )
    report = {
        "experiment": "strict F OOF row-local student screen",
        "official_train_only": True, "test_aggregate_used": False,
        "teacher_key": "p_f_stack", "baseline_key": "p_shared_stack",
        "seeds": list(SEEDS), "iterations": iterations, "folds": folds,
        "passed": passed,
        "decision": "build_strict_f_student_submission" if passed else "keep_current_champion",
        "gate": "student positive vs shared in 2023/2024; teacher correlation>=0.90; RMSE<=0.04",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    main(args.train.resolve(), args.oof_dir.resolve(), args.output.resolve(),
         args.task_type, args.iterations)
