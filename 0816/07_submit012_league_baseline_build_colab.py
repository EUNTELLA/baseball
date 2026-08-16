"""League-rate CatBoost 7시드 학습 및 submit012 기반 제출 ZIP 생성."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


SEEDS = [42, 7, 2024, 99, 1, 123, 777]
ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_ZIP = SCRIPT_DIR / "assets" / "submit012.zip"
# 2024 외삽 baseline 0.487742를 넣었을 때 예측 평균이 0.482827로 약 -0.004915.
# 2025 목표 평균 0.477에 같은 잔차를 적용해 입력 baseline을 학습 시 확정한다.
LEAGUE_BASELINE_2025 = 0.48191507866439055


def load_screen_module():
    path = SCRIPT_DIR / "06_submit012_league_baseline_colab.py"
    spec = importlib.util.spec_from_file_location("league_screen", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_params(task_type: str):
    params = dict(
        learning_rate=0.05,
        depth=6,
        verbose=100,
        eval_metric="Logloss",
        grow_policy="SymmetricTree",
    )
    if task_type == "GPU":
        params.update(task_type="GPU", devices="0")
    else:
        params.update(thread_count=-1)
    return params


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def patch_inference_script(path: Path) -> None:
    code = path.read_text(encoding="utf-8")
    old_pool = "pool = Pool(X, cat_features=[X.columns.get_loc(c) for c in cat_cols])"
    new_pool = '''cat_indices = [X.columns.get_loc(c) for c in cat_cols]
    pool = Pool(X, cat_features=cat_indices)
    baseline_rate = np.clip(meta["league_baseline_2025"], 1e-6, 1 - 1e-6)
    success_baseline = np.full(
        (len(X), 1), np.log(baseline_rate / (1 - baseline_rate))
    )
    success_pool = Pool(X, cat_features=cat_indices, baseline=success_baseline)'''
    old_function = "def avg_proba(prefix, seeds):"
    new_function = "def avg_proba(prefix, seeds, prediction_pool=pool):"
    old_predict = "ps.append(m.predict_proba(pool)[:, 1])"
    new_predict = "ps.append(m.predict_proba(prediction_pool)[:, 1])"
    old_success = 'p = np.clip(avg_proba("model_", meta["seeds"]), 1e-6, 1 - 1e-6)'
    new_success = 'p = np.clip(avg_proba("model_", meta["seeds"], success_pool), 1e-6, 1 - 1e-6)'
    for old in (old_pool, old_function, old_predict, old_success):
        if old not in code:
            raise RuntimeError(f"submit012 script 패턴을 찾지 못했습니다: {old}")
    code = code.replace(old_pool, new_pool)
    code = code.replace(old_function, new_function)
    code = code.replace(old_predict, new_predict)
    code = code.replace(old_success, new_success)
    path.write_text(code, encoding="utf-8")


def main(train_path: Path, output_zip: Path, work_dir: Path, task_type: str) -> None:
    screen = load_screen_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    train_mask = (frame["season"] <= 2023).to_numpy()
    valid_mask = (frame["season"] == 2024).to_numpy()
    global_mean = float(target[train_mask].mean())
    x = screen.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in screen.CAT_COLS:
        x[column] = x[column].astype(str)
    cat_indices = [x.columns.get_loc(column) for column in screen.CAT_COLS]

    rates_2019_2023 = (
        frame.loc[train_mask].groupby("season")[TARGET_COL].mean().to_dict()
    )
    forecast_2024 = screen.forecast_rate(rates_2019_2023, 2024)
    validation_train_baseline = logit(
        frame.loc[train_mask, "season"].map(rates_2019_2023).to_numpy()
    )[:, None]
    validation_baseline = np.full((int(valid_mask.sum()), 1), logit(forecast_2024))
    validation_train_pool = Pool(
        x.loc[train_mask], target[train_mask], cat_features=cat_indices,
        baseline=validation_train_baseline,
    )
    validation_pool = Pool(
        x.loc[valid_mask], target[valid_mask], cat_features=cat_indices,
        baseline=validation_baseline,
    )

    params = model_params(task_type)
    best_iterations = []
    validation_predictions = []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(
            **params, iterations=2000, early_stopping_rounds=100,
            random_seed=seed,
        )
        model.fit(
            validation_train_pool, eval_set=validation_pool, use_best_model=True
        )
        best_iteration = int(model.get_best_iteration())
        best_iterations.append(best_iteration)
        validation_predictions.append(model.predict_proba(validation_pool)[:, 1])
        print(
            f"validation seed={seed} iter={best_iteration} "
            f"seconds={time.perf_counter() - started:.1f}", flush=True,
        )
    validation_metrics = screen.bss(
        np.mean(validation_predictions, axis=0), target[valid_mask]
    )
    print("7-seed validation:", validation_metrics, flush=True)

    rates_all = frame.groupby("season")[TARGET_COL].mean().to_dict()
    full_baseline = logit(frame["season"].map(rates_all).to_numpy())[:, None]
    full_pool = Pool(x, target, cat_features=cat_indices, baseline=full_baseline)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    build_dir = work_dir / "package"
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(build_dir)
    model_dir = build_dir / "model"
    for old_model in model_dir.glob("model_*.cbm"):
        old_model.unlink()

    tags = []
    for seed, best_iteration in zip(SEEDS, best_iterations):
        tag = f"Sym_{seed}"
        tags.append(tag)
        model = CatBoostClassifier(
            **params,
            iterations=max(1, best_iteration + 1),
            random_seed=seed,
        )
        model.fit(full_pool)
        model.save_model(model_dir / f"model_{tag}.cbm")
        print(f"full model_{tag}.cbm iter={best_iteration + 1}", flush=True)

    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["seeds"] = tags
    meta["global_mean"] = global_mean
    meta["league_baseline"] = {
        "train_season_rates": {str(k): float(v) for k, v in rates_all.items()},
        "validation_forecast_2024": forecast_2024,
        "prediction_2025": LEAGUE_BASELINE_2025,
        "source": "training-time fixed; no test aggregate",
    }
    meta["league_baseline_2025"] = LEAGUE_BASELINE_2025
    meta.pop("logit_shift", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    patch_inference_script(build_dir / "script.py")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir))
    with zipfile.ZipFile(output_zip) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    report = {
        "base": "submit012 / LB 998.0030076995",
        "candidate": "7-seed league-rate baseline + submit012 fixed auxiliary offset",
        "task_type": task_type,
        "best_iterations": best_iterations,
        "validation": validation_metrics,
        "league_baseline_2025": LEAGUE_BASELINE_2025,
        "logit_shift_removed": True,
        "zip": str(output_zip),
        "zip_mib": output_zip.stat().st_size / 1024**2,
        "members": members,
    }
    report_path = output_zip.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/league_build"))
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    args = parser.parse_args()
    main(
        args.train.resolve(), args.output_zip.resolve(),
        args.work_dir.resolve(), args.task_type,
    )
