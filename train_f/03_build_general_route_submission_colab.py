"""현재 R 챔피언과 자체 6-seed 공통 F 경로를 결합한 제출 ZIP을 만든다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
SEEDS = (42, 7, 2024, 99, 1, 123)
ITERATIONS = 128
EXPECTED_SOURCE_SHA = "1a83c9f0903dbdee9daeab4d4c76402c1402da5150d9a037da4803753aa6288b"
ID, TARGET = "row_id", "control_success"


def load_features():
    spec = importlib.util.spec_from_file_location("deployment_route_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def optimal_shift(prediction, target):
    values = logit(prediction)
    target = np.asarray(target, float)
    low, high = -0.25, 0.25
    for _ in range(80):
        middle = (low + high) / 2
        if float(np.mean(sigmoid(values + middle) - target)) < 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def params(seed, task_type):
    result = {
        "iterations": ITERATIONS, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 1.0, "loss_function": "Logloss", "eval_metric": "Logloss",
        "random_seed": seed, "grow_policy": "SymmetricTree",
        "allow_writing_files": False, "verbose": False,
    }
    if task_type == "GPU":
        result.update(task_type="GPU", devices="0")
    else:
        result["thread_count"] = -1
    return result


def engineered(frame, target, season, before_year, module):
    prior = float(target[season < before_year].mean())
    values = module.engineer(frame.drop(columns=[ID, TARGET]), prior)
    for column in module.CAT_COLS:
        values[column] = values[column].astype(str)
    indices = [values.columns.get_loc(column) for column in module.CAT_COLS]
    return values, indices, prior


def inference_block():
    return '''
    # 공식 Train 전체로 학습한 공통 경로를 Futures 행에만 적용한다.
    general_meta_path = os.path.join(BASE, "model", "general_route_meta.json")
    if os.path.exists(general_meta_path):
        from catboost import CatBoostClassifier, Pool
        from general_route import engineer as general_engineer, prepare as general_prepare
        general_meta = json.load(open(general_meta_path, encoding="utf-8"))
        general_fe = general_engineer(
            test.drop(columns=[ID]), general_meta["global_mean"]
        )
        general_x = general_prepare(
            general_fe, general_meta["feature_cols"], general_meta["cat_cols"]
        )
        general_pool = Pool(
            general_x,
            cat_features=[general_x.columns.get_loc(c) for c in general_meta["cat_cols"]],
        )
        general_members = []
        for general_seed in general_meta["seeds"]:
            general_model = CatBoostClassifier()
            general_model.load_model(
                os.path.join(BASE, "model", f"general_route_{general_seed}.cbm")
            )
            general_members.append(general_model.predict_proba(general_pool)[:, 1])
        general_prediction = np.mean(general_members, axis=0)
        general_prediction = 1.0 / (1.0 + np.exp(-(
            np.log(np.clip(general_prediction, 1e-6, 1 - 1e-6)
                   / np.clip(1 - general_prediction, 1e-6, 1 - 1e-6))
            + general_meta["calibration_shift"]
        )))
        general_f_mask = test["game_type"].astype(str).eq("F").to_numpy()
        p[general_f_mask] = np.clip(
            general_prediction[general_f_mask], 1e-6, 1 - 1e-6
        )
'''


def run_package(package, test, sample, tag):
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=run_dir, capture_output=True,
        text=True, check=False, timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\n{completed.stdout}\n{completed.stderr}")
    return pd.read_csv(run_dir / "output" / "submission.csv"), completed.stdout.strip()


def main(source_zip, train_path, test_path, sample_path, output_zip, report_path, task_type):
    source_sha = digest(source_zip)
    if source_sha != EXPECTED_SOURCE_SHA:
        raise ValueError(f"원본 챔피언 SHA 불일치: {source_sha}")
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame[TARGET].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    module = load_features()
    calibration_features, calibration_cat, _ = engineered(
        frame, target, season, 2024, module
    )
    final_features = module.engineer(
        frame.drop(columns=[ID, TARGET]), float(target.mean())
    )
    for column in module.CAT_COLS:
        final_features[column] = final_features[column].astype(str)
    final_cat = [final_features.columns.get_loc(column) for column in module.CAT_COLS]
    history = season < 2024
    holdout = season == 2024
    calibration_train_pool = Pool(
        calibration_features.loc[history], target[history], cat_features=calibration_cat
    )
    calibration_valid_pool = Pool(calibration_features.loc[holdout], cat_features=calibration_cat)
    final_pool = Pool(final_features, target, cat_features=final_cat)

    with tempfile.TemporaryDirectory(prefix="general_route_build_") as temporary:
        root = Path(temporary)
        source_dir, candidate_dir = root / "source", root / "candidate"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        shutil.copytree(source_dir, candidate_dir)
        calibration_members, training = [], []
        for seed in SEEDS:
            started = time.perf_counter()
            calibration_model = CatBoostClassifier(**params(seed, task_type))
            calibration_model.fit(calibration_train_pool)
            calibration_members.append(
                calibration_model.predict_proba(calibration_valid_pool)[:, 1]
            )
            del calibration_model
            final_model = CatBoostClassifier(**params(seed, task_type))
            final_model.fit(final_pool)
            final_model.save_model(candidate_dir / "model" / f"general_route_{seed}.cbm")
            seconds = float(time.perf_counter() - started)
            training.append({"seed": seed, "iterations": ITERATIONS, "seconds": seconds})
            print(f"general route seed={seed} sec={seconds:.1f}", flush=True)
            del final_model
        calibration_prediction = np.mean(calibration_members, axis=0)
        calibration_shift = optimal_shift(calibration_prediction, target[holdout])
        metadata = {
            "seeds": list(SEEDS), "iterations": ITERATIONS,
            "global_mean": float(target.mean()), "calibration_shift": calibration_shift,
            "calibration_season": 2024, "feature_cols": list(final_features.columns),
            "cat_cols": list(module.CAT_COLS), "active_region": "F",
            "test_aggregate_used": False,
        }
        (candidate_dir / "model" / "general_route_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(FEATURE_PATH, candidate_dir / "general_route.py")
        script_path = candidate_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    pred_map = dict(zip(test[ID], p))\n"
        if marker not in script:
            raise RuntimeError("최종 예측 삽입 위치를 찾지 못했습니다.")
        script_path.write_text(
            script.replace(marker, inference_block() + "\n" + marker), encoding="utf-8"
        )
        test = pd.read_csv(test_path, encoding="utf-8-sig")
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        source_real, source_stdout = run_package(source_dir, test, sample, "source_real")
        candidate_real, candidate_stdout = run_package(candidate_dir, test, sample, "candidate_real")
        comparison = test[[ID, "game_type"]].merge(source_real, on=ID).merge(
            candidate_real, on=ID, suffixes=("_source", "_candidate")
        )
        delta = comparison[f"{TARGET}_candidate"] - comparison[f"{TARGET}_source"]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        if len(r_delta) and float(r_delta.abs().max()) > 1e-12:
            raise ValueError(f"R행 변경: {float(r_delta.abs().max())}")
        if len(f_delta) and float(f_delta.abs().max()) == 0:
            raise ValueError("F행이 변경되지 않았습니다.")
        smoke_test = test.iloc[[0]].copy()
        smoke_test["game_type"] = "F"
        smoke_sample = sample.iloc[[0]].copy()
        smoke_sample[ID] = smoke_test[ID].to_numpy()
        source_smoke, _ = run_package(source_dir, smoke_test, smoke_sample, "source_smoke")
        candidate_smoke, _ = run_package(candidate_dir, smoke_test, smoke_sample, "candidate_smoke")
        smoke_delta = float(candidate_smoke[TARGET].iloc[0] - source_smoke[TARGET].iloc[0])
        if smoke_delta == 0:
            raise ValueError("F smoke 행이 변경되지 않았습니다.")
        if candidate_real[TARGET].isna().any() or not candidate_real[TARGET].between(0, 1).all():
            raise ValueError("후보 예측 결측 또는 범위 오류")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(candidate_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(candidate_dir))
        with zipfile.ZipFile(output_zip) as archive:
            zip_error, members = archive.testzip(), archive.namelist()
        if zip_error is not None:
            raise RuntimeError(f"ZIP 손상: {zip_error}")
    report = {
        "experiment": "current R champion plus own six-seed general Futures route",
        "official_train_only": True, "test_aggregate_used": False,
        "source_zip": str(source_zip), "source_sha256": source_sha,
        "output_zip": str(output_zip), "output_sha256": digest(output_zip),
        "training": training, "metadata": metadata, "members": members,
        "zip_test_error": zip_error, "sample_rows": int(len(candidate_real)),
        "sample_missing": int(candidate_real[TARGET].isna().sum()),
        "sample_min": float(candidate_real[TARGET].min()),
        "sample_max": float(candidate_real[TARGET].max()),
        "sample_mean": float(candidate_real[TARGET].mean()),
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_mean_delta": float(f_delta.mean()) if len(f_delta) else None,
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "f_smoke_delta": smoke_delta, "source_stdout": source_stdout,
        "candidate_stdout": candidate_stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.train.resolve(), args.test.resolve(),
         args.sample.resolve(), args.output_zip.resolve(), args.report.resolve(), args.task_type)
