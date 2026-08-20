"""현재 최고 제출에 R 행 저강도 잔차 보정을 추가해 제출 ZIP을 만든다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "common" / "model_features.py"
SEEDS = (17, 42, 777)
SCALE = 0.025
ID_COL, TARGET_COL = "row_id", "control_success"


def load_features():
    spec = importlib.util.spec_from_file_location("model_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inference_block() -> str:
    return '''
    # 공식 Train의 2024 시간 안전 OOF 잔차로 학습한 R 행 전용 약한 보정.
    r_meta_path = os.path.join(BASE, "model", "r_residual_meta.json")
    if os.path.exists(r_meta_path):
        from catboost import CatBoostRegressor
        r_meta = json.load(open(r_meta_path, encoding="utf-8"))
        r_fe = engineer(test.drop(columns=[ID]), r_meta["global_mean"])
        r_x = prepare(r_fe, r_meta["feature_cols"], r_meta["cat_cols"])
        r_pool = Pool(r_x, cat_features=[r_x.columns.get_loc(c) for c in r_meta["cat_cols"]])
        r_members = []
        for seed in r_meta["seeds"]:
            r_model = CatBoostRegressor()
            r_model.load_model(os.path.join(BASE, "model", f"r_residual_{seed}.cbm"))
            r_members.append(r_model.predict(r_pool))
        r_correction = np.mean(r_members, axis=0)
        r_mask = test["game_type"].astype(str).eq("R").to_numpy()
        p[r_mask] = np.clip(p[r_mask] + r_meta["scale"] * r_correction[r_mask], 1e-6, 1 - 1e-6)
'''


def main(train_path: Path, anchor_path: Path, base_zip: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, task_type: str) -> None:
    if not base_zip.exists():
        raise FileNotFoundError(base_zip)
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    anchor = np.load(anchor_path)
    year_mask = frame["season"].astype(int).eq(2024).to_numpy()
    year_rows = frame.loc[year_mask].reset_index(drop=True)
    if len(year_rows) != len(anchor["row_id"]):
        raise ValueError("2024 anchor 행 수가 Train과 다릅니다.")
    if not np.array_equal(year_rows[ID_COL].to_numpy(), anchor["row_id"]):
        raise ValueError("2024 anchor row_id 순서가 Train과 다릅니다.")
    target = year_rows[TARGET_COL].astype(float).to_numpy()
    if not np.array_equal(target.astype(np.int8), anchor["target"]):
        raise ValueError("2024 anchor 정답 배열이 Train과 다릅니다.")

    feature_module = load_features()
    history_mask = frame["season"].astype(int).lt(2024).to_numpy()
    global_mean = float(frame.loc[history_mask, TARGET_COL].mean())
    features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), global_mean)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    feature_columns = list(features.columns)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    r_mask = year_mask & frame["game_type"].astype(str).eq("R").to_numpy()
    anchor_residual = np.full(len(frame), np.nan, dtype=float)
    anchor_residual[year_mask] = target - anchor["prediction"].astype(float)
    train_pool = Pool(
        features.loc[r_mask], anchor_residual[r_mask], cat_features=cat_indices
    )

    build_dir = output_zip.parent / "build_r_residual_probe"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    with zipfile.ZipFile(base_zip) as archive:
        archive.extractall(build_dir)
    training = []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostRegressor(
            iterations=1200, depth=7, learning_rate=0.025,
            loss_function="RMSE", l2_leaf_reg=20,
            random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=seed, task_type=task_type,
            devices="0" if task_type == "GPU" else None,
            thread_count=6, allow_writing_files=False, verbose=100,
        )
        model.fit(train_pool)
        seconds = float(time.perf_counter() - started)
        model.save_model(build_dir / "model" / f"r_residual_{seed}.cbm")
        training.append({"seed": seed, "seconds": seconds})
        print(f"R residual seed={seed} sec={seconds:.1f}", flush=True)
        del model
        gc.collect()

    r_meta = {
        "scale": SCALE, "seeds": list(SEEDS), "global_mean": global_mean,
        "feature_cols": feature_columns, "cat_cols": list(feature_module.CAT_COLS),
        "training_season": 2024, "training_rows": int(r_mask.sum()),
        "target": "control_success minus complete time-safe anchor prediction",
        "test_aggregate_used": False,
    }
    write_json(build_dir / "model" / "r_residual_meta.json", r_meta)
    script_path = build_dir / "script.py"
    script = script_path.read_text(encoding="utf-8")
    anchor_text = "    pred_map = dict(zip(test[ID], p))\n"
    if anchor_text not in script:
        raise RuntimeError("최종 확률 뒤의 삽입 위치를 찾지 못했습니다.")
    script_path.write_text(
        script.replace(anchor_text, inference_block() + "\n" + anchor_text), encoding="utf-8"
    )

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

    verify_dir = output_zip.parent / "verify_r_residual_probe"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    with zipfile.ZipFile(output_zip) as archive:
        archive.extractall(verify_dir)
    (verify_dir / "data").mkdir()
    shutil.copy2(test_path, verify_dir / "data" / "test.csv")
    shutil.copy2(sample_path, verify_dir / "data" / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=verify_dir,
        capture_output=True, text=True, check=True, timeout=600,
    )
    submission = pd.read_csv(verify_dir / "output" / "submission.csv")
    if submission[TARGET_COL].isna().any() or not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("샘플 추론 결과의 결측 또는 범위 오류")
    report = {
        "model": "current champion plus R-only three-seed residual at scale 0.025",
        "official_train_only": True, "test_aggregate_used": False,
        "base_zip": str(base_zip), "output_zip": str(output_zip),
        "anchor_path": str(anchor_path), "training": training, "r_meta": r_meta,
        "members": members, "zip_test_error": bad_member,
        "sample_rows": int(len(submission)), "sample_missing": int(submission[TARGET_COL].isna().sum()),
        "sample_min": float(submission[TARGET_COL].min()), "sample_max": float(submission[TARGET_COL].max()),
        "sample_mean": float(submission[TARGET_COL].mean()), "sample_stdout": completed.stdout.strip(),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.anchor.resolve(), args.base_zip.resolve(),
         args.test.resolve(), args.sample.resolve(), args.output_zip.resolve(),
         args.report.resolve(), args.task_type)
