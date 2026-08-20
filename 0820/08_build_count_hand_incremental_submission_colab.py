"""현재 1029 제출에 Train 2024 카운트×손 조합 잔차 차등표를 추가한다."""
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
from catboost import CatBoostClassifier, Pool

ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024, 99, 1, 123, 777)
SHRINKAGE, SCALE = 500.0, 0.3
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FEATURE_PATH = ROOT / "common" / "model_features.py"
BASE_ZIP = ROOT / "0819" / "results" / "submit_catboost_residual_differential.zip"
OUTPUT_ZIP = SCRIPT_DIR / "results" / "submit_catboost_count_hand_incremental.zip"
BUILD_DIR = SCRIPT_DIR / "results" / "build_count_hand_incremental"


def load_features():
    spec = importlib.util.spec_from_file_location("official_features", FEATURE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(FEATURE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params(seed: int) -> dict:
    return {"iterations": 2000, "learning_rate": 0.05, "depth": 6,
            "l2_leaf_reg": 1.0, "random_seed": seed, "verbose": 0,
            "loss_function": "Logloss", "eval_metric": "Logloss",
            "early_stopping_rounds": 100, "grow_policy": "SymmetricTree",
            "task_type": "GPU", "devices": "0"}


def count_hand_key(frame: pd.DataFrame) -> pd.Series:
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    same = frame["pitcher_hand"].astype(str).eq(frame["batter_hand"].astype(str))
    hand = pd.Series(np.where(same, "same", "opposite"), index=frame.index)
    return balls + "-" + strikes + "|" + hand


def inference_block() -> str:
    return '''
    # 공식 Train 2024 OOF의 카운트×손 조합 잔차 차등. 현재 test 행만 조회한다.
    count_hand_path = os.path.join(BASE, "model", "count_hand_differential.json")
    if os.path.exists(count_hand_path):
        count_hand_asset = json.load(open(count_hand_path, encoding="utf-8"))
        balls_key = pd.to_numeric(test["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
        strikes_key = pd.to_numeric(test["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
        hand_key = np.where(test["pitcher_hand"].astype(str) == test["batter_hand"].astype(str), "same", "opposite")
        row_key = balls_key + "-" + strikes_key + "|" + pd.Series(hand_key, index=test.index)
        count_hand_correction = row_key.map(count_hand_asset["table"]).fillna(0.0).to_numpy(dtype=float)
        p = np.clip(p + count_hand_correction, 1e-6, 1 - 1e-6)
'''


def main(train_path: Path, test_path: Path, sample_path: Path, report_path: Path, base_zip: Path) -> None:
    if not base_zip.exists():
        raise FileNotFoundError(
            f"기준 1029 ZIP이 없습니다: {base_zip}. "
            "Drive에 보관한 ZIP을 --base-zip으로 지정하거나 0819/03 빌더로 재생성하세요."
        )
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    train_mask, valid_mask = season < 2024, season == 2024
    feature_module = load_features()
    league_rate = float(target[train_mask].mean())
    x = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
    for column in feature_module.CAT_COLS:
        x[column] = x[column].astype(str)
    cat_indices = [x.columns.get_loc(column) for column in feature_module.CAT_COLS]
    train_pool = Pool(x.loc[train_mask], target[train_mask], cat_features=cat_indices)
    valid_pool = Pool(x.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
    predictions, iterations, seconds = [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params(seed))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        predictions.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(max(1, int(model.get_best_iteration()) + 1))
        seconds.append(float(time.perf_counter() - started))
        print(f"source fold=2024 seed={seed} iter={iterations[-1]} sec={seconds[-1]:.1f}", flush=True)
        del model
        gc.collect()
    prediction = np.mean(predictions, axis=0)
    residual = target[valid_mask].astype(float) - prediction
    source_key = count_hand_key(frame.loc[valid_mask])
    grouped = pd.DataFrame({"key": source_key.to_numpy(), "residual": residual}).groupby("key").residual.agg(["sum", "count"])
    global_mean = float(residual.mean())
    grouped["difference"] = SCALE * (grouped["sum"] - grouped["count"] * global_mean) / (grouped["count"] + SHRINKAGE)
    table = {str(key): float(value) for key, value in grouped["difference"].items()}

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    with zipfile.ZipFile(base_zip) as archive:
        archive.extractall(BUILD_DIR)
    asset = {"source": "official Train 2024 OOF only", "axis": "count×hand match",
             "shrinkage": SHRINKAGE, "scale": SCALE, "source_global_residual_mean_removed": global_mean,
             "test_aggregate_used": False, "table": table}
    (BUILD_DIR / "model" / "count_hand_differential.json").write_text(
        json.dumps(asset, ensure_ascii=False), encoding="utf-8")
    script_path = BUILD_DIR / "script.py"
    script = script_path.read_text(encoding="utf-8")
    anchor = '    off = meta.get("offset")\n'
    if anchor not in script:
        raise RuntimeError("기준 추론 코드의 offset 위치를 찾지 못했습니다.")
    script_path.write_text(script.replace(anchor, inference_block() + "\n" + anchor), encoding="utf-8")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_DIR))
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad_member, members = archive.testzip(), archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    verify_dir = SCRIPT_DIR / "results" / "verify_count_hand_incremental"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        archive.extractall(verify_dir)
    (verify_dir / "data").mkdir()
    shutil.copy2(test_path, verify_dir / "data" / "test.csv")
    shutil.copy2(sample_path, verify_dir / "data" / "sample_submission.csv")
    completed = subprocess.run([sys.executable, "script.py"], cwd=verify_dir,
                               capture_output=True, text=True, check=True, timeout=600)
    submission = pd.read_csv(verify_dir / "output" / "submission.csv")
    if submission[TARGET_COL].isna().any() or not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("샘플 추론 결과의 결측 또는 확률 범위 오류")
    report = {"model": "1029 residual differential + count-hand incremental differential",
              "official_train_only": True, "test_aggregate_used": False,
              "base_zip": str(base_zip), "output_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
              "source_fold": 2024, "seeds": list(SEEDS), "best_iterations": iterations,
              "seconds": seconds, "groups": len(table), "shrinkage": SHRINKAGE, "scale": SCALE,
              "members": members, "zip_test_error": bad_member, "sample_rows": int(len(submission)),
              "sample_missing": int(submission[TARGET_COL].isna().sum()),
              "sample_min": float(submission[TARGET_COL].min()), "sample_max": float(submission[TARGET_COL].max()),
              "sample_mean": float(submission[TARGET_COL].mean()), "sample_stdout": completed.stdout.strip()}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-zip", type=Path, default=BASE_ZIP)
    args = parser.parse_args()
    main(args.train.resolve(), args.test.resolve(), args.sample.resolve(), args.report.resolve(), args.base_zip.resolve())
