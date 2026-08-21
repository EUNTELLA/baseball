"""R 0.05 챔피언에 F 이전 유형 전환 잔차를 추가한 제출 ZIP을 만든다."""
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
from catboost import CatBoostRegressor, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024)
SCALE = 0.05
ROOT = Path(__file__).resolve().parents[1]
TRANSITION_PATH = ROOT / "0821" / "05_f_transition_residual_screen_colab.py"


def load_transition_module():
    spec = importlib.util.spec_from_file_location("f_transition_build", TRANSITION_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(TRANSITION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_block() -> str:
    return '''
    # 공식 Train의 2024 전체 anchor 잔차로 학습한 F 이전 유형 전환 보정.
    f_meta_path = os.path.join(BASE, "model", "f_transition_meta.json")
    if os.path.exists(f_meta_path):
        from catboost import CatBoostRegressor, Pool
        f_meta = json.load(open(f_meta_path, encoding="utf-8"))
        prior = f_meta["prior_type"]
        previous = test["pitcher_id"].astype(str).map(prior).fillna("NEW").astype(str)
        current = test["game_type"].astype(str)
        numeric = lambda name: pd.to_numeric(test[name], errors="coerce")
        f_x = pd.DataFrame({
            "game_type": current,
            "prior_type": previous,
            "transition": previous + ">" + current,
            "count": test["balls_before"].astype(str) + "-" + test["strikes_before"].astype(str),
            "hand": test["pitcher_hand"].astype(str) + "-" + test["batter_hand"].astype(str),
            "team_type": test["pitcher_team_id"].astype(str) + "|" + current,
            "base_prediction": p,
            "log_pitcher_n": np.log1p(numeric("asof_pitcher_n").fillna(0).clip(lower=0)),
            "career": numeric("asof_pitcher_success_rate"),
            "recent1": numeric("asof_pitcher_prev1_game_success_rate"),
            "recent3": numeric("asof_pitcher_prev3_game_success_rate"),
            "recent5": numeric("asof_pitcher_prev5_game_success_rate"),
            "middle": numeric("asof_pitcher_middle_rate"),
            "reverse": numeric("asof_pitcher_reverse_rate"),
            "li": numeric("li"),
            "inning": numeric("inning"),
            "runners": numeric("num_runners_on"),
        })
        for column in f_meta["cat_cols"]:
            f_x[column] = f_x[column].astype("string").fillna("__MISSING__").astype(str)
        f_pool = Pool(f_x, cat_features=[f_x.columns.get_loc(c) for c in f_meta["cat_cols"]])
        f_members = []
        for seed in f_meta["seeds"]:
            f_model = CatBoostRegressor()
            f_model.load_model(os.path.join(BASE, "model", f"f_transition_{seed}.cbm"))
            f_members.append(f_model.predict(f_pool))
        f_correction = np.mean(f_members, axis=0)
        f_mask = current.eq("F").to_numpy()
        p[f_mask] = np.clip(p[f_mask] + f_meta["scale"] * f_correction[f_mask], 1e-6, 1 - 1e-6)
'''


def main(train_path: Path, anchor_path: Path, base_zip: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    transition = load_transition_module()
    year_rows = frame.loc[frame["season"].astype(int).eq(2024)].reset_index(drop=True)
    anchor = np.load(anchor_path, allow_pickle=True)
    if len(year_rows) != len(anchor["row_id"]):
        raise ValueError("2024 anchor 행 수 불일치")
    if not np.array_equal(year_rows[ID_COL].astype(str).to_numpy(), anchor["row_id"].astype(str)):
        raise ValueError("2024 anchor row_id 순서 불일치")
    target = year_rows[TARGET_COL].astype(int).to_numpy()
    prediction = anchor["prediction"].astype(float)
    features = transition.transition_features(year_rows, prediction, frame, 2024)
    cat_indices = [features.columns.get_loc(column) for column in transition.CAT_COLS]
    train_pool = Pool(features, target - prediction, cat_features=cat_indices)

    with tempfile.TemporaryDirectory(prefix="f_transition_build_") as temporary:
        build_dir = Path(temporary) / "package"
        with zipfile.ZipFile(base_zip) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("기준 ZIP 손상")
            archive.extractall(build_dir)
        training = []
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostRegressor(
                iterations=250, depth=3, learning_rate=0.025,
                loss_function="RMSE", l2_leaf_reg=100, random_strength=0.2,
                bootstrap_type="Bernoulli", subsample=0.8,
                random_seed=seed, task_type=task_type,
                devices="0" if task_type == "GPU" else None,
                thread_count=6, allow_writing_files=False, verbose=50,
            )
            model.fit(train_pool)
            seconds = float(time.perf_counter() - started)
            model.save_model(build_dir / "model" / f"f_transition_{seed}.cbm")
            training.append({"seed": seed, "seconds": seconds})
            print(f"F transition seed={seed} sec={seconds:.1f}", flush=True)

        prior = transition.dominant_prior_type(frame, 2025).to_dict()
        meta = {
            "scale": SCALE, "seeds": list(SEEDS), "cat_cols": list(transition.CAT_COLS),
            "prior_type": {str(key): str(value) for key, value in prior.items()},
            "training_season": 2024, "training_rows": int(len(year_rows)),
            "test_aggregate_used": False,
        }
        (build_dir / "model" / "f_transition_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        script_path = build_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    pred_map = dict(zip(test[ID], p))\n"
        if marker not in script:
            raise RuntimeError("F 보정 삽입 위치를 찾지 못했습니다.")
        script_path.write_text(script.replace(marker, inference_block() + "\n" + marker), encoding="utf-8")

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

        verify_dir = Path(temporary) / "verify"
        with zipfile.ZipFile(output_zip) as archive:
            archive.extractall(verify_dir)
        (verify_dir / "data").mkdir()
        shutil.copy2(test_path, verify_dir / "data" / "test.csv")
        shutil.copy2(sample_path, verify_dir / "data" / "sample_submission.csv")
        completed = subprocess.run([sys.executable, "script.py"], cwd=verify_dir,
                                   capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(
                "샘플 추론 실패\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        submission = pd.read_csv(verify_dir / "output" / "submission.csv")
        if submission[TARGET_COL].isna().any() or not submission[TARGET_COL].between(0, 1).all():
            raise ValueError("샘플 추론 결측 또는 범위 오류")
        report = {
            "model": "R residual scale 0.05 champion plus F transition scale 0.05",
            "official_train_only": True, "test_aggregate_used": False,
            "base_zip": str(base_zip), "output_zip": str(output_zip),
            "anchor_path": str(anchor_path), "training": training, "meta": meta,
            "sha256": sha256(output_zip), "members": members, "zip_test_error": bad_member,
            "sample_rows": int(len(submission)), "sample_missing": int(submission[TARGET_COL].isna().sum()),
            "sample_min": float(submission[TARGET_COL].min()), "sample_max": float(submission[TARGET_COL].max()),
            "sample_mean": float(submission[TARGET_COL].mean()), "sample_stdout": completed.stdout.strip(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
