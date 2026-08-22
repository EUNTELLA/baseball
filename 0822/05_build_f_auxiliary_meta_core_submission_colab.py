"""검증된 현재 최고 ZIP에 F 보조확률 메타 core를 추가한다."""
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


ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = ROOT / "0822" / "04_f_auxiliary_meta_core_screen_colab.py"
SEEDS = (17, 42, 777)
SCALE = 0.025
EXPECTED_SHIFT = -0.0416386466
SOURCE_SHIFT = -0.03842671927234861
ID, TARGET = "row_id", "control_success"


def load_screen():
    spec = importlib.util.spec_from_file_location("f_meta_screen", SCREEN_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SCREEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inference_block() -> str:
    return '''
    # 공식 Train 2024 F행에서 학습한 보조확률 메타 core.
    f_meta_path = os.path.join(BASE, "model", "f_auxiliary_meta_core.json")
    if os.path.exists(f_meta_path):
        from catboost import CatBoostRegressor
        f_meta = json.load(open(f_meta_path, encoding="utf-8"))
        f_success = np.clip(avg_proba("model_", meta["seeds"]), 1e-6, 1 - 1e-6)
        f_mr = np.clip(avg_proba("mr_", f_meta["auxiliary_seeds"]), 1e-6, 1 - 1e-6)
        f_wayoff = np.clip(avg_proba("wayoff_", f_meta["auxiliary_seeds"]), 1e-6, 1 - 1e-6)
        f_failure = np.clip(1.0 - f_mr - f_wayoff, 1e-6, 1 - 1e-6)
        f_extra_shift = f_meta["verified_shift"] - f_meta["source_shift"]
        f_anchor = 1.0 / (1.0 + np.exp(-(
            np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            - f_extra_shift
        )))
        f_x = pd.DataFrame({
            "anchor_logit": np.log(f_anchor / (1 - f_anchor)),
            "success_logit": np.log(f_success / (1 - f_success)),
            "mr_logit": np.log(f_mr / (1 - f_mr)),
            "wayoff_logit": np.log(f_wayoff / (1 - f_wayoff)),
            "failure_complement_logit": np.log(f_failure / (1 - f_failure)),
            "success_minus_anchor": f_success - f_anchor,
            "failure_minus_anchor": f_failure - f_anchor,
            "mr_plus_wayoff": f_mr + f_wayoff,
            "count": test["balls_before"].astype(str) + "-" + test["strikes_before"].astype(str),
            "hand": test["pitcher_hand"].astype(str) + "-" + test["batter_hand"].astype(str),
            "base_state": test["base_state"].astype(str),
            "top_bottom": test["top_bottom"].astype(str),
            "pitcher_team_id": test["pitcher_team_id"].astype(str),
            "batter_team_id": test["batter_team_id"].astype(str),
            "inning": pd.to_numeric(test["inning"], errors="coerce"),
            "outs": pd.to_numeric(test["outs_before"], errors="coerce"),
            "runners": pd.to_numeric(test["num_runners_on"], errors="coerce"),
            "score_diff": pd.to_numeric(test["score_diff_pitcher_team"], errors="coerce"),
            "li": pd.to_numeric(test["li"], errors="coerce"),
            "pitcher_n": np.log1p(pd.to_numeric(test["asof_pitcher_n"], errors="coerce").fillna(0)),
            "pitcher_rate": pd.to_numeric(test["asof_pitcher_success_rate"], errors="coerce"),
            "recent1": pd.to_numeric(test["asof_pitcher_prev1_game_success_rate"], errors="coerce"),
            "recent5": pd.to_numeric(test["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
        })
        for f_column in f_meta["cat_cols"]:
            f_x[f_column] = f_x[f_column].astype("string").fillna("__MISSING__").astype(str)
        f_pool = Pool(f_x[f_meta["feature_cols"]], cat_features=f_meta["cat_cols"])
        f_members = []
        for f_seed in f_meta["seeds"]:
            f_model = CatBoostRegressor()
            f_model.load_model(os.path.join(BASE, "model", f"f_meta_{f_seed}.cbm"))
            f_members.append(f_model.predict(f_pool))
        f_correction = np.mean(f_members, axis=0)
        f_mask = test["game_type"].astype(str).eq("F").to_numpy()
        p[f_mask] = np.clip(
            p[f_mask] + f_meta["scale"] * f_correction[f_mask], 1e-6, 1 - 1e-6
        )
'''


def run(package: Path, test: pd.DataFrame, sample: pd.DataFrame, tag: str):
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data = run_dir / "data"
    data.mkdir(exist_ok=True)
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=run_dir, capture_output=True,
        text=True, check=False, timeout=600,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\n{completed.stdout}\n{completed.stderr}")
    return pd.read_csv(run_dir / "output" / "submission.csv"), completed.stdout.strip()


def main(source_zip: Path, component_dir: Path, train_path: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, task_type: str):
    screen = load_screen()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    rows = frame.loc[frame["season"].astype(int).eq(2024)].reset_index(drop=True)
    asset_file = np.load(component_dir / "components_2024.npz", allow_pickle=True)
    asset = {name: asset_file[name] for name in asset_file.files}
    if not np.array_equal(rows[ID].astype(str), asset["row_id"].astype(str)):
        raise ValueError("2024 구성요소 정렬 불일치")
    f_mask = rows["game_type"].astype(str).eq("F").to_numpy()
    x = screen.meta_features(rows, asset)
    base = screen.sigmoid(screen.logit(asset["anchor"].astype(float)) + screen.VERIFIED_SHIFT_DELTA)
    target = asset["target"].astype(float) - base
    train_pool = Pool(x.loc[f_mask], target[f_mask], cat_features=list(screen.CAT))
    with tempfile.TemporaryDirectory(prefix="f_meta_build_") as temporary:
        root = Path(temporary)
        source_dir, candidate_dir = root / "source", root / "candidate"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        shutil.copytree(source_dir, candidate_dir)
        meta = json.loads((candidate_dir / "model" / "meta.json").read_text(encoding="utf-8"))
        if abs(float(meta.get("logit_shift", 0)) - EXPECTED_SHIFT) > 1e-10:
            raise ValueError(f"원본 ZIP shift 불일치: {meta.get('logit_shift')}")
        training = []
        for seed in SEEDS:
            model = CatBoostRegressor(
                iterations=300, depth=3, learning_rate=0.02, loss_function="RMSE",
                l2_leaf_reg=100, random_strength=0.2, bootstrap_type="Bernoulli",
                subsample=0.8, random_seed=seed, task_type=task_type,
                devices="0" if task_type == "GPU" else None, thread_count=6,
                allow_writing_files=False, verbose=False,
            )
            started = time.perf_counter()
            model.fit(train_pool)
            seconds = float(time.perf_counter() - started)
            model.save_model(candidate_dir / "model" / f"f_meta_{seed}.cbm")
            training.append({"seed": seed, "seconds": seconds})
            print(f"F meta seed={seed} sec={seconds:.1f}", flush=True)
        f_meta = {
            "scale": SCALE, "seeds": list(SEEDS), "feature_cols": list(x.columns),
            "cat_cols": list(screen.CAT), "auxiliary_seeds": list(meta["offset"]["seeds"]),
            "source_shift": SOURCE_SHIFT, "verified_shift": EXPECTED_SHIFT,
            "training_season": 2024, "training_rows": int(f_mask.sum()),
            "target": "control_success minus verified-shift anchor", "test_aggregate_used": False,
        }
        (candidate_dir / "model" / "f_auxiliary_meta_core.json").write_text(
            json.dumps(f_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        script_path = candidate_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    # 공식 Train의 2024 시간 안전 OOF 잔차로 학습한 R 행 전용 약한 보정.\n"
        if marker not in script:
            raise RuntimeError("R residual 앞 삽입 위치를 찾지 못했습니다.")
        script_path.write_text(script.replace(marker, inference_block() + "\n" + marker), encoding="utf-8")
        test = pd.read_csv(test_path, encoding="utf-8-sig")
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        source_real, source_stdout = run(source_dir, test, sample, "source_real")
        candidate_real, candidate_stdout = run(candidate_dir, test, sample, "candidate_real")
        real = test[[ID, "game_type"]].merge(source_real, on=ID).merge(
            candidate_real, on=ID, suffixes=("_source", "_candidate")
        )
        real_delta = real[f"{TARGET}_candidate"] - real[f"{TARGET}_source"]
        r_delta = real_delta[real["game_type"].astype(str).eq("R")]
        if len(r_delta) and float(r_delta.abs().max()) > 1e-12:
            raise ValueError(f"R행이 변경됨: {float(r_delta.abs().max())}")
        # 제공 샘플에 F행이 없어도 F 경로가 실제로 동작하는지 합성 행으로 확인한다.
        smoke_test = test.iloc[[0]].copy()
        smoke_test["game_type"] = "F"
        smoke_sample = sample.iloc[[0]].copy()
        smoke_sample[ID] = smoke_test[ID].to_numpy()
        source_smoke, _ = run(source_dir, smoke_test, smoke_sample, "source_f_smoke")
        candidate_smoke, _ = run(candidate_dir, smoke_test, smoke_sample, "candidate_f_smoke")
        f_smoke_delta = float(candidate_smoke[TARGET].iloc[0] - source_smoke[TARGET].iloc[0])
        if f_smoke_delta == 0.0:
            raise ValueError("F 합성 행이 변경되지 않았습니다.")
        if candidate_real[TARGET].isna().any() or not candidate_real[TARGET].between(0, 1).all():
            raise ValueError("후보 결과 결측 또는 확률 범위 오류")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(candidate_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(candidate_dir))
        with zipfile.ZipFile(output_zip) as archive:
            zip_error = archive.testzip()
            members = archive.namelist()
        if zip_error is not None:
            raise RuntimeError(f"ZIP 손상: {zip_error}")
    report = {
        "experiment": "F auxiliary-probability meta core submission",
        "official_train_only": True, "test_aggregate_used": False,
        "source_zip": str(source_zip), "source_sha256": digest(source_zip),
        "output_zip": str(output_zip), "output_sha256": digest(output_zip),
        "training": training, "f_meta": f_meta, "members": members,
        "zip_test_error": zip_error, "sample_rows": int(len(candidate_real)),
        "sample_missing": int(candidate_real[TARGET].isna().sum()),
        "sample_min": float(candidate_real[TARGET].min()),
        "sample_max": float(candidate_real[TARGET].max()),
        "sample_mean": float(candidate_real[TARGET].mean()),
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_smoke_delta": f_smoke_delta,
        "source_stdout": source_stdout, "candidate_stdout": candidate_stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.component_dir.resolve(), args.train.resolve(),
         args.test.resolve(), args.sample.resolve(), args.output_zip.resolve(),
         args.report.resolve(), args.task_type)
