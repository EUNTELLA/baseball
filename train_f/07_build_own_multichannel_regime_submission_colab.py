"""검증된 자체 Futures 다중채널 residual regime 제출 ZIP을 만든다."""
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
SCREEN_PATH = ROOT / "train_f" / "06_own_multichannel_regime_screen_colab.py"
SEEDS = (17, 42, 777)
SCALE = 0.05
SOURCE_SHIFT = -0.03842671927234861
VERIFIED_SHIFT = -0.0416386466
ID, TARGET = "row_id", "control_success"


def load_screen():
    spec = importlib.util.spec_from_file_location("own_f_regime_screen", SCREEN_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SCREEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inference_block():
    return '''
    # 공식 Train으로 학습한 Futures 장기 다중채널 residual regime.
    own_f_path = os.path.join(BASE, "model", "own_f_multichannel_regime.json")
    if os.path.exists(own_f_path):
        from catboost import CatBoostRegressor
        own_f_meta = json.load(open(own_f_path, encoding="utf-8"))
        own_success = np.clip(avg_proba("model_", meta["seeds"]), 1e-6, 1 - 1e-6)
        own_mr = np.clip(avg_proba("mr_", own_f_meta["auxiliary_seeds"]), 1e-6, 1 - 1e-6)
        own_large = np.clip(avg_proba("wayoff_", own_f_meta["auxiliary_seeds"]), 1e-6, 1 - 1e-6)
        own_failure = np.clip(1.0 - own_mr - own_large, 1e-6, 1 - 1e-6)
        own_extra_shift = own_f_meta["verified_shift"] - own_f_meta["source_shift"]
        own_anchor = 1.0 / (1.0 + np.exp(-(
            np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            - own_extra_shift
        )))
        own_x = pd.DataFrame({
            "anchor_logit": np.log(own_anchor / (1 - own_anchor)),
            "success_logit": np.log(own_success / (1 - own_success)),
            "middle_reverse_logit": np.log(own_mr / (1 - own_mr)),
            "large_miss_logit": np.log(own_large / (1 - own_large)),
            "failure_complement_logit": np.log(own_failure / (1 - own_failure)),
            "success_minus_anchor": own_success - own_anchor,
            "failure_minus_anchor": own_failure - own_anchor,
            "failure_mass": own_mr + own_large,
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
            "recent3": pd.to_numeric(test["asof_pitcher_prev3_game_success_rate"], errors="coerce"),
            "recent5": pd.to_numeric(test["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
        })
        for own_column in own_f_meta["cat_cols"]:
            own_x[own_column] = own_x[own_column].astype("string").fillna("__MISSING__").astype(str)
        own_pool = Pool(own_x[own_f_meta["feature_cols"]], cat_features=own_f_meta["cat_cols"])
        own_members = []
        for own_seed in own_f_meta["seeds"]:
            own_model = CatBoostRegressor()
            own_model.load_model(os.path.join(BASE, "model", f"own_f_regime_{own_seed}.cbm"))
            own_members.append(own_model.predict(own_pool))
        own_correction = np.mean(own_members, axis=0)
        own_mask = test["game_type"].astype(str).eq("F").to_numpy()
        p[own_mask] = np.clip(
            p[own_mask] + own_f_meta["scale"] * own_correction[own_mask],
            1e-6, 1 - 1e-6
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
        text=True, check=False, timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\n{completed.stdout}\n{completed.stderr}")
    return pd.read_csv(run_dir / "output" / "submission.csv"), completed.stdout.strip()


def main(source_zip: Path, component_dir: Path, train_path: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, task_type: str):
    for required in (source_zip, train_path, test_path, sample_path):
        if not required.exists():
            raise FileNotFoundError(required)
    for year in (2022, 2023, 2024):
        component = component_dir / f"components_{year}.npz"
        if not component.exists():
            raise FileNotFoundError(component)
    screen = load_screen()
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    x_parts, residual_parts, weight_parts, training_rows = [], [], [], {}
    for year in (2022, 2023, 2024):
        rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
        loaded = np.load(component_dir / f"components_{year}.npz", allow_pickle=True)
        asset = {name: loaded[name] for name in loaded.files}
        if not np.array_equal(rows[ID].astype(str), asset["row_id"].astype(str)):
            raise ValueError(f"{year} row_id 정렬 불일치")
        mask = rows["game_type"].astype(str).eq("F").to_numpy()
        x = screen.features(rows, asset).loc[mask].reset_index(drop=True)
        base = screen.sigmoid(screen.logit(asset["anchor"][mask]) + screen.SHIFT_DELTA)
        residual = asset["target"][mask].astype(float) - base
        x_parts.append(x)
        residual_parts.append(residual)
        weight_parts.append(np.full(int(mask.sum()), 0.55 ** (2024 - year), dtype=float))
        training_rows[str(year)] = int(mask.sum())
    x_train = pd.concat(x_parts, ignore_index=True)
    residual_train = np.concatenate(residual_parts)
    train_weights = np.concatenate(weight_parts)
    train_pool = Pool(
        x_train, residual_train, weight=train_weights, cat_features=list(screen.CAT_COLS)
    )

    with tempfile.TemporaryDirectory(prefix="own_f_regime_build_") as temporary:
        root = Path(temporary)
        source_dir, candidate_dir = root / "source", root / "candidate"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        if not (source_dir / "script.py").exists():
            raise ValueError("원본 ZIP 최상위에 script.py가 없습니다.")
        shutil.copytree(source_dir, candidate_dir)
        base_meta = json.loads((candidate_dir / "model" / "meta.json").read_text(encoding="utf-8"))
        if abs(float(base_meta.get("logit_shift", 0)) - VERIFIED_SHIFT) > 1e-10:
            raise ValueError(f"원본 ZIP shift 불일치: {base_meta.get('logit_shift')}")
        training = []
        for seed in SEEDS:
            model = CatBoostRegressor(**screen.params(seed, task_type))
            started = time.perf_counter()
            model.fit(train_pool)
            seconds = float(time.perf_counter() - started)
            model.save_model(candidate_dir / "model" / f"own_f_regime_{seed}.cbm")
            training.append({"seed": seed, "seconds": seconds})
            print(f"own F regime seed={seed} sec={seconds:.1f}", flush=True)
        regime_meta = {
            "scale": SCALE, "recent_weight": 0.0, "seeds": list(SEEDS),
            "feature_cols": list(x_train.columns), "cat_cols": list(screen.CAT_COLS),
            "auxiliary_seeds": list(base_meta["offset"]["seeds"]),
            "source_shift": SOURCE_SHIFT, "verified_shift": VERIFIED_SHIFT,
            "training_rows_by_year": training_rows, "decay": 0.55,
            "target": "control_success minus verified-shift anchor",
            "official_train_only": True, "test_aggregate_used": False,
        }
        (candidate_dir / "model" / "own_f_multichannel_regime.json").write_text(
            json.dumps(regime_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        script_path = candidate_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    # 공식 Train의 2024 시간 안전 OOF 잔차로 학습한 R 행 전용 약한 보정.\n"
        if marker not in script:
            raise RuntimeError("F regime 삽입 위치를 찾지 못했습니다.")
        script_path.write_text(
            script.replace(marker, inference_block() + "\n" + marker), encoding="utf-8"
        )
        test = pd.read_csv(test_path, encoding="utf-8-sig")
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        source_real, source_stdout = run(source_dir, test, sample, "source_real")
        candidate_real, candidate_stdout = run(candidate_dir, test, sample, "candidate_real")
        comparison = test[[ID, "game_type"]].merge(source_real, on=ID).merge(
            candidate_real, on=ID, suffixes=("_source", "_candidate")
        )
        delta = comparison[f"{TARGET}_candidate"] - comparison[f"{TARGET}_source"]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        if len(r_delta) and float(r_delta.abs().max()) > 1e-12:
            raise ValueError(f"R행 변경: {float(r_delta.abs().max())}")
        smoke_test = test.iloc[[0]].copy()
        smoke_test["game_type"] = "F"
        smoke_sample = sample.iloc[[0]].copy()
        smoke_sample[ID] = smoke_test[ID].to_numpy()
        source_smoke, _ = run(source_dir, smoke_test, smoke_sample, "source_f_smoke")
        candidate_smoke, _ = run(candidate_dir, smoke_test, smoke_sample, "candidate_f_smoke")
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
        "experiment": "own Futures multichannel residual regime submission",
        "official_train_only": True, "test_aggregate_used": False,
        "source_zip": str(source_zip), "source_sha256": digest(source_zip),
        "output_zip": str(output_zip), "output_sha256": digest(output_zip),
        "training": training, "regime_meta": regime_meta,
        "members": members, "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_real)),
        "sample_missing": int(candidate_real[TARGET].isna().sum()),
        "sample_min": float(candidate_real[TARGET].min()),
        "sample_max": float(candidate_real[TARGET].max()),
        "sample_mean": float(candidate_real[TARGET].mean()),
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
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
