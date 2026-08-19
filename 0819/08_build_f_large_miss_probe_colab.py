"""1029 구성에 F 행 큰 이탈 보조 logit +0.025를 더한 탐색 제출본을 만든다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


TARGET_COL = "control_success"
DELTA_COEFFICIENT = 0.025
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_BUILDER_PATH = SCRIPT_DIR / "03_build_residual_differential_submission_colab.py"
BASE_ZIP = SCRIPT_DIR / "results" / "submit_catboost_residual_differential.zip"
OUTPUT_ZIP = SCRIPT_DIR / "results" / "submit_catboost_f_large_miss_0025.zip"
BUILD_DIR = SCRIPT_DIR / "results" / "build_f_large_miss_0025"


def load_builder():
    spec = importlib.util.spec_from_file_location("residual_submission_builder", BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inference_block() -> str:
    return f'''
    # 공식 Train 순방향 검증에서 고른 탐색 계수다. test에서는 현재 행의
    # game_type과 보조 모델 예측만 사용하며 다른 test 행을 집계하지 않는다.
    f_signal_path = os.path.join(BASE, "model", "f_large_miss_signal.json")
    if off and os.path.exists(f_signal_path):
        f_signal = json.load(open(f_signal_path, encoding="utf-8"))
        is_f = test["game_type"].astype(str).eq("F").to_numpy()
        large_miss_centered = (logit(avg_proba("wayoff_", off["seeds"]))
                               - off["mu_wayoff"])
        extra = np.where(is_f,
                         f_signal["delta_coefficient"] * large_miss_centered,
                         0.0)
        p = np.clip(1 / (1 + np.exp(-(logit(p) + extra))), 1e-6, 1 - 1e-6)
'''


def main(train: Path, test: Path, sample: Path, report_path: Path) -> None:
    builder = load_builder()
    base_report = SCRIPT_DIR / "results" / "f_large_miss_base_build.json"
    # Colab을 새로 시작해도 현재 1029 구조를 동일 코드로 재생성한다.
    builder.main(train, test, sample, base_report)
    if not BASE_ZIP.exists():
        raise FileNotFoundError(BASE_ZIP)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(BUILD_DIR)

    signal_meta = {
        "signal": "large_miss",
        "scope": "game_type F rows only",
        "delta_coefficient": DELTA_COEFFICIENT,
        "selection": {
            "fold_2023_bss_delta": 1.2352246650859797,
            "fold_2024_bss_delta": -0.10544318717053969,
            "mean_bss_delta": 0.56489073895772,
            "worst_bss_delta": -0.10544318717053969,
            "fold_2024_absolute_mean_error_delta": -6.620565052173344e-05,
        },
        "official_train_only": True,
        "test_aggregate_used": False,
        "purpose": "one-off leaderboard probe; not promoted by the local gate",
    }
    (BUILD_DIR / "model" / "f_large_miss_signal.json").write_text(
        json.dumps(signal_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    script_path = BUILD_DIR / "script.py"
    script = script_path.read_text(encoding="utf-8")
    anchor = "    shift = meta.get(\"logit_shift\")\n"
    if anchor not in script:
        raise RuntimeError("기준 추론 코드에서 shift 위치를 찾지 못했습니다.")
    script_path.write_text(
        script.replace(anchor, inference_block() + "\n" + anchor), encoding="utf-8"
    )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_DIR))
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    verify_dir = SCRIPT_DIR / "results" / "verify_f_large_miss_0025"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        archive.extractall(verify_dir)
    (verify_dir / "data").mkdir()
    shutil.copy2(test, verify_dir / "data" / "test.csv")
    shutil.copy2(sample, verify_dir / "data" / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=verify_dir,
        capture_output=True, text=True, check=True, timeout=600,
    )
    submission = pd.read_csv(verify_dir / "output" / "submission.csv")
    if submission[TARGET_COL].isna().any() or not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("샘플 추론 결과의 결측 또는 확률 범위 오류")
    report = {
        "model": "1029 residual-differential model plus F-only large-miss auxiliary logit",
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "output_zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "signal": signal_meta,
        "members": members,
        "zip_test_error": bad_member,
        "sample_rows": int(len(submission)),
        "sample_missing": int(submission[TARGET_COL].isna().sum()),
        "sample_min": float(submission[TARGET_COL].min()),
        "sample_max": float(submission[TARGET_COL].max()),
        "sample_mean": float(submission[TARGET_COL].mean()),
        "sample_stdout": completed.stdout.strip(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.train.resolve(), args.test.resolve(), args.sample.resolve(), args.report.resolve())
