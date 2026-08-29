"""현재 최고 ZIP에 R 실패확률 여집합 0.20 혼합을 추가한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


BLEND = 0.20
EXPECTED_R_SCALE = 0.075
EXPECTED_SHIFT = -0.0416386466
ID, TARGET = "row_id", "control_success"


def digest(path: Path) -> str:
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


def shift_to_mean(prediction: np.ndarray, target_mean: float) -> float:
    values = logit(prediction)
    low, high = -2.0, 2.0
    for _ in range(80):
        middle = (low + high) / 2
        if float(sigmoid(values + middle).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def inference_block() -> str:
    return '''
    # 공식 Train의 2024 OOF 구성요소로 고정한 R행 실패확률 여집합 혼합.
    fc_meta_path = os.path.join(BASE, "model", "failure_complement_blend.json")
    if os.path.exists(fc_meta_path):
        fc_meta = json.load(open(fc_meta_path, encoding="utf-8"))
        mr_probability = avg_proba("mr_", fc_meta["auxiliary_seeds"])
        wayoff_probability = avg_proba("wayoff_", fc_meta["auxiliary_seeds"])
        failure_complement = np.clip(
            1.0 - mr_probability - wayoff_probability, 1e-6, 1 - 1e-6
        )
        aligned_failure = 1.0 / (1.0 + np.exp(-(
            np.log(failure_complement / (1 - failure_complement))
            + fc_meta["alignment_shift"]
        )))
        extra_shift = fc_meta["verified_shift"] - fc_meta["source_shift"]
        original_anchor = 1.0 / (1.0 + np.exp(-(
            np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            - extra_shift
        )))
        rebuilt = original_anchor.copy()
        fc_r_mask = test["game_type"].astype(str).eq("R").to_numpy()
        rebuilt[fc_r_mask] = (
            (1.0 - fc_meta["blend"]) * original_anchor[fc_r_mask]
            + fc_meta["blend"] * aligned_failure[fc_r_mask]
        )
        p = np.clip(1.0 / (1.0 + np.exp(-(
            np.log(np.clip(rebuilt, 1e-6, 1 - 1e-6)
                   / np.clip(1 - rebuilt, 1e-6, 1 - 1e-6)) + extra_shift
        ))), 1e-6, 1 - 1e-6)
'''


def run_package(package: Path, test_path: Path, sample_path: Path, tag: str) -> tuple[pd.DataFrame, str]:
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(test_path, data_dir / "test.csv")
    shutil.copy2(sample_path, data_dir / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=run_dir, capture_output=True,
        text=True, check=False, timeout=600,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return pd.read_csv(run_dir / "output" / "submission.csv"), completed.stdout.strip()


def main(source_zip: Path, component_dir: Path, test_path: Path, sample_path: Path,
         output_zip: Path, report_path: Path) -> None:
    component = np.load(component_dir / "components_2024.npz", allow_pickle=True)
    failure = np.clip(
        1.0 - component["mr"].astype(float) - component["wayoff"].astype(float),
        1e-6, 1 - 1e-6,
    )
    alignment_shift = shift_to_mean(failure, float(component["anchor"].astype(float).mean()))
    with tempfile.TemporaryDirectory(prefix="failure_complement_build_") as temporary:
        root = Path(temporary)
        source_dir, candidate_dir = root / "source", root / "candidate"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        shutil.copytree(source_dir, candidate_dir)
        meta_path = candidate_dir / "model" / "meta.json"
        r_meta_path = candidate_dir / "model" / "r_residual_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        r_meta = json.loads(r_meta_path.read_text(encoding="utf-8"))
        if abs(float(meta.get("logit_shift", 0.0)) - EXPECTED_SHIFT) > 1e-10:
            raise ValueError(f"원본 ZIP shift 불일치: {meta.get('logit_shift')}")
        if abs(float(r_meta.get("scale", 0.0)) - EXPECTED_R_SCALE) > 1e-12:
            raise ValueError(f"원본 ZIP R scale 불일치: {r_meta.get('scale')}")
        shift_info = meta.get("global_shift_candidate", {})
        source_shift = float(shift_info.get("source_shift", -0.03842671927234861))
        auxiliary_seeds = list(meta["offset"]["seeds"])
        blend_meta = {
            "blend": BLEND, "region": "R", "alignment_shift": alignment_shift,
            "source_shift": source_shift, "verified_shift": EXPECTED_SHIFT,
            "auxiliary_seeds": auxiliary_seeds,
            "source": "official Train 2024 OOF components",
            "test_aggregate_used": False,
        }
        (candidate_dir / "model" / "failure_complement_blend.json").write_text(
            json.dumps(blend_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        script_path = candidate_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    # 공식 Train의 2024 시간 안전 OOF 잔차로 학습한 R 행 전용 약한 보정.\n"
        if marker not in script:
            raise RuntimeError("R residual 앞 삽입 위치를 찾지 못했습니다.")
        script_path.write_text(script.replace(marker, inference_block() + "\n" + marker), encoding="utf-8")
        source_submission, source_stdout = run_package(source_dir, test_path, sample_path, "source")
        candidate_submission, candidate_stdout = run_package(candidate_dir, test_path, sample_path, "candidate")
        test = pd.read_csv(test_path, encoding="utf-8-sig")[[ID, "game_type"]]
        comparison = test.merge(source_submission, on=ID).merge(
            candidate_submission, on=ID, suffixes=("_source", "_candidate")
        )
        delta = comparison[f"{TARGET}_candidate"] - comparison[f"{TARGET}_source"]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        if candidate_submission[TARGET].isna().any() or not candidate_submission[TARGET].between(0, 1).all():
            raise ValueError("후보 추론 결과 결측 또는 확률 범위 오류")
        if len(f_delta) and float(f_delta.abs().max()) > 1e-12:
            raise ValueError(f"F행이 변경됨: {float(f_delta.abs().max())}")
        if not len(r_delta) or float(r_delta.abs().max()) == 0.0:
            raise ValueError("R행이 변경되지 않았습니다.")
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
        "experiment": "R failure-complement 0.20 blend over verified-shift champion",
        "official_train_only": True, "test_aggregate_used": False,
        "source_zip": str(source_zip), "source_sha256": digest(source_zip),
        "output_zip": str(output_zip), "output_sha256": digest(output_zip),
        "blend_meta": blend_meta, "members": members, "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_submission)),
        "sample_missing": int(candidate_submission[TARGET].isna().sum()),
        "sample_min": float(candidate_submission[TARGET].min()),
        "sample_max": float(candidate_submission[TARGET].max()),
        "sample_mean": float(candidate_submission[TARGET].mean()),
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "r_mean_delta": float(r_delta.mean()), "r_max_absolute_delta": float(r_delta.abs().max()),
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
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.component_dir.resolve(), args.test.resolve(),
         args.sample.resolve(), args.output_zip.resolve(), args.report.resolve())
