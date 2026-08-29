"""검증된 R 실패여집합 제출 ZIP의 혼합 강도만 안전하게 재패키징한다."""
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

import pandas as pd


ID, TARGET = "row_id", "control_success"
EXPECTED_SOURCE_BLEND = 0.20
EXPECTED_R_SCALE = 0.075
EXPECTED_SHIFT = -0.0416386466


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_package(package: Path, test_path: Path, sample_path: Path, tag: str):
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
        raise RuntimeError(
            f"{tag} 추론 실패\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    submission = pd.read_csv(run_dir / "output" / "submission.csv")
    return submission, completed.stdout.strip()


def main(source_zip: Path, test_path: Path, sample_path: Path, blend: float,
         output_zip: Path, report_path: Path) -> None:
    if not 0.20 < blend <= 0.40:
        raise ValueError("확장 검증 범위인 0.20 < blend <= 0.40만 허용합니다.")
    with tempfile.TemporaryDirectory(prefix="failure_complement_repack_") as temporary:
        root = Path(temporary)
        source_dir, candidate_dir = root / "source", root / "candidate"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        shutil.copytree(source_dir, candidate_dir)

        fc_path = candidate_dir / "model" / "failure_complement_blend.json"
        meta_path = candidate_dir / "model" / "meta.json"
        r_meta_path = candidate_dir / "model" / "r_residual_meta.json"
        if not fc_path.exists():
            raise FileNotFoundError("원본 ZIP에 failure-complement 메타가 없습니다.")
        fc_meta = json.loads(fc_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        r_meta = json.loads(r_meta_path.read_text(encoding="utf-8"))
        if abs(float(fc_meta.get("blend", -1)) - EXPECTED_SOURCE_BLEND) > 1e-12:
            raise ValueError(f"원본 blend 불일치: {fc_meta.get('blend')}")
        if fc_meta.get("region") != "R":
            raise ValueError(f"원본 적용 영역 불일치: {fc_meta.get('region')}")
        if abs(float(r_meta.get("scale", 0)) - EXPECTED_R_SCALE) > 1e-12:
            raise ValueError(f"원본 R scale 불일치: {r_meta.get('scale')}")
        if abs(float(meta.get("logit_shift", 0)) - EXPECTED_SHIFT) > 1e-10:
            raise ValueError(f"원본 shift 불일치: {meta.get('logit_shift')}")

        fc_meta["blend"] = blend
        fc_meta["parent_blend"] = EXPECTED_SOURCE_BLEND
        fc_path.write_text(
            json.dumps(fc_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        source_submission, source_stdout = run_package(
            source_dir, test_path, sample_path, "source"
        )
        candidate_submission, candidate_stdout = run_package(
            candidate_dir, test_path, sample_path, "candidate"
        )
        test = pd.read_csv(test_path, encoding="utf-8-sig")[[ID, "game_type"]]
        comparison = test.merge(source_submission, on=ID).merge(
            candidate_submission, on=ID, suffixes=("_source", "_candidate")
        )
        delta = comparison[f"{TARGET}_candidate"] - comparison[f"{TARGET}_source"]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        if candidate_submission[TARGET].isna().any():
            raise ValueError("후보 결과에 결측이 있습니다.")
        if not candidate_submission[TARGET].between(0, 1).all():
            raise ValueError("후보 결과가 확률 범위를 벗어났습니다.")
        if len(f_delta) and float(f_delta.abs().max()) > 1e-12:
            raise ValueError(f"F행이 변경됐습니다: {float(f_delta.abs().max())}")
        if not len(r_delta) or float(r_delta.abs().max()) == 0:
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
        "experiment": f"R failure-complement strength {blend:.3f}",
        "models_retrained": False,
        "official_train_only": True,
        "test_aggregate_used": False,
        "source_zip": str(source_zip),
        "source_sha256": digest(source_zip),
        "source_blend": EXPECTED_SOURCE_BLEND,
        "candidate_blend": blend,
        "output_zip": str(output_zip),
        "output_sha256": digest(output_zip),
        "members": members,
        "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_submission)),
        "sample_missing": int(candidate_submission[TARGET].isna().sum()),
        "sample_min": float(candidate_submission[TARGET].min()),
        "sample_max": float(candidate_submission[TARGET].max()),
        "sample_mean": float(candidate_submission[TARGET].mean()),
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "r_mean_delta": float(r_delta.mean()),
        "r_max_absolute_delta": float(r_delta.abs().max()),
        "source_stdout": source_stdout,
        "candidate_stdout": candidate_stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.25)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.test.resolve(), args.sample.resolve(),
         args.blend, args.output_zip.resolve(), args.report.resolve())
