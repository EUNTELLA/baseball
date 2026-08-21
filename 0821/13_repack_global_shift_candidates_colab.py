"""R 0.075 최고 ZIP의 전역 로짓 shift 후보를 재학습 없이 생성한다."""
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


CANDIDATES = (
    ("verified", -0.0416386466),
    ("response_optimum", -0.04390362815914772),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def find_meta(root: Path) -> Path:
    matches = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "logit_shift" in payload:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"logit_shift 메타 파일 수가 1이 아님: {matches}")
    return matches[0]


def verify(candidate_dir: Path, test_path: Path, sample_path: Path) -> dict:
    verify_dir = candidate_dir.parent / f"verify_{candidate_dir.name}"
    shutil.copytree(candidate_dir, verify_dir)
    data_dir = verify_dir / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(test_path, data_dir / "test.csv")
    shutil.copy2(sample_path, data_dir / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=verify_dir,
        capture_output=True, text=True, check=False, timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "샘플 추론 실패\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    submission = pd.read_csv(verify_dir / "output" / "submission.csv")
    prediction = pd.to_numeric(submission["control_success"], errors="coerce")
    return {
        "rows": int(len(submission)), "missing": int(prediction.isna().sum()),
        "minimum": float(prediction.min()), "maximum": float(prediction.max()),
        "mean": float(prediction.mean()), "stdout": completed.stdout.strip(),
    }


def main(source_zip: Path, test_path: Path, sample_path: Path,
         output_dir: Path, report_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shift_candidates_") as temporary:
        root = Path(temporary)
        source_dir = root / "source"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        source_meta_path = find_meta(source_dir)
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        source_shift = float(source_meta["logit_shift"])
        results = []
        for name, shift in CANDIDATES:
            candidate_dir = root / name
            shutil.copytree(source_dir, candidate_dir)
            meta_path = candidate_dir / source_meta_path.relative_to(source_dir)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["logit_shift"] = shift
            meta["global_shift_candidate"] = {
                "source_shift": source_shift,
                "candidate_shift": shift,
                "delta": shift - source_shift,
                "selection": "own submission response curve",
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output_zip = output_dir / f"submit_catboost_r0075_shift_{name}.zip"
            with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(candidate_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(candidate_dir))
            with zipfile.ZipFile(output_zip) as archive:
                zip_error = archive.testzip()
                members = len(archive.namelist())
            verification = verify(candidate_dir, test_path, sample_path)
            if zip_error is not None or verification["missing"] or not (
                0 <= verification["minimum"] <= verification["maximum"] <= 1
            ):
                raise RuntimeError(f"후보 검증 실패: {name}")
            results.append({
                "name": name, "shift": shift, "delta_vs_source": shift - source_shift,
                "output_zip": str(output_zip), "sha256": digest(output_zip),
                "members": members, "zip_test_error": zip_error,
                "sample_verification": verification,
            })
    report = {
        "experiment": "R 0.075 global logit shift candidates",
        "models_retrained": False, "official_train_only": True,
        "test_aggregate_used": False, "source_zip": str(source_zip),
        "source_shift": source_shift, "candidates": results,
        "recommended_first": "verified",
        "reason": "기존 자체 제출에서 직접 확인된 shift를 먼저 사용하고 반응곡선 정점은 보관한다.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.test.resolve(), args.sample.resolve(),
         args.output_dir.resolve(), args.report.resolve())
