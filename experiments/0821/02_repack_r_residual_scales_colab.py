"""검증된 R 잔차 모델의 강도만 바꾼 제출 ZIP 두 개를 만든다."""
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


TARGET_COL = "control_success"
SCALES = (0.05, 0.075)


def scale_tag(scale: float) -> str:
    return f"{int(round(scale * 1000)):04d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(source_zip: Path, test_path: Path, sample_path: Path,
         output_dir: Path, report_path: Path) -> None:
    if not source_zip.exists():
        raise FileNotFoundError(source_zip)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    with tempfile.TemporaryDirectory(prefix="r_scale_") as temporary:
        root = Path(temporary)
        source_dir = root / "source"
        with zipfile.ZipFile(source_zip) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("원본 ZIP이 손상됐습니다.")
            archive.extractall(source_dir)

        meta_path = source_dir / "model" / "r_residual_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        required_models = [source_dir / "model" / f"r_residual_{seed}.cbm"
                           for seed in meta["seeds"]]
        if any(not path.exists() for path in required_models):
            raise FileNotFoundError("원본 ZIP에서 R 잔차 모델을 찾지 못했습니다.")

        for scale in SCALES:
            candidate_dir = root / f"candidate_{scale_tag(scale)}"
            shutil.copytree(source_dir, candidate_dir)
            candidate_meta_path = candidate_dir / "model" / "r_residual_meta.json"
            candidate_meta = dict(meta)
            candidate_meta["scale"] = scale
            candidate_meta_path.write_text(
                json.dumps(candidate_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_zip = output_dir / f"submit_catboost_r_residual_scale{scale_tag(scale)}.zip"
            if output_zip.exists():
                output_zip.unlink()
            with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(candidate_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(candidate_dir))
            with zipfile.ZipFile(output_zip) as archive:
                bad_member = archive.testzip()
                members = archive.namelist()
            if bad_member is not None:
                raise RuntimeError(f"ZIP 손상: {bad_member}")

            verify_dir = root / f"verify_{scale_tag(scale)}"
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
                raise ValueError(f"scale={scale} 샘플 추론 결과 오류")
            results.append({
                "scale": scale,
                "output_zip": str(output_zip),
                "sha256": sha256(output_zip),
                "members": len(members),
                "zip_test_error": bad_member,
                "sample_rows": int(len(submission)),
                "sample_missing": int(submission[TARGET_COL].isna().sum()),
                "sample_min": float(submission[TARGET_COL].min()),
                "sample_max": float(submission[TARGET_COL].max()),
                "sample_mean": float(submission[TARGET_COL].mean()),
                "sample_stdout": completed.stdout.strip(),
            })

    payload = {
        "experiment": "R-only residual scale candidates",
        "source_zip": str(source_zip),
        "models_retrained": False,
        "official_train_only": True,
        "test_aggregate_used": False,
        "candidates": results,
        "recommended_submission": "scale0050",
        "reason": "scale 0.025 transferred positively, while 0.05 is safer than 0.075 under the negative 2023 forward result",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
