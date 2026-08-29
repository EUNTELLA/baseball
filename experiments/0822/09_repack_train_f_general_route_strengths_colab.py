"""검증된 Futures general route의 강도 후보를 재학습 없이 만든다."""
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
DEFAULT_STRENGTHS = (0.75, 1.25)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_package(package: Path, test: pd.DataFrame, sample: pd.DataFrame, tag: str):
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=run_dir, capture_output=True,
        text=True, check=False, timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\n{completed.stdout}\n{completed.stderr}")
    return pd.read_csv(run_dir / "output" / "submission.csv"), completed.stdout.strip()


def strength_name(value: float) -> str:
    return f"{round(value * 100):03d}"


def main(source_zip: Path, test_path: Path, sample_path: Path,
         output_dir: Path, report_path: Path, strengths: tuple[float, ...]):
    source_sha = digest(source_zip)
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []

    with tempfile.TemporaryDirectory(prefix="general_strength_") as temporary:
        root = Path(temporary)
        source_dir = root / "source"
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_dir)
        meta_path = source_dir / "model" / "general_route_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if source_meta.get("active_region") != "F":
            raise ValueError("Futures 전용 general route ZIP이 아닙니다.")

        script_path = source_dir / "script.py"
        script = script_path.read_text(encoding="utf-8")
        old = '''        p[general_f_mask] = np.clip(
            general_prediction[general_f_mask], 1e-6, 1 - 1e-6
        )'''
        new = '''        general_base = p[general_f_mask].copy()
        general_strength = float(general_meta.get("route_strength", 1.0))
        p[general_f_mask] = np.clip(
            general_base + general_strength *
            (general_prediction[general_f_mask] - general_base),
            1e-6, 1 - 1e-6
        )'''
        if old not in script:
            raise RuntimeError("general route 강도 삽입 위치를 찾지 못했습니다.")

        for strength in strengths:
            if strength <= 0:
                raise ValueError(f"강도는 양수여야 합니다: {strength}")
            candidate_dir = root / f"candidate_{strength_name(strength)}"
            shutil.copytree(source_dir, candidate_dir)
            candidate_script = script.replace(old, new)
            (candidate_dir / "script.py").write_text(candidate_script, encoding="utf-8")
            candidate_meta = dict(source_meta)
            candidate_meta["route_strength"] = strength
            (candidate_dir / "model" / "general_route_meta.json").write_text(
                json.dumps(candidate_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            prediction, stdout = run_package(candidate_dir, test, sample, strength_name(strength))
            if prediction[TARGET].isna().any() or not prediction[TARGET].between(0, 1).all():
                raise ValueError(f"강도 {strength}: 예측 결측 또는 범위 오류")
            output_zip = output_dir / f"submit_catboost_rchampion_fgeneral6_s{strength_name(strength)}.zip"
            with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(candidate_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(candidate_dir))
            with zipfile.ZipFile(output_zip) as archive:
                zip_error = archive.testzip()
                members = len(archive.namelist())
            if zip_error is not None:
                raise RuntimeError(f"강도 {strength} ZIP 손상: {zip_error}")
            candidates.append({
                "strength": strength, "output_zip": str(output_zip),
                "sha256": digest(output_zip), "members": members,
                "zip_test_error": zip_error, "sample_rows": int(len(prediction)),
                "sample_missing": int(prediction[TARGET].isna().sum()),
                "sample_min": float(prediction[TARGET].min()),
                "sample_max": float(prediction[TARGET].max()),
                "sample_mean": float(prediction[TARGET].mean()), "stdout": stdout,
            })

    report = {
        "experiment": "Futures general route strength candidates",
        "models_retrained": False, "official_train_only": True,
        "test_aggregate_used": False, "source_zip": str(source_zip),
        "source_sha256": source_sha, "source_strength": 1.0,
        "source_public_score": 1050.1729660849, "candidates": candidates,
        "recommended_first": 0.75,
        "reason": "1.0의 서버 개선은 확인됐지만 F 경로의 시즌 변동성을 고려해 완화 후보를 먼저 비교한다.",
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
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.test.resolve(), args.sample.resolve(),
         args.output_dir.resolve(), args.report.resolve(), tuple(args.strengths))
