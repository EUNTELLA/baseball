"""단일 자체 패키지의 F행에 Train OOF로 확정한 고정 logit shift를 적용한다."""
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


ID, TARGET = "row_id", "control_success"


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_flat(source: Path, destination: Path):
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("원본 ZIP 무결성 오류")
        archive.extractall(destination)
    scripts = list(destination.rglob("script.py"))
    nested_zips = list(destination.rglob("*.zip"))
    if scripts != [destination / "script.py"]:
        raise ValueError(f"원본은 최상위 script.py 하나여야 합니다: {scripts}")
    if nested_zips:
        raise ValueError(f"중첩 ZIP이 있습니다: {nested_zips}")


def run_package(package: Path, test: pd.DataFrame, sample: pd.DataFrame, tag: str):
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
    result = pd.read_csv(run_dir / "output" / "submission.csv")
    return result, completed.stdout.strip()


def main(source_zip: Path, test_path: Path, sample_path: Path, output_zip: Path,
         report_path: Path, shift: float, dead_zone: float):
    if shift >= 0 or abs(shift) > 0.1:
        raise ValueError("이번 후보 shift는 -0.1보다 크고 0보다 작아야 합니다")
    source_sha = digest(source_zip)
    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)

    with tempfile.TemporaryDirectory(prefix="f_level_gate_") as temporary:
        root = Path(temporary)
        source = root / "source"
        candidate = root / "candidate"
        extract_flat(source_zip, source)
        shutil.copytree(source, candidate)

        marker = "    pred_map = dict(zip(test[ID], p))\n"
        script_path = candidate / "script.py"
        script = script_path.read_text(encoding="utf-8")
        if marker not in script:
            raise RuntimeError("최종 예측 삽입 위치를 찾지 못했습니다")
        block = f'''    # Train strict-forward OOF에서 확정한 F 전용 고정 level 보정
    f_level_shift = {shift!r}
    f_level_mask = test["game_type"].astype(str).eq("F").to_numpy()
    if f_level_mask.any():
        f_level_probability = np.clip(p[f_level_mask], 1e-6, 1 - 1e-6)
        f_level_logit = np.log(f_level_probability / (1 - f_level_probability))
        p[f_level_mask] = 1 / (1 + np.exp(-(f_level_logit + f_level_shift)))
'''
        script_path.write_text(script.replace(marker, block + marker), encoding="utf-8")
        meta = {
            "active_region": "F", "logit_shift": shift, "dead_zone": dead_zone,
            "source": "official Train strict-forward OOF", "test_aggregate_used": False,
        }
        (candidate / "model" / "f_prior_level_gate.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        source_prediction, source_stdout = run_package(source, test, sample, "source")
        candidate_prediction, candidate_stdout = run_package(candidate, test, sample, "candidate")
        comparison = test[[ID, "game_type"]].merge(source_prediction, on=ID).merge(
            candidate_prediction, on=ID, suffixes=("_source", "_candidate")
        )
        delta = comparison[f"{TARGET}_candidate"] - comparison[f"{TARGET}_source"]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        if len(r_delta) and float(r_delta.abs().max()) > 1e-12:
            raise ValueError(f"R행이 변경됐습니다: {float(r_delta.abs().max())}")

        smoke_test = test.iloc[[0]].copy()
        smoke_test["game_type"] = "F"
        smoke_sample = sample.loc[
            sample[ID].astype(str).eq(str(smoke_test[ID].iloc[0]))
        ].copy()
        source_smoke, _ = run_package(source, smoke_test, smoke_sample, "source_f_smoke")
        candidate_smoke, _ = run_package(candidate, smoke_test, smoke_sample, "candidate_f_smoke")
        f_smoke_delta = float(candidate_smoke[TARGET].iloc[0] - source_smoke[TARGET].iloc[0])
        if f_smoke_delta >= 0:
            raise ValueError(f"F smoke shift 방향 오류: {f_smoke_delta}")

        singleton_differences = []
        for index in range(min(8, len(test))):
            one_test = test.iloc[[index]].copy()
            one_sample = sample.loc[sample[ID].astype(str).eq(str(one_test[ID].iloc[0]))].copy()
            one_prediction, _ = run_package(candidate, one_test, one_sample, f"single_{index}")
            full_value = candidate_prediction.loc[
                candidate_prediction[ID].astype(str).eq(str(one_test[ID].iloc[0])), TARGET
            ].iloc[0]
            singleton_differences.append(abs(float(one_prediction[TARGET].iloc[0]) - float(full_value)))
        maximum_singleton_difference = max(singleton_differences, default=0.0)
        if maximum_singleton_difference > 1e-12:
            raise ValueError(f"행 독립성 검사 실패: {maximum_singleton_difference}")
        if candidate_prediction[TARGET].isna().any() or not candidate_prediction[TARGET].between(0, 1).all():
            raise ValueError("후보 예측 결측 또는 확률 범위 오류")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(candidate.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(candidate))
        with zipfile.ZipFile(output_zip) as archive:
            zip_error = archive.testzip()
            members = archive.namelist()
        if zip_error is not None:
            raise RuntimeError(f"후보 ZIP 손상: {zip_error}")

    report = {
        "experiment": "F prior-year level gate submission",
        "official_train_only": True, "test_aggregate_used": False,
        "source_zip": str(source_zip), "source_sha256": source_sha,
        "output_zip": str(output_zip), "output_sha256": digest(output_zip),
        "shift": shift, "dead_zone": dead_zone, "members": len(members),
        "script_count": sum(name == "script.py" for name in members),
        "nested_zip_count": sum(name.lower().endswith(".zip") for name in members),
        "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_prediction)),
        "sample_missing": int(candidate_prediction[TARGET].isna().sum()),
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_mean_delta": float(f_delta.mean()) if len(f_delta) else None,
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "f_smoke_delta": f_smoke_delta,
        "maximum_singleton_difference": maximum_singleton_difference,
        "source_stdout": source_stdout, "candidate_stdout": candidate_stdout,
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
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=-0.05)
    parser.add_argument("--dead-zone", type=float, default=0.02)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.test.resolve(), args.sample.resolve(),
         args.output_zip.resolve(), args.report.resolve(), args.shift, args.dead_zone)
