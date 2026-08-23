"""두 독립 추론 패키지를 실행해 R행만 보수적으로 혼합한 제출 ZIP을 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


WRAPPER = r'''from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BLEND = __BLEND__


def run_member(name):
    member = ROOT / name
    data = member / "data"
    output = member / "output"
    data.mkdir(exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    shutil.copy2(ROOT / "data" / "test.csv", data / "test.csv")
    sample = ROOT / "data" / "sample_submission.csv"
    if sample.exists():
        shutil.copy2(sample, data / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=member, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{name} inference failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    files = sorted(output.glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"{name} output CSV count={len(files)}")
    print(completed.stdout.strip(), flush=True)
    return pd.read_csv(files[0])


def prediction_column(frame):
    columns = [column for column in frame.columns if column != "row_id"]
    if len(columns) != 1:
        raise ValueError(f"prediction columns={columns}")
    return columns[0]


def main():
    test = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    current = run_member("current")
    alternate = run_member("alternate")
    if not current["row_id"].astype(str).equals(alternate["row_id"].astype(str)):
        raise ValueError("member row_id order mismatch")
    if not current["row_id"].astype(str).equals(test["row_id"].astype(str)):
        raise ValueError("test/output row_id order mismatch")
    current_column = prediction_column(current)
    alternate_column = prediction_column(alternate)
    base = current[current_column].to_numpy(float)
    other = alternate[alternate_column].to_numpy(float)
    result = base.copy()
    active = test["game_type"].astype(str).eq("R").to_numpy()
    result[active] = (1.0 - BLEND) * base[active] + BLEND * other[active]
    result = np.clip(result, 1e-6, 1 - 1e-6)
    destination = ROOT / "output"
    destination.mkdir(exist_ok=True)
    pd.DataFrame({"row_id": current["row_id"], current_column: result}).to_csv(
        destination / "submission.csv", index=False
    )
    print(
        f"Saved ./output/submission.csv rows={len(result)} "
        f"R={int(active.sum())} F={int((~active).sum())} blend={BLEND:.3f} mean={result.mean():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source, destination):
    if not zipfile.is_zipfile(source):
        raise ValueError(f"유효한 ZIP이 아닙니다: {source}")
    with zipfile.ZipFile(source) as archive:
        archive.extractall(destination)
    scripts = list(destination.rglob("script.py"))
    # 이미 두 추론 패키지를 감싼 제출물은 루트와 내부에 script.py가 함께 있다.
    # 루트 실행기가 있으면 그것을 완성 패키지의 진입점으로 우선한다.
    root_script = destination / "script.py"
    if root_script.exists():
        package = destination
    elif len(scripts) == 1:
        package = scripts[0].parent
    else:
        raise ValueError(f"{source}: 루트 script.py 없음, 전체 개수={len(scripts)}")
    if package != destination:
        temporary = destination.parent / f"{destination.name}_flat"
        shutil.move(str(package), temporary)
        shutil.rmtree(destination)
        shutil.move(str(temporary), destination)
    if not (destination / "requirements.txt").exists():
        raise FileNotFoundError(destination / "requirements.txt")


def requirements(*packages):
    lines = []
    names = set()
    for package in packages:
        for line in (package / "requirements.txt").read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            name = re.split(r"[<>=!~\[]", clean, maxsplit=1)[0].strip().lower().replace("_", "-")
            if clean and not clean.startswith("#") and name not in names:
                lines.append(clean)
                names.add(name)
    return "\n".join(lines) + "\n"


def zip_directory(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "output" not in path.relative_to(source).parts and "data" not in path.relative_to(source).parts:
                archive.write(path, path.relative_to(source).as_posix())


def verify(package, test_path, sample_path):
    data = package / "data"
    data.mkdir(exist_ok=True)
    shutil.copy2(test_path, data / "test.csv")
    shutil.copy2(sample_path, data / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=package, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(f"샘플 추론 실패\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    output = pd.read_csv(package / "output" / "submission.csv")
    return {
        "rows": int(len(output)), "missing": int(output.isna().sum().sum()),
        "stdout": completed.stdout.strip(),
    }


def main(current_zip, alternate_zip, test_path, sample_path, output_zip, report, blend):
    with tempfile.TemporaryDirectory(prefix="strict_r_blend_") as temporary:
        root = Path(temporary)
        package = root / "package"
        package.mkdir()
        current = package / "current"
        alternate = package / "alternate"
        current.mkdir()
        alternate.mkdir()
        extract(current_zip, current)
        extract(alternate_zip, alternate)
        (package / "script.py").write_text(
            WRAPPER.replace("__BLEND__", repr(blend)), encoding="utf-8"
        )
        (package / "requirements.txt").write_text(requirements(current, alternate), encoding="utf-8")
        verification = verify(package, test_path, sample_path)
        shutil.rmtree(package / "data", ignore_errors=True)
        shutil.rmtree(package / "output", ignore_errors=True)
        shutil.rmtree(current / "data", ignore_errors=True)
        shutil.rmtree(current / "output", ignore_errors=True)
        shutil.rmtree(alternate / "data", ignore_errors=True)
        shutil.rmtree(alternate / "output", ignore_errors=True)
        zip_directory(package, output_zip)
    with zipfile.ZipFile(output_zip) as archive:
        error = archive.testzip()
        members = archive.namelist()
    payload = {
        "experiment": "strict model-only R-only runtime blend",
        "official_train_only": True, "test_aggregate_used": False,
        "current_zip": str(current_zip), "current_sha256": sha256(current_zip),
        "alternate_zip": str(alternate_zip), "alternate_sha256": sha256(alternate_zip),
        "blend": blend, "active_region": "R", "f_rows_unchanged": True,
        "output_zip": str(output_zip), "output_sha256": sha256(output_zip),
        "members": len(members), "zip_test_error": error,
        "sample_verification": verification,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-zip", type=Path, required=True)
    parser.add_argument("--alternate-zip", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.blend <= 1:
        parser.error("--blend는 0보다 크고 1 이하여야 합니다")
    main(args.current_zip.resolve(), args.alternate_zip.resolve(), args.test.resolve(),
         args.sample.resolve(), args.output_zip.resolve(), args.report.resolve(), args.blend)
