"""Build an F-row post-regime profile correction package.

The frozen profile is learned from official Train OOF rows only.  Inference is
row-local: each test row uses only its own pitcher_team_id and count key plus
the precomputed Train profile table.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEGMENT_AUDIT = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
ID = "row_id"
TARGET = "control_success"
TRAIN_YEAR = 2024
VALID_AXES = {"pitcher_team_count"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_segment_module():
    spec = importlib.util.spec_from_file_location("segment_audit", SEGMENT_AUDIT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SEGMENT_AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def count_key(frame: pd.DataFrame) -> pd.Series:
    balls = first_existing(frame, ("balls_before", "balls", "ball_count"))
    strikes = first_existing(frame, ("strikes_before", "strikes", "strike_count"))
    if balls and strikes:
        return (
            frame[balls].fillna(-1).astype(int).astype(str)
            + "-"
            + frame[strikes].fillna(-1).astype(int).astype(str)
        )
    if "count" in frame.columns:
        return frame["count"].fillna("missing").astype(str)
    return pd.Series(["missing"] * len(frame), index=frame.index)


def add_axis(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    if axis not in VALID_AXES:
        raise ValueError(f"지원하지 않는 axis입니다: {axis}")
    result = frame.copy()
    result["pitcher_team_count"] = (
        result["pitcher_team_id"].fillna("missing").astype(str)
        + "|"
        + count_key(result).astype(str)
    )
    return result


def load_oof_frame(train_path: Path, own_oof_dir: Path, axis: str) -> pd.DataFrame:
    segment = load_segment_module()
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    oof = segment.load_oof(own_oof_dir, TRAIN_YEAR)
    rows = raw.loc[raw["season"].astype(int).eq(TRAIN_YEAR)].copy().reset_index(drop=True)
    if len(rows) != len(oof):
        raise ValueError(f"{TRAIN_YEAR} row count mismatch: train={len(rows)} oof={len(oof)}")
    if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
        raise ValueError(f"{TRAIN_YEAR} OOF row_id alignment failed")
    frame = pd.concat(
        [rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type", "pitcher_id"])],
        axis=1,
    )
    frame = frame.loc[frame["game_type"].astype(str).eq("F")].copy()
    frame = add_axis(frame, axis)
    return frame


def build_profile(frame: pd.DataFrame, axis: str, min_rows: int) -> pd.DataFrame:
    residual = frame["target"].to_numpy(float) - frame["p_champion"].to_numpy(float)
    profile = frame.assign(residual=residual).groupby(axis, observed=True).agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("p_champion", "mean"),
    )
    profile = profile.loc[profile["rows"].ge(min_rows)].copy()
    profile.index = profile.index.astype(str)
    return profile.sort_values(["rows", "residual_mean"], ascending=[False, False])


def extract_package(source_zip: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(source_zip):
        raise ValueError(f"유효한 ZIP이 아닙니다: {source_zip}")
    with zipfile.ZipFile(source_zip) as archive:
        error = archive.testzip()
        if error is not None:
            raise ValueError(f"원본 ZIP 무결성 오류: {error}")
        archive.extractall(destination)
    if not (destination / "script.py").exists():
        scripts = list(destination.rglob("script.py"))
        if len(scripts) != 1:
            raise ValueError(f"root script.py 없음, 전체 script.py 개수={len(scripts)}")
        package = scripts[0].parent
        temporary = destination.parent / f"{destination.name}_flat"
        shutil.move(str(package), temporary)
        shutil.rmtree(destination)
        shutil.move(str(temporary), destination)
    if not (destination / "script.py").exists():
        raise FileNotFoundError(destination / "script.py")
    if not (destination / "requirements.txt").exists():
        raise FileNotFoundError(destination / "requirements.txt")


def injection_block(probability_name: str) -> str:
    return f'''    # F post-regime frozen profile correction: official Train asset, row-local keys only
    _fprof_path = ROOT / "model" / "f_postregime_profile.json"
    if _fprof_path.exists():
        import json as _fprof_json
        with _fprof_path.open("r", encoding="utf-8") as _fprof_stream:
            _fprof_meta = _fprof_json.load(_fprof_stream)
        _fprof_mask = test["game_type"].astype(str).eq("F").to_numpy()
        if _fprof_mask.any():
            _fprof_balls_col = next((c for c in ("balls_before", "balls", "ball_count") if c in test.columns), None)
            _fprof_strikes_col = next((c for c in ("strikes_before", "strikes", "strike_count") if c in test.columns), None)
            if _fprof_balls_col is not None and _fprof_strikes_col is not None:
                _fprof_count = (
                    test[_fprof_balls_col].fillna(-1).astype(int).astype(str)
                    + "-"
                    + test[_fprof_strikes_col].fillna(-1).astype(int).astype(str)
                )
            elif "count" in test.columns:
                _fprof_count = test["count"].fillna("missing").astype(str)
            else:
                _fprof_count = pd.Series(["missing"] * len(test), index=test.index)
            _fprof_key = test["pitcher_team_id"].fillna("missing").astype(str) + "|" + _fprof_count.astype(str)
            _fprof_table = _fprof_meta["profiles"]
            _fprof_rows = _fprof_key.map(lambda x: _fprof_table.get(str(x), {{}}).get("rows", 0)).to_numpy(float)
            _fprof_resid = _fprof_key.map(lambda x: _fprof_table.get(str(x), {{}}).get("residual_mean", 0.0)).to_numpy(float)
            _fprof_delta = (
                _fprof_meta["scale"]
                * (_fprof_rows / (_fprof_rows + _fprof_meta["shrinkage"]))
                * _fprof_resid
            )
            _fprof_delta = np.clip(_fprof_delta, -_fprof_meta["cap"], _fprof_meta["cap"])
            _fprof_probability = {probability_name}
            _fprof_probability[_fprof_mask] = np.clip(
                _fprof_probability[_fprof_mask] + _fprof_delta[_fprof_mask],
                1e-6,
                1 - 1e-6,
            )
            {probability_name} = _fprof_probability
'''


def patch_script(script_path: Path) -> str:
    script = script_path.read_text(encoding="utf-8")
    normal_marker = "    pred_map = dict(zip(test[ID], p))\n"
    wrapper_marker = "    destination = ROOT / \"output\"\n"
    if normal_marker in script:
        script_path.write_text(script.replace(normal_marker, injection_block("p") + normal_marker), encoding="utf-8")
        return "probability_array_p"
    if wrapper_marker in script:
        script_path.write_text(script.replace(wrapper_marker, injection_block("result") + wrapper_marker), encoding="utf-8")
        return "wrapper_result"
    raise RuntimeError("최종 예측 삽입 위치를 찾지 못했습니다")


def run_package(package: Path, test: pd.DataFrame, sample: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, str]:
    run_dir = package.parent / f"verify_{tag}"
    shutil.copytree(package, run_dir)
    data = run_dir / "data"
    data.mkdir(exist_ok=True)
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "script.py"],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    if completed.returncode:
        raise RuntimeError(f"{tag} 추론 실패\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    output = pd.read_csv(run_dir / "output" / "submission.csv")
    return output, completed.stdout.strip()


def zip_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if not path.is_file() or "data" in relative.parts or "output" in relative.parts:
                continue
            archive.write(path, relative.as_posix())


def prediction_column(frame: pd.DataFrame) -> str:
    columns = [column for column in frame.columns if column != ID]
    if len(columns) != 1:
        raise ValueError(f"prediction columns={columns}")
    return columns[0]


def verify_independence(package: Path, test: pd.DataFrame, sample: pd.DataFrame,
                        full_prediction: pd.DataFrame) -> float:
    column = prediction_column(full_prediction)
    indices = list(dict.fromkeys([0, 1, 2, len(test) // 2, len(test) - 2, len(test) - 1]))
    indices = [index for index in indices if 0 <= index < len(test)]
    differences = []
    for index in indices[:8]:
        one_test = test.iloc[[index]].copy()
        one_sample = sample.loc[sample[ID].astype(str).eq(str(one_test[ID].iloc[0]))].copy()
        if one_sample.empty:
            one_sample = sample.iloc[[0]].copy()
            one_sample[ID] = one_test[ID].iloc[0]
        one_prediction, _ = run_package(package, one_test, one_sample, f"single_{index}")
        full_value = full_prediction.loc[
            full_prediction[ID].astype(str).eq(str(one_test[ID].iloc[0])), column
        ].iloc[0]
        differences.append(abs(float(one_prediction[column].iloc[0]) - float(full_value)))
    return max(differences, default=0.0)


def main(source_zip: Path, train_path: Path, own_oof_dir: Path, test_path: Path,
         sample_path: Path, output_zip: Path, report_path: Path, axis: str,
         shrinkage: float, scale: float, cap: float, min_rows: int) -> None:
    if shrinkage <= 0 or scale <= 0 or cap <= 0:
        raise ValueError("shrinkage, scale, cap은 양수여야 합니다")
    train_frame = load_oof_frame(train_path, own_oof_dir, axis)
    profile = build_profile(train_frame, axis, min_rows)
    if profile.empty:
        raise ValueError("profile table is empty")

    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)

    with tempfile.TemporaryDirectory(prefix="f_postregime_profile_") as temporary:
        root = Path(temporary)
        source = root / "source"
        candidate = root / "candidate"
        extract_package(source_zip, source)
        shutil.copytree(source, candidate)

        insertion_mode = patch_script(candidate / "script.py")
        model_dir = candidate / "model"
        model_dir.mkdir(exist_ok=True)
        profile_payload = {
            "axis": axis,
            "training_year": TRAIN_YEAR,
            "training_f_rows": int(len(train_frame)),
            "profile_count": int(len(profile)),
            "min_rows": int(min_rows),
            "shrinkage": float(shrinkage),
            "scale": float(scale),
            "cap": float(cap),
            "official_train_only": True,
            "test_aggregate_used": False,
            "profiles": {
                str(key): {
                    "rows": int(row["rows"]),
                    "residual_mean": float(row["residual_mean"]),
                    "target_mean": float(row["target_mean"]),
                    "prediction_mean": float(row["prediction_mean"]),
                }
                for key, row in profile.iterrows()
            },
        }
        (model_dir / "f_postregime_profile.json").write_text(
            json.dumps(profile_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        source_prediction, source_stdout = run_package(source, test, sample, "source")
        candidate_prediction, candidate_stdout = run_package(candidate, test, sample, "candidate")
        source_column = prediction_column(source_prediction)
        candidate_column = prediction_column(candidate_prediction)
        comparison = test[[ID, "game_type"]].merge(
            source_prediction.rename(columns={source_column: "source"}), on=ID
        ).merge(
            candidate_prediction.rename(columns={candidate_column: "candidate"}), on=ID
        )
        delta = comparison["candidate"] - comparison["source"]
        r_delta = delta[comparison["game_type"].astype(str).eq("R")]
        f_delta = delta[comparison["game_type"].astype(str).eq("F")]
        if len(r_delta) and float(r_delta.abs().max()) > 1e-12:
            raise ValueError(f"R행이 변경됐습니다: {float(r_delta.abs().max())}")
        if candidate_prediction[candidate_column].isna().any():
            raise ValueError("후보 예측에 결측이 있습니다")
        if not candidate_prediction[candidate_column].between(0, 1).all():
            raise ValueError("후보 예측이 확률 범위를 벗어났습니다")

        smoke_test = test.iloc[[0]].copy()
        smoke_test["game_type"] = "F"
        smoke_test["pitcher_team_id"] = str(next(iter(profile.index))).split("|", 1)[0]
        smoke_sample = sample.loc[sample[ID].astype(str).eq(str(smoke_test[ID].iloc[0]))].copy()
        if smoke_sample.empty:
            smoke_sample = sample.iloc[[0]].copy()
            smoke_sample[ID] = smoke_test[ID].iloc[0]
        source_smoke, _ = run_package(source, smoke_test, smoke_sample, "source_f_smoke")
        candidate_smoke, _ = run_package(candidate, smoke_test, smoke_sample, "candidate_f_smoke")
        f_smoke_delta = float(
            candidate_smoke[prediction_column(candidate_smoke)].iloc[0]
            - source_smoke[prediction_column(source_smoke)].iloc[0]
        )
        maximum_singleton_difference = verify_independence(candidate, test, sample, candidate_prediction)
        if maximum_singleton_difference > 1e-12:
            raise ValueError(f"행 독립성 검사 실패: {maximum_singleton_difference}")

        shutil.rmtree(candidate / "data", ignore_errors=True)
        shutil.rmtree(candidate / "output", ignore_errors=True)
        zip_directory(candidate, output_zip)

    with zipfile.ZipFile(output_zip) as archive:
        zip_error = archive.testzip()
        members = archive.namelist()
    if zip_error is not None:
        raise RuntimeError(f"후보 ZIP 손상: {zip_error}")

    report = {
        "experiment": "F post-regime profile submission",
        "official_train_only": True,
        "test_aggregate_used": False,
        "source_zip": str(source_zip),
        "source_sha256": sha256(source_zip),
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "axis": axis,
        "training_year": TRAIN_YEAR,
        "training_f_rows": int(len(train_frame)),
        "profile_count": int(len(profile)),
        "min_rows": int(min_rows),
        "shrinkage": float(shrinkage),
        "scale": float(scale),
        "cap": float(cap),
        "insertion_mode": insertion_mode,
        "members": len(members),
        "script_count": sum(Path(name).name == "script.py" for name in members),
        "root_script": "script.py" in members,
        "nested_zip_count": sum(name.lower().endswith(".zip") for name in members),
        "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_prediction)),
        "sample_missing": int(candidate_prediction[candidate_column].isna().sum()),
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_mean_delta": float(f_delta.mean()) if len(f_delta) else None,
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
        "f_smoke_delta": f_smoke_delta,
        "maximum_singleton_difference": maximum_singleton_difference,
        "source_stdout": source_stdout,
        "candidate_stdout": candidate_stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--own-oof-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--axis", choices=sorted(VALID_AXES), default="pitcher_team_count")
    parser.add_argument("--shrinkage", type=float, default=10000.0)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--cap", type=float, default=0.01)
    parser.add_argument("--min-rows", type=int, default=300)
    args = parser.parse_args()
    main(
        args.source_zip.resolve(),
        args.train.resolve(),
        args.own_oof_dir.resolve(),
        args.test.resolve(),
        args.sample.resolve(),
        args.output_zip.resolve(),
        args.report.resolve(),
        args.axis,
        args.shrinkage,
        args.scale,
        args.cap,
        args.min_rows,
    )
