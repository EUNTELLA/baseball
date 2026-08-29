"""Build an R-only segment residual correction package.

Segments are frozen from the 2024 own champion OOF file.  Inference only uses
the current test row fields and the frozen segment table.
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
AUDIT_PATH = ROOT / "train_r" / "03_r_segment_error_audit_colab.py"
ID = "row_id"
TARGET = "control_success"
TRAIN_YEAR = 2024


def load_audit_module():
    spec = importlib.util.spec_from_file_location("r_segment_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_flat(source: Path, destination: Path):
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("source zip integrity failed")
        archive.extractall(destination)
    scripts = list(destination.rglob("script.py"))
    nested_zips = list(destination.rglob("*.zip"))
    if scripts != [destination / "script.py"]:
        raise ValueError(f"source must contain one root script.py: {scripts}")
    if nested_zips:
        raise ValueError(f"source contains nested zip files: {nested_zips}")


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
        raise RuntimeError(f"{tag} inference failed\n{completed.stdout}\n{completed.stderr}")
    result = pd.read_csv(run_dir / "output" / "submission.csv")
    return result, completed.stdout.strip()


def build_segments(train_path: Path, oof_dir: Path, axis: str, min_rows: int, audit):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    oof = audit.load_oof(oof_dir, TRAIN_YEAR)
    rows = raw.loc[raw["season"].astype(int).eq(TRAIN_YEAR)].copy().reset_index(drop=True)
    if not np.array_equal(rows[ID].astype(str).to_numpy(), oof[ID].astype(str).to_numpy()):
        raise ValueError(f"{TRAIN_YEAR} row_id alignment failed")
    frame = pd.concat(
        [rows.reset_index(drop=True), oof.drop(columns=[ID, "game_type", "pitcher_id"])],
        axis=1,
    )
    frame["season"] = TRAIN_YEAR
    frame = audit.add_axes(frame)
    frame = frame.loc[frame["game_type"].astype(str).eq("R")].copy()
    residual = frame["target"].to_numpy(float) - frame["p_champion"].to_numpy(float)
    grouped = frame.assign(residual=residual).groupby(axis, observed=True).agg(
        rows=("residual", "size"),
        residual_mean=("residual", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("p_champion", "mean"),
    )
    grouped = grouped.loc[grouped["rows"].ge(min_rows)].copy()
    if grouped.empty:
        raise ValueError(f"no {axis} segments with min_rows={min_rows}")
    return {
        str(index): {
            "rows": int(row["rows"]),
            "residual_mean": float(row["residual_mean"]),
            "target_mean": float(row["target_mean"]),
            "prediction_mean": float(row["prediction_mean"]),
        }
        for index, row in grouped.iterrows()
    }


def injection_block(axis: str):
    return f'''
    # Frozen R segment residual correction from official Train OOF.
    import json as _rseg_json
    import numpy as _rseg_np
    import pandas as _rseg_pd

    def _rseg_first_existing(_frame, _names):
        for _name in _names:
            if _name in _frame.columns:
                return _name
        return None

    def _rseg_axis_p_hand(_test, _probability):
        _bins = _rseg_np.array([0.0, 0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 1.0])
        _p_band = _rseg_np.digitize(
            _rseg_np.clip(_probability, 1e-6, 1 - 1e-6), _bins[1:-1], right=False
        ).astype(str)
        _pitcher_hand = _rseg_first_existing(_test, ("pitcher_hand", "pitcher_side", "p_throws"))
        _batter_hand = _rseg_first_existing(_test, ("batter_hand", "stand", "batter_side"))
        if _pitcher_hand and _batter_hand:
            _hand = _test[_pitcher_hand].astype(str) + "_" + _test[_batter_hand].astype(str)
        elif _batter_hand:
            _hand = _test[_batter_hand].astype(str)
        else:
            _hand = _rseg_pd.Series(["missing"] * len(_test), index=_test.index, dtype="string")
        return (
            _rseg_pd.Series(_p_band, index=_test.index, dtype="string")
            + "|"
            + _hand.astype("string")
        ).astype(str).to_numpy()

    with open("model/r_segment_correction.json", "r", encoding="utf-8") as _rseg_stream:
        _rseg_meta = _rseg_json.load(_rseg_stream)
    if _rseg_meta.get("axis") != {axis!r}:
        raise RuntimeError("R segment axis mismatch")
    _rseg_mask = test["game_type"].astype(str).eq("R").to_numpy()
    if _rseg_mask.any():
        _rseg_keys = _rseg_axis_p_hand(test, p)
        _rseg_delta = _rseg_np.zeros(len(test), dtype=float)
        _rseg_alpha = float(_rseg_meta["alpha"])
        _rseg_scale = float(_rseg_meta["scale"])
        _rseg_segments = _rseg_meta["segments"]
        for _rseg_index, _rseg_key in enumerate(_rseg_keys):
            if not _rseg_mask[_rseg_index]:
                continue
            _rseg_item = _rseg_segments.get(str(_rseg_key))
            if _rseg_item is None:
                continue
            _rseg_rows = float(_rseg_item["rows"])
            _rseg_shrink = _rseg_rows / (_rseg_rows + _rseg_alpha)
            _rseg_delta[_rseg_index] = _rseg_scale * _rseg_shrink * float(_rseg_item["residual_mean"])
        p[_rseg_mask] = _rseg_np.clip(p[_rseg_mask] + _rseg_delta[_rseg_mask], 1e-6, 1 - 1e-6)
'''


def main(source_zip: Path, train_path: Path, oof_dir: Path, test_path: Path, sample_path: Path,
         output_zip: Path, report_path: Path, axis: str, alpha: float, scale: float, min_rows: int):
    if axis != "axis_p_hand":
        raise ValueError("only axis_p_hand is supported for this builder")
    audit = load_audit_module()
    segments = build_segments(train_path, oof_dir, axis, min_rows, audit)
    source_sha = digest(source_zip)
    test = pd.read_csv(test_path, encoding="utf-8-sig", low_memory=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", low_memory=False)

    with tempfile.TemporaryDirectory(prefix="r_segment_build_") as temporary:
        root = Path(temporary)
        source = root / "source"
        candidate = root / "candidate"
        extract_flat(source_zip, source)
        shutil.copytree(source, candidate)

        script_path = candidate / "script.py"
        script = script_path.read_text(encoding="utf-8")
        marker = "    pred_map = dict(zip(test[ID], p))\n"
        if marker not in script:
            raise RuntimeError("prediction insertion marker not found")
        script_path.write_text(
            script.replace(marker, injection_block(axis) + marker),
            encoding="utf-8",
        )
        meta = {
            "axis": axis,
            "alpha": alpha,
            "scale": scale,
            "min_rows": min_rows,
            "train_year": TRAIN_YEAR,
            "active_region": "R",
            "source": "official Train OOF",
            "test_aggregate_used": False,
            "segments": segments,
        }
        model_dir = candidate / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "r_segment_correction.json").write_text(
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
        if len(f_delta) and float(f_delta.abs().max()) > 1e-12:
            raise ValueError(f"F rows changed: {float(f_delta.abs().max())}")
        if candidate_prediction[TARGET].isna().any() or not candidate_prediction[TARGET].between(0, 1).all():
            raise ValueError("candidate has missing or out-of-range probabilities")

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
            raise ValueError(f"row independence failed: {maximum_singleton_difference}")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(candidate.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(candidate))
        with zipfile.ZipFile(output_zip) as archive:
            zip_error = archive.testzip()
            members = archive.namelist()
        if zip_error is not None:
            raise RuntimeError(f"candidate zip integrity failed: {zip_error}")

    report = {
        "experiment": "R segment residual correction submission",
        "official_train_only": True,
        "test_aggregate_used": False,
        "source_zip": str(source_zip),
        "source_sha256": source_sha,
        "output_zip": str(output_zip),
        "output_sha256": digest(output_zip),
        "axis": axis,
        "alpha": alpha,
        "scale": scale,
        "min_rows": min_rows,
        "segment_count": len(segments),
        "members": len(members),
        "script_count": sum(name == "script.py" for name in members),
        "nested_zip_count": sum(name.lower().endswith(".zip") for name in members),
        "zip_test_error": zip_error,
        "sample_rows": int(len(candidate_prediction)),
        "sample_missing": int(candidate_prediction[TARGET].isna().sum()),
        "r_mean_delta": float(r_delta.mean()) if len(r_delta) else None,
        "r_max_absolute_delta": float(r_delta.abs().max()) if len(r_delta) else None,
        "f_max_absolute_delta": float(f_delta.abs().max()) if len(f_delta) else None,
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
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--axis", default="axis_p_hand")
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--min-rows", type=int, default=5000)
    args = parser.parse_args()
    main(args.source_zip.resolve(), args.train.resolve(), args.oof_dir.resolve(),
         args.test.resolve(), args.sample.resolve(), args.output_zip.resolve(),
         args.report.resolve(), args.axis, args.alpha, args.scale, args.min_rows)
