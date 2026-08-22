"""현재 챔피언 parity 예측 위에서 strict model-only R 혼합을 재검증한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_MODULE = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
COMPARE_MODULE = ROOT / "0823" / "01_strict_anchor_comparison_colab.py"
PAIRS = ((2022, 2023), (2023, 2024))
BLENDS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
FAILURE_BLEND = 0.20


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def align(strict, current, year):
    strict_ids = pd.Index(np.asarray(strict["row_id"]).astype(str))
    current_ids = pd.Index(np.asarray(current["row_id"]).astype(str))
    if strict_ids.has_duplicates or current_ids.has_duplicates:
        raise ValueError(f"{year} row_id 중복")
    positions = strict_ids.get_indexer(current_ids)
    if np.any(positions < 0) or len(strict_ids) != len(current_ids):
        raise ValueError(f"{year} row_id 구성 불일치")
    if not np.allclose(np.asarray(strict["target"])[positions], current["target"]):
        raise ValueError(f"{year} target 불일치")
    return np.asarray(strict["model_only"], float)[positions]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(strict_dir, component_dir, train_path, output, task_type):
    champion_module = load_module(CHAMPION_MODULE, "champion_validation")
    compare_module = load_module(COMPARE_MODULE, "strict_comparison")
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    feature_module = champion_module.load_features()
    league_rate = float(frame.loc[frame["season"].astype(int).lt(2022), "control_success"].mean())
    features = feature_module.engineer(frame.drop(columns=["row_id", "control_success"]), league_rate)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    assets = {year: champion_module.load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    strict_assets = {year: compare_module.load_alternate(strict_dir, year) for year in (2022, 2023, 2024)}
    folds = []
    for calibration_year, validation_year in PAIRS:
        calibration, valid = assets[calibration_year], assets[validation_year]
        correction, valid_rows, seconds = champion_module.train_correction(
            frame, features, cat_indices, calibration_year, validation_year,
            calibration, valid, task_type,
        )
        target = valid["target"].astype(float)
        anchor = valid["anchor"].astype(float)
        valid_r = valid_rows["game_type"].astype(str).eq("R").to_numpy()
        alignment_shift = champion_module.shift_to_mean(
            calibration["failure_complement"].astype(float),
            float(calibration["anchor"].astype(float).mean()),
        )
        aligned_failure = champion_module.sigmoid(
            champion_module.logit(valid["failure_complement"].astype(float)) + alignment_shift
        )
        champion = anchor.copy()
        champion[valid_r] = (
            (1 - FAILURE_BLEND) * champion[valid_r] + FAILURE_BLEND * aligned_failure[valid_r]
        )
        champion = champion_module.sigmoid(
            champion_module.logit(champion) + champion_module.VERIFIED_SHIFT_DELTA
        )
        champion[valid_r] = np.clip(
            champion[valid_r] + champion_module.R_SCALE * correction, 1e-6, 1 - 1e-6
        )
        alternate = align(strict_assets[validation_year], valid, validation_year)
        baseline_score = champion_module.bss(champion, target)
        candidates = []
        for blend in BLENDS:
            candidate = champion.copy()
            candidate[valid_r] = (
                (1 - blend) * champion[valid_r] + blend * alternate[valid_r]
            )
            candidates.append({
                "blend": blend,
                "bss_delta": champion_module.bss(candidate, target) - baseline_score,
                "pitcher_bootstrap_probability": champion_module.bootstrap(
                    valid_rows["pitcher_id"].to_numpy(), champion, candidate, target,
                    823200 + validation_year + int(blend * 1000),
                ),
                "absolute_mean_error_delta": (
                    abs(float(candidate.mean()) - float(target.mean()))
                    - abs(float(champion.mean()) - float(target.mean()))
                ),
            })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "champion_bss": baseline_score, "r_training_seconds": seconds,
            "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} complete", flush=True)

    summaries = []
    for blend in BLENDS:
        rows = [next(row for row in fold["candidates"] if row["blend"] == blend) for fold in folds]
        deltas = [float(row["bss_delta"]) for row in rows]
        probabilities = [float(row["pitcher_bootstrap_probability"]) for row in rows]
        ratio = min(map(abs, deltas)) / max(map(abs, deltas)) if max(map(abs, deltas)) else 0.0
        summaries.append({
            "blend": blend, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "worst_delta": min(deltas), "magnitude_ratio": ratio,
            "minimum_pitcher_bootstrap_probability": min(probabilities),
            "passed": bool(min(deltas) >= 1 and ratio >= 0.25 and min(probabilities) >= 0.80),
        })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "strict model-only R blend over current champion parity",
        "official_train_only": True, "test_aggregate_used": False,
        "fixed_failure_complement_blend": FAILURE_BLEND,
        "fixed_r_scale": champion_module.R_SCALE,
        "fixed_verified_shift_delta": champion_module.VERIFIED_SHIFT_DELTA,
        "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "build_strict_anchor_submission" if passed else "keep_current_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries,
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


def main(strict_source, component_dir, train_path, output, task_type):
    if strict_source.is_dir():
        run(strict_source, component_dir, train_path, output, task_type)
        return
    with tempfile.TemporaryDirectory(prefix="strict_champion_") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(strict_source) as archive:
            archive.extractall(directory)
        run(directory, component_dir, train_path, output, task_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-source", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.strict_source.resolve(), args.component_dir.resolve(), args.train.resolve(),
         args.output.resolve(), args.task_type)
