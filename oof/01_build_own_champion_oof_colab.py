"""현재 자체 챔피언 구조의 2023·2024 strict-forward OOF를 저장한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENERAL_PATH = ROOT / "train_f" / "02_general_route_reconstruction_colab.py"
BASELINE_PATH = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
SEEDS = (42, 7, 2024, 99, 1, 123)
YEARS = (2023, 2024)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(train_path: Path, component_dir: Path, output_dir: Path, task_type: str):
    general = load_module(GENERAL_PATH, "own_oof_general")
    baseline = load_module(BASELINE_PATH, "own_oof_r_baseline")
    feature_module = general.load_features()
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame["control_success"].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    prior = float(target[season < 2022].mean())
    features = feature_module.engineer(frame.drop(columns=["row_id", "control_success"]), prior)
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    assets = {year: baseline.load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for year in YEARS:
        rows = frame.loc[season == year].reset_index(drop=True)
        asset = assets[year]
        if not np.array_equal(rows["row_id"].astype(str), asset["row_id"].astype(str)):
            raise ValueError(f"{year} row_id 불일치")

        general_raw, general_shift, iterations, general_seconds = general.train_fold(
            frame, target, season, year, SEEDS, task_type, feature_module, 128
        )
        p_f_general = general.sigmoid(general.logit(general_raw) + general_shift)

        calibration = assets[year - 1]
        correction, valid_rows, r_seconds = baseline.train_correction(
            frame, features, cat_indices, year - 1, year,
            calibration, asset, task_type,
        )
        if not np.array_equal(rows["row_id"].astype(str), valid_rows["row_id"].astype(str)):
            raise ValueError(f"{year} R correction row_id 불일치")

        is_r = rows["game_type"].astype(str).eq("R").to_numpy()
        is_f = ~is_r
        raw_anchor = asset["anchor"].astype(float)
        alignment_shift = baseline.shift_to_mean(
            calibration["failure_complement"].astype(float),
            float(calibration["anchor"].astype(float).mean()),
        )
        aligned_failure = baseline.sigmoid(
            baseline.logit(asset["failure_complement"].astype(float)) + alignment_shift
        )
        mixed_anchor = raw_anchor.copy()
        mixed_anchor[is_r] = 0.8 * raw_anchor[is_r] + 0.2 * aligned_failure[is_r]
        p_before_r = baseline.sigmoid(
            baseline.logit(mixed_anchor) + baseline.VERIFIED_SHIFT_DELTA
        )
        c_r_residual = np.zeros(len(rows), dtype=float)
        c_r_residual[is_r] = correction
        p_r_final = p_before_r.copy()
        p_r_final[is_r] = np.clip(
            p_r_final[is_r] + baseline.R_SCALE * correction, 1e-6, 1 - 1e-6
        )
        p_champion = p_r_final.copy()
        p_champion[is_f] = np.clip(p_f_general[is_f], 1e-6, 1 - 1e-6)

        output = output_dir / f"own_champion_oof_{year}.npz"
        np.savez_compressed(
            output,
            row_id=rows["row_id"].astype(str).to_numpy(),
            target=asset["target"].astype(np.float32),
            game_type=rows["game_type"].astype(str).to_numpy(),
            pitcher_id=rows["pitcher_id"].astype(str).to_numpy(),
            p_raw_anchor=raw_anchor.astype(np.float32),
            p_failure_complement=asset["failure_complement"].astype(np.float32),
            p_aligned_failure=aligned_failure.astype(np.float32),
            p_before_r=p_before_r.astype(np.float32),
            c_r_residual=c_r_residual.astype(np.float32),
            p_r_final=p_r_final.astype(np.float32),
            p_f_general6=p_f_general.astype(np.float32),
            p_champion=p_champion.astype(np.float32),
        )
        summary = {
            "year": year, "rows": int(len(rows)), "r_rows": int(is_r.sum()),
            "f_rows": int(is_f.sum()), "output": str(output),
            "general_iterations": iterations, "general_seconds": general_seconds,
            "general_calibration_shift": float(general_shift),
            "failure_alignment_shift": float(alignment_shift),
            "r_seconds": r_seconds,
            "champion_bss": baseline.bss(p_champion, asset["target"]),
            "r_bss": baseline.bss(p_champion[is_r], asset["target"][is_r]),
            "f_bss": baseline.bss(p_champion[is_f], asset["target"][is_f]),
        }
        summaries.append(summary)
        (output_dir / "run_summary.json").write_text(
            json.dumps({"status": "running", "folds": summaries}, ensure_ascii=False,
                       indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved {output}", flush=True)

    report = {
        "experiment": "own champion strict-forward OOF",
        "official_train_only": True, "test_aggregate_used": False,
        "structure": "R shift+residual075+failure020; F general6 min128",
        "general_seeds": list(SEEDS), "folds": summaries,
        "files": [str(output_dir / f"own_champion_oof_{year}.npz") for year in YEARS],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.component_dir.resolve(), args.output_dir.resolve(), args.task_type)
