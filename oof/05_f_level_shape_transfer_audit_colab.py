"""F OOF의 시즌별 level 오차와 행별 shape 신호를 분리해 감사한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = (2021, 2022, 2023, 2024)


def read(path):
    asset = np.load(path, allow_pickle=True)
    return {key: asset[key] for key in asset.files}


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    prior = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (prior * (1 - prior))))


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.asarray(value, dtype=float)))


def optimal_shift(target, prediction):
    grid = np.linspace(-0.20, 0.20, 1601)
    logits = logit(prediction)
    losses = [np.mean((target - sigmoid(logits + shift)) ** 2) for shift in grid]
    return float(grid[int(np.argmin(losses))])


def asset_path(oof_dir, year):
    historical = oof_dir / f"own_f_base_oof_{year}.npz"
    champion = oof_dir / f"own_champion_oof_{year}.npz"
    return historical if historical.exists() else champion


def main(oof_dir: Path, output: Path):
    folds = []
    for year in YEARS:
        path = asset_path(oof_dir, year)
        if not path.exists():
            raise FileNotFoundError(path)
        asset = read(path)
        mask = asset["game_type"].astype(str) == "F"
        target = asset["target"].astype(float)[mask]
        key = "p_f_general6" if "p_f_general6" in asset else "p_champion"
        prediction = asset[key].astype(float)[mask]
        shift = optimal_shift(target, prediction)
        shifted = sigmoid(logit(prediction) + shift)
        residual = target - prediction
        centered_target = target - target.mean()
        centered_prediction = prediction - prediction.mean()
        correlation = float(np.corrcoef(centered_target, centered_prediction)[0, 1])
        folds.append({
            "year": year,
            "rows": int(mask.sum()),
            "prediction_key": key,
            "target_mean": float(target.mean()),
            "prediction_mean": float(prediction.mean()),
            "mean_error": float(residual.mean()),
            "prediction_std": float(prediction.std()),
            "target_prediction_correlation": correlation,
            "raw_bss": bss(target, prediction),
            "oracle_level_shift": shift,
            "oracle_level_bss": bss(target, shifted),
            "oracle_level_delta": bss(target, shifted) - bss(target, prediction),
        })

    shift_signs = [np.sign(item["oracle_level_shift"]) for item in folds[1:]]
    stable_level = len(set(shift_signs)) == 1
    latest_correlations = [item["target_prediction_correlation"] for item in folds[-2:]]
    decision = (
        "screen_strict_prior_year_level_calibration"
        if stable_level
        else "screen_season_state_route_without_static_level_correction"
    )
    report = {
        "experiment": "F OOF level-shape transfer audit",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": folds,
        "level_shift_sign_stable_2022_2024": bool(stable_level),
        "latest_shape_correlations": latest_correlations,
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.oof_dir.resolve(), args.output.resolve())
