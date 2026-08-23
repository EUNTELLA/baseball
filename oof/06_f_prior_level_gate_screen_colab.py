"""직전 시즌 F level 오차에 dead-zone과 cap을 적용하는 strict-forward 감사."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VALID_YEARS = (2023, 2024)
DEAD_ZONES = (0.02, 0.03, 0.05)
CAPS = (0.025, 0.05, 0.075)


def read(path):
    asset = np.load(path, allow_pickle=True)
    return {key: asset[key] for key in asset.files}


def path_for(directory, year):
    history = directory / f"own_f_base_oof_{year}.npz"
    champion = directory / f"own_champion_oof_{year}.npz"
    return history if history.exists() else champion


def prediction(asset):
    key = "p_f_general6" if "p_f_general6" in asset else "p_champion"
    return asset[key].astype(float)


def logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.asarray(value, float)))


def bss(target, probability):
    target = np.asarray(target, float)
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    prior = target.mean()
    return float(1e5 * (1 - np.mean((target - probability) ** 2) / (prior * (1 - prior))))


def optimal_shift(target, probability):
    grid = np.linspace(-0.5, 0.5, 4001)
    base = logit(probability)
    losses = [np.mean((target - sigmoid(base + shift)) ** 2) for shift in grid]
    return float(grid[int(np.argmin(losses))])


def bootstrap(ids, target, base, candidate, seed, count=500):
    gain = (target - base) ** 2 - (target - candidate) ** 2
    grouped = pd.DataFrame({"id": pd.Series(ids).astype(str), "gain": gain, "rows": 1})
    grouped = grouped.groupby("id", observed=True)[["gain", "rows"]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(count):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += grouped[sample, 0].sum() / grouped[sample, 1].sum() > 0
    return float(positive / count)


def main(oof_dir: Path, output: Path):
    assets = {year: read(path_for(oof_dir, year)) for year in (2022, 2023, 2024)}
    prior_shifts = {}
    for year, asset in assets.items():
        mask = asset["game_type"].astype(str) == "F"
        prior_shifts[year] = optimal_shift(
            asset["target"].astype(float)[mask], prediction(asset)[mask]
        )

    summaries = []
    for dead_zone in DEAD_ZONES:
        for cap in CAPS:
            folds = []
            for year in VALID_YEARS:
                asset = assets[year]
                mask = asset["game_type"].astype(str) == "F"
                target = asset["target"].astype(float)
                base = prediction(asset)
                raw_shift = prior_shifts[year - 1]
                applied_shift = 0.0 if abs(raw_shift) < dead_zone else float(np.clip(raw_shift, -cap, cap))
                candidate = base.copy()
                candidate[mask] = sigmoid(logit(base[mask]) + applied_shift)
                folds.append({
                    "year": year,
                    "prior_year_shift": raw_shift,
                    "applied_shift": applied_shift,
                    "overall_delta": bss(target, candidate) - bss(target, base),
                    "f_delta": bss(target[mask], candidate[mask]) - bss(target[mask], base[mask]),
                    "bootstrap": bootstrap(
                        asset["pitcher_id"].astype(str)[mask], target[mask], base[mask],
                        candidate[mask], 826000 + year + int(cap * 1000),
                    ) if applied_shift else 1.0,
                })
            passed = (
                min(fold["overall_delta"] for fold in folds) >= 0
                and min(fold["f_delta"] for fold in folds) >= 0
                and min(fold["bootstrap"] for fold in folds) >= 0.8
                and folds[-1]["overall_delta"] >= 1
            )
            summaries.append({
                "dead_zone": dead_zone,
                "cap": cap,
                "fold_2023_delta": folds[0]["overall_delta"],
                "fold_2024_delta": folds[1]["overall_delta"],
                "fold_2023_f_delta": folds[0]["f_delta"],
                "fold_2024_f_delta": folds[1]["f_delta"],
                "minimum_bootstrap": min(fold["bootstrap"] for fold in folds),
                "folds": folds,
                "passed": bool(passed),
            })

    summaries.sort(
        key=lambda item: (item["passed"], min(item["fold_2023_delta"], item["fold_2024_delta"]),
                          item["fold_2024_delta"]), reverse=True
    )
    selected = next((item for item in summaries if item["passed"]), None)
    deployment_source_shift = prior_shifts[2024]
    report = {
        "experiment": "F prior-year level dead-zone gate",
        "official_train_only": True,
        "test_aggregate_used": False,
        "prior_year_oracle_shifts": prior_shifts,
        "selected": selected,
        "top": summaries[:9],
        "deployment_source_shift": deployment_source_shift,
        "deployment_shift": (
            float(np.clip(deployment_source_shift, -selected["cap"], selected["cap"]))
            if selected and abs(deployment_source_shift) >= selected["dead_zone"] else 0.0
        ),
        "decision": "build_f_level_gate_submission" if selected else "keep_own_champion_oof",
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
