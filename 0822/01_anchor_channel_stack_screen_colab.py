"""시간 전방 구성요소 채널을 현재 anchor에 저강도로 결합해 선별한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = ((2022, 2023), (2023, 2024))
CHANNELS = ("success", "failure_complement", "base_final", "base_offset")
REGIONS = ("all", "F", "R")
BLENDS = (0.025, 0.05, 0.10, 0.15, 0.20)
BOOTSTRAPS = 500


def load_components(directory: Path, year: int) -> dict[str, np.ndarray]:
    asset = np.load(directory / f"components_{year}.npz", allow_pickle=True)
    result = {name: asset[name] for name in asset.files}
    result["failure_complement"] = np.clip(
        1.0 - result["mr"].astype(float) - result["wayoff"].astype(float), 1e-6, 1 - 1e-6
    )
    return result


def logit(value):
    value = np.clip(np.asarray(value, float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.asarray(value, float)))


def shift_to_mean(prediction: np.ndarray, target_mean: float) -> float:
    values = logit(prediction)
    low, high = -2.0, 2.0
    for _ in range(80):
        middle = (low + high) / 2
        if float(sigmoid(values + middle).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def bss(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def bootstrap(ids, base, candidate, target, seed):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"id": ids.astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, counts = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / counts[sample].sum() > 0)
    return float(positive / BOOTSTRAPS)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(component_dir: Path, train_path: Path, output: Path) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig", usecols=["row_id", "season", "pitcher_id"])
    assets = {year: load_components(component_dir, year) for year in (2022, 2023, 2024)}
    folds = []
    for calibration_year, validation_year in PAIRS:
        calibration, valid = assets[calibration_year], assets[validation_year]
        valid_rows = frame.loc[frame["season"].astype(int).eq(validation_year)].reset_index(drop=True)
        if not np.array_equal(valid_rows["row_id"].astype(str).to_numpy(), valid["row_id"].astype(str)):
            raise ValueError(f"{validation_year} 구성요소 row_id 정렬 불일치")
        target = valid["target"].astype(float)
        anchor = valid["anchor"].astype(float)
        game_type = valid["game_type"].astype(str)
        base_score = bss(anchor, target)
        candidates = []
        channel_correlations = {}
        for channel in CHANNELS:
            calibration_channel = calibration[channel].astype(float)
            valid_channel = valid[channel].astype(float)
            shift = shift_to_mean(calibration_channel, float(calibration["anchor"].astype(float).mean()))
            aligned = sigmoid(logit(valid_channel) + shift)
            channel_correlations[channel] = float(np.corrcoef(target - anchor, target - aligned)[0, 1])
            for region in REGIONS:
                active = np.ones(len(target), dtype=bool) if region == "all" else game_type == region
                for blend in BLENDS:
                    prediction = anchor.copy()
                    prediction[active] = ((1 - blend) * anchor[active] + blend * aligned[active])
                    candidates.append({
                        "channel": channel, "region": region, "blend": blend,
                        "bss_delta": bss(prediction, target) - base_score,
                        "pitcher_bootstrap_probability": bootstrap(
                            valid_rows["pitcher_id"].to_numpy(), anchor, prediction, target,
                            822100 + validation_year + int(blend * 1000) + len(channel) + len(region),
                        ),
                        "absolute_mean_error_delta": (
                            abs(float(prediction.mean()) - float(target.mean()))
                            - abs(float(anchor.mean()) - float(target.mean()))
                        ),
                    })
        folds.append({
            "calibration_year": calibration_year, "validation_year": validation_year,
            "channel_error_correlations": channel_correlations, "candidates": candidates,
        })
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={validation_year} candidates={len(candidates)}", flush=True)

    summaries = []
    for channel in CHANNELS:
        for region in REGIONS:
            for blend in BLENDS:
                rows = [next(item for item in fold["candidates"] if item["channel"] == channel
                             and item["region"] == region and item["blend"] == blend) for fold in folds]
                deltas = [float(item["bss_delta"]) for item in rows]
                probabilities = [float(item["pitcher_bootstrap_probability"]) for item in rows]
                ratio = min(abs(value) for value in deltas) / max(abs(value) for value in deltas) if max(map(abs, deltas)) else 0.0
                passed = min(deltas) >= 1.0 and ratio >= 0.25 and min(probabilities) >= 0.80
                summaries.append({
                    "channel": channel, "region": region, "blend": blend,
                    "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                    "worst_delta": min(deltas), "magnitude_ratio": ratio,
                    "minimum_pitcher_bootstrap_probability": min(probabilities),
                    "passed": bool(passed),
                })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "time-forward anchor component channel stack screen",
        "official_train_only": True, "test_aggregate_used": False,
        "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_anchor_reconstruction" if passed else "keep_r0075_verified_shift_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries[:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.component_dir.resolve(), args.train.resolve(), args.output.resolve())
