"""strict Futures stack의 shared 대비 증분과 저장 checkpoint 구성을 감사한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = (2022, 2023, 2024)


def locate(directory: Path, year: int) -> Path:
    for pattern in (f"strict_f_regime075_oof_{year}.npz", f"*oof*{year}*.npz"):
        found = sorted(set(directory.rglob(pattern)))
        if found:
            return found[0]
    raise FileNotFoundError(f"{year} OOF를 찾지 못했습니다: {directory}")


def pick(asset, *names, required=True):
    for name in names:
        if name in asset.files:
            return asset[name]
    if required:
        raise KeyError(f"필요한 키가 없습니다: {names}; 현재 키={asset.files}")
    return None


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def describe(name, value, target, shared):
    value = np.asarray(value, float)
    return {
        "name": name,
        "bss": bss(value, target),
        "delta_vs_shared": bss(value, target) - bss(shared, target),
        "mean": float(value.mean()),
        "target_mean": float(target.mean()),
        "mean_delta_vs_shared": float(value.mean() - shared.mean()),
        "prediction_correlation_with_shared": float(np.corrcoef(value, shared)[0, 1]),
        "residual_correlation_with_shared": float(
            np.corrcoef(target - value, target - shared)[0, 1]
        ),
    }


def main(source: Path, output: Path):
    folds = []
    for year in YEARS:
        path = locate(source, year)
        asset = np.load(path, allow_pickle=True)
        target = np.asarray(pick(asset, "target", "y", "control_success"), float)
        game_type = np.asarray(pick(asset, "game_type", "league", "type")).astype(str)
        f_mask = game_type == "F"
        shared = np.asarray(pick(asset, "p_shared_stack"), float)[f_mask]
        futures = np.asarray(pick(asset, "p_f_stack"), float)[f_mask]
        stages = [describe("shared_stack", shared, target[f_mask], shared)]
        stages.append(describe("futures_stack", futures, target[f_mask], shared))
        optional = (
            "p_shared_adaptive", "p_f_adaptive", "p_surface",
            "p_model_only", "p_deployment",
        )
        for name in optional:
            value = pick(asset, name, required=False)
            if value is not None:
                stages.append(describe(name, np.asarray(value)[f_mask], target[f_mask], shared))

        checkpoint_paths = sorted(source.rglob(f"f_components_{year}.npz"))
        checkpoints = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            channel_rows = []
            for key in checkpoint.files:
                value = np.asarray(checkpoint[key])
                row = {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                if value.ndim == 1 and len(value) in (int(f_mask.sum()), len(target)):
                    numeric = value.astype(float)
                    if len(value) == len(target):
                        numeric = numeric[f_mask]
                    row.update({
                        "mean": float(np.nanmean(numeric)),
                        "std": float(np.nanstd(numeric)),
                        "correlation_with_stack_delta": float(np.corrcoef(
                            np.nan_to_num(numeric), futures - shared
                        )[0, 1]) if np.nanstd(numeric) > 0 else None,
                    })
                channel_rows.append(row)
            checkpoints.append({"path": str(checkpoint_path), "channels": channel_rows})
        folds.append({
            "year": year,
            "oof_path": str(path),
            "f_rows": int(f_mask.sum()),
            "available_oof_keys": list(asset.files),
            "stages": stages,
            "checkpoints": checkpoints,
        })
        print(f"year={year} F rows={int(f_mask.sum())} checkpoints={len(checkpoints)}", flush=True)

    stack_deltas = [
        next(row for row in fold["stages"] if row["name"] == "futures_stack")["delta_vs_shared"]
        for fold in folds
    ]
    payload = {
        "experiment": "Futures stack component attribution",
        "official_train_only": True,
        "test_aggregate_used": False,
        "folds": folds,
        "futures_stack_deltas_vs_shared": stack_deltas,
        "decision": (
            "reconstruct_futures_stack_channels"
            if min(stack_deltas) > 0 else "stop_futures_stack_reconstruction"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "futures_stack_deltas_vs_shared": stack_deltas,
        "checkpoint_keys": {
            str(fold["year"]): [
                row["key"] for checkpoint in fold["checkpoints"] for row in checkpoint["channels"]
            ] for fold in folds
        },
        "decision": payload["decision"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
