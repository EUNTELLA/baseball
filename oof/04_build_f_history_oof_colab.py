"""2021·2022 F general6 strict-forward OOF를 동일한 형식으로 생성한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_f" / "02_general_route_reconstruction_colab.py"
SEEDS = (42, 7, 2024, 99, 1, 123)


def load_route_module():
    spec = importlib.util.spec_from_file_location("f_history_route", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    prior = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (prior * (1 - prior))))


def main(train_path: Path, output_dir: Path, years, task_type: str):
    route = load_route_module()
    features = route.load_features()
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame["control_success"].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    available = set(np.unique(season).tolist())
    reports = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        if year not in available or year - 1 not in available:
            raise ValueError(f"{year}: 목표 시즌 또는 직전 시즌 데이터가 없습니다")
        if not np.any(season < year - 1):
            raise ValueError(f"{year}: iteration 선택용 과거 학습 시즌이 없습니다")

        prediction, shift, iterations, seconds = route.train_fold(
            frame, target, season, year, SEEDS, task_type, features, 128
        )
        rows = frame.loc[season == year].reset_index(drop=True)
        calibrated = route.sigmoid(route.logit(prediction) + shift)
        f_mask = rows["game_type"].astype(str).eq("F").to_numpy()
        output = output_dir / f"own_f_base_oof_{year}.npz"
        np.savez_compressed(
            output,
            row_id=rows["row_id"].astype(str).to_numpy(),
            target=rows["control_success"].astype(np.float32).to_numpy(),
            game_type=rows["game_type"].astype(str).to_numpy(),
            pitcher_id=rows["pitcher_id"].astype(str).to_numpy(),
            p_f_general6=calibrated.astype(np.float32),
        )
        report = {
            "year": int(year),
            "rows": int(len(rows)),
            "f_rows": int(f_mask.sum()),
            "seeds": list(SEEDS),
            "iterations": [int(value) for value in iterations],
            "seconds": seconds,
            "calibration_shift": float(shift),
            "f_bss": bss(rows.loc[f_mask, "control_success"], calibrated[f_mask]),
            "output": str(output),
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    summary = {"experiment": "F general6 historical strict-forward OOF", "folds": reports}
    (output_dir / "own_f_history_oof.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=(2021, 2022))
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output_dir.resolve(), args.years, args.task_type)
