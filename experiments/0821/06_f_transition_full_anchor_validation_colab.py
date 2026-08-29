"""전체 시간 안전 anchor 위에서 F 전환 잔차의 2024 증분을 확인한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ID_COL, TARGET_COL = "row_id", "control_success"
SEEDS = (42, 7, 2024)
SCALES = (0.025, 0.05, 0.075)
ROOT = Path(__file__).resolve().parents[1]
TRANSITION_PATH = ROOT / "0821" / "05_f_transition_residual_screen_colab.py"


def load_transition_module():
    spec = importlib.util.spec_from_file_location("f_transition_screen", TRANSITION_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(TRANSITION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    return {
        "brier": brier,
        "bss_score": float(1e5 * (1 - brier / (rate * (1 - rate)))),
        "prediction_mean": float(prediction.mean()), "target_mean": rate,
    }


def aligned_anchor(frame: pd.DataFrame, year: int, path: Path):
    rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
    asset = np.load(path, allow_pickle=True)
    if len(rows) != len(asset["row_id"]):
        raise ValueError(f"{year} anchor 행 수 불일치")
    if not np.array_equal(rows[ID_COL].astype(str).to_numpy(), asset["row_id"].astype(str)):
        raise ValueError(f"{year} anchor row_id 순서 불일치")
    target = rows[TARGET_COL].astype(int).to_numpy()
    if not np.array_equal(target.astype(np.int8), asset["target"].astype(np.int8)):
        raise ValueError(f"{year} anchor 정답 불일치")
    return rows, target, asset["prediction"].astype(float)


def main(train_path: Path, anchor_dir: Path, output: Path, task_type: str) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    transition = load_transition_module()
    source_rows, source_target, source_prediction = aligned_anchor(
        frame, 2023, anchor_dir / "anchor_2023.npz"
    )
    valid_rows, valid_target, valid_prediction = aligned_anchor(
        frame, 2024, anchor_dir / "anchor_2024.npz"
    )
    source_x = transition.transition_features(source_rows, source_prediction, frame, 2023)
    valid_x = transition.transition_features(valid_rows, valid_prediction, frame, 2024)
    cat_indices = [source_x.columns.get_loc(column) for column in transition.CAT_COLS]
    train_pool = Pool(source_x, source_target - source_prediction, cat_features=cat_indices)
    valid_pool = Pool(valid_x, cat_features=cat_indices)
    members = []
    for seed in SEEDS:
        model = CatBoostRegressor(
            iterations=250, depth=3, learning_rate=0.025,
            loss_function="RMSE", l2_leaf_reg=100, random_strength=0.2,
            bootstrap_type="Bernoulli", subsample=0.8,
            random_seed=seed, task_type=task_type,
            devices="0" if task_type == "GPU" else None,
            thread_count=6, allow_writing_files=False, verbose=False,
        )
        model.fit(train_pool)
        members.append(model.predict(valid_pool))
    correction = np.mean(members, axis=0)
    valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
    correction[~valid_f] = 0.0
    baseline = metric(valid_prediction, valid_target)
    candidates = []
    for scale in SCALES:
        prediction = np.clip(valid_prediction + scale * correction, 1e-6, 1 - 1e-6)
        result = metric(prediction, valid_target)
        candidates.append({
            "scale": scale, "metrics": result,
            "bss_delta": result["bss_score"] - baseline["bss_score"],
            "absolute_mean_error_delta": abs(result["prediction_mean"] - result["target_mean"])
            - abs(baseline["prediction_mean"] - baseline["target_mean"]),
        })
    selected = next(row for row in candidates if row["scale"] == 0.05)
    neighbors_positive = all(row["bss_delta"] > 0 for row in candidates)
    passed = (selected["bss_delta"] >= 1.0 and neighbors_positive
              and selected["absolute_mean_error_delta"] <= 0.001)
    report = {
        "experiment": "F transition residual on complete anchor",
        "official_train_only": True, "test_aggregate_used": False,
        "r_scale_fixed": 0.05,
        "note": "R-only correction does not change the F-only Brier delta",
        "source_rows": int(len(source_rows)), "valid_rows": int(len(valid_rows)),
        "valid_f_rows": int(valid_f.sum()), "baseline": baseline,
        "candidates": candidates, "selected": selected,
        "decision": "build_f_transition_submission" if passed else "keep_r_scale0050_champion",
        "gate": "scale 0.05 delta >= +1, 0.025/0.05/0.075 all positive, mean-error delta <= 0.001",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.anchor_dir.resolve(), args.output.resolve(), args.task_type)
