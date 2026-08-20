"""현재 최고 구성 위에서 F 행의 실패유형 보조 신호 추가 계수를 선별한다.

검증 연도 Y의 성공·MR·큰 이탈 확률은 Y 이전 시즌만 학습해 생성한다.
기존 전역 MR/큰 이탈 offset과 과거 예측 오차 보정 3종은 그대로 유지하고,
F 행에만 중심화된 보조 logit의 추가 계수를 적용한다.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
VALIDATION_FOLDS = (2023, 2024)
DELTA_COEFFICIENTS = (-0.10, -0.05, -0.025, 0.025, 0.05, 0.10)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
COMMON_PATH = ROOT / "0817" / "03_catboost_full_pipeline_walkforward_colab.py"
SCREEN_PATH = SCRIPT_DIR / "01_catboost_residual_differential_screen_colab.py"
FAILURE_LABEL_PATH = ROOT / "common" / "failure_labels.py"
BASE_CONFIG = {
    "name": "catboost_d6_lr05_l2_1",
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 1.0,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_module("full_pipeline_common", COMMON_PATH)
    screen = load_module("error_adjustment_screen", SCREEN_PATH)
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    is_f = frame["game_type"].astype(str).eq("F").to_numpy()
    contexts = screen.contexts(frame)
    failure_module = load_module("failure_labels", FAILURE_LABEL_PATH)
    recovered = failure_module.recover_failure_labels(frame)
    have = recovered["middle"].notna().to_numpy()
    mr_target = (
        (recovered["middle"].eq(1) | recovered["reverse"].eq(1))
        .fillna(False).astype(int).to_numpy()
    )
    large_miss_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    success_oof = np.full(len(frame), np.nan, dtype=float)
    report = {
        "experiment": "F-row auxiliary failure-signal incremental coefficient screen",
        "official_train_only": True,
        "test_aggregate_used": False,
        "baseline": "1029 structure: three past-error adjustments plus global auxiliary offset and train-only shift",
        "signals": ["mr", "large_miss"],
        "delta_coefficients": list(DELTA_COEFFICIENTS),
        "pretraining": [],
        "fold_results": [],
    }

    # 2023의 차등표에 필요한 2021·2022 성공 OOF를 미리 만든다.
    for fold in (2021, 2022):
        train_mask = season < fold
        valid_mask = season == fold
        features, cat_indices, global_mean = common.engineer(
            frame, feature_module, train_mask, target
        )
        prediction, iterations, seconds = common.train_inner_and_predict(
            features, cat_indices, target, np.ones(len(frame), dtype=bool),
            train_mask, valid_mask, BASE_CONFIG, task_type,
            f"source fold={fold} success",
        )
        success_oof[valid_mask] = prediction
        report["pretraining"].append({
            "fold": fold,
            "global_mean": global_mean,
            "best_iterations": iterations,
            "seconds": seconds,
        })
        write_json(output, report)
        del features
        gc.collect()

    for validation_year in VALIDATION_FOLDS:
        calibration_year = validation_year - 1
        inner_train = season < calibration_year
        calibration = season == calibration_year
        outer_train = season < validation_year
        validation = season == validation_year
        print(f"\n===== fold {validation_year}: calibrate {calibration_year} =====", flush=True)
        inner_x, inner_ci, _ = common.engineer(frame, feature_module, inner_train, target)
        outer_x, outer_ci, _ = common.engineer(frame, feature_module, outer_train, target)
        predictions, training = {}, {}
        for name, labels, eligible, config in (
            ("success", target, np.ones(len(frame), dtype=bool), BASE_CONFIG),
            ("mr", mr_target, have, common.AUX_CONFIG),
            ("large_miss", large_miss_target, have, common.AUX_CONFIG),
        ):
            inner_prediction, iterations, inner_seconds = common.train_inner_and_predict(
                inner_x, inner_ci, labels, eligible, inner_train, calibration,
                config, task_type, f"fold={validation_year} {name}",
            )
            outer_prediction, outer_seconds = common.train_outer_and_predict(
                outer_x, outer_ci, labels, eligible, outer_train, validation,
                config, iterations, task_type, f"fold={validation_year} {name}",
            )
            predictions[name] = {"inner": inner_prediction, "outer": outer_prediction}
            training[name] = {
                "best_iterations": iterations,
                "inner_seconds": inner_seconds,
                "outer_seconds": outer_seconds,
            }
        success_oof[validation] = predictions["success"]["outer"]

        residual = target.astype(float) - success_oof
        source = np.isin(season, (validation_year - 2, validation_year - 1))
        if np.isnan(residual[source]).any():
            raise RuntimeError(f"fold={validation_year} 차등표 원천 OOF 예측 누락")
        additions = {}
        for name, shrinkage in screen.AXES:
            table = screen.differential_table(
                pitcher, contexts[name], residual, source, shrinkage
            )
            additions[name] = screen.apply_table(table, pitcher, contexts[name], validation)
        three_axis = additions["hand"] + additions["two_strikes"] + additions["runners_on"]

        offset = common.fit_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["large_miss"]["inner"], target[calibration], have[calibration],
        )
        inner_offset = common.apply_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["large_miss"]["inner"], offset,
        )
        corrected_success = np.clip(
            predictions["success"]["outer"] + three_axis, 1e-6, 1 - 1e-6
        )
        current_offset = common.apply_offset(
            corrected_success, predictions["mr"]["outer"],
            predictions["large_miss"]["outer"], offset,
        )
        forecast = common.select_alpha_and_forecast(frame, validation_year)
        shift = common.fixed_shift(inner_offset, forecast["forecast"])
        current_final = common.sigmoid(common.logit(current_offset) + shift)
        y_valid = target[validation]
        current_metrics = common.metrics(current_final, y_valid)
        f_valid = is_f[validation]
        centered = {
            "mr": common.logit(predictions["mr"]["outer"]) - offset["mu_mr"],
            "large_miss": (
                common.logit(predictions["large_miss"]["outer"]) - offset["mu_wayoff"]
            ),
        }
        candidates = []
        for signal, values in centered.items():
            for coefficient in DELTA_COEFFICIENTS:
                extra_logit = np.where(f_valid, coefficient * values, 0.0)
                candidate = common.sigmoid(common.logit(current_offset) + shift + extra_logit)
                candidate_metrics = common.metrics(candidate, y_valid)
                candidates.append({
                    "signal": signal,
                    "delta_coefficient": coefficient,
                    "metrics": candidate_metrics,
                    "bss_delta_vs_current": candidate_metrics["score"] - current_metrics["score"],
                    "absolute_mean_error_delta": (
                        abs(candidate_metrics["mean_error"]) - abs(current_metrics["mean_error"])
                    ),
                })
        report["fold_results"].append({
            "validation_year": validation_year,
            "source_seasons": [validation_year - 2, validation_year - 1],
            "f_rows": int(f_valid.sum()),
            "f_share": float(f_valid.mean()),
            "training": training,
            "offset": offset,
            "shift": shift,
            "current": current_metrics,
            "candidates": candidates,
        })
        write_json(output, report)
        del inner_x, outer_x, predictions
        gc.collect()

    summaries = []
    for signal in report["signals"]:
        for coefficient in DELTA_COEFFICIENTS:
            rows = [
                next(
                    item for item in fold["candidates"]
                    if item["signal"] == signal and item["delta_coefficient"] == coefficient
                )
                for fold in report["fold_results"]
            ]
            deltas = [float(row["bss_delta_vs_current"]) for row in rows]
            mean_errors = [float(row["absolute_mean_error_delta"]) for row in rows]
            summaries.append({
                "signal": signal,
                "delta_coefficient": coefficient,
                "fold_2023_delta": deltas[0],
                "fold_2024_delta": deltas[1],
                "mean_delta": float(np.mean(deltas)),
                "worst_delta": float(np.min(deltas)),
                "fold_2024_absolute_mean_error_delta": mean_errors[1],
                "both_positive": bool(min(deltas) > 0),
            })
    stable = [
        row for row in summaries
        if row["both_positive"]
        and row["fold_2024_delta"] >= 3.0
        and row["fold_2024_absolute_mean_error_delta"] <= 0.001
    ]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = summaries
    report["selected"] = selected
    report["decision"] = "validate_selected_f_signal" if selected else "keep_1029_champion"
    report["gate"] = (
        "same signal/coefficient improves 2023 and 2024; 2024 BSS >= +3; "
        "2024 absolute mean-error deterioration <= 0.001"
    )
    write_json(output, report)
    print(json.dumps({"selected": selected, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
