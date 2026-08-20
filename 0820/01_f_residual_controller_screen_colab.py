"""현재 최고 구성 위에서 F 행 전용 잔차 제어기의 순방향 증분을 검증한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ID_COL = "row_id"
TARGET_COL = "control_success"
VALIDATION_FOLDS = (2023, 2024)
SCALES = (0.05, 0.10, 0.15, 0.20, 0.30)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
COMMON_PATH = ROOT / "0817" / "03_catboost_full_pipeline_walkforward_colab.py"
SCREEN_PATH = ROOT / "0819" / "01_catboost_residual_differential_screen_colab.py"
LABEL_PATH = ROOT / "0816" / "reference_catboost_best" / "recovered_labels.csv.gz"
BASE_CONFIG = {"name": "d6_lr05_l2_1", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0}


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


def controller_frame(features: pd.DataFrame, base_prediction: np.ndarray) -> pd.DataFrame:
    result = features.copy()
    result["base_prediction"] = np.asarray(base_prediction, dtype=float)
    return result


def fit_f_controller(
    source_frames: list[pd.DataFrame], source_targets: list[np.ndarray],
    source_is_f: list[np.ndarray], cat_cols: list[str], seed: int,
    task_type: str,
) -> CatBoostRegressor:
    x = pd.concat(source_frames, ignore_index=True)
    y = np.concatenate(source_targets)
    f = np.concatenate(source_is_f)
    cat_indices = [x.columns.get_loc(column) for column in cat_cols]
    params = {
        "iterations": 250, "depth": 4, "learning_rate": 0.025,
        "loss_function": "RMSE", "l2_leaf_reg": 100.0,
        "random_strength": 0.2, "bootstrap_type": "Bernoulli",
        "subsample": 0.8, "random_seed": seed,
        "allow_writing_files": False, "verbose": 0,
    }
    if task_type == "GPU":
        params.update(task_type="GPU", devices="0")
    else:
        params["thread_count"] = -1
    model = CatBoostRegressor(**params)
    model.fit(Pool(x.loc[f], y[f], cat_features=cat_indices))
    return model


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_module("full_pipeline_common", COMMON_PATH)
    screen = load_module("error_adjustment_screen", SCREEN_PATH)
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    is_f = frame["game_type"].astype(str).eq("F").to_numpy()
    contexts = screen.contexts(frame)
    recovered = frame[[ID_COL]].merge(pd.read_csv(LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = (
        (recovered["middle"].eq(1) | recovered["reverse"].eq(1))
        .fillna(False).astype(int).to_numpy()
    )
    large_miss_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    success_oof = np.full(len(frame), np.nan, dtype=float)
    fold_features: dict[int, pd.DataFrame] = {}
    report = {
        "experiment": "F-row residual controller over current 1029 structure",
        "official_train_only": True,
        "test_aggregate_used": False,
        "controller": {"depth": 4, "iterations": 250, "learning_rate": 0.025, "l2_leaf_reg": 100.0},
        "scales": list(SCALES), "pretraining": [], "fold_results": [],
    }

    for fold in (2021, 2022):
        train_mask, valid_mask = season < fold, season == fold
        features, cat_indices, global_mean = common.engineer(frame, feature_module, train_mask, target)
        prediction, iterations, seconds = common.train_inner_and_predict(
            features, cat_indices, target, np.ones(len(frame), dtype=bool),
            train_mask, valid_mask, BASE_CONFIG, task_type, f"source fold={fold} success",
        )
        success_oof[valid_mask] = prediction
        fold_features[fold] = controller_frame(features.loc[valid_mask].reset_index(drop=True), prediction)
        report["pretraining"].append({"fold": fold, "global_mean": global_mean, "best_iterations": iterations, "seconds": seconds})
        write_json(output, report)
        del features
        gc.collect()

    for validation_year in VALIDATION_FOLDS:
        calibration_year = validation_year - 1
        inner_train, calibration = season < calibration_year, season == calibration_year
        outer_train, validation = season < validation_year, season == validation_year
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
            training[name] = {"best_iterations": iterations, "inner_seconds": inner_seconds, "outer_seconds": outer_seconds}
        success_oof[validation] = predictions["success"]["outer"]
        fold_features[validation_year] = controller_frame(
            outer_x.loc[validation].reset_index(drop=True), predictions["success"]["outer"]
        )

        residual = target.astype(float) - success_oof
        source_years = (validation_year - 2, validation_year - 1)
        source = np.isin(season, source_years)
        if np.isnan(residual[source]).any():
            raise RuntimeError(f"fold={validation_year} 원천 OOF 예측 누락")
        additions = {}
        for name, shrinkage in screen.AXES:
            table = screen.differential_table(pitcher, contexts[name], residual, source, shrinkage)
            additions[name] = screen.apply_table(table, pitcher, contexts[name], validation)
        corrected_success = np.clip(
            predictions["success"]["outer"] + additions["hand"]
            + additions["two_strikes"] + additions["runners_on"], 1e-6, 1 - 1e-6
        )
        offset = common.fit_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["large_miss"]["inner"], target[calibration], have[calibration],
        )
        inner_offset = common.apply_offset(
            predictions["success"]["inner"], predictions["mr"]["inner"],
            predictions["large_miss"]["inner"], offset,
        )
        current_offset = common.apply_offset(
            corrected_success, predictions["mr"]["outer"], predictions["large_miss"]["outer"], offset,
        )
        forecast = common.select_alpha_and_forecast(frame, validation_year)
        shift = common.fixed_shift(inner_offset, forecast["forecast"])
        current = common.sigmoid(common.logit(current_offset) + shift)
        current_metrics = common.metrics(current, target[validation])

        source_frames = [fold_features[year] for year in source_years]
        source_targets = [
            target[season == year].astype(float) - success_oof[season == year]
            for year in source_years
        ]
        source_f = [is_f[season == year] for year in source_years]
        started = time.perf_counter()
        controller = fit_f_controller(
            source_frames, source_targets, source_f, list(feature_module.CAT_COLS),
            820000 + validation_year, task_type,
        )
        validation_x = fold_features[validation_year]
        cat_indices = [validation_x.columns.get_loc(column) for column in feature_module.CAT_COLS]
        correction = controller.predict(Pool(validation_x, cat_features=cat_indices))
        controller_seconds = float(time.perf_counter() - started)
        candidates = []
        for scale in SCALES:
            candidate = np.clip(current + np.where(is_f[validation], scale * correction, 0.0), 1e-6, 1 - 1e-6)
            metrics = common.metrics(candidate, target[validation])
            candidates.append({
                "scale": scale, "metrics": metrics,
                "bss_delta_vs_current": metrics["score"] - current_metrics["score"],
                "absolute_mean_error_delta": abs(metrics["mean_error"]) - abs(current_metrics["mean_error"]),
            })
        report["fold_results"].append({
            "validation_year": validation_year, "source_years": list(source_years),
            "f_rows": int(is_f[validation].sum()), "training": training,
            "controller_seconds": controller_seconds,
            "correction_f_mean": float(correction[is_f[validation]].mean()),
            "correction_f_std": float(correction[is_f[validation]].std()),
            "current": current_metrics, "candidates": candidates,
        })
        write_json(output, report)
        del inner_x, outer_x, predictions, controller
        gc.collect()

    summaries = []
    for scale in SCALES:
        rows = [next(item for item in fold["candidates"] if item["scale"] == scale) for fold in report["fold_results"]]
        deltas = [float(row["bss_delta_vs_current"]) for row in rows]
        errors = [float(row["absolute_mean_error_delta"]) for row in rows]
        summaries.append({
            "scale": scale, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
            "fold_2024_absolute_mean_error_delta": errors[1], "both_positive": bool(min(deltas) > 0),
        })
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 3.0 and row["fold_2024_absolute_mean_error_delta"] <= 0.001]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = summaries
    report["selected"] = selected
    report["decision"] = "continue_f_residual_full_build" if selected else "keep_1029_champion"
    report["gate"] = "same scale improves 2023 and 2024; 2024 BSS >= +3; mean-error deterioration <= 0.001"
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
