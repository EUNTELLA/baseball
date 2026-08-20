"""과거 시즌 예측 오차 보정 3종을 기존 offset·shift 전체 과정에서 검증한다."""
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
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
COMMON_PATH = ROOT / "0817" / "03_catboost_full_pipeline_walkforward_colab.py"
SCREEN_PATH = SCRIPT_DIR / "01_catboost_residual_differential_screen_colab.py"
COUNT_HAND_PATH = ROOT / "0820" / "07_count_hand_leverage_differential_screen_colab.py"
TRACKMAN_FEATURE_PATH = ROOT / "0820" / "13_trackman_prior_feature_screen_colab.py"
LABEL_PATH = ROOT / "0816" / "reference_catboost_best" / "recovered_labels.csv.gz"
BASE_CONFIG = {"name": "catboost_d6_lr05_l2_1", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 1.0}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extended_metrics(common, prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return common.metrics(np.clip(prediction, 1e-6, 1 - 1e-6), target)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(
    train_path: Path,
    output: Path,
    task_type: str,
    selected_axes: tuple[str, ...] = ("hand", "two_strikes", "runners_on"),
    adjustment_weight: float = 1.0,
    include_count_hand: bool = False,
    count_hand_shrinkage: float = 500.0,
    count_hand_scale: float = 0.3,
    trackman_path: Path | None = None,
    mapping_path: Path | None = None,
    trackman_blend: float = 1.0,
) -> None:
    common = load_module("full_pipeline_common", COMMON_PATH)
    screen = load_module("residual_differential_screen", SCREEN_PATH)
    count_hand_module = load_module("count_hand_differential", COUNT_HAND_PATH) if include_count_hand else None
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    axis_contexts = screen.contexts(frame)
    recovered = frame[[ID_COL]].merge(pd.read_csv(LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = ((recovered["middle"] == 1) | (recovered["reverse"] == 1)).fillna(False).astype(int).to_numpy()
    wayoff_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    include_trackman = trackman_path is not None and mapping_path is not None
    trackman_attached = None
    trackman_columns: list[str] = []
    if include_trackman:
        trackman_module = load_module("trackman_prior_features", TRACKMAN_FEATURE_PATH)
        assets = trackman_module.build_prior_assets(frame, trackman_path, mapping_path)
        merge_key = pd.DataFrame({"pitcher_id": frame["pitcher_id"].astype(str), "season": season,
                                  "_row_order": np.arange(len(frame))})
        trackman_attached = (merge_key.merge(assets, on=["pitcher_id", "season"], how="left", sort=False)
                             .sort_values("_row_order").reset_index(drop=True))
        trackman_columns = [column for column in trackman_attached.columns if column.startswith("tm_")]
        reliable = (trackman_attached["tm_mapping_similarity"].ge(0.90)
                    & trackman_attached["tm_mapping_margin"].ge(0.02)
                    & trackman_attached["tm_prior_n"].ge(200.0))
        for column in trackman_columns:
            trackman_attached[column] = pd.to_numeric(trackman_attached[column], errors="coerce").where(reliable)
    success_oof = np.full(len(frame), np.nan, dtype=float)
    report = {
        "experiment": "OOF residual differential 3-axis in complete 997 pipeline",
        "official_train_only": True,
        "test_aggregate_used": False,
        "axes": {name: shrinkage for name, shrinkage in screen.AXES},
        "selected_axes": list(selected_axes),
        "adjustment_weight": adjustment_weight,
        "incremental_count_hand": {
            "enabled": include_count_hand,
            "shrinkage": count_hand_shrinkage,
            "scale": count_hand_scale,
            "source": "previous season only",
        },
        "trackman_success_replacement": {
            "enabled": include_trackman, "blend": trackman_blend,
            "similarity_floor": 0.90, "margin_floor": 0.02, "prior_n_floor": 200.0,
        },
        "seeds": list(common.SEEDS),
        "pretraining": [],
        "results": [],
    }

    # 2023 검증표의 원천인 2021·2022 성공 OOF를 먼저 만든다.
    for fold in (2021, 2022):
        train_mask = season < fold
        valid_mask = season == fold
        features, cat_indices, global_mean = common.engineer(
            frame, feature_module, train_mask, target
        )
        prediction, iterations, seconds = common.train_inner_and_predict(
            features,
            cat_indices,
            target,
            np.ones(len(frame), dtype=bool),
            train_mask,
            valid_mask,
            BASE_CONFIG,
            task_type,
            f"source fold={fold} success",
        )
        success_oof[valid_mask] = prediction
        report["pretraining"].append({
            "fold": fold,
            "global_mean": global_mean,
            "best_iterations": iterations,
            "seconds": seconds,
            "metrics": extended_metrics(common, prediction, target[valid_mask]),
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
        inner_x, inner_ci, inner_mean = common.engineer(frame, feature_module, inner_train, target)
        outer_x, outer_ci, outer_mean = common.engineer(frame, feature_module, outer_train, target)
        inner_tm_x, outer_tm_x = None, None
        if include_trackman:
            inner_tm_x, outer_tm_x = inner_x.copy(), outer_x.copy()
            for column in trackman_columns:
                values = trackman_attached[column].to_numpy()
                inner_tm_x[column], outer_tm_x[column] = values, values

        predictions = {}
        training = {}
        for name, labels, eligible, config in (
            ("success", target, np.ones(len(frame), dtype=bool), BASE_CONFIG),
            ("mr", mr_target, have, common.AUX_CONFIG),
            ("wayoff", wayoff_target, have, common.AUX_CONFIG),
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
            if name == "success" and include_trackman:
                tm_inner, tm_iterations, tm_inner_seconds = common.train_inner_and_predict(
                    inner_tm_x, inner_ci, labels, eligible, inner_train, calibration,
                    config, task_type, f"fold={validation_year} success Trackman",
                )
                tm_outer, tm_outer_seconds = common.train_outer_and_predict(
                    outer_tm_x, outer_ci, labels, eligible, outer_train, validation,
                    config, tm_iterations, task_type, f"fold={validation_year} success Trackman",
                )
                predictions["success_trackman"] = {"inner": tm_inner, "outer": tm_outer}
                training["success_trackman"] = {
                    "best_iterations": tm_iterations,
                    "inner_seconds": tm_inner_seconds, "outer_seconds": tm_outer_seconds,
                }
        success_oof[validation] = predictions["success"]["outer"]

        residual = target.astype(float) - success_oof
        source = np.isin(season, (validation_year - 2, validation_year - 1))
        if np.isnan(residual[source]).any():
            raise RuntimeError(f"fold={validation_year} 표 원천 OOF 예측 누락")
        additions, tables = {}, {}
        for name, shrinkage in screen.AXES:
            table = screen.differential_table(
                pitcher, axis_contexts[name], residual, source, shrinkage
            )
            additions[name] = screen.apply_table(
                table, pitcher, axis_contexts[name], validation
            )
            tables[name] = {
                "shrinkage": shrinkage,
                "pitchers": int(len(table)),
                "median_absolute_difference": float(table.abs().median()) if len(table) else 0.0,
            }
        correction = adjustment_weight * sum(
            (additions[name] for name in selected_axes), np.zeros(int(validation.sum()))
        )
        count_hand_correction = np.zeros(int(validation.sum()))
        if include_count_hand:
            count_hand_keys = count_hand_module.keys(frame)["count_hand"]
            previous_season = season == (validation_year - 1)
            count_hand_correction, count_hand_groups = count_hand_module.differential(
                count_hand_keys.loc[previous_season], residual[previous_season],
                count_hand_keys.loc[validation], count_hand_shrinkage,
            )
            count_hand_correction *= count_hand_scale
            tables["count_hand"] = {
                "shrinkage": count_hand_shrinkage,
                "scale": count_hand_scale,
                "groups": count_hand_groups,
                "source_season": validation_year - 1,
            }

        calibration_target = target[calibration]
        offset = common.fit_offset(
            predictions["success"]["inner"],
            predictions["mr"]["inner"],
            predictions["wayoff"]["inner"],
            calibration_target,
            have[calibration],
        )
        inner_offset = common.apply_offset(
            predictions["success"]["inner"],
            predictions["mr"]["inner"],
            predictions["wayoff"]["inner"],
            offset,
        )
        blended_inner_success = predictions["success"]["inner"]
        blended_outer_success = predictions["success"]["outer"]
        candidate_offset_parameters = offset
        candidate_inner_offset = inner_offset
        if include_trackman:
            blended_inner_success = ((1.0 - trackman_blend) * predictions["success"]["inner"]
                                     + trackman_blend * predictions["success_trackman"]["inner"])
            blended_outer_success = ((1.0 - trackman_blend) * predictions["success"]["outer"]
                                     + trackman_blend * predictions["success_trackman"]["outer"])
            candidate_offset_parameters = common.fit_offset(
                blended_inner_success, predictions["mr"]["inner"], predictions["wayoff"]["inner"],
                calibration_target, have[calibration],
            )
            candidate_inner_offset = common.apply_offset(
                blended_inner_success, predictions["mr"]["inner"], predictions["wayoff"]["inner"],
                candidate_offset_parameters,
            )
        incumbent_success = np.clip(predictions["success"]["outer"] + correction, 1e-6, 1 - 1e-6)
        baseline_success = incumbent_success if (include_count_hand or include_trackman) else predictions["success"]["outer"]
        baseline_offset = common.apply_offset(
            baseline_success,
            predictions["mr"]["outer"],
            predictions["wayoff"]["outer"],
            offset,
        )
        candidate_incumbent = np.clip(blended_outer_success + correction, 1e-6, 1 - 1e-6)
        corrected_success = np.clip(candidate_incumbent + count_hand_correction, 1e-6, 1 - 1e-6)
        candidate_offset = common.apply_offset(
            corrected_success,
            predictions["mr"]["outer"],
            predictions["wayoff"]["outer"],
            candidate_offset_parameters,
        )
        forecast = common.select_alpha_and_forecast(frame, validation_year)
        shift = common.fixed_shift(inner_offset, forecast["forecast"])
        candidate_shift = common.fixed_shift(candidate_inner_offset, forecast["forecast"])
        baseline_final = common.sigmoid(common.logit(baseline_offset) + shift)
        candidate_final = common.sigmoid(common.logit(candidate_offset) + candidate_shift)
        validation_target = target[validation]
        baseline_metrics = extended_metrics(common, baseline_final, validation_target)
        candidate_metrics = extended_metrics(common, candidate_final, validation_target)
        fold_result = {
            "validation_year": validation_year,
            "source_seasons": [validation_year - 2, validation_year - 1],
            "inner_global_mean": inner_mean,
            "outer_global_mean": outer_mean,
            "training": training,
            "tables": tables,
            "offset": offset,
            "rate_forecast": forecast,
            "shift": shift,
            "candidate_offset": candidate_offset_parameters,
            "candidate_shift": candidate_shift,
            "correction": {
                "mean": float(correction.mean()),
                "std": float(correction.std()),
                "min": float(correction.min()),
                "max": float(correction.max()),
            },
            "count_hand_correction": {
                "mean": float(count_hand_correction.mean()),
                "std": float(count_hand_correction.std()),
                "min": float(count_hand_correction.min()),
                "max": float(count_hand_correction.max()),
            },
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "bss_delta": candidate_metrics["score"] - baseline_metrics["score"],
            "absolute_mean_error_delta": (
                abs(candidate_metrics["mean_error"]) - abs(baseline_metrics["mean_error"])
            ),
        }
        report["results"].append(fold_result)
        print(
            f"fold={validation_year} full BSS delta={fold_result['bss_delta']:+.2f}",
            flush=True,
        )
        write_json(output, report)
        del inner_x, outer_x, predictions
        if inner_tm_x is not None:
            del inner_tm_x, outer_tm_x
        gc.collect()

    bss_deltas = [float(row["bss_delta"]) for row in report["results"]]
    mean_error_deltas = [float(row["absolute_mean_error_delta"]) for row in report["results"]]
    passed = (
        min(bss_deltas) > 0.0
        and bss_deltas[-1] >= 5.0
        and mean_error_deltas[-1] <= 0.001
    )
    report["summary"] = {
        "bss_deltas": bss_deltas,
        "absolute_mean_error_deltas": mean_error_deltas,
        "decision": "build_residual_differential_submission" if passed else "keep_997_baseline",
        "gate": "2023/2024 BSS positive, 2024 BSS>=+5, 2024 abs mean error delta<=0.001",
    }
    write_json(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument(
        "--axes", default="hand,two_strikes,runners_on",
        help="쉼표로 구분한 적용 보정: hand,two_strikes,runners_on",
    )
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--include-count-hand", action="store_true")
    parser.add_argument("--count-hand-shrinkage", type=float, default=500.0)
    parser.add_argument("--count-hand-scale", type=float, default=0.3)
    parser.add_argument("--trackman", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--trackman-blend", type=float, default=1.0)
    args = parser.parse_args()
    selected_axes = tuple(item.strip() for item in args.axes.split(",") if item.strip())
    allowed_axes = {name for name, _ in load_module("axis_check", SCREEN_PATH).AXES}
    if not selected_axes or any(name not in allowed_axes for name in selected_axes):
        parser.error(f"--axes는 {sorted(allowed_axes)} 중 하나 이상이어야 합니다.")
    if args.weight <= 0:
        parser.error("--weight는 0보다 커야 합니다.")
    if (args.trackman is None) != (args.mapping is None):
        parser.error("--trackman과 --mapping은 함께 지정해야 합니다.")
    if not 0.0 <= args.trackman_blend <= 1.0:
        parser.error("--trackman-blend는 0과 1 사이여야 합니다.")
    main(
        args.train.resolve(), args.output.resolve(), args.task_type,
        selected_axes, args.weight, args.include_count_hand,
        args.count_hand_shrinkage, args.count_hand_scale,
        args.trackman.resolve() if args.trackman else None,
        args.mapping.resolve() if args.mapping else None,
        args.trackman_blend,
    )
