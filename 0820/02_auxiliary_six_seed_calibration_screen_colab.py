"""MR·큰 이탈 보조 채널의 3/6시드 평균과 Train 전역 shift 재계산을 비교한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ID_COL, TARGET_COL = "row_id", "control_success"
VALIDATION_FOLDS = (2023, 2024)
AUX_SEEDS = (42, 7, 2024, 99, 1, 123)
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


def train_auxiliary(common, inner_x, inner_ci, outer_x, outer_ci, labels, eligible,
                    inner_train, calibration, outer_train, validation, task_type, label):
    pools = {
        "inner_train": Pool(inner_x.loc[inner_train & eligible], labels[inner_train & eligible], cat_features=inner_ci),
        "calibration": Pool(inner_x.loc[calibration & eligible], labels[calibration & eligible], cat_features=inner_ci),
        "inner_prediction": Pool(inner_x.loc[calibration], cat_features=inner_ci),
        "outer_train": Pool(outer_x.loc[outer_train & eligible], labels[outer_train & eligible], cat_features=outer_ci),
        "outer_prediction": Pool(outer_x.loc[validation], cat_features=outer_ci),
    }
    inner_members, outer_members, iterations = [], [], []
    inner_seconds, outer_seconds = [], []
    for seed in AUX_SEEDS:
        started = time.perf_counter()
        model = CatBoostClassifier(**common.model_params(common.AUX_CONFIG, seed, 2000, task_type, True))
        model.fit(pools["inner_train"], eval_set=pools["calibration"], use_best_model=True)
        inner_members.append(model.predict_proba(pools["inner_prediction"])[:, 1])
        iteration = max(1, int(model.get_best_iteration()) + 1)
        iterations.append(iteration)
        inner_seconds.append(float(time.perf_counter() - started))
        del model
        gc.collect()
        started = time.perf_counter()
        model = CatBoostClassifier(**common.model_params(common.AUX_CONFIG, seed, iteration, task_type, False))
        model.fit(pools["outer_train"])
        outer_members.append(model.predict_proba(pools["outer_prediction"])[:, 1])
        outer_seconds.append(float(time.perf_counter() - started))
        print(f"{label} seed={seed} iter={iteration} inner={inner_seconds[-1]:.1f}s outer={outer_seconds[-1]:.1f}s", flush=True)
        del model
        gc.collect()
    inner, outer = np.stack(inner_members), np.stack(outer_members)
    predictions = {
        "inner_3": inner[:3].mean(axis=0), "outer_3": outer[:3].mean(axis=0),
        "inner_6": inner.mean(axis=0), "outer_6": outer.mean(axis=0),
    }
    training = {
        "seeds": list(AUX_SEEDS), "best_iterations": iterations,
        "inner_seconds": inner_seconds, "outer_seconds": outer_seconds,
        "outer_member_mean_std": float(outer.std(axis=0).mean()),
    }
    return predictions, training


def main(train_path: Path, output: Path, task_type: str) -> None:
    common = load_module("full_pipeline_common", COMMON_PATH)
    screen = load_module("error_adjustment_screen", SCREEN_PATH)
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    pitcher = frame["pitcher_id"].to_numpy()
    contexts = screen.contexts(frame)
    recovered = frame[[ID_COL]].merge(pd.read_csv(LABEL_PATH), on=ID_COL, how="left")
    have = recovered["middle"].notna().to_numpy()
    mr_target = ((recovered["middle"].eq(1) | recovered["reverse"].eq(1)).fillna(False).astype(int).to_numpy())
    large_miss_target = ((target == 0) & (mr_target == 0)).astype(int)
    feature_module = common.load_features_module()
    success_oof = np.full(len(frame), np.nan, dtype=float)
    report = {
        "experiment": "auxiliary 3-vs-6 seed ensemble and train-only shift audit",
        "official_train_only": True, "test_aggregate_used": False,
        "success_seeds": list(common.SEEDS), "auxiliary_seeds": list(AUX_SEEDS),
        "pretraining": [], "fold_results": [],
    }

    for fold in (2021, 2022):
        train_mask, valid_mask = season < fold, season == fold
        features, cat_indices, global_mean = common.engineer(frame, feature_module, train_mask, target)
        prediction, iterations, seconds = common.train_inner_and_predict(
            features, cat_indices, target, np.ones(len(frame), dtype=bool), train_mask,
            valid_mask, BASE_CONFIG, task_type, f"source fold={fold} success",
        )
        success_oof[valid_mask] = prediction
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
        success_inner, success_iterations, success_inner_seconds = common.train_inner_and_predict(
            inner_x, inner_ci, target, np.ones(len(frame), dtype=bool), inner_train,
            calibration, BASE_CONFIG, task_type, f"fold={validation_year} success",
        )
        success_outer, success_outer_seconds = common.train_outer_and_predict(
            outer_x, outer_ci, target, np.ones(len(frame), dtype=bool), outer_train,
            validation, BASE_CONFIG, success_iterations, task_type, f"fold={validation_year} success",
        )
        success_oof[validation] = success_outer
        auxiliary, training = {}, {
            "success": {"best_iterations": success_iterations, "inner_seconds": success_inner_seconds, "outer_seconds": success_outer_seconds}
        }
        for name, labels in (("mr", mr_target), ("large_miss", large_miss_target)):
            auxiliary[name], training[name] = train_auxiliary(
                common, inner_x, inner_ci, outer_x, outer_ci, labels, have,
                inner_train, calibration, outer_train, validation, task_type,
                f"fold={validation_year} {name}",
            )

        residual = target.astype(float) - success_oof
        source = np.isin(season, (validation_year - 2, validation_year - 1))
        if np.isnan(residual[source]).any():
            raise RuntimeError(f"fold={validation_year} 차등표 원천 OOF 누락")
        additions = {}
        for name, shrinkage in screen.AXES:
            table = screen.differential_table(pitcher, contexts[name], residual, source, shrinkage)
            additions[name] = screen.apply_table(table, pitcher, contexts[name], validation)
        corrected_success = np.clip(
            success_outer + additions["hand"] + additions["two_strikes"] + additions["runners_on"],
            1e-6, 1 - 1e-6,
        )
        forecast = common.select_alpha_and_forecast(frame, validation_year)
        variants = {}
        for count in (3, 6):
            suffix = str(count)
            offset = common.fit_offset(
                success_inner, auxiliary["mr"][f"inner_{suffix}"],
                auxiliary["large_miss"][f"inner_{suffix}"], target[calibration], have[calibration],
            )
            inner_offset = common.apply_offset(
                success_inner, auxiliary["mr"][f"inner_{suffix}"],
                auxiliary["large_miss"][f"inner_{suffix}"], offset,
            )
            outer_offset = common.apply_offset(
                corrected_success, auxiliary["mr"][f"outer_{suffix}"],
                auxiliary["large_miss"][f"outer_{suffix}"], offset,
            )
            variants[count] = {
                "offset": offset, "outer_offset": outer_offset,
                "shift": common.fixed_shift(inner_offset, forecast["forecast"]),
            }
        baseline = common.sigmoid(common.logit(variants[3]["outer_offset"]) + variants[3]["shift"])
        candidates_raw = {
            "aux6_shared_shift": common.sigmoid(common.logit(variants[6]["outer_offset"]) + variants[3]["shift"]),
            "aux6_train_recomputed_shift": common.sigmoid(common.logit(variants[6]["outer_offset"]) + variants[6]["shift"]),
        }
        y_valid = target[validation]
        baseline_metrics = common.metrics(baseline, y_valid)
        candidates = []
        for name, prediction in candidates_raw.items():
            metrics = common.metrics(prediction, y_valid)
            candidates.append({
                "name": name, "metrics": metrics,
                "bss_delta_vs_aux3": metrics["score"] - baseline_metrics["score"],
                "absolute_mean_error_delta": abs(metrics["mean_error"]) - abs(baseline_metrics["mean_error"]),
            })
        report["fold_results"].append({
            "validation_year": validation_year, "training": training, "forecast": forecast,
            "aux3": {"metrics": baseline_metrics, "offset": variants[3]["offset"], "shift": variants[3]["shift"]},
            "aux6": {"offset": variants[6]["offset"], "shift": variants[6]["shift"]},
            "candidates": candidates,
        })
        write_json(output, report)
        del inner_x, outer_x, auxiliary
        gc.collect()

    summaries = []
    for name in ("aux6_shared_shift", "aux6_train_recomputed_shift"):
        rows = [next(item for item in fold["candidates"] if item["name"] == name) for fold in report["fold_results"]]
        deltas = [float(row["bss_delta_vs_aux3"]) for row in rows]
        errors = [float(row["absolute_mean_error_delta"]) for row in rows]
        summaries.append({
            "name": name, "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
            "mean_delta": float(np.mean(deltas)), "worst_delta": float(np.min(deltas)),
            "fold_2024_absolute_mean_error_delta": errors[1], "both_positive": bool(min(deltas) > 0),
        })
    stable = [row for row in summaries if row["both_positive"] and row["fold_2024_delta"] >= 3.0 and row["fold_2024_absolute_mean_error_delta"] <= 0.001]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"], report["selected"] = summaries, selected
    report["decision"] = "continue_auxiliary_six_seed_build" if selected else "keep_1029_champion"
    report["gate"] = "2023/2024 positive; 2024 >= +3; mean-error deterioration <= 0.001"
    write_json(output, report)
    print(json.dumps({"summaries": summaries, "selected": selected, "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
