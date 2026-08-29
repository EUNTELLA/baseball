"""투수·타자 as-of와 상황 prior를 결합한 계층형 기본 예측을 전방 검증한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import Ridge

ID_COL, TARGET_COL = "row_id", "control_success"
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 7, 2024)
ALPHAS = (100.0, 1000.0, 10000.0)
BLENDS = (0.10, 0.20, 0.30, 0.40, 0.50)
CONTEXT_STRENGTHS = (100.0, 500.0, 1500.0)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("direct_catboost_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(frame: pd.DataFrame, column: str, fallback: float) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(fallback).to_numpy(float)


def smooth_rate(frame: pd.DataFrame, entity: str, league: float, strength: float) -> np.ndarray:
    n = np.clip(number(frame, f"asof_{entity}_n", 0.0), 0.0, None)
    rate = np.clip(number(frame, f"asof_{entity}_success_rate", league), 0.0, 1.0)
    return (n * rate + strength * league) / (n + strength)


def context_key(frame: pd.DataFrame) -> pd.Series:
    count = (pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
             + "-" + pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str))
    same = np.where(frame["pitcher_hand"].astype(str).eq(frame["batter_hand"].astype(str)), "same", "opposite")
    return frame["game_type"].astype(str) + "|" + count + "|" + pd.Series(same, index=frame.index)


def context_prior(key: pd.Series, target: np.ndarray, train_mask: np.ndarray,
                  valid_mask: np.ndarray, league: float, strength: float) -> tuple[np.ndarray, np.ndarray]:
    table = pd.DataFrame({"key": key.loc[train_mask].to_numpy(), "y": target[train_mask]}).groupby("key").y.agg(["sum", "count"])
    sums = key.loc[train_mask].map(table["sum"]).to_numpy(float)
    counts = key.loc[train_mask].map(table["count"]).to_numpy(float)
    train_prior = (sums - target[train_mask] + strength * league) / (counts - 1.0 + strength)
    valid_sum = key.loc[valid_mask].map(table["sum"]).fillna(0.0).to_numpy(float)
    valid_count = key.loc[valid_mask].map(table["count"]).fillna(0.0).to_numpy(float)
    valid_prior = (valid_sum + strength * league) / (valid_count + strength)
    return train_prior, valid_prior


def hierarchy_matrix(frame: pd.DataFrame, league: float, context: np.ndarray) -> np.ndarray:
    p30, p100 = smooth_rate(frame, "pitcher", league, 30.0), smooth_rate(frame, "pitcher", league, 100.0)
    b30, b100 = smooth_rate(frame, "batter", league, 30.0), smooth_rate(frame, "batter", league, 100.0)
    career = number(frame, "asof_pitcher_success_rate", league)
    recent1 = number(frame, "asof_pitcher_prev1_game_success_rate", league)
    recent3 = number(frame, "asof_pitcher_prev3_game_success_rate", league)
    recent5 = number(frame, "asof_pitcher_prev5_game_success_rate", league)
    return np.column_stack([
        p30, p100, b30, b100, context,
        recent1 - career, recent3 - career, recent5 - career,
        np.log1p(np.clip(number(frame, "asof_pitcher_n", 0.0), 0.0, None)),
        number(frame, "li", 1.0), number(frame, "score_diff_pitcher_team", 0.0),
        np.ones(len(frame)),
    ])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, output: Path, task_type: str) -> None:
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    keys = context_key(frame)
    report = {"experiment": "independent hierarchical prior Ridge stack",
              "official_train_only": True, "external_code_predictions_or_coefficients_used": False,
              "test_aggregate_used": False, "fold_results": []}

    for fold in FOLDS:
        train_mask, valid_mask = season < fold, season == fold
        league = float(target[train_mask].mean())
        features = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league)
        for column in feature_module.CAT_COLS:
            features[column] = features[column].astype(str)
        cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
        train_pool = Pool(features.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool = Pool(features.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        direct_members, iterations = [], []
        for seed in SEEDS:
            model = CatBoostClassifier(**base.classifier_params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            direct_members.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            del model
            gc.collect()
        direct = np.mean(direct_members, axis=0)
        baseline_metrics = base.metric(direct, target[valid_mask])
        candidates = []
        for context_strength in CONTEXT_STRENGTHS:
            train_context, valid_context = context_prior(keys, target, train_mask, valid_mask, league, context_strength)
            context_all = np.full(len(frame), league, dtype=float)
            context_all[train_mask], context_all[valid_mask] = train_context, valid_context
            matrix = hierarchy_matrix(frame, league, context_all)
            train_x, valid_x = matrix[train_mask], matrix[valid_mask]
            mean, std = train_x.mean(axis=0), train_x.std(axis=0)
            std[std == 0] = 1.0
            train_z, valid_z = (train_x - mean) / std, (valid_x - mean) / std
            weights = np.power(0.55, (fold - 1) - season[train_mask])
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(train_z, target[train_mask], sample_weight=weights)
                hierarchical = np.clip(model.predict(valid_z), 1e-6, 1 - 1e-6)
                for blend in BLENDS:
                    prediction = np.clip((1.0 - blend) * direct + blend * hierarchical, 1e-6, 1 - 1e-6)
                    metrics = base.metric(prediction, target[valid_mask])
                    candidates.append({"context_strength": context_strength, "alpha": alpha, "blend": blend,
                                       "hierarchical_metrics": base.metric(hierarchical, target[valid_mask]),
                                       "metrics": metrics, "bss_delta": metrics["bss_score"] - baseline_metrics["bss_score"],
                                       "error_correlation": float(np.corrcoef(target[valid_mask] - direct,
                                                                              target[valid_mask] - hierarchical)[0, 1])})
        report["fold_results"].append({"fold": fold, "direct_best_iterations": iterations,
                                       "baseline": baseline_metrics, "candidates": candidates})
        print(f"fold={fold} direct_iter={iterations} candidates={len(candidates)}", flush=True)
        write_json(output, report)
        del features, train_pool, valid_pool
        gc.collect()

    summaries = []
    for strength in CONTEXT_STRENGTHS:
        for alpha in ALPHAS:
            for blend in BLENDS:
                rows = [next(row for row in fold["candidates"] if row["context_strength"] == strength
                             and row["alpha"] == alpha and row["blend"] == blend)
                        for fold in report["fold_results"]]
                deltas = [float(row["bss_delta"]) for row in rows]
                summaries.append({"context_strength": strength, "alpha": alpha, "blend": blend,
                                  "fold_2022_delta": deltas[0], "fold_2023_delta": deltas[1],
                                  "fold_2024_delta": deltas[2], "mean_delta": float(np.mean(deltas)),
                                  "worst_delta": float(np.min(deltas)), "all_positive": bool(min(deltas) > 0)})
    stable = [row for row in summaries if row["all_positive"] and row["fold_2024_delta"] >= 5.0]
    selected = max(stable, key=lambda row: (row["worst_delta"], row["mean_delta"])) if stable else None
    report["summaries"] = sorted(summaries, key=lambda row: (row["worst_delta"], row["mean_delta"]), reverse=True)
    report["selected"] = selected
    report["decision"] = "continue_hierarchical_stack_full_pipeline" if selected else "reject_hierarchical_prior_stack"
    report["gate"] = "same setting positive in 2022/2023/2024 and 2024 >= +5"
    write_json(output, report)
    print(json.dumps({"selected": selected, "top": report["summaries"][:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.output.resolve(), args.task_type)
