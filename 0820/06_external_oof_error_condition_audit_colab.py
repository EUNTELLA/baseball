"""참조 OOF와 기준 OOF의 오차 차이를 행 조건별로 진단한다.

참조 예측은 조건 탐색에만 사용하며 제출 예측, 보정값, ZIP에는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ID_COL, TARGET_COL = "row_id", "control_success"
YEARS = (2023, 2024)
SEEDS = (42, 7, 2024)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_PATH = SCRIPT_DIR / "04_dynamic_pitcher_baseline_residual_screen_colab.py"


def load_base():
    spec = importlib.util.spec_from_file_location("dynamic_base_screen", BASE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_frame(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_n = pd.to_numeric(rows.get("asof_pitcher_n"), errors="coerce").fillna(0)
    li = pd.to_numeric(rows.get("li"), errors="coerce")
    inning = pd.to_numeric(rows.get("inning"), errors="coerce")
    runners = pd.to_numeric(rows.get("num_runners_on"), errors="coerce").fillna(0)
    balls = pd.to_numeric(rows.get("balls_before"), errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(rows.get("strikes_before"), errors="coerce").fillna(-1).astype(int)
    pitcher_hand = rows.get("pitcher_hand", pd.Series("NA", index=rows.index)).astype(str)
    batter_hand = rows.get("batter_hand", pd.Series("NA", index=rows.index)).astype(str)
    return pd.DataFrame({
        "game_type": "game_type=" + rows.get("game_type", pd.Series("NA", index=rows.index)).astype(str),
        "count": "count=" + balls.astype(str) + "-" + strikes.astype(str),
        "hand_match": "hand=" + np.where(pitcher_hand.eq(batter_hand), "same", "opposite"),
        "runners": "runners=" + np.where(runners.gt(0), "on", "empty"),
        "inning_band": "inning=" + pd.cut(inning, [-np.inf, 3, 6, np.inf], labels=["early", "middle", "late"]).astype(str),
        "history_band": "history=" + pd.cut(numeric_n, [-np.inf, 99, 399, 999, np.inf], labels=["0-99", "100-399", "400-999", "1000+"]).astype(str),
        "leverage_band": "leverage=" + pd.cut(li, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "medium", "high"]).astype(str),
        "pitcher_hand": "pitcher_hand=" + pitcher_hand,
        "batter_hand": "batter_hand=" + batter_hand,
    }, index=rows.index)


def aggregate(conditions: pd.DataFrame, advantage: np.ndarray, year: int) -> list[dict]:
    records = []
    for axis in conditions.columns:
        values = conditions[axis].fillna(f"{axis}=missing").astype(str).to_numpy()
        for value in np.unique(values):
            mask = values == value
            if mask.sum() < 3000:
                continue
            mean = float(np.mean(advantage[mask]))
            records.append({
                "year": year, "axis": axis, "condition": value,
                "rows": int(mask.sum()), "coverage": float(mask.mean()),
                "external_brier_advantage": mean,
                "bss_equivalent_advantage": float(1e5 * mean / 0.25),
                "external_row_win_rate": float(np.mean(advantage[mask] > 0)),
            })
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, external_root: Path, output: Path, task_type: str) -> None:
    anchor_path = external_root / "evaluation" / "anchors" / "adaptive_gate.npz"
    if not anchor_path.exists():
        raise FileNotFoundError(f"외부 진단 앵커가 없습니다: {anchor_path}")
    anchor = np.load(anchor_path, allow_pickle=False)
    base = load_base()
    feature_module = base.load_features_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    target = frame[TARGET_COL].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    report = {
        "experiment": "reference OOF row-condition error audit",
        "diagnostic_only": True, "official_train_only_for_our_models": True,
        "external_predictions_used_in_submission": False,
        "external_fixed_coefficients_copied": False,
        "anchor": str(anchor_path), "folds": [], "condition_rows": [],
    }

    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        rows = frame.loc[valid_mask].reset_index(drop=True)
        y = target[valid_mask]
        external_y = np.asarray(anchor[f"y{str(year)[-2:]}"])
        external_p = np.asarray(anchor[f"p{str(year)[-2:]}"])
        external_pitcher = np.asarray(anchor[f"pitcher{str(year)[-2:]}"]).astype(str)
        local_pitcher = rows["pitcher_id"].astype(str).to_numpy()
        if len(y) != len(external_y) or not np.array_equal(y.astype(np.float32), external_y.astype(np.float32)):
            raise ValueError(f"{year} 참조 OOF target 행 정렬이 공식 Train과 다릅니다.")
        if not np.array_equal(local_pitcher, external_pitcher):
            raise ValueError(f"{year} 참조 OOF pitcher_id 행 정렬이 공식 Train과 다릅니다.")

        league_rate = float(target[train_mask].mean())
        x = feature_module.engineer(frame.drop(columns=[ID_COL, TARGET_COL]), league_rate)
        for column in feature_module.CAT_COLS:
            x[column] = x[column].astype(str)
        cat_indices = [x.columns.get_loc(column) for column in feature_module.CAT_COLS]
        train_pool = Pool(x.loc[train_mask], target[train_mask], cat_features=cat_indices)
        valid_pool = Pool(x.loc[valid_mask], target[valid_mask], cat_features=cat_indices)
        members, iterations = [], []
        for seed in SEEDS:
            model = CatBoostClassifier(**base.classifier_params(seed, task_type))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            members.append(model.predict_proba(valid_pool)[:, 1])
            iterations.append(max(1, int(model.get_best_iteration()) + 1))
            del model
            gc.collect()
        ours = np.mean(members, axis=0)
        advantage = (ours - y) ** 2 - (external_p - y) ** 2
        condition_rows = aggregate(condition_frame(rows), advantage, year)
        report["condition_rows"].extend(condition_rows)
        report["folds"].append({
            "year": year, "rows": int(len(y)), "alignment_verified": True,
            "our_best_iterations": iterations,
            "our_brier": float(np.mean((ours - y) ** 2)),
            "external_diagnostic_brier": float(np.mean((external_p - y) ** 2)),
            "external_brier_advantage": float(np.mean(advantage)),
        })
        write_json(output, report)
        del x, train_pool, valid_pool
        gc.collect()

    table = pd.DataFrame(report["condition_rows"])
    pivot = table.pivot_table(index=["axis", "condition"], columns="year", values="bss_equivalent_advantage")
    counts = table.groupby(["axis", "condition"])["rows"].min()
    stable = pivot.dropna().copy()
    stable["worst_bss_equivalent_advantage"] = stable[list(YEARS)].min(axis=1)
    stable["mean_bss_equivalent_advantage"] = stable[list(YEARS)].mean(axis=1)
    stable["minimum_rows"] = counts
    stable = stable[stable["worst_bss_equivalent_advantage"] > 0].sort_values(
        ["worst_bss_equivalent_advantage", "minimum_rows"], ascending=False
    )
    report["stable_external_advantage_conditions"] = [
        {"axis": idx[0], "condition": idx[1], **{str(k): float(v) for k, v in row.items()}}
        for idx, row in stable.head(20).iterrows()
    ]
    report["decision"] = "design_independent_features_from_stable_conditions" if len(stable) else "no_stable_condition_found"
    write_json(output, report)
    print(json.dumps({
        "folds": report["folds"],
        "top_stable_conditions": report["stable_external_advantage_conditions"][:10],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.external_root.resolve(), args.output.resolve(), args.task_type)
