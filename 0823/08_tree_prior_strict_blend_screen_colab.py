"""자체 Tree-Prior 모델과 저장 anchor의 strict-forward R/F 혼합을 검증한다."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


YEARS = (2022, 2023, 2024)
WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
BOOTSTRAPS = 500
DROP = {"control_success", "row_id", "season"}
COMPACT_COLUMNS = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "game_type",
    "pitcher_hand", "batter_hand", "balls_before", "strikes_before", "inning",
    "outs", "num_runners_on", "score_diff", "li", "top_bottom",
    "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_batter_n", "asof_batter_success_rate",
)


def bss(prediction, target):
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, float)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def load_anchor(directory: Path, year: int):
    patterns = (f"strict_f_regime075_oof_{year}.npz", f"*oof*{year}*.npz")
    paths = []
    for pattern in patterns:
        paths.extend(directory.rglob(pattern))
        if paths:
            break
    if not paths:
        raise FileNotFoundError(f"{year} strict OOF를 찾지 못했습니다: {directory}")
    asset = np.load(sorted(set(paths))[0], allow_pickle=True)
    def pick(*names):
        for name in names:
            if name in asset.files:
                return asset[name]
        raise KeyError(f"필요한 키가 없습니다: {names}; 현재 키={asset.files}")
    return {
        "row_id": pick("row_id", "row_ids", "id"),
        "target": pick("target", "y", "control_success").astype(float),
        "anchor": pick("p_model_only").astype(float),
    }


def numeric_frame(frame: pd.DataFrame, columns=None, medians=None):
    source = frame.drop(columns=[c for c in DROP if c in frame], errors="ignore")
    if columns is None:
        columns = list(source.columns)
    result = pd.DataFrame(index=source.index)
    for column in columns:
        value = source[column] if column in source else pd.Series(np.nan, index=source.index)
        if pd.api.types.is_numeric_dtype(value):
            result[column] = pd.to_numeric(value, errors="coerce")
        else:
            # 문자열 자체의 결정적 해시만 사용하며 검증행 집계나 빈도는 사용하지 않는다.
            result[column] = pd.util.hash_pandas_object(
                value.astype("string").fillna("__MISSING__"), index=False
            ).to_numpy(np.uint64) % np.uint64(1_000_003)
    result = result.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = result.median(axis=0).fillna(0.0)
    result = result.fillna(medians).astype(np.float32)
    return result, columns, medians


def pitcher_prior(rows: pd.DataFrame, global_rate: float, strength: float = 200.0):
    n = pd.to_numeric(rows.get("asof_pitcher_n", 0), errors="coerce").fillna(0).clip(lower=0)
    rate = pd.to_numeric(
        rows.get("asof_pitcher_success_rate", global_rate), errors="coerce"
    ).fillna(global_rate).clip(0, 1)
    return ((rate.to_numpy(float) * n.to_numpy(float) + global_rate * strength)
            / (n.to_numpy(float) + strength))


def train_alternative(raw: pd.DataFrame, year: int, estimators: int, workers: int,
                      max_train_rows: int):
    train = raw.loc[raw["season"].astype(int).lt(year)].reset_index(drop=True)
    valid = raw.loc[raw["season"].astype(int).eq(year)].reset_index(drop=True)
    if max_train_rows > 0 and len(train) > max_train_rows:
        # 정답 비율을 유지한 결정적 표본이며 검증연도 정보는 사용하지 않는다.
        train = train.groupby("control_success", group_keys=False, observed=True).apply(
            lambda part: part.sample(
                n=max(1, round(max_train_rows * len(part) / len(train))),
                random_state=823000 + year,
            )
        ).sort_index().head(max_train_rows).reset_index(drop=True)
    target = train["control_success"].to_numpy(np.int8)
    global_rate = float(target.mean())
    available = [column for column in COMPACT_COLUMNS if column in train.columns]
    x_train, columns, medians = numeric_frame(train, available)
    x_valid, _, _ = numeric_frame(valid, columns, medians)
    model = ExtraTreesClassifier(
        n_estimators=estimators, max_depth=14, min_samples_leaf=100,
        max_features=0.8, class_weight=None, bootstrap=False,
        random_state=823042 + year, n_jobs=workers,
    )
    started = time.perf_counter()
    model.fit(x_train, target)
    tree_probability = model.predict_proba(x_valid)[:, 1]
    prior_probability = pitcher_prior(valid, global_rate)
    # 서로 다른 두 자체 추정량을 고정 비율로 합쳐 두 번째 모델을 정의한다.
    prediction = np.clip(0.75 * tree_probability + 0.25 * prior_probability, 1e-6, 1 - 1e-6)
    return valid, prediction, float(time.perf_counter() - started), len(columns)


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


def choose_weight(records, region):
    eligible = []
    for weight in WEIGHTS:
        deltas = [row[region][str(weight)]["delta"] for row in records]
        if min(deltas) >= 0:
            eligible.append((float(np.mean(deltas)), weight))
    if not eligible:
        return 0.0
    best = max(value for value, _ in eligible)
    # 최고 과거 개선의 90%를 달성하는 가장 작은 혼합비를 선택한다.
    return float(min(weight for value, weight in eligible if value >= 0.90 * best))


def main(train_path: Path, anchor_source: Path, output: Path,
         estimators: int, workers: int, max_train_rows: int):
    raw = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    folds = []
    predictions = {}
    for year in YEARS:
        anchor = load_anchor(anchor_source, year)
        valid, alternative, seconds, feature_count = train_alternative(
            raw, year, estimators, workers, max_train_rows
        )
        predictions[year] = alternative
        if not np.array_equal(valid["row_id"].astype(str), anchor["row_id"].astype(str)):
            raise ValueError(f"{year} row_id 정렬 불일치")
        target = anchor["target"].astype(float)
        base = anchor["anchor"].astype(float)
        fold = {"year": year, "seconds": seconds, "feature_count": feature_count}
        for region in ("R", "F"):
            mask = valid["game_type"].astype(str).eq(region).to_numpy()
            base_score = bss(base[mask], target[mask])
            region_result = {}
            for weight in WEIGHTS:
                candidate = (1 - weight) * base[mask] + weight * alternative[mask]
                region_result[str(weight)] = {
                    "delta": bss(candidate, target[mask]) - base_score,
                    "rows": int(mask.sum()),
                }
            fold[region] = region_result
        folds.append(fold)
        print(f"year={year} complete sec={seconds:.1f}", flush=True)

    selection_source = [row for row in folds if row["year"] in (2022, 2023)]
    selected = {region: choose_weight(selection_source, region) for region in ("R", "F")}
    audit = next(row for row in folds if row["year"] == 2024)
    valid_2024 = raw.loc[raw["season"].astype(int).eq(2024)].reset_index(drop=True)
    anchor_2024 = load_anchor(anchor_source, 2024)
    alternative_2024 = predictions[2024]
    target_2024 = anchor_2024["target"].astype(float)
    base_2024 = anchor_2024["anchor"].astype(float)
    candidate_2024 = base_2024.copy()
    for region, weight in selected.items():
        mask = valid_2024["game_type"].astype(str).eq(region).to_numpy()
        candidate_2024[mask] = (
            (1 - weight) * base_2024[mask] + weight * alternative_2024[mask]
        )
    delta_2024 = bss(candidate_2024, target_2024) - bss(base_2024, target_2024)
    probability = bootstrap(
        valid_2024["pitcher_id"].to_numpy(), base_2024, candidate_2024,
        target_2024, 823500,
    )
    report = {
        "experiment": "own tree-prior strict-forward blend",
        "official_train_only": True,
        "test_aggregate_used": False,
        "selection_years": [2022, 2023],
        "audit_year": 2024,
        "weights": list(WEIGHTS),
        "estimators": estimators,
        "max_train_rows": max_train_rows,
        "folds": folds,
        "selected_weights": selected,
        "audit_2024": {
            "bss_delta": delta_2024,
            "pitcher_bootstrap_probability": probability,
            "R_delta": audit["R"][str(selected["R"])]["delta"],
            "F_delta": audit["F"][str(selected["F"])]["delta"],
        },
        "decision": (
            "continue_tree_prior_full_pipeline"
            if delta_2024 >= 1 and probability >= 0.80 else "keep_current_champion"
        ),
        "gate": "weights selected on 2022-2023 only; 2024 delta>=+1 and bootstrap>=0.80",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_weights": selected, "audit_2024": report["audit_2024"],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--anchor-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=40)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=400000)
    args = parser.parse_args()
    main(args.train.resolve(), args.anchor_source.resolve(), args.output.resolve(),
         args.estimators, args.workers, args.max_train_rows)
