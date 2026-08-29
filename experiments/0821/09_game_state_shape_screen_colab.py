"""경기 상태별 OOF 오차의 수준 제거 형태 신호를 시간 전방 선별한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL, TARGET_COL = "row_id", "control_success"
PAIRS = ((2022, 2023), (2023, 2024))
SHRINKAGES = (500.0, 1500.0, 3000.0)
SCALES = (0.10, 0.25, 0.50, 0.75)
BOOTSTRAPS = 500


def aligned_anchor(frame: pd.DataFrame, year: int, anchor_dir: Path):
    rows = frame.loc[frame["season"].astype(int).eq(year)].reset_index(drop=True)
    asset = np.load(anchor_dir / f"anchor_{year}.npz", allow_pickle=True)
    if len(rows) != len(asset["row_id"]):
        raise ValueError(f"{year} anchor 행 수 불일치")
    if not np.array_equal(rows[ID_COL].astype(str).to_numpy(), asset["row_id"].astype(str)):
        raise ValueError(f"{year} anchor row_id 순서 불일치")
    target = rows[TARGET_COL].astype(int).to_numpy()
    if not np.array_equal(target.astype(np.int8), asset["target"].astype(np.int8)):
        raise ValueError(f"{year} anchor 정답 불일치")
    return rows, target, asset["prediction"].astype(float)


def state_keys(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = lambda name: pd.to_numeric(rows[name], errors="coerce")
    inning = pd.cut(numeric("inning"), [-np.inf, 3, 6, np.inf], labels=("early", "middle", "late")).astype(str)
    score = pd.cut(numeric("score_diff_pitcher_team"), [-np.inf, -2, 2, np.inf],
                   labels=("behind", "close", "ahead")).astype(str)
    leverage = pd.cut(numeric("li"), [-np.inf, 0.7, 1.5, np.inf],
                      labels=("low", "medium", "high")).astype(str)
    outs = numeric("outs_before").fillna(-1).astype(int).astype(str)
    side = rows["top_bottom"].astype(str)
    bases = rows["base_state"].astype(str)
    league = rows["game_type"].astype(str)
    return pd.DataFrame({
        "inning_side_outs": league + "|" + inning + "|" + side + "|" + outs,
        "inning_score": league + "|" + inning + "|" + score,
        "bases_outs": league + "|" + bases + "|" + outs,
        "leverage_score": league + "|" + leverage + "|" + score,
        "inning_bases_leverage": league + "|" + inning + "|" + bases + "|" + leverage,
    })


def centered_lookup(source_rows: pd.DataFrame, source_key: pd.Series, residual: np.ndarray,
                    valid_key: pd.Series, shrinkage: float) -> tuple[np.ndarray, int]:
    league = source_rows["game_type"].astype(str).reset_index(drop=True)
    table = pd.DataFrame({"key": source_key.to_numpy(), "league": league, "residual": residual})
    parent = table.groupby("league", observed=True)["residual"].mean()
    grouped = table.groupby(["league", "key"], observed=True)["residual"].agg(["sum", "count"]).reset_index()
    grouped["difference"] = (
        grouped["sum"] - grouped["count"] * grouped["league"].map(parent)
    ) / (grouped["count"] + shrinkage)
    mapping = grouped.set_index("key")["difference"]
    return valid_key.map(mapping).fillna(0.0).to_numpy(float), int(len(grouped))


def bss(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    rate = float(np.mean(target))
    return float(1e5 * (1 - np.mean((prediction - target) ** 2) / (rate * (1 - rate))))


def pitcher_bootstrap(rows: pd.DataFrame, base: np.ndarray, candidate: np.ndarray,
                      target: np.ndarray, seed: int) -> float:
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"pitcher": rows["pitcher_id"].astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("pitcher", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, counts = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / counts[sample].sum() > 0)
    return float(positive / BOOTSTRAPS)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(train_path: Path, anchor_dir: Path, output: Path) -> None:
    frame = pd.read_csv(train_path, encoding="utf-8-sig")
    anchors = {year: aligned_anchor(frame, year, anchor_dir) for year in (2022, 2023, 2024)}
    folds = []
    for source_year, valid_year in PAIRS:
        source_rows, source_target, source_prediction = anchors[source_year]
        valid_rows, valid_target, valid_prediction = anchors[valid_year]
        source_keys, valid_keys = state_keys(source_rows), state_keys(valid_rows)
        residual = source_target - source_prediction
        base_score = bss(valid_prediction, valid_target)
        candidates = []
        for axis in source_keys.columns:
            for shrinkage in SHRINKAGES:
                correction, groups = centered_lookup(
                    source_rows, source_keys[axis], residual, valid_keys[axis], shrinkage
                )
                for scale in SCALES:
                    candidate = np.clip(valid_prediction + scale * correction, 1e-6, 1 - 1e-6)
                    applied = scale * correction
                    candidates.append({
                        "axis": axis, "shrinkage": shrinkage, "scale": scale, "groups": groups,
                        "bss_delta": bss(candidate, valid_target) - base_score,
                        "pitcher_bootstrap_probability": pitcher_bootstrap(
                            valid_rows, valid_prediction, candidate, valid_target,
                            seed=821000 + valid_year + int(shrinkage) + int(scale * 100),
                        ),
                        "correction_mean": float(applied.mean()),
                        "correction_std": float(applied.std()),
                        "residual_correlation": float(np.corrcoef(applied, valid_target - valid_prediction)[0, 1]),
                    })
        folds.append({"source_year": source_year, "valid_year": valid_year, "candidates": candidates})
        write_json(output, {"status": "running", "folds": folds})
        print(f"fold={valid_year} candidates={len(candidates)}", flush=True)

    summaries = []
    for axis in state_keys(anchors[2022][0]).columns:
        for shrinkage in SHRINKAGES:
            for scale in SCALES:
                rows = [next(item for item in fold["candidates"]
                             if item["axis"] == axis and item["shrinkage"] == shrinkage
                             and item["scale"] == scale) for fold in folds]
                deltas = [float(item["bss_delta"]) for item in rows]
                probabilities = [float(item["pitcher_bootstrap_probability"]) for item in rows]
                ratio = min(abs(value) for value in deltas) / max(abs(value) for value in deltas) if max(map(abs, deltas)) else 0.0
                passed = min(deltas) >= 1.0 and ratio >= 0.25 and min(probabilities) >= 0.80
                summaries.append({
                    "axis": axis, "shrinkage": shrinkage, "scale": scale,
                    "fold_2023_delta": deltas[0], "fold_2024_delta": deltas[1],
                    "worst_delta": min(deltas), "magnitude_ratio": ratio,
                    "minimum_pitcher_bootstrap_probability": min(probabilities),
                    "passed": bool(passed),
                })
    summaries.sort(key=lambda row: (row["passed"], row["worst_delta"], row["magnitude_ratio"]), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    report = {
        "experiment": "row-local game-state shape residual screen",
        "official_train_only": True, "test_aggregate_used": False,
        "level_centering": "source-season game_type mean only",
        "folds": folds, "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "continue_game_state_full_pipeline" if passed else "keep_r_scale0050_champion",
        "gate": "each fold >=+1, magnitude ratio >=0.25, pitcher bootstrap probability >=0.80",
    }
    write_json(output, report)
    print(json.dumps({"selected": report["selected"], "top": summaries[:10],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.train.resolve(), args.anchor_dir.resolve(), args.output.resolve())
