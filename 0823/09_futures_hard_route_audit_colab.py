"""현재 R 챔피언을 고정하고 Futures 행만 별도 strict 예측으로 교체한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_SOURCE = ROOT / "0822" / "02_failure_complement_champion_validation_colab.py"
YEARS = (2023, 2024)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate(directory: Path, year: int) -> Path:
    for pattern in (f"strict_f_regime075_oof_{year}.npz", f"*oof*{year}*.npz"):
        found = sorted(set(directory.rglob(pattern)))
        if found:
            return found[0]
    raise FileNotFoundError(f"{year} strict OOF를 찾지 못했습니다: {directory}")


def pick(asset, *names):
    for name in names:
        if name in asset.files:
            return asset[name]
    raise KeyError(f"필요한 키가 없습니다: {names}; 현재 키={asset.files}")


def load_route(directory: Path, year: int):
    path = locate(directory, year)
    asset = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "row_id": pick(asset, "row_id", "row_ids", "id").astype(str),
        "target": pick(asset, "target", "y", "control_success").astype(float),
        "shared_stack": pick(asset, "p_shared_stack").astype(float),
        "f_stack": pick(asset, "p_f_stack").astype(float),
        "model_only": pick(asset, "p_model_only").astype(float),
    }


def bootstrap(ids, base, candidate, target, seed, count=1000):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    grouped = pd.DataFrame({"id": ids.astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).agg({"gain": "sum", "n": "sum"})
    sums, rows = grouped["gain"].to_numpy(float), grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    positive = 0
    for _ in range(count):
        sample = rng.integers(0, len(grouped), len(grouped))
        positive += bool(sums[sample].sum() / rows[sample].sum() > 0)
    return float(positive / count)


def main(train_path: Path, component_dir: Path, route_source: Path,
         output: Path, task_type: str):
    module = load_module(CHAMPION_SOURCE, "champion_validation")
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    feature_module = module.load_features()
    league_rate = float(frame.loc[frame["season"].astype(int).lt(2022), "control_success"].mean())
    features = feature_module.engineer(
        frame.drop(columns=["row_id", "control_success"]), league_rate
    )
    for column in feature_module.CAT_COLS:
        features[column] = features[column].astype(str)
    cat_indices = [features.columns.get_loc(column) for column in feature_module.CAT_COLS]
    assets = {year: module.load_asset(component_dir, year) for year in (2022, 2023, 2024)}
    reports = []
    for calibration_year, validation_year in module.PAIRS:
        calibration, valid = assets[calibration_year], assets[validation_year]
        correction, valid_rows, seconds = module.train_correction(
            frame, features, cat_indices, calibration_year, validation_year,
            calibration, valid, task_type,
        )
        target = valid["target"].astype(float)
        anchor = valid["anchor"].astype(float)
        r_mask = valid_rows["game_type"].astype(str).eq("R").to_numpy()
        f_mask = valid_rows["game_type"].astype(str).eq("F").to_numpy()
        champion = module.sigmoid(module.logit(anchor) + module.VERIFIED_SHIFT_DELTA)
        champion[r_mask] = np.clip(
            champion[r_mask] + module.R_SCALE * correction, 1e-6, 1 - 1e-6
        )
        # 서버에서 확인된 R 실패여집합 0.20까지 현재 챔피언 parity에 포함한다.
        alignment_shift = module.shift_to_mean(
            calibration["failure_complement"].astype(float),
            float(calibration["anchor"].astype(float).mean()),
        )
        aligned = module.sigmoid(
            module.logit(valid["failure_complement"].astype(float)) + alignment_shift
        )
        rebuilt = anchor.copy()
        rebuilt[r_mask] = 0.80 * rebuilt[r_mask] + 0.20 * aligned[r_mask]
        champion = module.sigmoid(module.logit(rebuilt) + module.VERIFIED_SHIFT_DELTA)
        champion[r_mask] = np.clip(
            champion[r_mask] + module.R_SCALE * correction, 1e-6, 1 - 1e-6
        )

        route = load_route(route_source, validation_year)
        if not np.array_equal(valid_rows["row_id"].astype(str), route["row_id"]):
            raise ValueError(f"{validation_year} route row_id 정렬 불일치")
        if not np.allclose(target, route["target"], atol=0, rtol=0):
            raise ValueError(f"{validation_year} route target 정렬 불일치")
        base_all, base_f = module.bss(champion, target), module.bss(champion[f_mask], target[f_mask])
        candidates = []
        for name in ("shared_stack", "f_stack", "model_only"):
            candidate = champion.copy()
            candidate[f_mask] = np.clip(route[name][f_mask], 1e-6, 1 - 1e-6)
            r_delta = float(np.max(np.abs(candidate[r_mask] - champion[r_mask])))
            if r_delta != 0.0:
                raise ValueError(f"R행이 변경됐습니다: {r_delta}")
            candidates.append({
                "route": name,
                "overall_bss_delta": module.bss(candidate, target) - base_all,
                "f_bss_delta": module.bss(candidate[f_mask], target[f_mask]) - base_f,
                "f_pitcher_bootstrap_probability": bootstrap(
                    valid_rows.loc[f_mask, "pitcher_id"].to_numpy(), champion[f_mask],
                    candidate[f_mask], target[f_mask], 823900 + validation_year,
                ),
                "r_max_absolute_delta": r_delta,
                "f_prediction_mean_delta": float(candidate[f_mask].mean() - champion[f_mask].mean()),
            })
        reports.append({
            "year": validation_year,
            "rows": int(len(valid_rows)),
            "f_rows": int(f_mask.sum()),
            "r_training_seconds": seconds,
            "source": route["path"],
            "candidates": candidates,
        })
        print(f"year={validation_year} complete", flush=True)

    summaries = []
    for name in ("shared_stack", "f_stack", "model_only"):
        rows = [next(row for row in fold["candidates"] if row["route"] == name) for fold in reports]
        summaries.append({
            "route": name,
            "fold_2023_overall_delta": rows[0]["overall_bss_delta"],
            "fold_2024_overall_delta": rows[1]["overall_bss_delta"],
            "fold_2023_f_delta": rows[0]["f_bss_delta"],
            "fold_2024_f_delta": rows[1]["f_bss_delta"],
            "minimum_f_pitcher_bootstrap_probability": min(
                row["f_pitcher_bootstrap_probability"] for row in rows
            ),
            "passed": bool(
                min(row["overall_bss_delta"] for row in rows) >= 1
                and min(row["f_bss_delta"] for row in rows) >= 1
                and min(row["f_pitcher_bootstrap_probability"] for row in rows) >= 0.80
            ),
        })
    summaries.sort(key=lambda row: (row["passed"], min(
        row["fold_2023_overall_delta"], row["fold_2024_overall_delta"]
    )), reverse=True)
    passed = [row for row in summaries if row["passed"]]
    payload = {
        "experiment": "R champion plus Futures hard route audit",
        "official_train_only": True,
        "test_aggregate_used": False,
        "r_route": "current champion parity",
        "f_routes": ["shared_stack", "f_stack", "model_only"],
        "folds": reports,
        "summaries": summaries,
        "selected": passed[0] if passed else None,
        "decision": "reconstruct_selected_futures_route" if passed else "keep_current_champion",
        "gate": "R unchanged; 2023/2024 overall and F delta>=+1; F pitcher bootstrap>=0.80",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": payload["selected"], "summaries": summaries,
                      "decision": payload["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--route-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.train.resolve(), args.component_dir.resolve(), args.route_source.resolve(),
         args.output.resolve(), args.task_type)
