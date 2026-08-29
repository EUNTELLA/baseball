"""두 strict-forward OOF anchor를 연도 이전 정보만으로 비교·혼합한다."""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = (2022, 2023, 2024)
R_WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
F_WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
BOOTSTRAPS = 1000


def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    rate = float(y.mean())
    return float(1e5 * (1 - np.mean((y - p) ** 2) / (rate * (1 - rate))))


def pick(asset, names, required=True):
    for name in names:
        if name in asset.files:
            return asset[name]
    if required:
        raise KeyError(f"필요한 키가 없습니다: {names}; 현재 키={asset.files}")
    return None


def locate(directory: Path, year: int) -> Path:
    patterns = (
        f"strict_f_regime075_oof_{year}.npz",
        f"*oof*{year}*.npz",
    )
    found = []
    for pattern in patterns:
        found.extend(directory.rglob(pattern))
        if found:
            break
    if not found:
        raise FileNotFoundError(f"{year} strict OOF NPZ를 찾지 못했습니다: {directory}")
    return sorted(set(found))[0]


def load_alternate(directory: Path, year: int):
    path = locate(directory, year)
    asset = np.load(path, allow_pickle=True)
    row_id = pick(asset, ("row_id", "row_ids", "id"))
    target = pick(asset, ("target", "y", "control_success"))
    game_type = pick(asset, ("game_type", "league", "type"), required=False)
    model_only = pick(asset, ("p_model_only",))
    deployment = pick(asset, ("p_deployment",), required=False)
    shared = pick(asset, ("p_shared_stack",), required=False)
    futures = pick(asset, ("p_f_stack",), required=False)
    if game_type is None:
        grouped = None
    elif shared is not None and futures is not None:
        grouped = np.where(np.asarray(game_type).astype(str) == "F", futures, shared)
    else:
        grouped = None
    return {
        "path": str(path), "row_id": row_id, "target": np.asarray(target, float),
        "game_type": None if game_type is None else np.asarray(game_type).astype(str),
        "model_only": np.asarray(model_only, float),
        "deployment": None if deployment is None else np.asarray(deployment, float),
        "group_anchor": None if grouped is None else np.asarray(grouped, float),
    }


def load_current(directory: Path, year: int):
    path = directory / f"components_{year}.npz"
    asset = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "row_id": pick(asset, ("row_id", "row_ids", "id")),
        "target": np.asarray(pick(asset, ("target", "y", "control_success")), float),
        "game_type": np.asarray(pick(asset, ("game_type", "league", "type"))).astype(str),
        "prediction": np.asarray(pick(asset, ("anchor", "prediction", "p")), float),
    }


def metric_rows(year, y, game_type, predictions):
    rows = []
    for region in ("all", "R", "F"):
        mask = np.ones(len(y), bool) if region == "all" else game_type == region
        for name, prediction in predictions.items():
            if prediction is None:
                continue
            rows.append({
                "year": year, "region": region, "model": name,
                "rows": int(mask.sum()), "brier": float(np.mean((y[mask] - prediction[mask]) ** 2)),
                "bss": bss(y[mask], prediction[mask]),
                "prediction_mean": float(prediction[mask].mean()),
                "target_mean": float(y[mask].mean()),
            })
    return rows


def choose_weight(years, data, region, grid):
    rows = []
    for weight in grid:
        deltas = []
        for year in years:
            fold = data[year]
            mask = fold["game_type"] == region
            prediction = (1 - weight) * fold["current"][mask] + weight * fold["alternate"][mask]
            deltas.append(bss(fold["target"][mask], prediction) - bss(fold["target"][mask], fold["current"][mask]))
        objective = float(np.mean(deltas) + 0.35 * np.min(deltas))
        rows.append({
            "weight": weight, "objective": objective,
            "mean_delta_bss": float(np.mean(deltas)), "worst_delta_bss": float(np.min(deltas)),
            **{f"delta_{year}": delta for year, delta in zip(years, deltas)},
        })
    rows.sort(key=lambda row: (row["objective"], -row["weight"]), reverse=True)
    return float(rows[0]["weight"]), rows


def blend_by_group(fold, r_weight, f_weight):
    weights = np.where(fold["game_type"] == "F", f_weight, r_weight)
    return (1 - weights) * fold["current"] + weights * fold["alternate"]


def bootstrap(ids, y, base, candidate, seed=82301):
    gain = (base - y) ** 2 - (candidate - y) ** 2
    grouped = pd.DataFrame({"id": ids.astype(str), "gain": gain, "n": 1})
    grouped = grouped.groupby("id", observed=True).sum()
    sums = grouped["gain"].to_numpy(float)
    counts = grouped["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(grouped), size=(BOOTSTRAPS, len(grouped)))
    values = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {
        "positive_probability": float(np.mean(values > 0)),
        "mean_brier_gain": float(values.mean()),
        "ci95_brier_gain": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(strict_dir: Path, component_dir: Path, train_path: Path, output: Path):
    train = pd.read_csv(train_path, usecols=["row_id", "season", "pitcher_id"], low_memory=False)
    data, metrics, correlations = {}, [], []
    for year in YEARS:
        alternate = load_alternate(strict_dir, year)
        current = load_current(component_dir, year)
        if not np.array_equal(current["row_id"].astype(str), alternate["row_id"].astype(str)):
            raise ValueError(f"{year} row_id 정렬 불일치")
        if not np.allclose(current["target"], alternate["target"]):
            raise ValueError(f"{year} target 불일치")
        game_type = alternate["game_type"] if alternate["game_type"] is not None else current["game_type"]
        candidate = alternate["model_only"]
        data[year] = {
            "target": current["target"], "game_type": game_type,
            "current": current["prediction"], "alternate": candidate,
        }
        predictions = {
            "current_anchor": current["prediction"], "alternate_model_only": candidate,
            "alternate_deployment": alternate["deployment"],
            "alternate_group_anchor": alternate["group_anchor"],
        }
        metrics.extend(metric_rows(year, current["target"], game_type, predictions))
        for region in ("all", "R", "F"):
            mask = np.ones(len(game_type), bool) if region == "all" else game_type == region
            correlations.append({
                "year": year, "region": region,
                "error_correlation": float(np.corrcoef(
                    current["target"][mask] - current["prediction"][mask],
                    current["target"][mask] - candidate[mask],
                )[0, 1]),
            })

    strict = []
    for validation_year, selection_years in ((2023, (2022,)), (2024, (2022, 2023))):
        r_weight, r_grid = choose_weight(selection_years, data, "R", R_WEIGHTS)
        f_weight, f_grid = choose_weight(selection_years, data, "F", F_WEIGHTS)
        fold = data[validation_year]
        prediction = blend_by_group(fold, r_weight, f_weight)
        strict.append({
            "validation_year": validation_year, "selection_years": list(selection_years),
            "selected_r_weight": r_weight, "selected_f_weight": f_weight,
            "current_bss": bss(fold["target"], fold["current"]),
            "blend_bss": bss(fold["target"], prediction),
            "bss_delta": bss(fold["target"], prediction) - bss(fold["target"], fold["current"]),
            "r_weight_grid": r_grid, "f_weight_grid": f_grid,
        })

    final_r_weight, final_r_grid = choose_weight(YEARS, data, "R", R_WEIGHTS)
    final_f_weight, final_f_grid = choose_weight(YEARS, data, "F", F_WEIGHTS)
    latest = data[2024]
    latest_prediction = blend_by_group(
        latest, strict[-1]["selected_r_weight"], strict[-1]["selected_f_weight"]
    )
    latest_rows = train.loc[train["season"].astype(int).eq(2024)].reset_index(drop=True)
    report = {
        "experiment": "strict-forward alternate anchor comparison",
        "official_train_only": True, "test_aggregate_used": False,
        "metrics": metrics, "error_correlations": correlations,
        "strict_year_selection": strict,
        "latest_year_pitcher_bootstrap": bootstrap(
            latest_rows["pitcher_id"].to_numpy(), latest["target"], latest["current"], latest_prediction
        ),
        "deployment_weight_recommendation": {
            "r_weight": final_r_weight, "f_weight": final_f_weight,
            "method": "mean annual delta BSS + 0.35 * worst annual delta BSS",
            "r_grid": final_r_grid, "f_grid": final_f_grid,
        },
    }
    deltas = [row["bss_delta"] for row in strict]
    probability = report["latest_year_pitcher_bootstrap"]["positive_probability"]
    report["decision"] = (
        "continue_alternate_anchor_deployment"
        if min(deltas) > 0 and probability >= 0.80 and (final_r_weight > 0 or final_f_weight > 0)
        else "keep_current_champion"
    )
    report["gate"] = "2023/2024 strict delta positive, 2024 pitcher bootstrap>=0.80, final weight>0"
    write_json(output, report)
    print(json.dumps({
        "strict_year_selection": strict,
        "latest_year_pitcher_bootstrap": report["latest_year_pitcher_bootstrap"],
        "deployment_r_weight": final_r_weight, "deployment_f_weight": final_f_weight,
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


def main(strict_source: Path, component_dir: Path, train_path: Path, output: Path):
    if strict_source.is_dir():
        run(strict_source, component_dir, train_path, output)
        return
    with tempfile.TemporaryDirectory(prefix="strict_anchor_") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(strict_source) as archive:
            archive.extractall(directory)
        run(directory, component_dir, train_path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-source", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.strict_source.resolve(), args.component_dir.resolve(), args.train.resolve(), args.output.resolve())
