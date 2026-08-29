"""F general route와 F-regime route의 strict-forward 혼합비를 고른다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENERAL_PATH = ROOT / "train_f" / "02_general_route_reconstruction_colab.py"
YEARS = (2023, 2024)
WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
SEEDS = (42, 7, 2024, 99, 1, 123)


def load_module():
    spec = importlib.util.spec_from_file_location("general_route", GENERAL_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(GENERAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path):
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def find_strict(directory: Path, year: int):
    matches = sorted(directory.rglob(f"strict_f_regime075_oof_{year}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{year} strict OOF 개수={len(matches)}: {directory}")
    return matches[0]


def bss(target, prediction):
    target = np.asarray(target, float)
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    return float(1e5 * (1 - np.mean((target - prediction) ** 2) / (rate * (1 - rate))))


def bootstrap(pitcher, target, baseline, candidate, seed, repeats=500):
    table = pd.DataFrame({
        "pitcher": pd.Series(pitcher).astype(str),
        "gain": (np.asarray(target, float) - np.asarray(baseline, float)) ** 2
                - (np.asarray(target, float) - np.asarray(candidate, float)) ** 2,
    }).groupby("pitcher", observed=True).gain.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    values = table.to_numpy(float)
    gains = []
    for _ in range(repeats):
        sampled = values[rng.integers(0, len(values), len(values))]
        gains.append(float(sampled[:, 0].sum() / sampled[:, 1].sum()))
    return float(np.mean(np.asarray(gains) > 0))


def pick(asset, keys):
    for key in keys:
        if key in asset:
            return np.asarray(asset[key])
    raise KeyError(f"필요 키 없음: {keys}; 실제={sorted(asset)}")


def run(strict_dir: Path, component_dir: Path, train_path: Path,
        output: Path, task_type: str):
    general = load_module()
    frame = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    target = frame["control_success"].astype(int).to_numpy()
    season = frame["season"].astype(int).to_numpy()
    feature_module = general.load_features()
    folds = []

    for year in YEARS:
        raw, shift, iterations, seconds = general.train_fold(
            frame, target, season, year, SEEDS, task_type, feature_module, 128
        )
        general_prediction = general.sigmoid(general.logit(raw) + shift)
        rows = frame.loc[season == year].reset_index(drop=True)
        component = general.load_component(component_dir, year)
        strict = load_npz(find_strict(strict_dir, year))
        component_ids = component["row_id"].astype(str)
        strict_ids = pick(strict, ("row_id", "row_ids")).astype(str)
        if not np.array_equal(rows["row_id"].astype(str), component_ids):
            raise ValueError(f"{year} component row_id 불일치")
        if not np.array_equal(component_ids, strict_ids):
            raise ValueError(f"{year} strict row_id 불일치")
        y = component["target"].astype(float)
        strict_prediction = pick(strict, ("p_f_stack",)).astype(float)
        mask = rows["game_type"].astype(str).eq("F").to_numpy()
        baseline = np.clip(general_prediction[mask], 1e-6, 1 - 1e-6)
        alternate = np.clip(strict_prediction[mask], 1e-6, 1 - 1e-6)
        y_f = y[mask]
        base_f_score = bss(y_f, baseline)
        base_all = general.sigmoid(general.logit(component["anchor"]) + general.SHIFT_DELTA)
        base_all[mask] = baseline
        base_all_score = bss(y, base_all)
        candidates = []
        for weight in WEIGHTS:
            f_prediction = np.clip((1 - weight) * baseline + weight * alternate, 1e-6, 1 - 1e-6)
            full_prediction = base_all.copy()
            full_prediction[mask] = f_prediction
            candidates.append({
                "weight": weight,
                "overall_bss_delta": bss(y, full_prediction) - base_all_score,
                "f_bss_delta": bss(y_f, f_prediction) - base_f_score,
                "f_pitcher_bootstrap_probability": bootstrap(
                    rows.loc[mask, "pitcher_id"].to_numpy(), y_f, baseline,
                    f_prediction, 823800 + year + int(weight * 1000),
                ),
                "f_prediction_mean_delta": float(f_prediction.mean() - baseline.mean()),
            })
        folds.append({
            "year": year,
            "general_iterations": iterations,
            "general_seconds": seconds,
            "general_calibration_shift": float(shift),
            "f_rows": int(mask.sum()),
            "candidates": candidates,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "folds": folds},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"fold={year} complete", flush=True)

    summaries = []
    for weight in WEIGHTS[1:]:
        rows = [next(item for item in fold["candidates"] if item["weight"] == weight)
                for fold in folds]
        deltas = [item["overall_bss_delta"] for item in rows]
        f_deltas = [item["f_bss_delta"] for item in rows]
        probability = min(item["f_pitcher_bootstrap_probability"] for item in rows)
        ratio = min(abs(deltas)) / max(abs(deltas)) if max(abs(deltas)) else 1.0
        passed = bool(min(deltas) > 0 and min(f_deltas) > 0 and probability >= 0.80)
        summaries.append({
            "weight": weight,
            "fold_2023_overall_delta": deltas[0],
            "fold_2024_overall_delta": deltas[1],
            "fold_2023_f_delta": f_deltas[0],
            "fold_2024_f_delta": f_deltas[1],
            "worst_overall_delta": min(deltas),
            "magnitude_ratio": ratio,
            "minimum_f_pitcher_bootstrap_probability": probability,
            "passed": passed,
        })
    summaries.sort(key=lambda item: (item["passed"], item["worst_overall_delta"],
                                     item["magnitude_ratio"]), reverse=True)
    selected = next((item for item in summaries if item["passed"]), None)
    report = {
        "experiment": "strict-forward F route blend audit",
        "official_train_only": True,
        "test_aggregate_used": False,
        "baseline": "six-seed general route, minimum 128 iterations",
        "weights": list(WEIGHTS),
        "folds": folds,
        "summaries": summaries,
        "selected": selected,
        "decision": "select_single_f_blend" if selected else "keep_fgeneral6",
        "gate": "2023/2024 overall and F delta positive; minimum F pitcher bootstrap>=0.80",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "summaries": summaries,
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


def main(strict_source: Path, component_dir: Path, train_path: Path,
         output: Path, task_type: str):
    if strict_source.is_dir():
        run(strict_source, component_dir, train_path, output, task_type)
        return
    with tempfile.TemporaryDirectory(prefix="strict_f_blend_") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(strict_source) as archive:
            archive.extractall(directory)
        run(directory, component_dir, train_path, output, task_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-source", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    main(args.strict_source.resolve(), args.component_dir.resolve(), args.train.resolve(),
         args.output.resolve(), args.task_type)
