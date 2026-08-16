"""공식 train 시즌 추세만으로 고정 shift를 계산해 CatBoost 제출 ZIP 생성."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
TEST_PATH = ROOT / "open" / "data" / "test.csv"
SAMPLE_PATH = ROOT / "open" / "data" / "sample_submission.csv"
BASE_ZIP = Path(__file__).resolve().parent / "assets" / "submit012.zip"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "submit_catboost_train_trend_shift"
BUILD_DIR = RESULTS_DIR / "build" / OUTPUT_STEM
OUTPUT_ZIP = RESULTS_DIR / f"{OUTPUT_STEM}.zip"
ALPHA_GRID = np.linspace(0.0, 1.0, 101)
FORECAST_FOLDS = (2022, 2023, 2024)
TARGET_COL = "control_success"


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def choose_alpha(season_rates: dict[int, float]) -> tuple[float, list[dict[str, float]]]:
    losses = []
    for alpha in ALPHA_GRID:
        predictions = []
        actuals = []
        for year in FORECAST_FOLDS:
            prediction = season_rates[year - 1] + alpha * (
                season_rates[year - 1] - season_rates[year - 2]
            )
            predictions.append(prediction)
            actuals.append(season_rates[year])
        losses.append(float(np.mean((np.asarray(predictions) - actuals) ** 2)))
    best_index = int(np.argmin(losses))
    best_alpha = float(ALPHA_GRID[best_index])
    rows = [
        {"alpha": float(alpha), "forecast_mse": loss}
        for alpha, loss in zip(ALPHA_GRID, losses)
    ]
    return best_alpha, rows


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    season_rates = (
        train.groupby("season")[TARGET_COL].mean().astype(float).to_dict()
    )
    alpha, alpha_results = choose_alpha(season_rates)
    target_2025 = season_rates[2024] + alpha * (
        season_rates[2024] - season_rates[2023]
    )

    reference_prediction = pd.read_csv(
        Path(__file__).resolve().parent
        / "reference_catboost_best" / "artifacts" / "sub010.csv.gz"
    )[TARGET_COL].to_numpy(dtype=float)
    reference_mean = float(reference_prediction.mean())
    logits = np.log(
        np.clip(reference_prediction, 1e-6, 1 - 1e-6)
        / (1 - np.clip(reference_prediction, 1e-6, 1 - 1e-6))
    )
    objective = lambda shift: float(sigmoid(logits + shift).mean()) - target_2025
    logit_shift = float(brentq(objective, -1.0, 1.0, xtol=1e-12))
    shifted_mean = float(sigmoid(logits + logit_shift).mean())

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        archive.extractall(BUILD_DIR)
    meta_path = BUILD_DIR / "model" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["logit_shift"] = logit_shift
    meta["shift_provenance"] = {
        "data": "official train.csv only",
        "season_rates": {str(year): rate for year, rate in season_rates.items()},
        "forecast_formula": "rate_2025 = rate_2024 + alpha * (rate_2024 - rate_2023)",
        "alpha_selection": "grid 0.00..1.00 step 0.01; minimum walk-forward MSE for 2022-2024",
        "alpha": alpha,
        "target_2025": target_2025,
        "reference_prediction_mean": reference_mean,
        "test_aggregate_used": False,
        "external_data_used": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_DIR))
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        members = archive.namelist()
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"ZIP 손상: {bad_member}")

    validation_dir = RESULTS_DIR / f"validation_{OUTPUT_STEM}"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        archive.extractall(validation_dir)
    (validation_dir / "data").mkdir()
    shutil.copy2(TEST_PATH, validation_dir / "data" / "test.csv")
    shutil.copy2(SAMPLE_PATH, validation_dir / "data" / "sample_submission.csv")
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=validation_dir,
        capture_output=True, text=True, check=True, timeout=600,
    )
    output = pd.read_csv(validation_dir / "output" / "submission.csv")
    if output[TARGET_COL].isna().any() or not output[TARGET_COL].between(0, 1).all():
        raise ValueError("샘플 추론 결과의 결측 또는 확률 범위가 잘못되었습니다.")

    report = {
        "model": "CatBoost d6 FE10 7-seed + MR/wayoff offset + official-train trend shift",
        "official_train_only": True,
        "external_data_used": False,
        "test_aggregate_used": False,
        "season_rates": {str(year): rate for year, rate in season_rates.items()},
        "alpha": alpha,
        "alpha_selection_folds": list(FORECAST_FOLDS),
        "target_2025": target_2025,
        "reference_prediction_mean": reference_mean,
        "logit_shift": logit_shift,
        "shifted_reference_mean": shifted_mean,
        "alpha_grid_results": alpha_results,
        "zip": str(OUTPUT_ZIP.relative_to(ROOT)),
        "zip_mib": OUTPUT_ZIP.stat().st_size / 1024**2,
        "members": members,
        "zip_test_error": bad_member,
        "sample_rows": int(len(output)),
        "sample_missing": int(output[TARGET_COL].isna().sum()),
        "sample_min": float(output[TARGET_COL].min()),
        "sample_max": float(output[TARGET_COL].max()),
        "sample_stdout": completed.stdout.strip(),
    }
    report_path = RESULTS_DIR / f"{OUTPUT_STEM}.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2))
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
