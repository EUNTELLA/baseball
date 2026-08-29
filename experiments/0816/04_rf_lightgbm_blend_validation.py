from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from importlib.util import spec_from_file_location, module_from_spec


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RF_RESULTS = ROOT / "0814" / "results"
WEIGHTS = np.arange(0.0, 0.51, 0.05)


def load_validation_module():
    path = Path(__file__).resolve().parent / "03_lightgbm_time_validation.py"
    spec = spec_from_file_location("lightgbm_validation", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_validation_module()
    frame = pd.read_csv(module.TRAIN_PATH, encoding="utf-8-sig")
    features = [c for c in frame.columns if c not in (module.ID_COL, module.TARGET_COL)]
    config = module.CONFIGS["leaves15"]
    y23, lgb23, _, _ = module.fit_predict(frame, features, (2021, 2022), 2023, config)
    y24, lgb24, _, _ = module.fit_predict(frame, features, (2022, 2023), 2024, config)

    rf23_data = np.load(RF_RESULTS / "08_base_oof_2023.npz")
    rf24_data = np.load(RF_RESULTS / "03_seed_ensemble_oof.npz")
    rf23 = rf23_data["prediction"].astype(float)
    rf24 = rf24_data["predictions"][0].astype(float)
    if not (np.array_equal(y23, rf23_data["y"]) and np.array_equal(y24, rf24_data["y"])):
        raise ValueError("LightGBM과 RF OOF 행 순서가 일치하지 않습니다.")

    rows = []
    for lgb_weight in WEIGHTS:
        rf_weight = 1.0 - lgb_weight
        blend23 = rf_weight * rf23 + lgb_weight * lgb23
        blend24 = rf_weight * rf24 + lgb_weight * lgb24
        strict_model = LinearRegression().fit(blend23[:, None], y23)
        oracle_model = LinearRegression().fit(blend24[:, None], y24)
        row = {
            "rf_weight": float(rf_weight),
            "lightgbm_weight": float(lgb_weight),
            "strict": module.metrics(y24, strict_model.predict(blend24[:, None])),
            "oracle": module.metrics(y24, oracle_model.predict(blend24[:, None])),
            "oracle_slope": float(oracle_model.coef_[0]),
            "oracle_intercept": float(oracle_model.intercept_),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    baseline = rows[0]
    eligible = [
        row for row in rows[1:]
        if row["strict"]["brier"] < baseline["strict"]["brier"]
        and row["oracle"]["brier"] < baseline["oracle"]["brier"]
    ]
    selected = min(
        eligible,
        key=lambda row: (row["oracle"]["brier"], row["strict"]["brier"]),
        default=None,
    )
    payload = {
        "selection_rule": "must beat RF-only in both; choose best oracle Brier then strict Brier",
        "rf_baseline": baseline,
        "selected": selected,
        "should_package": selected is not None,
        "runs": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "04_rf_lightgbm_blend_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 선택 결과 ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
