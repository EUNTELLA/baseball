from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = Path(__file__).resolve().parent / "results" / "03_seed_ensemble_oof.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET_COL = "control_success"
FIXED_OFFSET = -0.009683759059887942
N_SPLITS = 5


def competition_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    rate = float(y.mean())
    brier = float(np.mean((prediction - y) ** 2))
    reference = rate * (1.0 - rate)
    return {
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
        "prediction_mean": float(prediction.mean()),
    }


def fit_predict(method: str, train_p: np.ndarray, train_y: np.ndarray, valid_p: np.ndarray) -> np.ndarray:
    if method == "affine":
        model = LinearRegression().fit(train_p[:, None], train_y)
        return model.predict(valid_p[:, None])
    if method == "quadratic_ridge":
        train_x = np.column_stack([train_p, train_p**2])
        valid_x = np.column_stack([valid_p, valid_p**2])
        model = Ridge(alpha=10.0).fit(train_x, train_y)
        return model.predict(valid_x)
    if method == "platt":
        eps = 1e-6
        train_logit = np.log(np.clip(train_p, eps, 1 - eps) / np.clip(1 - train_p, eps, 1 - eps))
        valid_logit = np.log(np.clip(valid_p, eps, 1 - eps) / np.clip(1 - valid_p, eps, 1 - eps))
        model = LogisticRegression(C=1.0, max_iter=1000).fit(train_logit[:, None], train_y)
        return model.predict_proba(valid_logit[:, None])[:, 1]
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return model.fit(train_p, train_y).predict(valid_p)
    raise ValueError(method)


def fit_final_parameters(method: str, p: np.ndarray, y: np.ndarray) -> dict[str, object]:
    if method == "affine":
        model = LinearRegression().fit(p[:, None], y)
        return {"intercept": float(model.intercept_), "slope": float(model.coef_[0])}
    if method == "quadratic_ridge":
        model = Ridge(alpha=10.0).fit(np.column_stack([p, p**2]), y)
        return {"intercept": float(model.intercept_), "coefficients": model.coef_.tolist()}
    if method == "platt":
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
        model = LogisticRegression(C=1.0, max_iter=1000).fit(logit[:, None], y)
        return {"intercept": float(model.intercept_[0]), "coefficient": float(model.coef_[0, 0])}
    return {"note": "isotonic requires serialized threshold arrays"}


def main() -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError("먼저 03_seed_ensemble_cv.py를 실행하세요.")
    data = np.load(OOF_PATH)
    y = data["y"].astype(float)
    prediction = data["predictions"][0].astype(float)
    methods = ("affine", "quadratic_ridge", "platt", "isotonic")
    cv_prediction = {method: np.zeros_like(prediction) for method in methods}
    folds = np.array_split(np.arange(len(y)), N_SPLITS)

    for valid_index in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[valid_index] = False
        for method in methods:
            cv_prediction[method][valid_index] = fit_predict(
                method,
                prediction[train_mask],
                y[train_mask],
                prediction[valid_index],
            )

    rows = [
        {"method": "fixed_offset", **competition_metrics(y, prediction + FIXED_OFFSET)}
    ]
    for method in methods:
        rows.append({"method": method, **competition_metrics(y, cv_prediction[method])})
    result = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    best_method = str(result.iloc[0]["method"])
    payload = {
        "cv": "5 contiguous folds within 2024 OOF",
        "selection_rule": "minimum cross-validated Brier",
        "results": result.to_dict(orient="records"),
        "best_method": best_method,
        "best_final_parameters": (
            fit_final_parameters(best_method, prediction, y)
            if best_method != "fixed_offset"
            else {"offset": FIXED_OFFSET}
        ),
    }
    result.to_csv(RESULTS_DIR / "05_calibration_methods_cv.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "05_calibration_methods_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
