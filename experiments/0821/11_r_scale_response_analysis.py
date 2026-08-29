"""자체 제출 세 점으로 R 잔차 강도의 보수적 반응곡선을 계산한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OBSERVATIONS = (
    {"scale": 0.0, "score": 1029.0832235020, "name": "R 보정 전 기준"},
    {"scale": 0.025, "score": 1031.5033329643, "name": "R 잔차 0.025"},
    {"scale": 0.05, "score": 1033.0126318779, "name": "R 잔차 0.05"},
)
CANDIDATES = (0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.10)


def main(output: Path) -> None:
    scales = np.asarray([row["scale"] for row in OBSERVATIONS], float)
    scores = np.asarray([row["score"] for row in OBSERVATIONS], float)
    quadratic, linear, intercept = np.polyfit(scales, scores, 2)
    optimum = float(-linear / (2.0 * quadratic))
    predictions = [
        {
            "scale": scale,
            "predicted_score": float(quadratic * scale ** 2 + linear * scale + intercept),
            "predicted_delta_vs_0050": float(
                quadratic * scale ** 2 + linear * scale + intercept - scores[-1]
            ),
        }
        for scale in CANDIDATES
    ]
    eligible = [row for row in predictions if row["scale"] <= 0.075]
    recommended = max(eligible, key=lambda row: row["predicted_score"])
    report = {
        "experiment": "R residual scale response from own submissions",
        "observations": list(OBSERVATIONS),
        "quadratic": {
            "a": float(quadratic), "b": float(linear), "c": float(intercept),
            "unconstrained_optimum_scale": optimum,
        },
        "candidate_predictions": predictions,
        "recommended": recommended,
        "decision": "submit_existing_scale0075_candidate",
        "risk": "세 점만 사용한 서버 반응 근사이므로 예상 점수는 보장되지 않는다.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
