"""샘플 데이터로 투수별 제구력 평가 결과를 출력한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseball_platform.quality.pitch_validator import load_and_validate_pitches
from baseball_platform.transforms.control_metrics import (
    aggregate_by_pitcher,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="투수 제구력 평가")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_pitches.csv",
        help="투구 좌표 CSV 경로",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "control_zone.json",
        help="평가 설정 JSON 경로",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pitches = load_and_validate_pitches(args.input)
    metrics = aggregate_by_pitcher(pitches, config)
    print(
        json.dumps(
            [metric.to_dict() for metric in metrics],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
