"""목표 좌표와 실제 투구 좌표로 제구력 지표를 계산한다."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from baseball_platform.quality.pitch_validator import Pitch


@dataclass(frozen=True)
class ControlConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    valid_radius: float
    minimum_pitch_count: int
    accuracy_weight: float
    consistency_weight: float
    valid_pitch_weight: float

    @property
    def maximum_error(self) -> float:
        return math.hypot(self.x_max - self.x_min, self.y_max - self.y_min)


@dataclass(frozen=True)
class PitchResult:
    pitch_id: str
    pitcher_id: str
    distance_error: float
    is_valid_pitch: bool
    is_in_strike_zone: bool


@dataclass(frozen=True)
class PitcherMetrics:
    pitcher_id: str
    pitch_count: int
    mean_error: float
    rmse: float
    error_stddev: float
    valid_pitch_rate: float
    strike_zone_rate: float
    accuracy_score: float
    consistency_score: float
    valid_pitch_score: float
    control_score: float
    has_sufficient_sample: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def load_config(path: str | Path) -> ControlConfig:
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)

    zone = raw["zone"]
    weights = raw["score_weights"]
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("score_weights의 합은 1이어야 합니다.")
    if raw["valid_radius"] <= 0:
        raise ValueError("valid_radius는 0보다 커야 합니다.")
    if raw["minimum_pitch_count"] <= 0:
        raise ValueError("minimum_pitch_count는 0보다 커야 합니다.")
    if zone["x_min"] >= zone["x_max"] or zone["y_min"] >= zone["y_max"]:
        raise ValueError("스트라이크존의 최소 좌표는 최대 좌표보다 작아야 합니다.")

    return ControlConfig(
        x_min=float(zone["x_min"]),
        x_max=float(zone["x_max"]),
        y_min=float(zone["y_min"]),
        y_max=float(zone["y_max"]),
        valid_radius=float(raw["valid_radius"]),
        minimum_pitch_count=int(raw["minimum_pitch_count"]),
        accuracy_weight=float(weights["accuracy"]),
        consistency_weight=float(weights["consistency"]),
        valid_pitch_weight=float(weights["valid_pitch"]),
    )


def evaluate_pitch(pitch: Pitch, config: ControlConfig) -> PitchResult:
    error = math.hypot(
        pitch.actual_x - pitch.target_x,
        pitch.actual_y - pitch.target_y,
    )
    in_zone = (
        config.x_min <= pitch.actual_x <= config.x_max
        and config.y_min <= pitch.actual_y <= config.y_max
    )
    return PitchResult(
        pitch_id=pitch.pitch_id,
        pitcher_id=pitch.pitcher_id,
        distance_error=error,
        is_valid_pitch=error <= config.valid_radius,
        is_in_strike_zone=in_zone,
    )


def aggregate_by_pitcher(
    pitches: Iterable[Pitch], config: ControlConfig
) -> list[PitcherMetrics]:
    grouped: dict[str, list[PitchResult]] = {}
    for pitch in pitches:
        result = evaluate_pitch(pitch, config)
        grouped.setdefault(pitch.pitcher_id, []).append(result)

    metrics = [
        _calculate_pitcher_metrics(pitcher_id, results, config)
        for pitcher_id, results in grouped.items()
    ]
    return sorted(metrics, key=lambda item: item.pitcher_id)


def _calculate_pitcher_metrics(
    pitcher_id: str,
    results: list[PitchResult],
    config: ControlConfig,
) -> PitcherMetrics:
    errors = [result.distance_error for result in results]
    count = len(errors)
    mean_error = statistics.fmean(errors)
    rmse = math.sqrt(statistics.fmean(error**2 for error in errors))
    error_stddev = statistics.pstdev(errors)
    valid_pitch_rate = _percentage(
        sum(result.is_valid_pitch for result in results), count
    )
    strike_zone_rate = _percentage(
        sum(result.is_in_strike_zone for result in results), count
    )

    accuracy_score = _inverse_error_score(mean_error, config.maximum_error)
    consistency_score = _inverse_error_score(
        error_stddev, config.maximum_error
    )
    valid_pitch_score = valid_pitch_rate
    control_score = (
        accuracy_score * config.accuracy_weight
        + consistency_score * config.consistency_weight
        + valid_pitch_score * config.valid_pitch_weight
    )

    return PitcherMetrics(
        pitcher_id=pitcher_id,
        pitch_count=count,
        mean_error=_rounded(mean_error),
        rmse=_rounded(rmse),
        error_stddev=_rounded(error_stddev),
        valid_pitch_rate=_rounded(valid_pitch_rate),
        strike_zone_rate=_rounded(strike_zone_rate),
        accuracy_score=_rounded(accuracy_score),
        consistency_score=_rounded(consistency_score),
        valid_pitch_score=_rounded(valid_pitch_score),
        control_score=_rounded(control_score),
        has_sufficient_sample=count >= config.minimum_pitch_count,
    )


def _inverse_error_score(error: float, maximum_error: float) -> float:
    return max(0.0, min(100.0, (1 - error / maximum_error) * 100))


def _percentage(part: int, whole: int) -> float:
    return part / whole * 100


def _rounded(value: float) -> float:
    return round(value, 3)
