from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from baseball_platform.quality.pitch_validator import (
    Pitch,
    PitchValidationError,
    load_and_validate_pitches,
)
from baseball_platform.transforms.control_metrics import (
    ControlConfig,
    aggregate_by_pitcher,
    evaluate_pitch,
)


CONFIG = ControlConfig(
    x_min=-1,
    x_max=1,
    y_min=-1,
    y_max=1,
    valid_radius=0.3,
    minimum_pitch_count=2,
    accuracy_weight=0.5,
    consistency_weight=0.3,
    valid_pitch_weight=0.2,
)


def make_pitch(
    pitch_id: str,
    actual_x: float,
    actual_y: float,
    target_x: float = 0,
    target_y: float = 0,
) -> Pitch:
    return Pitch(
        pitch_id=pitch_id,
        pitcher_id="TEST01",
        pitched_at=datetime.fromisoformat("2026-07-28T19:30:00+09:00"),
        target_x=target_x,
        target_y=target_y,
        actual_x=actual_x,
        actual_y=actual_y,
        zone_width=2,
        zone_height=2,
        pitch_type="FASTBALL",
    )


class ControlMetricsTest(unittest.TestCase):
    def test_pitch_on_target_is_valid_and_in_zone(self) -> None:
        result = evaluate_pitch(make_pitch("P1", 0, 0), CONFIG)

        self.assertEqual(result.distance_error, 0)
        self.assertTrue(result.is_valid_pitch)
        self.assertTrue(result.is_in_strike_zone)

    def test_valid_radius_boundary_is_inclusive(self) -> None:
        result = evaluate_pitch(make_pitch("P1", 0.3, 0), CONFIG)

        self.assertTrue(result.is_valid_pitch)

    def test_aggregation_calculates_rates_and_sample_status(self) -> None:
        pitches = [make_pitch("P1", 0, 0), make_pitch("P2", 0.6, 0)]

        metrics = aggregate_by_pitcher(pitches, CONFIG)[0]

        self.assertEqual(metrics.pitch_count, 2)
        self.assertEqual(metrics.mean_error, 0.3)
        self.assertEqual(metrics.rmse, 0.424)
        self.assertEqual(metrics.valid_pitch_rate, 50)
        self.assertEqual(metrics.strike_zone_rate, 100)
        self.assertTrue(metrics.has_sufficient_sample)
        self.assertGreaterEqual(metrics.control_score, 0)
        self.assertLessEqual(metrics.control_score, 100)


class PitchValidatorTest(unittest.TestCase):
    def test_duplicate_pitch_id_is_rejected(self) -> None:
        content = (
            "pitch_id,pitcher_id,pitched_at,target_x,target_y,actual_x,"
            "actual_y,zone_width,zone_height,pitch_type\n"
            "P1,TEST,2026-07-28T19:30:00+09:00,0,0,0,0,2,2,FASTBALL\n"
            "P1,TEST,2026-07-28T19:31:00+09:00,0,0,0,0,2,2,FASTBALL\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pitches.csv"
            path.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(PitchValidationError, "중복 pitch_id"):
                load_and_validate_pitches(path)


if __name__ == "__main__":
    unittest.main()
