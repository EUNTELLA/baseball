from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baseball_platform.contracts import load_contract
from baseball_platform.pipeline import PROJECT_ROOT, run
from baseball_platform.quality.dataset_validator import (
    DatasetValidationError,
    validate_submission,
)
from baseball_platform.transforms.temporal_features import add_leakage_safe_history
from baseball_platform.validation.temporal_split import expanding_season_folds


class TemporalFeatureTest(unittest.TestCase):
    def test_current_target_is_not_used_in_current_feature(self) -> None:
        rows = [
            {"pitcher_id": "P1", "target": "1"},
            {"pitcher_id": "P1", "target": "0"},
        ]

        result = add_leakage_safe_history(rows)

        self.assertEqual(result[0]["history_pitch_count"], 0)
        self.assertEqual(result[0]["history_success_rate"], 0.5)
        self.assertEqual(result[1]["history_pitch_count"], 1)
        self.assertEqual(result[1]["history_success_rate"], 1.0)

    def test_temporal_fold_has_only_past_seasons(self) -> None:
        rows = [
            {"season": year, "target": 1}
            for year in range(2019, 2025)
        ]

        folds = expanding_season_folds(rows)

        for fold in folds:
            self.assertTrue(
                all(
                    int(row["season"]) < fold.validation_season
                    for row in fold.train_rows
                )
            )


class SubmissionValidationTest(unittest.TestCase):
    def test_out_of_range_probability_is_rejected(self) -> None:
        rows = [
            {"pitch_id": "P1", "control_success_probability": "1.2"}
        ]
        with self.assertRaises(DatasetValidationError):
            validate_submission(rows, ["P1"])


class EndToEndPipelineTest(unittest.TestCase):
    def test_pipeline_generates_valid_submission(self) -> None:
        contract = load_contract(
            PROJECT_ROOT / "configs" / "provisional_schema.json"
        )
        self.assertEqual(contract.schema_version, "provisional-v1")
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory))

            self.assertEqual(result["train_rows"], 1440)
            self.assertEqual(result["test_rows"], 240)
            self.assertTrue(Path(str(result["submission"])).exists())
            self.assertTrue(Path(str(result["fold_results"])).exists())
            self.assertTrue(Path(str(result["leaderboard"])).exists())
            self.assertTrue(Path(str(result["dashboard"])).exists())
            self.assertIn(
                result["best_model"],
                {"mean_probability", "logistic_regression"},
            )


if __name__ == "__main__":
    unittest.main()
