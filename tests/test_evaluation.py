from __future__ import annotations

import unittest

from baseball_platform.evaluation import compare_models
from baseball_platform.models.baseline import MeanProbabilityModel
from baseball_platform.validation.temporal_split import TemporalFold


class EvaluationTest(unittest.TestCase):
    def test_lower_log_loss_model_ranks_first(self) -> None:
        fold = TemporalFold(
            train_rows=[
                {"target": 0},
                {"target": 1},
                {"target": 1},
                {"target": 1},
            ],
            validation_rows=[
                {"target": 1},
                {"target": 1},
                {"target": 0},
                {"target": 1},
            ],
            validation_season=2024,
        )

        results, leaderboard = compare_models(
            {
                "model_a": MeanProbabilityModel,
                "model_b": MeanProbabilityModel,
            },
            [fold],
            target_column="target",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([row.rank for row in leaderboard], [1, 2])
        self.assertLessEqual(
            leaderboard[0].mean_log_loss,
            leaderboard[1].mean_log_loss,
        )


if __name__ == "__main__":
    unittest.main()
