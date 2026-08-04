from __future__ import annotations

import unittest

from baseball_platform.evaluation import brier_skill_score, compare_models
from baseball_platform.validation.temporal_split import TemporalFold


class RowProbabilityModel:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, rows, target_column) -> None:
        pass

    def predict_proba(self, rows) -> list[float]:
        return [float(row[self.column]) for row in rows]


class EvaluationTest(unittest.TestCase):
    def test_official_brier_skill_score(self) -> None:
        targets = [0, 1, 0, 1]

        self.assertEqual(brier_skill_score(targets, [0, 1, 0, 1]), 100000)
        self.assertEqual(brier_skill_score(targets, [0.5] * 4), 0)
        self.assertEqual(brier_skill_score(targets, [1, 0, 1, 0]), 0)

    def test_higher_oof_brier_skill_score_ranks_first(self) -> None:
        fold = TemporalFold(
            train_rows=[
                {"target": 0},
                {"target": 1},
                {"target": 1},
                {"target": 1},
            ],
            validation_rows=[
                {"target": 1, "strong": 0.9, "weak": 0.5},
                {"target": 1, "strong": 0.8, "weak": 0.5},
                {"target": 0, "strong": 0.1, "weak": 0.5},
                {"target": 1, "strong": 0.9, "weak": 0.5},
            ],
            validation_season=2024,
        )

        results, leaderboard = compare_models(
            {
                "weak": lambda: RowProbabilityModel("weak"),
                "strong": lambda: RowProbabilityModel("strong"),
            },
            [fold],
            target_column="target",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([row.rank for row in leaderboard], [1, 2])
        self.assertEqual(leaderboard[0].model, "strong")
        self.assertGreater(
            leaderboard[0].oof_brier_skill_score,
            leaderboard[1].oof_brier_skill_score,
        )


if __name__ == "__main__":
    unittest.main()
