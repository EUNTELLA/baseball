from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from baseball_platform.competition_data import discover_competition_files


class CompetitionDataTest(unittest.TestCase):
    def test_discovers_train_test_and_sample_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root / "train.csv", ["pitch_id", "control_success"])
            self._write(root / "test.csv", ["pitch_id", "feature"])
            self._write(
                root / "sample_submission.csv",
                ["pitch_id", "control_success"],
            )

            files = discover_competition_files(root, require_train=True)

            self.assertEqual(files.train, root / "train.csv")
            self.assertEqual(files.test, root / "test.csv")
            self.assertEqual(
                files.sample_submission, root / "sample_submission.csv"
            )

    @staticmethod
    def _write(path: Path, columns: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerow({column: "0" for column in columns})


if __name__ == "__main__":
    unittest.main()
