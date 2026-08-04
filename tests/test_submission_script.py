from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import script


class ProbabilityModel:
    def predict_proba(self, rows):
        return [0.2, 0.8]


class SubmissionScriptTest(unittest.TestCase):
    def test_preserves_sample_format_and_writes_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample_submission.csv"
            with sample.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["pitch_id", "control_success"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"pitch_id": "P1", "control_success": "0"},
                        {"pitch_id": "P2", "control_success": "0"},
                    ]
                )
            output = root / "output" / "submission.csv"

            probabilities = script.predict(
                ProbabilityModel(), [{}, {}], [], "rows"
            )
            with patch.object(script, "OUTPUT_PATH", output):
                script.write_submission(
                    sample, probabilities, "control_success"
                )

            columns, rows = script.read_csv(output)
            self.assertEqual(columns, ["pitch_id", "control_success"])
            self.assertEqual([row["pitch_id"] for row in rows], ["P1", "P2"])
            self.assertEqual(
                [row["control_success"] for row in rows],
                ["0.2000000000", "0.8000000000"],
            )


if __name__ == "__main__":
    unittest.main()
