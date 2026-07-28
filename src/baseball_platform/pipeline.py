"""합성 데이터로 학습부터 제출 파일 생성까지 실행하는 임시 파이프라인."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseball_platform.collectors.synthetic_data import generate_synthetic_datasets
from baseball_platform.contracts import load_contract
from baseball_platform.evaluation import compare_models, write_comparison_results
from baseball_platform.models.baseline import (
    LogisticBaseline,
    MeanProbabilityModel,
)
from baseball_platform.quality.dataset_validator import (
    read_csv_rows,
    validate_dataset,
    validate_submission,
)
from baseball_platform.transforms.temporal_features import add_leakage_safe_history
from baseball_platform.validation.temporal_split import expanding_season_folds
from baseball_platform.visualization import create_model_comparison_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(output_directory: Path) -> dict[str, object]:
    contract = load_contract(PROJECT_ROOT / "configs" / "provisional_schema.json")
    train_path, test_path = generate_synthetic_datasets(
        output_directory, contract
    )
    raw_train = read_csv_rows(train_path)
    raw_test = read_csv_rows(test_path)
    validate_dataset(raw_train, contract, is_train=True)
    validate_dataset(raw_test, contract, is_train=False)

    combined = raw_train + raw_test
    featured = add_leakage_safe_history(
        combined, target_column=contract.target_column
    )
    train_rows = featured[: len(raw_train)]
    test_rows = featured[len(raw_train) :]
    numeric = list(contract.numeric_features) + [
        "history_pitch_count",
        "history_success_rate",
        "recent_20_success_rate",
    ]
    categorical = list(contract.categorical_features)

    folds = expanding_season_folds(train_rows)
    model_factories = {
        "mean_probability": MeanProbabilityModel,
        "logistic_regression": lambda: LogisticBaseline(
            numeric, categorical
        ),
    }
    fold_results, leaderboard = compare_models(
        model_factories,
        folds,
        target_column=contract.target_column,
    )
    fold_results_path, leaderboard_path = write_comparison_results(
        output_directory, fold_results, leaderboard
    )
    dashboard_path = create_model_comparison_dashboard(
        output_directory / "model_comparison.png",
        fold_results,
        leaderboard,
    )

    final_model = model_factories[leaderboard[0].model]()
    final_model.fit(train_rows, contract.target_column)
    probabilities = final_model.predict_proba(test_rows)
    submission_path = output_directory / "synthetic_submission.csv"
    submission_rows = [
        {
            "pitch_id": row["pitch_id"],
            "control_success_probability": f"{probability:.8f}",
        }
        for row, probability in zip(test_rows, probabilities, strict=True)
    ]
    with submission_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["pitch_id", "control_success_probability"],
        )
        writer.writeheader()
        writer.writerows(submission_rows)
    validate_submission(
        submission_rows, [str(row["pitch_id"]) for row in test_rows]
    )
    return {
        "schema_version": contract.schema_version,
        "best_model": leaderboard[0].model,
        "best_mean_log_loss": leaderboard[0].mean_log_loss,
        "fold_results": str(fold_results_path),
        "leaderboard": str(leaderboard_path),
        "dashboard": str(dashboard_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "submission": str(submission_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="임시 제구 확률 예측 파이프라인")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
