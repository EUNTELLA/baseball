"""모든 후보 모델을 동일한 시간 Fold와 지표로 비교한다."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from baseball_platform.validation.temporal_split import TemporalFold


@dataclass(frozen=True)
class FoldResult:
    model: str
    validation_season: int
    train_rows: int
    validation_rows: int
    validation_success_rate: float
    log_loss: float
    brier_score: float
    brier_skill_score: float
    roc_auc: float


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    model: str
    mean_log_loss: float
    std_log_loss: float
    mean_brier_score: float
    oof_brier_score: float
    oof_brier_skill_score: float
    mean_roc_auc: float
    fold_count: int


ModelFactory = Callable[[], object]


def brier_skill_score(
    targets: list[int], probabilities: list[float]
) -> float:
    """대회 공식 Brier Skill Score를 계산한다."""
    if not targets or len(targets) != len(probabilities):
        raise ValueError("정답과 예측은 길이가 같은 비어 있지 않은 값이어야 합니다.")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("정답은 0 또는 1이어야 합니다.")
    if any(
        not math.isfinite(probability) or not 0 <= probability <= 1
        for probability in probabilities
    ):
        raise ValueError("예측 확률은 유한한 0~1 값이어야 합니다.")

    brier = statistics.fmean(
        (probability - target) ** 2
        for target, probability in zip(targets, probabilities, strict=True)
    )
    success_rate = statistics.fmean(targets)
    reference_brier = success_rate * (1 - success_rate)
    if reference_brier == 0:
        return 0.0
    return max(0.0, 100000 * (1 - brier / reference_brier))


def compare_models(
    model_factories: dict[str, ModelFactory],
    folds: list[TemporalFold],
    *,
    target_column: str,
) -> tuple[list[FoldResult], list[LeaderboardRow]]:
    fold_results: list[FoldResult] = []
    for model_name, factory in model_factories.items():
        for fold in folds:
            model = factory()
            model.fit(fold.train_rows, target_column)
            probabilities = model.predict_proba(fold.validation_rows)
            targets = [
                int(row[target_column]) for row in fold.validation_rows
            ]
            fold_results.append(
                FoldResult(
                    model=model_name,
                    validation_season=fold.validation_season,
                    train_rows=len(fold.train_rows),
                    validation_rows=len(fold.validation_rows),
                    validation_success_rate=statistics.fmean(targets),
                    log_loss=log_loss(targets, probabilities),
                    brier_score=brier_score_loss(targets, probabilities),
                    brier_skill_score=brier_skill_score(
                        targets, probabilities
                    ),
                    roc_auc=roc_auc_score(targets, probabilities),
                )
            )

    leaderboard = _summarize(fold_results)
    return fold_results, leaderboard


def write_comparison_results(
    output_directory: str | Path,
    fold_results: list[FoldResult],
    leaderboard: list[LeaderboardRow],
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fold_path = output / "fold_results.csv"
    leaderboard_path = output / "leaderboard.csv"
    _write_dataclasses(fold_path, fold_results)
    _write_dataclasses(leaderboard_path, leaderboard)
    return fold_path, leaderboard_path


def _summarize(results: list[FoldResult]) -> list[LeaderboardRow]:
    grouped: dict[str, list[FoldResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    summaries = []
    for model, model_results in grouped.items():
        log_losses = [result.log_loss for result in model_results]
        summaries.append(
            {
                "model": model,
                "mean_log_loss": statistics.fmean(log_losses),
                "std_log_loss": statistics.pstdev(log_losses),
                "mean_brier_score": statistics.fmean(
                    result.brier_score for result in model_results
                ),
                "oof_brier_score": _weighted_mean(
                    model_results, "brier_score"
                ),
                "oof_brier_skill_score": _oof_brier_skill_score(
                    model_results
                ),
                "mean_roc_auc": statistics.fmean(
                    result.roc_auc for result in model_results
                ),
                "fold_count": len(model_results),
            }
        )
    summaries.sort(
        key=lambda item: (
            -item["oof_brier_skill_score"], item["oof_brier_score"]
        )
    )
    return [
        LeaderboardRow(
            rank=rank,
            model=str(item["model"]),
            mean_log_loss=round(float(item["mean_log_loss"]), 6),
            std_log_loss=round(float(item["std_log_loss"]), 6),
            mean_brier_score=round(float(item["mean_brier_score"]), 6),
            oof_brier_score=round(float(item["oof_brier_score"]), 6),
            oof_brier_skill_score=round(
                float(item["oof_brier_skill_score"]), 6
            ),
            mean_roc_auc=round(float(item["mean_roc_auc"]), 6),
            fold_count=int(item["fold_count"]),
        )
        for rank, item in enumerate(summaries, start=1)
    ]


def _weighted_mean(results: list[FoldResult], field: str) -> float:
    total_rows = sum(result.validation_rows for result in results)
    return sum(
        float(getattr(result, field)) * result.validation_rows
        for result in results
    ) / total_rows


def _oof_brier_skill_score(results: list[FoldResult]) -> float:
    brier = _weighted_mean(results, "brier_score")
    success_rate = _weighted_mean(results, "validation_success_rate")
    reference_brier = success_rate * (1 - success_rate)
    if reference_brier == 0:
        return 0.0
    return max(0.0, 100000 * (1 - brier / reference_brier))


def _write_dataclasses(path: Path, rows: list[object]) -> None:
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
