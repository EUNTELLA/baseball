"""미래 시즌을 과거 시즌 학습에 섞지 않는 분할."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalFold:
    train_rows: list[dict[str, object]]
    validation_rows: list[dict[str, object]]
    validation_season: int


def expanding_season_folds(
    rows: list[dict[str, object]],
    validation_seasons: tuple[int, ...] = (2023, 2024),
) -> list[TemporalFold]:
    folds: list[TemporalFold] = []
    for validation_season in validation_seasons:
        train = [
            row for row in rows if int(row["season"]) < validation_season
        ]
        validation = [
            row for row in rows if int(row["season"]) == validation_season
        ]
        if not train or not validation:
            raise ValueError(
                f"{validation_season} 시즌 분할에 필요한 데이터가 없습니다."
            )
        folds.append(TemporalFold(train, validation, validation_season))
    return folds
