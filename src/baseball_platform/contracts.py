"""공식 데이터 매핑 전 사용하는 임시 데이터 계약."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataContract:
    schema_version: str
    id_columns: tuple[str, ...]
    time_columns: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_column: str
    train_seasons: tuple[int, ...]
    test_season: int

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    @property
    def required_train_columns(self) -> set[str]:
        return set(
            self.id_columns
            + self.time_columns
            + self.feature_columns
            + (self.target_column,)
        )

    @property
    def required_test_columns(self) -> set[str]:
        return set(self.id_columns + self.time_columns + self.feature_columns)


def load_contract(path: str | Path) -> DataContract:
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    return DataContract(
        schema_version=raw["schema_version"],
        id_columns=tuple(raw["id_columns"]),
        time_columns=tuple(raw["time_columns"]),
        numeric_features=tuple(raw["numeric_features"]),
        categorical_features=tuple(raw["categorical_features"]),
        target_column=raw["target_column"],
        train_seasons=tuple(raw["train_seasons"]),
        test_season=raw["test_season"],
    )
