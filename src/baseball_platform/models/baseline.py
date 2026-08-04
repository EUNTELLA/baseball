"""평균 확률과 Logistic Regression 기준 모델."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MeanProbabilityModel:
    probability: float = 0.5

    def fit(self, rows: list[dict[str, object]], target_column: str) -> None:
        targets = [int(row[target_column]) for row in rows]
        self.probability = sum(targets) / len(targets)

    def predict_proba(self, rows: list[dict[str, object]]) -> list[float]:
        return [self.probability] * len(rows)


class LogisticBaseline:
    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> None:
        self.features = numeric_features + categorical_features
        numeric_indices = list(range(len(numeric_features)))
        categorical_indices = list(
            range(len(numeric_features), len(self.features))
        )
        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric_indices,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="UNKNOWN",
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_indices,
                ),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("preprocessing", preprocessing),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def fit(self, rows: list[dict[str, object]], target_column: str) -> None:
        x = _as_matrix(rows, self.features)
        y = [int(row[target_column]) for row in rows]
        self.pipeline.fit(x, y)

    def predict_proba(self, rows: list[dict[str, object]]) -> list[float]:
        x = _as_matrix(rows, self.features)
        return self.pipeline.predict_proba(x)[:, 1].tolist()

def _as_matrix(
    rows: list[dict[str, object]], features: list[str]
) -> list[list[object]]:
    return [[row.get(feature) for feature in features] for row in rows]
