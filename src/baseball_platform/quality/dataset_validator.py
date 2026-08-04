"""학습·평가 데이터 및 제출 파일의 기본 계약을 검증한다."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from baseball_platform.contracts import DataContract


class DatasetValidationError(ValueError):
    pass


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_dataset(
    rows: list[dict[str, str]],
    contract: DataContract,
    *,
    is_train: bool,
) -> None:
    if not rows:
        raise DatasetValidationError("데이터가 비어 있습니다.")

    columns = set(rows[0])
    required = (
        contract.required_train_columns
        if is_train
        else contract.required_test_columns
    )
    missing = required - columns
    if missing:
        raise DatasetValidationError(
            f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}"
        )
    if not is_train and contract.target_column in columns:
        raise DatasetValidationError(
            f"평가 데이터에는 {contract.target_column}가 없어야 합니다."
        )

    expected_seasons = (
        set(contract.train_seasons) if is_train else {contract.test_season}
    )
    seen_pitch_ids: set[str] = set()
    previous_date: str | None = None
    last_sequence_by_game: dict[str, int] = {}
    for number, row in enumerate(rows, start=2):
        pitch_id = row["pitch_id"].strip()
        if not pitch_id or pitch_id in seen_pitch_ids:
            raise DatasetValidationError(
                f"{number}행: pitch_id가 비었거나 중복입니다."
            )
        seen_pitch_ids.add(pitch_id)

        try:
            season = int(row["season"])
            sequence = int(row["pitch_sequence"])
            balls = int(row["balls"])
            strikes = int(row["strikes"])
            outs = int(row["outs"])
        except ValueError as exc:
            raise DatasetValidationError(
                f"{number}행: 정수형 값이 올바르지 않습니다."
            ) from exc

        if season not in expected_seasons:
            raise DatasetValidationError(
                f"{number}행: 허용되지 않은 시즌 {season}"
            )
        if not (0 <= balls <= 3 and 0 <= strikes <= 2 and 0 <= outs <= 2):
            raise DatasetValidationError(
                f"{number}행: 볼·스트라이크·아웃 범위가 잘못됐습니다."
            )

        game_date = row["game_date"]
        if previous_date is not None and game_date < previous_date:
            raise DatasetValidationError("데이터가 날짜 순서로 정렬되지 않았습니다.")
        previous_date = game_date
        game_id = row["game_id"]
        previous_sequence = last_sequence_by_game.get(game_id)
        if previous_sequence is not None and sequence <= previous_sequence:
            raise DatasetValidationError(
                f"{number}행: 경기 내 투구 순서가 증가하지 않습니다."
            )
        last_sequence_by_game[game_id] = sequence

        if is_train:
            try:
                target = int(row[contract.target_column])
            except ValueError as exc:
                raise DatasetValidationError(
                    f"{number}행: {contract.target_column}가 정수가 아닙니다."
                ) from exc
            if target not in (0, 1):
                raise DatasetValidationError(
                    f"{number}행: {contract.target_column}는 0 또는 1이어야 합니다."
                )


def validate_submission(
    rows: list[dict[str, str]], expected_pitch_ids: list[str]
) -> None:
    if [row.get("pitch_id") for row in rows] != expected_pitch_ids:
        raise DatasetValidationError("제출 파일의 투구 ID 또는 행 순서가 다릅니다.")
    for number, row in enumerate(rows, start=2):
        try:
            probability = float(row["control_success_probability"])
        except (KeyError, ValueError) as exc:
            raise DatasetValidationError(
                f"{number}행: 예측 확률이 올바르지 않습니다."
            ) from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise DatasetValidationError(
                f"{number}행: 예측 확률은 0~1이어야 합니다."
            )
