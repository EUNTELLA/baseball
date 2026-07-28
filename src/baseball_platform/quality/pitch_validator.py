"""투구 좌표 입력 데이터의 형식과 값 범위를 검증한다."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REQUIRED_FIELDS = {
    "pitch_id",
    "pitcher_id",
    "pitched_at",
    "target_x",
    "target_y",
    "actual_x",
    "actual_y",
    "zone_width",
    "zone_height",
    "pitch_type",
}


@dataclass(frozen=True)
class Pitch:
    pitch_id: str
    pitcher_id: str
    pitched_at: datetime
    target_x: float
    target_y: float
    actual_x: float
    actual_y: float
    zone_width: float
    zone_height: float
    pitch_type: str


class PitchValidationError(ValueError):
    """입력 파일에 분석할 수 없는 값이 있을 때 발생한다."""


def load_and_validate_pitches(path: str | Path) -> list[Pitch]:
    """CSV를 읽고 검증된 투구 목록을 반환한다."""
    csv_path = Path(path)
    errors: list[str] = []
    pitches: list[Pitch] = []
    seen_ids: set[str] = set()

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_fields = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise PitchValidationError(f"필수 열이 없습니다: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                pitch = _parse_pitch(row, row_number)
                if pitch.pitch_id in seen_ids:
                    raise PitchValidationError(
                        f"{row_number}행: 중복 pitch_id '{pitch.pitch_id}'"
                    )
                seen_ids.add(pitch.pitch_id)
                pitches.append(pitch)
            except (PitchValidationError, TypeError, ValueError) as exc:
                errors.append(str(exc))

    if errors:
        raise PitchValidationError("\n".join(errors))
    if not pitches:
        raise PitchValidationError("투구 데이터가 없습니다.")
    return pitches


def _parse_pitch(row: dict[str, str], row_number: int) -> Pitch:
    text_fields = ("pitch_id", "pitcher_id", "pitched_at", "pitch_type")
    for field in text_fields:
        if not (row.get(field) or "").strip():
            raise PitchValidationError(f"{row_number}행: {field} 값이 비어 있습니다.")

    try:
        pitched_at = datetime.fromisoformat(row["pitched_at"])
        target_x = float(row["target_x"])
        target_y = float(row["target_y"])
        actual_x = float(row["actual_x"])
        actual_y = float(row["actual_y"])
        zone_width = float(row["zone_width"])
        zone_height = float(row["zone_height"])
    except (TypeError, ValueError) as exc:
        raise PitchValidationError(
            f"{row_number}행: 날짜 또는 숫자 형식이 올바르지 않습니다."
        ) from exc

    if zone_width <= 0 or zone_height <= 0:
        raise PitchValidationError(
            f"{row_number}행: zone_width와 zone_height는 0보다 커야 합니다."
        )
    if not (-1 <= target_x <= 1 and -1 <= target_y <= 1):
        raise PitchValidationError(
            f"{row_number}행: 목표 좌표는 정규화 존(-1~1) 안에 있어야 합니다."
        )

    return Pitch(
        pitch_id=row["pitch_id"].strip(),
        pitcher_id=row["pitcher_id"].strip(),
        pitched_at=pitched_at,
        target_x=target_x,
        target_y=target_y,
        actual_x=actual_x,
        actual_y=actual_y,
        zone_width=zone_width,
        zone_height=zone_height,
        pitch_type=row["pitch_type"].strip().upper(),
    )
