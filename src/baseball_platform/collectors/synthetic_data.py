"""공식 데이터가 없을 때 파이프라인 검증에 사용할 합성 데이터를 만든다."""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

from baseball_platform.contracts import DataContract


def generate_synthetic_datasets(
    output_directory: str | Path,
    contract: DataContract,
    *,
    rows_per_season: int = 240,
    seed: int = 42,
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    all_rows: list[dict[str, object]] = []
    pitcher_rates = {f"P{i:02d}": rng.uniform(0.48, 0.75) for i in range(1, 9)}

    for season in (*contract.train_seasons, contract.test_season):
        season_start = date(season, 3, 20)
        for index in range(rows_per_season):
            pitcher_id = f"P{index % 8 + 1:02d}"
            game_number = index // 24
            game_date = season_start + timedelta(days=game_number)
            balls = rng.randrange(4)
            strikes = rng.randrange(3)
            outs = rng.randrange(3)
            leverage = round(rng.uniform(0.4, 3.0), 4)
            velocity = rng.gauss(144 + (index % 8), 2.2)
            release_height = rng.gauss(1.78, 0.08)
            success_logit = (
                math.log(pitcher_rates[pitcher_id] / (1 - pitcher_rates[pitcher_id]))
                - 0.20 * balls
                + 0.12 * strikes
                - 0.08 * max(0, leverage - 1)
                - 0.35 * abs(release_height - 1.78)
            )
            success_probability = 1 / (1 + math.exp(-success_logit))
            row: dict[str, object] = {
                "pitch_id": f"{season}_{index + 1:05d}",
                "game_id": f"G{season}_{game_number + 1:04d}",
                "pitcher_id": pitcher_id,
                "batter_id": f"B{index % 20 + 1:03d}",
                "team_id": f"T{index % 4 + 1:02d}",
                "season": season,
                "game_date": game_date.isoformat(),
                "pitch_sequence": index + 1,
                "balls": balls,
                "strikes": strikes,
                "outs": outs,
                "inning": index % 9 + 1,
                "score_diff": rng.randrange(-5, 6),
                "runner_1b": rng.randrange(2),
                "runner_2b": rng.randrange(2),
                "runner_3b": rng.randrange(2),
                "win_expectancy": round(rng.uniform(0.05, 0.95), 4),
                "leverage_index": leverage,
                "asof_success_rate": round(pitcher_rates[pitcher_id], 4),
                "recent_game_success_rate": round(
                    min(1, max(0, pitcher_rates[pitcher_id] + rng.gauss(0, 0.08))), 4
                ),
                "asof_fastball_rate": round(rng.uniform(0.35, 0.75), 4),
                "asof_velocity": round(velocity, 3),
                "asof_spin_rate": round(rng.gauss(2250, 160), 3),
                "asof_horizontal_movement": round(rng.gauss(0, 18), 3),
                "asof_vertical_movement": round(rng.gauss(35, 9), 3),
                "asof_release_height": round(release_height, 4),
                "asof_release_side": round(rng.gauss(-0.4, 0.25), 4),
                "asof_extension": round(rng.gauss(1.85, 0.12), 4),
                "pitch_type_history": rng.choice(
                    ["FASTBALL", "SLIDER", "CURVEBALL", "CHANGEUP"]
                ),
            }
            if season in contract.train_seasons:
                row[contract.target_column] = int(rng.random() < success_probability)
            all_rows.append(row)

    train_rows = [
        row for row in all_rows if row["season"] in contract.train_seasons
    ]
    test_rows = [
        row for row in all_rows if row["season"] == contract.test_season
    ]
    train_path = output / "synthetic_train.csv"
    test_path = output / "synthetic_test.csv"
    _write_rows(train_path, train_rows)
    _write_rows(test_path, test_rows)
    return train_path, test_path


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
