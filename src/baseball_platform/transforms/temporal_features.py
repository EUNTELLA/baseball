"""현재 행의 target을 사용하지 않는 과거 시점 특징을 생성한다."""

from __future__ import annotations

from collections import defaultdict, deque


def add_leakage_safe_history(
    rows: list[dict[str, str]],
    *,
    target_column: str = "target",
    window: int = 20,
) -> list[dict[str, object]]:
    histories: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    transformed: list[dict[str, object]] = []

    for row in rows:
        pitcher_id = row["pitcher_id"]
        history = histories[pitcher_id]
        successes, count = totals[pitcher_id]
        output: dict[str, object] = dict(row)
        output["history_pitch_count"] = count
        output["history_success_rate"] = (
            successes / count if count else 0.5
        )
        output[f"recent_{window}_success_rate"] = (
            sum(history) / len(history) if history else 0.5
        )
        transformed.append(output)

        raw_target = row.get(target_column, "")
        if raw_target not in ("", None):
            target = int(raw_target)
            totals[pitcher_id][0] += target
            totals[pitcher_id][1] += 1
            history.append(target)

    return transformed
