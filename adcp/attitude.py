from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AttitudeSample:
    source_id: str
    time: datetime
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    heading_unit: str = "deg"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def shortest_heading_delta(left: float, right: float) -> float:
    return ((right - left + 540.0) % 360.0) - 180.0


def interpolate_attitude(
    target_time: datetime,
    samples: list[AttitudeSample],
    maximum_gap_seconds: float = 1.5,
) -> tuple[dict[str, Any] | None, list[str]]:
    ordered = sorted(samples, key=lambda item: item.time)
    diagnostics: list[str] = []
    exact = next((sample for sample in ordered if sample.time == target_time), None)
    if exact:
        return {
            "heading_deg": exact.heading_deg % 360.0,
            "pitch_deg": exact.pitch_deg,
            "roll_deg": exact.roll_deg,
            "mode": "exact",
            "source_ids": [exact.source_id],
        }, diagnostics

    after = next((sample for sample in ordered if sample.time > target_time), None)
    before = max((sample for sample in ordered if sample.time < target_time), key=lambda s: s.time, default=None)
    if before is None or after is None:
        return None, ["attitude cannot be extrapolated; interpolation requires samples on both sides"]
    gap = (after.time - before.time).total_seconds()
    if gap > maximum_gap_seconds:
        return None, [
            f"attitude gap {gap:.1f}s exceeds {maximum_gap_seconds:.1f}s; interpolation is forbidden across the gap"
        ]
    fraction = (target_time - before.time).total_seconds() / gap
    heading = (before.heading_deg + shortest_heading_delta(before.heading_deg, after.heading_deg) * fraction) % 360.0
    return {
        "heading_deg": heading,
        "pitch_deg": before.pitch_deg + (after.pitch_deg - before.pitch_deg) * fraction,
        "roll_deg": before.roll_deg + (after.roll_deg - before.roll_deg) * fraction,
        "mode": "interpolated_shortest_heading",
        "source_ids": [before.source_id, after.source_id],
        "gap_seconds": gap,
    }, diagnostics
