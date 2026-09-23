"""Attitude interpolation and ordered instrument/ship/earth rotations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import convert_angle


def _r(angle: float, axis: str) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_chain(pitch_rad: float, roll_rad: float, heading_true_rad: float,
                   order: str = "zyx") -> List[Tuple[str, np.ndarray]]:
    if order == "zyx":
        return [("heading/z", _r(heading_true_rad, "z")),
                ("pitch/y", _r(pitch_rad, "y")),
                ("roll/x", _r(roll_rad, "x"))]
    if order == "zxy":
        return [("heading/z", _r(heading_true_rad, "z")),
                ("roll/x", _r(roll_rad, "x")),
                ("pitch/y", _r(pitch_rad, "y"))]
    raise ValueError(f"unknown rotation order: {order}")


def instrument_to_ship(vector: Sequence[float], pitch_rad: float, roll_rad: float,
                       order: str = "zyx") -> Tuple[np.ndarray, List[dict]]:
    chain = rotation_chain(pitch_rad, roll_rad, 0.0, order)
    result = np.asarray(vector, dtype=float)
    stages = []
    for name, matrix in chain:
        if name.startswith("heading"):
            continue
        result = matrix @ result
        stages.append({"stage": f"instrument->ship:{name}", "matrix_version": f"intrinsic-{order}",
                       "vector_mps": result.tolist()})
    return result, stages


def ship_to_earth(vector: Sequence[float], heading_true_rad: float) -> np.ndarray:
    return _r(heading_true_rad, "z") @ np.asarray(vector, dtype=float)


@dataclass
class InterpResult:
    values: Dict[str, float]
    method: str
    bounded_by: Tuple[str, str]
    crossed_zero_heading: bool
    diagnostic: Optional[str] = None


def _interp_scalar(samples: List[dict], ping_time: float) -> Optional[float]:
    prior = [s for s in samples if s["ping_time"] <= ping_time]
    after = [s for s in samples if s["ping_time"] >= ping_time]
    if not prior or not after:
        return None
    a, b = max(prior, key=lambda s: s["ping_time"]), min(after, key=lambda s: s["ping_time"])
    if a["ping_time"] == b["ping_time"]:
        return float(a["raw_value"])
    span = b["ping_time"] - a["ping_time"]
    if span <= 0:
        return None
    ratio = (ping_time - a["ping_time"]) / span
    return float(a["raw_value"]) + (float(b["raw_value"]) - float(a["raw_value"])) * ratio


def interpolate_attitude(samples: List[dict], ping_time: float,
                         max_gap_s: float = 2.5) -> InterpResult:
    """Linear interpolation inside a gap only.

    Heading is unwrapped locally, so a 359°→1° transition crosses the
    representation boundary without a 358° synthetic turn. Missing samples
    are never bridged across a gap larger than the source cadence limit.
    """
    exact = [s for s in samples if abs(s["ping_time"] - ping_time) < 1e-9]
    if exact:
        s = exact[0]
        return InterpResult({s["field"]: float(s["raw_value"])}, "exact",
                            (s["record_id"], s["record_id"]), False)

    prior = [s for s in samples if s["ping_time"] < ping_time]
    after = [s for s in samples if s["ping_time"] > ping_time]
    if not prior or not after:
        return InterpResult({}, "none", ("", ""), False, "attitude sample outside source interval")
    a = max(prior, key=lambda s: s["ping_time"])
    b = min(after, key=lambda s: s["ping_time"])
    gap = b["ping_time"] - a["ping_time"]
    if gap > max_gap_s:
        return InterpResult({}, "gap-rejected", (a["record_id"], b["record_id"]), False,
                            f"attitude gap {gap:g}s exceeds interpolation limit {max_gap_s:g}s")

    by_field: Dict[str, List[dict]] = {}
    for sample in (a, b):
        by_field.setdefault(sample["field"], []).append(sample)
    values: Dict[str, float] = {}
    crossed = False
    for field, pair in by_field.items():
        if len(pair) != 2:
            continue
        lo, hi = pair if pair[0]["ping_time"] < pair[1]["ping_time"] else (pair[1], pair[0])
        x, y = float(lo["raw_value"]), float(hi["raw_value"])
        if field == "heading":
            wrapped_delta = (y - x + 180.0) % 360.0 - 180.0
            crossed = abs(wrapped_delta) < abs(y - x) - 1e-9
            y = x + wrapped_delta
        ratio = (ping_time - lo["ping_time"]) / (hi["ping_time"] - lo["ping_time"])
        values[field] = x + (y - x) * ratio
    if values.get("heading", 0.0) < 0:
        values["heading"] %= 360.0
    return InterpResult(values, "linear-bounded", (a["record_id"], b["record_id"]), crossed)
