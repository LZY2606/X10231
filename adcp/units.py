"""Units and coordinate datums.

Every domain quantity carries a unit (scalars) or a coordinate frame
(vectors). Raw import values keep their original representation; only the
derived layer normalizes (e.g. cm/s -> m/s, deg -> rad). Values that cannot
be converted stay in place with a diagnostic instead of being dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

VELOCITY_TO_MPS = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
}

ANGLE_TO_RAD = {
    "deg": math.pi / 180.0,
    "rad": 1.0,
}

LENGTH_TO_M = {
    "m": 1.0,
    "cm": 0.01,
    "dm": 0.1,
}

# Frame / datum labels used across derived vectors.
FRAME_BEAM = "beam"            # along-beam radial components
FRAME_INSTRUMENT = "instrument"  # (x: beam3->4, y: beam1->2, z: up)
FRAME_SHIP = "ship"            # instrument after mounting angles
FRAME_EARTH = "earth"          # ship after heading/pitch/roll + declination
FRAME_GEOGRAPHIC = "earth"     # alias for human display


class ConversionError(ValueError):
    """Raised when a raw value cannot be normalized to the target unit."""


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str

    def to(self, target: str, table: dict[str, float] | None = None) -> "Quantity":
        factor = (table or {}).get(self.unit)
        if factor is None or target not in (table or {}):
            raise ConversionError(f"cannot convert {self.unit!r} to {target!r}")
        return Quantity(self.value * factor / table[target], target)


def to_mps(value: float, unit: str) -> float:
    if unit not in VELOCITY_TO_MPS:
        raise ConversionError(f"unknown velocity unit {unit!r}")
    return value * VELOCITY_TO_MPS[unit]


def to_rad(value: float, unit: str) -> float:
    if unit not in ANGLE_TO_RAD:
        raise ConversionError(f"unknown angle unit {unit!r}")
    return value * ANGLE_TO_RAD[unit]


def to_m(value: float, unit: str) -> float:
    if unit not in LENGTH_TO_M:
        raise ConversionError(f"unknown length unit {unit!r}")
    return value * LENGTH_TO_M[unit]


def wrap180(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


def shortest_arc_delta(next_deg: float, prev_deg: float) -> float:
    """Signed shortest-arc difference (next-prev) in (-180, 180]."""
    return wrap180(next_deg - prev_deg)
