"""Units and coordinate-frame bookkeeping.

Domain quantities always carry a unit or coordinate datum. Raw representations
are preserved at import time; normalization happens only in the derived layer.
"""
from __future__ import annotations

from dataclasses import dataclass

ANGLE_UNITS = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 0.0, "radian": 0.0}
VELOCITY_UNITS = {"m/s": 1.0, "mps": 1.0, "cm/s": 0.01, "mm/s": 0.001}
TIME_UNITS = {"s": 1.0, "sec": 1.0, "ms": 0.001}
LENGTH_UNITS = {"m": 1.0, "dm": 0.1, "cm": 0.01, "ft": 0.3048}


class Unconvertible(ValueError):
    """Raised when a raw value cannot be converted to the canonical unit."""


@dataclass(frozen=True)
class Quantity:
    """A value paired with its physical unit and, optionally, coordinate basis."""

    value: float
    unit: str
    basis: str | None = None

    def to(self, unit: str) -> "Quantity":
        return Quantity(convert(self.value, self.unit, unit), unit, self.basis)


def convert(value: float, src: str, dst: str) -> float:
    src = src.strip()
    dst = dst.strip()
    if src in VELOCITY_UNITS and dst in VELOCITY_UNITS:
        return value * VELOCITY_UNITS[src] / VELOCITY_UNITS[dst]
    if src in LENGTH_UNITS and dst in LENGTH_UNITS:
        return value * LENGTH_UNITS[src] / LENGTH_UNITS[dst]
    if src in TIME_UNITS and dst in TIME_UNITS:
        return value * TIME_UNITS[src] / TIME_UNITS[dst]
    if src in ANGLE_UNITS and dst in ANGLE_UNITS:
        v = value
        if src.startswith("rad"):
            import math

            v = math.degrees(v)
        if dst.startswith("rad"):
            import math

            v = math.radians(v)
        return v
    raise Unconvertible(f"cannot convert {src!r} -> {dst!r}")


def try_number(raw) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise Unconvertible(f"not numeric: {raw!r}")
    import math

    if math.isnan(v) or math.isinf(v):
        raise Unconvertible(f"not finite: {raw!r}")
    return v
