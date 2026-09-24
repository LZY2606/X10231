"""Units and coordinate datums.

Raw import stores the original representation verbatim.  The derivation layer
asks :func:`normalize` for a canonical value + unit; when a value cannot be
converted the original representation stays in place together with a
diagnostic instead of being dropped for plotting convenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# SI / canonical target units used by the derivation layer.
TO_SI = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "deg": 1.0,   # angles normalise below, unit retained as deg
    "rad": None,  # handled specially
    "count": 1.0,
    "percent": 1.0,
    "dB": 1.0,
    "s": 1.0,
    "ms": 0.001,
}


class UnitError(ValueError):
    """Raised when a value cannot be converted to the requested unit."""


@dataclass(frozen=True)
class Quantity:
    """A normalised domain quantity: value plus unit/datum identity."""

    value: float
    unit: str
    datum: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "datum": self.datum}


def normalize(raw: Any, unit: str, datum: str | None = None) -> Quantity:
    """Convert ``raw`` expressed in ``unit`` to canonical derivation units.

    Angles are always expressed in degrees by the derivation layer.
    Raises :class:`UnitError` for unknown units so callers can keep the raw
    representation with a diagnostic instead of silently dropping data.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise UnitError(f"non-numeric value {raw!r}") from exc

    if unit == "rad":
        import math

        return Quantity(math.degrees(value), "deg", datum)
    factor = TO_SI.get(unit)
    if factor is None:
        raise UnitError(f"unknown unit {unit!r}")
    converted = value * factor
    if unit == "deg":
        return Quantity(converted, "deg", datum)
    target = "m/s" if unit.endswith("/s") else ("s" if unit in ("s", "ms") else "m")
    if unit in ("count", "percent", "dB"):
        target = unit
    return Quantity(converted, target, datum)


def wrap_deg_0_360(angle: float) -> float:
    """Wrap any angle into [0, 360)."""
    return angle % 360.0


def wrap_deg_180(angle: float) -> float:
    """Wrap any angle into (-180, 180]."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 360.0 if wrapped == -180.0 else wrapped


def unwrap_deg(series: list[float]) -> list[float]:
    """Remove 359 -> 1 style jumps so interpolation sees a continuous path."""
    if not series:
        return []
    out = [float(series[0])]
    for value in series[1:]:
        delta = (value - out[-1] + 180.0) % 360.0 - 180.0
        out.append(out[-1] + delta)
    return out
