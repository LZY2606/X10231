"""Units / datums.

All domain quantities carry a unit or datum.  Imports keep the original
representation; the derived layer is the *only* place normalization happens.
Values that cannot be converted stay where they are and carry a diagnostic
instead of being silently dropped for plotting.
"""
from __future__ import annotations

import math
from typing import Any

# velocity: factor multiplies the stored number to obtain m/s
VELOCITY_TO_MPS: dict[str, float] = {
    "m/s": 1.0,
    "m s-1": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "knot": 0.5144444444,
    "knots": 0.5144444444,
}

# angle: factor multiplies the stored number to obtain degrees
ANGLE_TO_DEG: dict[str, float] = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "rad": 180.0 / math.pi,
    "radian": 180.0 / math.pi,
}

LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "cm": 0.01,
    "dm": 0.1,
}


class ConversionError(ValueError):
    def __init__(self, message: str, raw: Any, unit: str):
        super().__init__(message)
        self.raw = raw
        self.unit = unit


def _finite_number(raw: Any) -> float:
    if isinstance(raw, bool):
        raise ConversionError("boolean is not a measurement", raw, "")
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            raise ConversionError("non-finite measurement", raw, "")
        return value
    raise ConversionError("non-numeric measurement", raw, "")


def to_mps(raw: Any, unit: str) -> tuple[float | None, str | None]:
    """Normalize a velocity to m/s.

    Returns (value_mps, diagnostic).  Exactly one element is None.
    """
    try:
        value = _finite_number(raw)
        factor = VELOCITY_TO_MPS.get(unit)
        if factor is None:
            return None, f"unknown velocity unit {unit!r}; kept at source"
        return value * factor, None
    except ConversionError as exc:
        return None, str(exc)


def to_deg(raw: Any, unit: str) -> tuple[float | None, str | None]:
    try:
        value = _finite_number(raw)
        factor = ANGLE_TO_DEG.get(unit)
        if factor is None:
            return None, f"unknown angle unit {unit!r}; kept at source"
        return value * factor, None
    except ConversionError as exc:
        return None, str(exc)


def to_meters(raw: Any, unit: str) -> tuple[float | None, str | None]:
    try:
        value = _finite_number(raw)
        factor = LENGTH_TO_M.get(unit)
        if factor is None:
            return None, f"unknown length unit {unit!r}; kept at source"
        return value * factor, None
    except ConversionError as exc:
        return None, str(exc)


def wrap_360(deg: float) -> float:
    """Wrap an angle in degrees to [0, 360)."""
    return deg % 360.0


def wrap_pm180(deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0
