"""Domain quantities: units, coordinate datums, and raw-preserving conversion.

Import layer stores the *original representation* exactly as received
(:class:`RawValue`).  Derived layers normalise via :func:`normalise`; values that
cannot be converted stay in place with a diagnostic rather than being silently
dropped to keep a plot working.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# Linear unit conversions to SI canonical base.
# velocities -> m/s; lengths -> m; time -> s; angles -> degrees.
_UNIT_FACTORS: Dict[str, float] = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "knot": 0.5144444444444445,
    "m": 1.0,
    "dm": 0.1,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "s": 1.0,
    "ms": 0.001,
}

_ANGLE_FACTORS: Dict[str, float] = {
    "deg": 1.0,
    "degree": 1.0,
    "rad": 180.0 / math.pi,
}

HEADING_DATUMS = ("magnetic", "true")
VELOCITY_FRAMES = ("beam", "instrument", "ship", "earth")
EARTH_DATUMS = ("ENU", "NED", "XYZ")


@dataclass(frozen=True)
class RawValue:
    """Exactly what the importer saw. Never mutated by downstream code."""

    value: Any
    unit: Optional[str]
    datum: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Quantity:
    """A normalised scalar with its unit and optional coordinate datum."""

    value: float
    unit: str
    datum: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


def convert(raw: RawValue):
    """Convert a RawValue to canonical units.

    Returns ``(value, None)`` or ``(None, diagnostic)``.  The raw value is never
    lost: the diagnostic carries the original representation.
    """
    if raw.value is None:
        return None, Diagnostic("null_value", "value is missing/null", raw.to_dict())
    try:
        number = float(raw.value)
    except (TypeError, ValueError):
        return None, Diagnostic(
            "not_numeric",
            "cannot interpret %r as numeric" % (raw.value,),
            raw.to_dict(),
        )
    if math.isnan(number):
        return None, Diagnostic("nan_value", "value is NaN", raw.to_dict())

    unit = (raw.unit or "").strip()
    if unit in _ANGLE_FACTORS:
        return number * _ANGLE_FACTORS[unit], None
    if unit in _UNIT_FACTORS:
        return number * _UNIT_FACTORS[unit], None
    if unit == "":
        return number, Diagnostic(
            "undeclared_unit",
            "no unit declared; value passed through without scaling",
            raw.to_dict(),
        )
    return None, Diagnostic(
        "unknown_unit", "no conversion rule for unit %r" % (unit,), raw.to_dict()
    )


def normalise(raw: RawValue, canonical_unit):
    """Convert raw to ``canonical_unit``; on failure the caller keeps the raw."""
    value, diag = convert(raw)
    if value is None:
        return None, diag
    if canonical_unit not in ("m/s", "m", "s", "deg"):
        return None, Diagnostic(
            "unsupported_canonical",
            "canonical unit %r is not supported" % (canonical_unit,),
            raw.to_dict(),
        )
    return Quantity(value=value, unit=canonical_unit, datum=raw.datum), diag


@dataclass(frozen=True)
class Vec:
    """A 3-vector tagged with its coordinate frame and datum."""

    x: float
    y: float
    z: float
    frame: str
    datum: Optional[str] = None
    unit: str = "m/s"

    def array(self):
        return (self.x, self.y, self.z)

    def to_dict(self):
        return asdict(self)


def as_deg(rad):
    return rad * 180.0 / math.pi


def as_rad(deg):
    return deg * math.pi / 180.0


def wrap_360(angle_deg):
    """Wrap any angle into ``[0, 360)`` without losing information."""
    return angle_deg % 360.0


def angle_diff_short(start_deg, end_deg):
    """Shortest signed difference ``end - start`` accounting for the 360 wrap."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0
