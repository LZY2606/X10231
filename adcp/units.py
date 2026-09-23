"""Quantities with explicit unit / datum.

Raw representations are preserved at import time; only the derivation layer
normalizes via :func:`to_si`.  Values that cannot be converted stay where they
are and carry a diagnostic instead of being dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

# SI target unit -> set of accepted source units
_FACTORS: Dict[str, Dict[str, float]] = {
    "m/s": {"mm/s": 1e-3, "cm/s": 1e-2, "knot": 0.5144444444},
    "m": {"mm": 1e-3, "cm": 1e-2, "km": 1e3, "ft": 0.3048},
    "deg": {"rad": 180.0 / math.pi, "arcmin": 1.0 / 60.0},
    "deg_C": {},
    "count": {},
    "s": {"ms": 1e-3, "min": 60.0, "h": 3600.0},
    "Hz": {"kHz": 1e3, "MHz": 1e6},
    "nT": {"uT": 1e3},
}

# Angle-like quantities whose values must be wrapped into [0, 360).
_WRAP = {"deg"}


class UnitError(ValueError):
    """Raised when a raw quantity cannot be normalized to the target unit."""


@dataclass(frozen=True)
class Quant:
    """A scalar kept in its original representation plus explicit unit."""

    value: Any
    unit: str
    si: Optional[float] = None
    diagnostic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"value": self.value, "unit": self.unit}
        if self.si is not None:
            out["si"] = self.si
        if self.diagnostic is not None:
            out["diagnostic"] = self.diagnostic
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Quant":
        return cls(
            value=d.get("value"),
            unit=d.get("unit", ""),
            si=d.get("si"),
            diagnostic=d.get("diagnostic"),
        )

    @property
    def ok(self) -> bool:
        return self.si is not None and self.diagnostic is None


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees into ``[0, 360)``."""
    return a % 360.0


def _angle_delta(a: float, b: float) -> float:
    """Signed shortest-path delta ``b - a`` for two degree angles."""
    return (b - a + 180.0) % 360.0 - 180.0


def to_si(value: Any, unit: Optional[str], si_unit: str) -> Quant:
    """Convert ``value`` expressed in ``unit`` to ``si_unit``.

    A missing/unparseable value or an unknown unit produces a :class:`Quant`
    whose ``diagnostic`` explains the failure; conversion never raises for
    ordinary bad data so that unconvertible values remain in place.
    """
    if value is None or value == "":
        return Quant(value=value, unit=unit or "", diagnostic="MISSING_VALUE")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return Quant(value=value, unit=unit or "", diagnostic="NON_NUMERIC")
    if math.isnan(x):
        return Quant(value=value, unit=unit or "", diagnostic="NAN")
    if unit is None:
        return Quant(value=value, unit="", diagnostic="MISSING_UNIT")
    if unit == si_unit:
        si = x
    elif unit in _FACTORS.get(si_unit, {}):
        si = x * _FACTORS[si_unit][unit]
    else:
        return Quant(value=value, unit=unit, diagnostic=f"INCOMPATIBLE_UNIT({unit}->{si_unit})")
    if si_unit in _WRAP:
        si = wrap_deg(si)
    return Quant(value=value, unit=unit, si=si)


def lerp_angle(a: float, b: float, frac: float) -> float:
    """Linear interpolation between two degree angles on the shortest path."""
    return wrap_deg(a + _angle_delta(a, b) * frac)


def lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac
