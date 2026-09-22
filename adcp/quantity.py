"""Domain quantities that always carry their unit and/or datum.

Raw representation is preserved verbatim on import. The derived layer asks for
normalized SI values; a value that cannot be converted is kept in place with a
diagnostic instead of being dropped so the UI can still draw everything else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Linear speed factors to m/s.
SPEED_TO_SI = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
}
# Angle factors to radians.
ANGLE_TO_RAD = {
    "deg": math.pi / 180.0,
    "rad": 1.0,
}
# Length factors to metres.
LENGTH_TO_SI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048}
# Correlation / amplitude are dimensionless counts; their scale is explicit.
COUNT_TO_SI = {"count": 1.0, "percent": 0.01}


@dataclass
class Quantity:
    """A measurement preserved as imported plus an optional SI normalization."""

    raw: Any
    unit: str
    datum: Optional[str] = None          # e.g. "magnetic" / "true", frame name
    si_value: Optional[float] = None
    si_unit: Optional[str] = None
    convertible: bool = True
    diagnostic: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "datum": self.datum,
            "si_value": self.si_value,
            "si_unit": self.si_unit,
            "convertible": self.convertible,
            "diagnostic": self.diagnostic,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Quantity":
        return Quantity(**{k: data.get(k) for k in (
            "raw", "unit", "datum", "si_value", "si_unit",
            "convertible", "diagnostic", "meta")})


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def speed(raw: Any, unit: str = "m/s", datum: Optional[str] = None) -> Quantity:
    value = _finite(raw)
    factor = SPEED_TO_SI.get(unit)
    if value is None or factor is None:
        return Quantity(raw, unit, datum, convertible=False, si_unit="m/s",
                        diagnostic="unconvertible-speed")
    return Quantity(raw, unit, datum, si_value=value * factor, si_unit="m/s")


def angle(raw: Any, unit: str = "deg", datum: Optional[str] = None) -> Quantity:
    value = _finite(raw)
    factor = ANGLE_TO_RAD.get(unit)
    if value is None or factor is None:
        return Quantity(raw, unit, datum, convertible=False, si_unit="rad",
                        diagnostic="unconvertible-angle")
    return Quantity(raw, unit, datum, si_value=value * factor, si_unit="rad")


def length(raw: Any, unit: str = "m") -> Quantity:
    value = _finite(raw)
    factor = LENGTH_TO_SI.get(unit)
    if value is None or factor is None:
        return Quantity(raw, unit, None, convertible=False, si_unit="m",
                        diagnostic="unconvertible-length")
    return Quantity(raw, unit, None, si_value=value * factor, si_unit="m")


def count(raw: Any, unit: str = "count") -> Quantity:
    value = _finite(raw)
    factor = COUNT_TO_SI.get(unit)
    if value is None or factor is None:
        return Quantity(raw, unit, None, convertible=False, si_unit="count",
                        diagnostic="unconvertible-count")
    return Quantity(raw, unit, None, si_value=value * factor, si_unit="count")
