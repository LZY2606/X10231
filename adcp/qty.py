"""Quantities: every domain value carries a unit or datum.

Raw representation is preserved at import time; the derived layer is the only
place where normalization happens. Values that cannot be converted stay in
place with a diagnostic instead of being dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# canonical units used inside the derived layer
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "length": "m",
    "time": "s_since_epoch",
    "dimensionless": "1",
}

# conservative, explicit conversion table; anything not listed here is
# deliberately left un-converted with a diagnostic (no silent guessing)
CONVERSIONS = {
    ("mm/s", "m/s"): 0.001,
    ("cm/s", "m/s"): 0.01,
    ("m/s", "m/s"): 1.0,
    ("knot", "m/s"): 0.5144444,
    ("deg", "deg"): 1.0,
    ("rad", "deg"): 180.0 / math.pi,
    ("degree", "deg"): 1.0,
    ("m", "m"): 1.0,
    ("dm", "m"): 0.1,
    ("cm", "m"): 0.01,
    ("1", "1"): 1.0,
    ("count", "count"): 1.0,
    ("dB", "dB"): 1.0,
}


class ConversionError(ValueError):
    pass


@dataclass
class Qty:
    """A value together with the unit / datum it was expressed in."""

    value: Any
    unit: str
    raw: Optional[str] = None
    converted: bool = False
    diagnostic: Optional[str] = None
    datum: Optional[str] = None  # e.g. "magnetic" / "true" for headings

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Qty":
        return cls(**d)

    def display(self, digits: int = 4) -> str:
        if self.value is None:
            return f"— ({self.diagnostic or 'unconvertible'})"
        if isinstance(self.value, float):
            return f"{self.value:.{digits}g} {self.unit}"
        return f"{self.value} {self.unit}"


def normalize(q: Qty, canonical_unit: str) -> Qty:
    """Return a normalized copy of ``q`` or an in-place diagnostic copy."""
    if q.value is None:
        return _diag(q, canonical_unit, "value is null")
    pair = (q.unit, canonical_unit)
    factor = CONVERSIONS.get(pair)
    if factor is None:
        out = Qty(
            value=None,
            unit=canonical_unit,
            raw=q.raw,
            converted=False,
            diagnostic=(
                f"no conversion registered {q.unit!r} -> {canonical_unit!r}; "
                f"original kept at source"
            ),
            datum=q.datum,
        )
        return out
    try:
        val = float(q.value) * factor
    except (TypeError, ValueError):
        return _diag(q, canonical_unit, f"non-numeric value {q.value!r}")
    return Qty(value=val, unit=canonical_unit, raw=q.raw, converted=True, datum=q.datum)


def _diag(q: Qty, unit: str, msg: str) -> Qty:
    return Qty(value=None, unit=unit, raw=q.raw, converted=False, diagnostic=msg, datum=q.datum)


def wrap360(deg: float) -> float:
    """Wrap an angle in degrees to [0, 360)."""
    return deg % 360.0


def angular_delta(a: float, b: float) -> float:
    """Signed shortest-arc difference a - b in (-180, 180] degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def lerp_angle(a: float, b: float, t: float) -> float:
    """Linear interpolation along the shortest arc (survives 0/360 crossing)."""
    return wrap360(a + angular_delta(b, a) * t)
