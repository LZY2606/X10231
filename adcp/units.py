"""Quantity representation: keep the raw value, normalise only in derived layers.

Every domain quantity carries a unit/coordinate datum. A value that cannot be
converted is kept in place with a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

_TO_CANON: Dict[str, Dict[str, float]] = {
    "velocity": {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knot": 0.514444},
    "angle": {"deg": 1.0, "rad": 57.29577951308232},
    "length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048},
    "count": {"count": 1.0},
}
CANONICAL = {"velocity": "m/s", "angle": "deg", "length": "m",
             "count": "count"}
_KIND_BY_UNIT = {u: kind for kind, units in _TO_CANON.items() for u in units}


@dataclass(frozen=True)
class Quantity:
    value: Any
    unit: str
    kind: str
    canonical_value: Optional[float]
    diagnostic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "canonical_value": self.canonical_value,
            "diagnostic": self.diagnostic,
        }

    @property
    def ok(self) -> bool:
        return self.diagnostic is None and self.canonical_value is not None


def quantity(value: Any, unit: str, allowed: Optional[str] = None) -> Quantity:
    """Parse a raw measurement. Unconvertible values stay in place."""
    kind = _KIND_BY_UNIT.get(unit)
    if kind is None:
        return Quantity(value, unit, kind or "unknown", None,
                        f"unknown unit: {unit!r}")
    if allowed is not None and kind != allowed:
        return Quantity(value, unit, kind, None,
                        f"unit {unit!r} is not {allowed}")
    try:
        num = float(value)
    except (TypeError, ValueError):
        return Quantity(value, unit, kind, None, "non-numeric value")
    return Quantity(num, unit, kind, num * _TO_CANON[kind][unit], None)


def as_canonical(value: Any, unit: str, allowed: str) -> float:
    q = quantity(value, unit, allowed)
    if not q.ok:
        raise ValueError(q.diagnostic)
    return q.canonical_value
