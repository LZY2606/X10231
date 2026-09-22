"""Units and coordinate bases.

Every domain value keeps its *original* representation at import time; only the
derived layer normalizes.  A value that cannot be converted stays in place with
a diagnostic instead of being silently dropped for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# factor to a canonical unit, per dimension
_TO_CANONICAL = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "knot": 0.514444,
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "degree": 1.0,
    "radian": 180.0 / math.pi,
    "s": 1.0,
    "ms": 0.001,
    "decimal_degrees": 1.0,
}

_DIMENSION = {
    "m/s": "velocity",
    "cm/s": "velocity",
    "mm/s": "velocity",
    "knot": "velocity",
    "m": "length",
    "cm": "length",
    "mm": "length",
    "ft": "length",
    "degree": "angle",
    "radian": "angle",
    "s": "time",
    "ms": "time",
    "decimal_degrees": "angle",
}


class UnitError(ValueError):
    """Raised when two units are incompatible or a unit is unknown."""


def dimension_of(unit: str) -> str:
    try:
        return _DIMENSION[unit]
    except KeyError as exc:
        raise UnitError("unknown unit: %r" % unit) from exc


def convert(value: float, src_unit: str, dst_unit: str) -> float:
    """Convert ``value`` between two registered units of one dimension."""
    if src_unit == dst_unit:
        return value
    a = _TO_CANONICAL.get(src_unit)
    b = _TO_CANONICAL.get(dst_unit)
    if a is None or b is None:
        raise UnitError("cannot convert %r -> %r" % (src_unit, dst_unit))
    if dimension_of(src_unit) != dimension_of(dst_unit):
        raise UnitError("dimension mismatch: %r vs %r" % (src_unit, dst_unit))
    return value * a / b


@dataclass
class Quantity:
    """A scalar that carries its unit."""

    value: float
    unit: str

    def to(self, unit: str) -> "Quantity":
        return Quantity(convert(self.value, self.unit, unit), unit)

    @property
    def dimension(self) -> str:
        return dimension_of(self.unit)

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}


@dataclass
class RawValue:
    """A value exactly as imported, plus an optional normalized view.

    ``raw``/``raw_unit`` are never rewritten.  ``normalized`` is None when the
    value cannot be converted; ``diagnostic`` then explains why.
    """

    raw: float
    raw_unit: str
    normalized: Optional[Quantity] = None
    diagnostic: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "raw_unit": self.raw_unit,
            "normalized": self.normalized.to_dict() if self.normalized else None,
            "diagnostic": self.diagnostic,
        }


def normalize(raw: float, raw_unit: str, canonical_unit: str) -> RawValue:
    """Best-effort normalization that never throws.

    Inconvertible values remain at their original location with a diagnostic.
    """
    try:
        q = Quantity(raw, raw_unit).to(canonical_unit)
        return RawValue(raw=raw, raw_unit=raw_unit, normalized=q)
    except UnitError as exc:
        return RawValue(raw=raw, raw_unit=raw_unit, normalized=None, diagnostic=str(exc))


# Coordinate bases are first-class: every derived vector names the basis that
# its components live in, so different frames can never be mistaken for values.
BASES = ("beam", "instrument", "ship", "earth")

# Conventional transform chain; alternatives are fingerprinted explicitly.
CANONICAL_ORDER = ("instrument", "ship", "earth")


def basis_ok(basis: str) -> bool:
    return basis in BASES
