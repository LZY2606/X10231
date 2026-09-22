"""Units and coordinate/reference-frame metadata.

Design rule from the case study:

* Import **preserves the raw representation** of every domain quantity
  (value + its declared unit / datum).  Nothing is silently rescaled.
* Only the *derived* layer normalises into canonical SI units.
* Values that cannot be converted (unknown unit, incompatible dimension) stay
  where they were and carry a diagnostic instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# Canonical base units per dimension.
# dim -> (canonical unit, {unit: linear (scale, offset) to canonical})
_CONVERSIONS = {
    "velocity": ("m/s", {"m/s": (1.0, 0.0), "cm/s": (0.01, 0.0),
                         "mm/s": (0.001, 0.0), "knot": (0.514444, 0.0)}),
    "length": ("m", {"m": (1.0, 0.0), "cm": (0.01, 0.0), "mm": (0.001, 0.0)}),
    "angle": ("deg", {"deg": (1.0, 0.0), "rad": (57.29577951308232, 0.0)}),
    "time": ("s", {"s": (1.0, 0.0), "ms": (0.001, 0.0)}),
    "dimensionless": ("1", {"1": (1.0, 0.0), "%": (0.01, 0.0)}),
}

# Units that are inherently raw/count-like and never rescaled.
_COUNT_UNITS = {"count", "dB", "flag"}


@dataclass(frozen=True)
class Quantity:
    """A value that always carries its unit or coordinate datum.

    ``raw`` keeps the imported representation verbatim; ``value``/``unit`` are
    the normalised derived representation and are only populated after a
    successful :func:`normalise`.
    """

    value: Optional[float]
    unit: str
    dim: str
    raw_value: Optional[float] = None
    raw_unit: Optional[str] = None
    basis: Optional[str] = None  # coordinate/datum annotation, free text key
    diagnostic: Optional[str] = None

    @property
    def convertible(self) -> bool:
        return self.diagnostic is None and self.value is not None

    def as_tuple(self) -> Tuple[Optional[float], str, Optional[str]]:
        return self.value, self.unit, self.basis


def dimension_of(unit: str) -> str:
    """Return the dimension name for a known unit, else ``'unknown'``."""
    for dim, (_, table) in _CONVERSIONS.items():
        if unit in table:
            return dim
    if unit in _COUNT_UNITS:
        return "count"
    return "unknown"


def normalise(value, unit, basis=None) -> Quantity:
    """Convert a single measurement into canonical units without losing raw.

    Unknown units are retained in place with a ``diagnostic``; the caller must
    not silently replace them with zero.
    """
    raw_value, raw_unit = value, unit
    if value is None:
        return Quantity(None, unit, "unknown", raw_value, raw_unit, basis,
                        diagnostic="missing value")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return Quantity(None, str(unit), "unknown", raw_value, raw_unit,
                        basis, diagnostic="non-numeric value: %r" % (value,))
    dim = dimension_of(unit)
    if dim == "count":
        return Quantity(x, str(unit), "count", raw_value, raw_unit, basis)
    if dim == "unknown":
        return Quantity(None, str(unit), "unknown", raw_value, raw_unit,
                        basis, diagnostic="unknown unit %r; raw value retained"
                        % (unit,))
    canon, table = _CONVERSIONS[dim]
    scale, _offset = table[unit]
    return Quantity(x * scale, canon, dim, raw_value, raw_unit, basis)


def convert(value, unit, to_unit):
    """Explicit conversion with dimension compatibility check."""
    q = normalise(value, unit)
    if not q.convertible:
        return q
    to_dim = dimension_of(to_unit)
    if to_dim != q.dim:
        return Quantity(None, to_unit, to_dim, q.raw_value, q.raw_unit,
                        q.basis,
                        diagnostic="incompatible conversion %s(%s)->%s"
                        % (q.dim, unit, to_unit))
    scale, _ = _CONVERSIONS[q.dim][1][to_unit]
    out = q.value / scale
    return Quantity(out, to_unit, q.dim, q.raw_value, q.raw_unit, q.basis)


@dataclass(frozen=True)
class Vec3:
    """3-vector with an explicit coordinate basis label.

    Vectors with different bases must never be treated as comparable numbers;
    the basis string is part of the value identity.
    """

    x: float
    y: float
    z: float
    basis: str
    unit: str = "m/s"

    def as_array(self):
        import numpy as np
        return np.array([self.x, self.y, self.z], dtype=float)
