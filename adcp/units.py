"""Unit registry.

Raw records keep their original unit/basis. Normalization happens only in the
derived layer. Values whose unit cannot be converted stay in place and carry a
diagnostic instead of being silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# unit -> (dimension, factor to canonical unit)
UNITS = {
    "m/s": ("velocity", 1.0),
    "mm/s": ("velocity", 0.001),
    "cm/s": ("velocity", 0.01),
    "deg": ("angle", 1.0),
    "rad": ("angle", 180.0 / math.pi),
    "count": ("counts", 1.0),
    "dB": ("level", 1.0),
    "percent": ("ratio", 1.0),
    "m": ("length", 1.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "s": ("time", 1.0),
}

CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "counts": "count",
    "level": "dB",
    "ratio": "percent",
    "length": "m",
    "time": "s",
}


@dataclass
class Quantity:
    value: float | None
    unit: str
    basis: str = "raw"          # e.g. raw / beam / instrument / ship / earth
    diagnostic: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "value": self.value,
            "unit": self.unit,
            "basis": self.basis,
            "diagnostic": self.diagnostic,
            "extra": self.extra,
        }


def normalize(value, unit):
    """Return (canonical_value, canonical_unit, diagnostic|None).

    Unconvertible units return the raw value unchanged plus a diagnostic; the
    caller must keep the raw record in place.
    """
    if value is None:
        return None, unit, "missing value"
    info = UNITS.get(unit)
    if info is None:
        return value, unit, f"unconvertible unit '{unit}': value left in raw position"
    dim, factor = info
    return value * factor, CANONICAL[dim], None
