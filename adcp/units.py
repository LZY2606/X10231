"""Domain quantities with units / coordinate datum.

Raw representations are preserved at import time; normalization happens
only in the derived layer.  Values that cannot be converted stay in place
and carry a diagnostic instead of being dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# unit -> (canonical_unit, factor)  value_canonical = value * factor
_UNIT_TABLE = {
    "mm/s": ("m/s", 0.001),
    "m/s": ("m/s", 1.0),
    "cm/s": ("m/s", 0.01),
    "count": ("count", 1.0),
    "dB": ("dB", 1.0),
    "deg": ("deg", 1.0),
    "rad": ("deg", 57.29577951308232),
    "m": ("m", 1.0),
    "percent": ("percent", 1.0),
    "s": ("s", 1.0),
}


@dataclass
class Quantity:
    """A measured or configured domain quantity.

    ``raw`` is the textual representation exactly as imported; ``value`` is
    the numeric payload in ``unit``.  ``datum`` names the coordinate basis
    (e.g. ``beam``, ``instrument``, ``ship``, ``earth:true-north``).
    """
    value: Optional[float]
    unit: str
    datum: str
    raw: str = ""
    diagnostic: Optional[str] = None

    def to_canonical(self) -> "Quantity":
        """Return a normalized copy, or self with a diagnostic if unknown."""
        if self.value is None:
            return Quantity(None, self.unit, self.datum, self.raw,
                            diagnostic=self.diagnostic or "missing-value")
        entry = _UNIT_TABLE.get(self.unit)
        if entry is None:
            return Quantity(self.value, self.unit, self.datum, self.raw,
                            diagnostic=f"unknown-unit:{self.unit}")
        canon_unit, factor = entry
        return Quantity(self.value * factor, canon_unit, self.datum, self.raw)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Quantity":
        return Quantity(d.get("value"), d.get("unit", ""), d.get("datum", ""),
                        d.get("raw", ""), d.get("diagnostic"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Quantity":
        return Quantity.from_dict(json.loads(s))
