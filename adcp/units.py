"""Quantities always carry a unit / coordinate basis.

Raw representation from the source file is preserved on import; the derived
layers are the only place where normalization happens. Values that cannot be
converted stay at the original position together with a diagnostic instead of
being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Canonical units used after normalization in the derived layer.
UNITS = {
    "velocity": "m/s",
    "angle_deg": "deg",
    "angle_rad": "rad",
    "time": "s",
    "distance": "m",
    "correlation": "counts",
    "amplitude": "counts",
}

BASES = {
    "beam": "beam frame (positive away from transducer)",
    "instrument": "instrument frame (x forward, y starboard, z up)",
    "ship": "ship frame after mounting alignment",
    "earth_enu": "earth frame ENU (z up, true heading)",
    "magnetic": "magnetic reference",
    "true": "true geographic reference",
}


@dataclass
class Quantity:
    """A domain value together with its unit, reference basis and raw text."""

    value: Optional[float]
    unit: str
    basis: str = ""
    raw: str = ""
    diagnostic: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and self.diagnostic == ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "basis": self.basis,
            "raw": self.raw,
            "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Quantity":
        return cls(
            value=data.get("value"),
            unit=data.get("unit", ""),
            basis=data.get("basis", ""),
            raw=data.get("raw", ""),
            diagnostic=data.get("diagnostic", ""),
        )


def parse_quantity(raw: Any, unit: str, basis: str = "") -> Quantity:
    """Parse an imported cell without losing the original representation."""
    raw_text = str(raw)
    if raw is None or (isinstance(raw, str) and raw.strip() in {"", "NA", "NaN", "null", "None", "-"}):
        return Quantity(None, unit, basis, raw_text, diagnostic="missing_value")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Quantity(None, unit, basis, raw_text, diagnostic=f"unparseable:{raw_text!r}")
    if value != value:  # NaN
        return Quantity(None, unit, basis, raw_text, diagnostic="nan_value")
    return Quantity(value, unit, basis, raw_text)
