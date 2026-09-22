"""Domain quantities: raw representation is preserved; normalization happens
only at the derived layer. Unconvertible values stay in place with a diagnostic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_NUM_PREFIX = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

_UNIT_TO_SI = {
    "m/s": ("m/s", 1.0),
    "mm/s": ("m/s", 1e-3),
    "cm/s": ("m/s", 1e-2),
    "deg": ("deg", 1.0),
    "rad": ("deg", 180.0 / math.pi),
    "count": ("count", 1.0),
    "dB": ("dB", 1.0),
    "m": ("m", 1.0),
    "s": ("s", 1.0),
    "1": ("1", 1.0),
}


@dataclass
class Quantity:
    """A domain value that always carries unit / datum and its raw text."""

    raw: Any
    unit: str
    datum: str = ""
    value: Optional[float] = None  # normalized SI value; None when unconvertible
    diag: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.value is not None and self.diag is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "datum": self.datum,
            "value": self.value,
            "diag": self.diag,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Quantity":
        return Quantity(
            raw=d.get("raw"),
            unit=d.get("unit", "1"),
            datum=d.get("datum", ""),
            value=d.get("value"),
            diag=d.get("diag"),
            meta=d.get("meta", {}),
        )


def parse_quantity(raw: Any, unit: str, datum: str = "") -> Quantity:
    """Parse a raw representation into a normalized Quantity.

    Keeps the raw text untouched; if conversion fails the value stays None and
    a diagnostic is attached instead of dropping the record.
    """
    q = Quantity(raw=raw, unit=unit, datum=datum)
    if raw is None:
        q.diag = "missing"
        return q
    if isinstance(raw, str):
        text = raw.strip()
        if text.upper() in ("NA", "N/A", "BAD", ""):
            q.diag = "flagged:" + (text or "empty")
            return q
        match = _NUM_PREFIX.match(text)
        if not match:
            q.diag = "unparseable"
            return q
        number = float(match.group(1))
        # tolerate trailing unit annotations like "12.5 mm/s"
        tail = text[match.end():].strip()
        if tail:
            q.meta["raw_unit_hint"] = tail
    elif isinstance(raw, (int, float)):
        number = float(raw)
        if math.isnan(number) or math.isinf(number):
            q.diag = "non-finite"
            return q
    else:
        q.diag = "unsupported-type"
        return q
    conv = _UNIT_TO_SI.get(unit)
    if conv is None:
        q.diag = "unknown-unit:" + unit
        return q
    si_unit, factor = conv
    q.value = number * factor
    q.meta["si_unit"] = si_unit
    return q
