"""Units and coordinate-reference tagging.

Every domain quantity carries its raw representation, a unit (or coordinate
reference) and a status. Raw values are preserved verbatim on import; only the
derived layer normalizes them. A value that cannot be converted stays at its
origin with a diagnostic instead of being dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Canonical (normalized) units used by the derived layer.
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "time": "s",
    "distance": "m",
    "correlation": "counts",
    "amplitude": "counts",
}

# Linear conversion factor -> canonical unit.
_FACTORS: dict[tuple[str, str], float] = {
    ("mm/s", "m/s"): 1e-3,
    ("cm/s", "m/s"): 1e-2,
    ("m/s", "m/s"): 1.0,
    ("knot", "m/s"): 0.5144444444444445,
    ("deg", "deg"): 1.0,
    ("rad", "deg"): 180.0 / math.pi,
    ("arcdeg", "deg"): 1.0,
    ("s", "s"): 1.0,
    ("ms", "s"): 1e-3,
    ("m", "m"): 1.0,
    ("dm", "m"): 0.1,
    ("cm", "m"): 0.01,
    ("counts", "counts"): 1.0,
}


@dataclass
class Quantity:
    """A measured value with its raw text, unit and conversion status."""

    raw: str
    unit: str | None = None
    value: float | None = None
    canonical_unit: str | None = None
    status: str = "ok"  # ok | unconvertible
    diagnostic: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "value": self.value,
            "canonical_unit": self.canonical_unit,
            "status": self.status,
            "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Quantity":
        return cls(
            raw=data["raw"],
            unit=data.get("unit"),
            value=data.get("value"),
            canonical_unit=data.get("canonical_unit"),
            status=data.get("status", "ok"),
            diagnostic=data.get("diagnostic"),
        )


def parse_quantity(
    raw: Any,
    unit: str | None,
    canonical_unit: str,
    *,
    allow_nan: bool = False,
) -> Quantity:
    """Parse a raw value in a given source unit into the canonical unit.

    Unparseable / unconvertible values produce an ``unconvertible`` Quantity
    with a diagnostic; they are never silently dropped or coerced to zero.
    """
    raw_text = str(raw)
    if unit is None:
        return Quantity(
            raw=raw_text,
            unit=None,
            value=None,
            canonical_unit=canonical_unit,
            status="unconvertible",
            diagnostic="missing unit",
        )
    key = (unit, canonical_unit)
    factor = _FACTORS.get(key)
    if factor is None:
        return Quantity(
            raw=raw_text,
            unit=unit,
            value=None,
            canonical_unit=canonical_unit,
            status="unconvertible",
            diagnostic=f"no conversion {unit} -> {canonical_unit}",
        )
    try:
        parsed = float(raw_text)
    except (TypeError, ValueError):
        return Quantity(
            raw=raw_text,
            unit=unit,
            value=None,
            canonical_unit=canonical_unit,
            status="unconvertible",
            diagnostic=f"cannot parse {raw_text!r} as float",
        )
    if not allow_nan and not math.isfinite(parsed):
        return Quantity(
            raw=raw_text,
            unit=unit,
            value=None,
            canonical_unit=canonical_unit,
            status="unconvertible",
            diagnostic="non-finite value",
        )
    return Quantity(
        raw=raw_text,
        unit=unit,
        value=parsed * factor,
        canonical_unit=canonical_unit,
        status="ok",
    )
