"""Units and coordinate datums.

Domain quantities always carry a unit or coordinate datum.  Import keeps the
original representation; only the derived layer normalises to the working
system (SI radians / m s-1).  Values that cannot be converted are kept in
place with a diagnostic instead of being silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


ANGLE_RAW_TO_RAD = {
    "deg": math.pi / 180.0,
    "degree": math.pi / 180.0,
    "degrees": math.pi / 180.0,
    "rad": 1.0,
    "radian": 1.0,
    "radians": 1.0,
}

VELOCITY_RAW_TO_MPS = {
    "m/s": 1.0,
    "mps": 1.0,
    "m s-1": 1.0,
    "mm/s": 1e-3,
    "mmps": 1e-3,
    "cm/s": 1e-2,
}

SCALE_TO_ONE = {
    "percent": 1.0 / 100.0,
    "fraction": 1.0,
    "counts": None,  # pass through untouched (instrument-specific)
}


@dataclass(frozen=True)
class Quantity:
    """A value with its raw representation and working-system normalisation."""

    raw: Any
    unit: str
    value: Optional[float]
    diagnostic: Optional[str] = None

    def is_valid(self) -> bool:
        return self.value is not None and self.diagnostic is None

    def to_payload(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "value": self.value,
            "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Quantity":
        return cls(
            raw=payload["raw"],
            unit=payload["unit"],
            value=payload.get("value"),
            diagnostic=payload.get("diagnostic"),
        )


def _finite_number(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):  # bool is an int subclass; reject it
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isfinite(value):
            return value
        return None
    return None


def angle(raw: Any, unit: str = "deg") -> Quantity:
    value = _finite_number(raw)
    diagnostic = None
    if value is None:
        diagnostic = f"unconvertible angle {raw!r} in {unit!r}; raw value retained"
        return Quantity(raw=raw, unit=unit, value=None, diagnostic=diagnostic)
    factor = ANGLE_RAW_TO_RAD.get(unit.lower())
    if factor is None:
        return Quantity(raw=raw, unit=unit, value=None,
                        diagnostic=f"unknown angle unit {unit!r}; raw value retained")
    return Quantity(raw=raw, unit=unit, value=value * factor)


def velocity(raw: Any, unit: str = "m/s") -> Quantity:
    value = _finite_number(raw)
    if value is None:
        return Quantity(raw=raw, unit=unit, value=None,
                        diagnostic=f"unconvertible velocity {raw!r}; raw value retained")
    factor = VELOCITY_RAW_TO_MPS.get(unit.lower())
    if factor is None:
        return Quantity(raw=raw, unit=unit, value=None,
                        diagnostic=f"unknown velocity unit {unit!r}; raw value retained")
    return Quantity(raw=raw, unit=unit, value=value * factor)


def correlation(raw: Any, unit: str = "counts") -> Quantity:
    value = _finite_number(raw)
    if value is None:
        return Quantity(raw=raw, unit=unit, value=None,
                        diagnostic="non-numeric correlation; raw value retained")
    factor = SCALE_TO_ONE.get(unit.lower())
    if factor is None:  # e.g. raw instrument counts
        return Quantity(raw=raw, unit=unit, value=value, diagnostic=None)
    return Quantity(raw=raw, unit=unit, value=value * factor)


def amplitude(raw: Any, unit: str = "counts") -> Quantity:
    value = _finite_number(raw)
    if value is None:
        return Quantity(raw=raw, unit=unit, value=None,
                        diagnostic="non-numeric amplitude; raw value retained")
    return Quantity(raw=raw, unit=unit, value=value)


def normalize_deg(raw: Any, unit: str = "deg") -> float:
    """Normalise an angle directly to degrees (for DMS-capable callers)."""
    q = angle(raw, unit)
    if q.value is None:
        raise ValueError(q.diagnostic)
    return q.value * 180.0 / math.pi
