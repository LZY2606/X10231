"""Unit registry and quantity carrying.

Every domain quantity travels with its unit (for scalar families) or datum /
coordinate frame (for vectors). Raw import values keep their original
representation; normalisation to SI happens only in the derived layer.
"""
from __future__ import annotations

from dataclasses import dataclass

# Multiplier to SI for every supported unit, keyed by family.
_FAMILIES: dict[str, dict[str, float]] = {
    "velocity": {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knot": 0.514444444},
    "angle": {"deg": 1.0, "rad": 57.29577951308232},
    "length": {"m": 1.0, "cm": 0.01, "dm": 0.1, "ft": 0.3048},
    "time": {"s": 1.0, "ms": 0.001},
    "correlation": {"counts_255": 1.0, "percent": 2.55},
    "amplitude": {"counts": 1.0, "dB": 1.0},
    "dimensionless": {"1": 1.0},
}


class UnitError(ValueError):
    """Raised when a unit is unknown or a conversion crosses families."""


def register(family: str, unit: str, factor_to_si: float) -> None:
    _FAMILIES.setdefault(family, {})[unit] = factor_to_si


def family_of(unit: str) -> str:
    for family, units in _FAMILIES.items():
        if unit in units:
            return family
    raise UnitError(f"unknown unit: {unit!r}")


def to_si(value: float, unit: str) -> float:
    return value * _FAMILIES[family_of(unit)][unit]


def convert(value: float, src_unit: str, dst_unit: str) -> float:
    family = family_of(src_unit)
    if dst_unit not in _FAMILIES[family]:
        raise UnitError(f"cannot convert {src_unit!r} to {dst_unit!r}")
    return value * _FAMILIES[family][src_unit] / _FAMILIES[family][dst_unit]


@dataclass(frozen=True)
class Q:
    """A scalar with an explicit unit; ``raw`` preserves the imported text."""

    value: float | None
    unit: str
    raw: str | None = None

    def si(self) -> float | None:
        if self.value is None:
            return None
        return to_si(self.value, self.unit)

    def as_unit(self, unit: str) -> "Q":
        if self.value is None:
            return Q(None, unit, self.raw)
        return Q(convert(self.value, self.unit, unit), unit, self.raw)

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "raw": self.raw}

    @staticmethod
    def from_dict(d: dict) -> "Q":
        return Q(d["value"], d["unit"], d.get("raw"))
