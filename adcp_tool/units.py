"""Unit-aware value primitives.

Raw imports preserve their original representation (value + unit/basis). Only
the derivation layer normalises units. Values that cannot be converted stay at
their origin with a diagnostic instead of being dropped for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Optional

# angle bases normalised to radians
ANGLE_TO_RAD = {"deg": 0.017453292519943295, "rad": 1.0}
VELOCITY_TO_MPS = {"m/s": 1.0, "m s-1": 1.0, "cm/s": 0.01, "mm/s": 0.001}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return False


@dataclass(frozen=True)
class Q:
    """A measured/derived quantity with its original unit and coordinate basis."""

    value: float
    unit: str
    basis: str = ""
    raw: Any = None  # original textual representation as imported

    def missing(self) -> bool:
        return _is_missing(self.value)

    def to(self, unit: str) -> "Q":
        factor = _convert_factor(self.unit, unit)
        return Q(self.value * factor, unit, self.basis, self.raw)

    def normalise(self, kind: str) -> "Q":
        target = {"angle": "rad", "velocity": "m/s"}[kind]
        return self.to(target)


@dataclass(frozen=True)
class Unconvertible:
    """A raw value that cannot be converted; it stays in place with a reason."""

    raw: Any
    unit: str
    reason: str
    basis: str = ""

    def missing(self) -> bool:
        return True


Value = Q | Unconvertible


def quantity(raw: Any, unit: str, basis: str = "") -> Value:
    """Build a Q from a raw representation, keeping the raw text on the node."""
    if _is_missing(raw):
        return Q(float("nan"), unit, basis, raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Unconvertible(raw=raw, unit=unit, reason="not-a-number", basis=basis)
    if isnan(value):
        return Q(value, unit, basis, raw)
    return Q(value, unit, basis, raw)


def _convert_factor(src: str, dst: str) -> float:
    if src == dst:
        return 1.0
    if src in ANGLE_TO_RAD and dst in ANGLE_TO_RAD:
        return ANGLE_TO_RAD[src] / ANGLE_TO_RAD[dst]
    if src in VELOCITY_TO_MPS and dst in VELOCITY_TO_MPS:
        return VELOCITY_TO_MPS[src] / VELOCITY_TO_MPS[dst]
    raise KeyError(f"no conversion {src!r} -> {dst!r}")


def convert(v: Value, unit: str) -> tuple[Optional[float], Optional[str]]:
    """Return (converted value, diagnostic). Never raises on bad source data."""
    if isinstance(v, Unconvertible):
        return None, v.reason
    try:
        return v.to(unit).value, None
    except KeyError:
        return None, f"no conversion {v.unit!r} -> {unit!r}"
