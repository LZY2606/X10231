"""Raw representation preservation and derived-layer normalization.

Import keeps the original numeric representation, unit and any diagnostic
exactly as recorded. Normalization to canonical (SI where one exists) only
happens in the derivation layer; values that cannot be converted stay in
place and carry a diagnostic instead of being dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Linear scales only: canonical_value = raw_value * scale.
# Counts / percent / dB have no physical SI conversion: their canonical unit
# is the recorded dimensionless/engineering unit itself (scale 1).
_SCALES: dict[str, tuple[float, str]] = {
    # velocity
    "m/s": (1.0, "m/s"),
    "cm/s": (0.01, "m/s"),
    "mm/s": (1.0e-3, "m/s"),
    # angles
    "deg": (math.pi / 180.0, "rad"),
    "rad": (1.0, "rad"),
    # depth / distance
    "m": (1.0, "m"),
    "dm": (0.1, "m"),
    "cm": (0.01, "m"),
    # dimensionless engineering units preserved as-is
    "count": (1.0, "count"),
    "counts": (1.0, "count"),
    "percent": (1.0, "percent"),
    "%": (1.0, "percent"),
    "dB": (1.0, "dB"),
    # flags
    "flag": (1.0, "flag"),
    "bool": (1.0, "flag"),
}


@dataclass(frozen=True)
class Raw:
    """An imported measurement in its original representation."""

    value: Any
    unit: str | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"value": self.value, "unit": self.unit}
        if self.note is not None:
            out["note"] = self.note
        return out

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "Raw":
        return Raw(value=obj["value"], unit=obj.get("unit"), note=obj.get("note"))


@dataclass(frozen=True)
class ConversionResult:
    ok: bool
    value: float | None
    canonical_unit: str | None
    raw: Raw
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "canonical_unit": self.canonical_unit,
            "raw": self.raw.to_dict(),
            "diagnostic": self.diagnostic,
        }


def normalize(raw: Raw) -> ConversionResult:
    """Normalize one raw measurement. Never raises for unconvertible input."""
    if raw is None:
        return ConversionResult(False, None, None, Raw(None, None), "missing_value")
    if raw.value is None:
        return ConversionResult(False, None, None, raw, "missing_value")
    if raw.unit is None:
        return ConversionResult(False, None, None, raw, "missing_unit")
    key = raw.unit.strip()
    spec = _SCALES.get(key)
    if spec is None:
        return ConversionResult(False, None, None, raw, f"unknown_unit:{key}")
    scale, canonical = spec
    try:
        number = float(raw.value)
    except (TypeError, ValueError):
        return ConversionResult(False, None, None, raw, "non_numeric_value")
    if not math.isfinite(number):
        return ConversionResult(False, None, None, raw, "non_finite_value")
    return ConversionResult(True, number * scale, canonical, raw)
