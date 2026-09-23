"""Domain quantities: every value carries a unit or coordinate datum.

Raw representations are preserved on import; normalisation to a canonical unit
happens only in the derivation layer. Values that cannot be converted stay in
place (raw string + diagnostic) instead of being dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical unit per physical kind.
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "length": "m",
    "correlation": "count",
    "amplitude": "count",
    "time": "s",
    "density": "kg/m^3",
}

# Linear conversion factors to the canonical unit.
TO_CANON = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "deg": 1.0,
    "rad": 57.29577951308232,
    "m": 1.0,
    "cm": 0.01,
    "dm": 0.1,
    "ft": 0.3048,
    "s": 1.0,
    "ms": 0.001,
}

KINDS = {
    "m/s": "velocity", "cm/s": "velocity", "mm/s": "velocity",
    "deg": "angle", "rad": "angle",
    "m": "length", "cm": "length", "dm": "length", "ft": "length",
    "s": "time", "ms": "time",
}


@dataclass
class Diagnostic:
    code: str
    message: str
    raw: Any = None

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "raw": self.raw}


@dataclass
class Quantity:
    """A value with its unit/datum, the original representation, and flags."""

    value: float | None
    unit: str
    raw: Any = None
    datum: str | None = None          # coordinate datum for angles, e.g. MAG/TRUE
    kind: str | None = None
    converted: bool = False
    source_unit: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.value is not None and not self.diagnostics

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "datum": self.datum,
            "kind": self.kind,
            "raw": self.raw,
            "converted": self.converted,
            "source_unit": self.source_unit,
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }


def parse_quantity(raw: Any, unit: str, datum: str | None = None) -> Quantity:
    """Parse a raw cell into a canonical Quantity without discarding the raw text.

    Non-numeric or blank sentinels ("", "BAD", "nan", "NA") keep their original
    representation and carry a diagnostic rather than raising.
    """
    q = Quantity(value=None, unit=unit, raw=raw, datum=datum)
    kind = KINDS.get(unit)
    q.kind = kind
    if unit not in TO_CANON:
        q.diagnostics.append(
            Diagnostic("unknown_unit", f"unit {unit!r} has no conversion", raw)
        )
        return q
    try:
        if isinstance(raw, bool):
            raise ValueError("boolean is not a measurement")
        num = float(raw)
    except (TypeError, ValueError):
        q.diagnostics.append(
            Diagnostic("unparseable", "value cannot be parsed as a number", raw)
        )
        return q
    if num != num:  # NaN
        q.diagnostics.append(Diagnostic("nan", "value is NaN", raw))
        return q
    factor = TO_CANON[unit]
    if factor != 1.0:
        q.source_unit = unit
        q.converted = True
    q.value = num * factor
    canon = CANONICAL.get(kind)
    if canon:
        q.unit = canon
    return q


def convert(q: Quantity, unit: str) -> Quantity:
    """Convert an already-canonical quantity to another compatible unit."""
    if q.value is None:
        return q
    if q.unit not in TO_CANON or unit not in TO_CANON:
        out = Quantity(q.value, q.unit, q.raw, q.datum, q.kind, q.converted,
                       q.source_unit, list(q.diagnostics))
        out.diagnostics.append(
            Diagnostic("unknown_unit", f"cannot convert {q.unit!r}->{unit!r}", q.raw)
        )
        return out
    if KINDS.get(q.unit) != KINDS.get(unit):
        out = Quantity(q.value, q.unit, q.raw, q.datum, q.kind, q.converted,
                       q.source_unit, list(q.diagnostics))
        out.diagnostics.append(
            Diagnostic("kind_mismatch",
                       f"{q.unit!r} and {unit!r} are different physical kinds", q.raw)
        )
        return out
    value = q.value / TO_CANON[unit]
    return Quantity(value, unit, q.raw, q.datum, q.kind, True, q.source_unit,
                    list(q.diagnostics))
