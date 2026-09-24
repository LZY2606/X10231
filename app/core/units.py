"""Units and coordinate datums.

All domain quantities carry a unit and (where relevant) a coordinate datum.
Importers preserve the original representation; only the derivation layer
normalizes. Values that cannot be converted stay in place with diagnostics
instead of being silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# canonical unit -> {alias -> factor multiplying value to reach canonical}
_TO_CANONICAL: dict[str, dict[str, float]] = {
    "m/s": {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001},
    "deg": {"deg": 1.0, "rad": 180.0 / math.pi, "degree": 1.0, "degrees": 1.0},
    "counts": {"counts": 1.0, "count": 1.0},
    "dB": {"dB": 1.0, "db": 1.0},
    "m": {"m": 1.0, "cm": 0.01, "dm": 0.1},
}
CANONICAL_BY_ALIAS: dict[str, str] = {
    alias: canon for canon, aliases in _TO_CANONICAL.items() for alias in aliases
}


def normalize(raw: Any, unit: Optional[str]) -> tuple[Optional[float], list[str]]:
    """Return canonical value for ``raw`` expressed in ``unit``.

    A missing value or an unrecognized unit yields ``None`` plus diagnostics;
    the caller keeps the raw representation at the original location.
    """
    diagnostics: list[str] = []
    if raw is None or raw == "":
        diagnostics.append("missing value; raw representation retained")
        return None, diagnostics
    try:
        value = float(raw)
    except (TypeError, ValueError):
        diagnostics.append(f"non-numeric value {raw!r}; left in place, not plotted")
        return None, diagnostics
    if math.isnan(value):
        diagnostics.append("NaN value; left in place")
        return None, diagnostics
    if unit is None:
        diagnostics.append("unit missing; value kept raw without normalization")
        return None, diagnostics
    canonical = CANONICAL_BY_ALIAS.get(unit)
    if canonical is None:
        diagnostics.append(f"unknown unit {unit!r}; value left in place")
        return None, diagnostics
    return value * _TO_CANONICAL[canonical][unit], diagnostics


@dataclass
class Quantity:
    """A measurement with unit/datum metadata and raw provenance."""

    value: Optional[float]
    unit: Optional[str]
    raw: Any = None
    datum: Optional[str] = None
    valid: bool = True
    diagnostics: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any, unit: Optional[str], datum: Optional[str] = None) -> "Quantity":
        value, diagnostics = normalize(raw, unit)
        return cls(
            value=value,
            unit=CANONICAL_BY_ALIAS.get(unit or "") if value is not None else unit,
            raw=raw,
            datum=datum,
            valid=value is not None,
            diagnostics=diagnostics,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Quantity":
        return cls(**data)
