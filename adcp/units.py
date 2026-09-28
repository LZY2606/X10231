"""Quantities with units / coordinate datums.

Raw layer keeps the imported representation verbatim (`raw`, `raw_unit`).
The derived layer calls `normalize` to reach canonical SI units. Values that
cannot be converted stay in place with an `UNCONVERTIBLE` diagnostic instead
of being dropped (and instead of poisoning plots).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "correlation": "fraction",
    "amplitude": "dB",
    "depth": "m",
    "time": "s",
}

# conversion factor -> canonical SI representation
_VELOCITY = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knot": 0.5144444}
_DEPTH = {"m": 1.0, "cm": 0.01, "dm": 0.1}
_ANGLE = {"deg": 1.0, "rad": 180.0 / math.pi}
_CORR = {"fraction": 1.0, "percent": 0.01, "counts255": 1.0 / 255.0}
_AMP = {"dB": 1.0, "counts_dB": 1.0, "amp": 0.61}  # RDI amplitude counts ~0.61 dB/count

_TABLE = {"velocity": _VELOCITY, "depth": _DEPTH, "angle": _ANGLE,
          "correlation": _CORR, "amplitude": _AMP}

UNCONVERTIBLE = "UNCONVERTIBLE"


@dataclass
class Diagnostic:
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class Quantity:
    value: Any                 # float | list[float] | None when unconvertible
    unit: str                  # canonical unit
    kind: str                  # velocity / angle / ...
    frame: str | None = None   # beam/instrument/ship/earthENU ...
    raw: Any = None
    raw_unit: str | None = None
    diagnostics: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {"value": self.value, "unit": self.unit, "kind": self.kind}
        if self.frame:
            d["frame"] = self.frame
        if self.raw is not None or self.raw_unit is not None:
            d["raw"] = self.raw
            d["raw_unit"] = self.raw_unit
        if self.diagnostics:
            d["diagnostics"] = [x.as_dict() if isinstance(x, Diagnostic) else x
                                for x in self.diagnostics]
        return d


def normalize(value: Any, unit: str, kind: str, frame: str | None = None) -> Quantity:
    """Convert one scalar into canonical units; keep raw on failure."""
    table = _TABLE.get(kind)
    factor = table.get(unit) if table else None
    if factor is None:
        return Quantity(None, CANONICAL.get(kind, unit), kind, frame,
                        raw=value, raw_unit=unit,
                        diagnostics=[Diagnostic(UNCONVERTIBLE,
                                                f"no conversion {unit!r} -> {kind}")])
    try:
        out = float(value) * factor
        if math.isnan(out) or math.isinf(out):
            raise ValueError("non-finite")
    except (TypeError, ValueError):
        return Quantity(None, CANONICAL[kind], kind, frame, raw=value, raw_unit=unit,
                        diagnostics=[Diagnostic(UNCONVERTIBLE,
                                                f"cannot interpret {value!r} as {unit}")])
    return Quantity(out, CANONICAL[kind], kind, frame, raw=value, raw_unit=unit)


def normalize_vec(values, unit: str, kind: str, frame: str | None = None):
    """Vector convenience: converts element-wise, never drops bad elements."""
    qs = [normalize(v, unit, kind, frame) for v in values]
    out = [q.value for q in qs]
    diags = [None if not q.diagnostics else q.diagnostics[0].as_dict() for q in qs]
    return out, diags
