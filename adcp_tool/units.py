"""Quantities with units / coordinate datums.

The import layer keeps the *original* representation untouched.  Only the
derivation layer normalises values; values that cannot be converted stay in
place and carry a diagnostic instead of being silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# --- coordinate / frame datums (never silently converted) -------------------
FRAME_BEAM = "beam"
FRAME_INSTRUMENT = "instrument"
FRAME_SHIP = "ship"
FRAME_EARTH = "earth"
FRAME_DEPTH = "depth"

# canonical internal units
CANON = {
    "velocity": "m/s",
    "angle": "deg",
    "length": "m",
    "correlation": "percent",
    "amplitude": "dB",
}

_SPEED = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knot": 0.5144444, "kn": 0.5144444}
_ANGLE = {"deg": 1.0, "degree": 1.0, "rad": 180.0 / math.pi,
         "radians": 180.0 / math.pi, "counts": 0.01, "deg*100": 0.01}
_LENGTH = {"m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 0.001}
_CORR = {"percent": 1.0, "%": 1.0, "counts": 100.0 / 255.0, "counts8": 100.0 / 255.0}
_AMP = {"dB": 1.0, "db": 1.0, "counts": 0.45}

_TABLES = {"velocity": _SPEED, "angle": _ANGLE, "length": _LENGTH,
           "correlation": _CORR, "amplitude": _AMP}

_MISSING = {"", "n/a", "na", "nan", "none", "null", "-", "--"}


@dataclass(frozen=True)
class Quantity:
    """A raw measurement: numeric or text value plus its stated unit/datum."""
    raw: Any
    unit: Optional[str]
    kind: str
    coord_ref: Optional[str] = None

    @property
    def is_missing(self) -> bool:
        if self.raw is None:
            return True
        if isinstance(self.raw, str) and self.raw.strip().lower() in _MISSING:
            return True
        return False


def quantity(spec: Any, kind: str, coord_ref: Optional[str] = None) -> Quantity:
    """Build a Quantity from fixture forms: number, str or {value, unit}."""
    if isinstance(spec, dict):
        return Quantity(raw=spec.get("value"), unit=spec.get("unit"),
                        kind=kind, coord_ref=spec.get("coord_ref", coord_ref))
    if isinstance(spec, Quantity):
        return spec
    return Quantity(raw=spec, unit=None, kind=kind, coord_ref=coord_ref)


def normalize(q: Any, kind: str, coord_ref: Optional[str] = None):
    """Return (canonical_value_or_None, diagnostics_list).

    A missing/unparseable/unknown-unit value is *not* dropped: None is
    returned together with a diagnostic and the caller keeps the raw value.
    """
    qq = q if isinstance(q, Quantity) else quantity(q, kind, coord_ref)
    if qq.is_missing:
        return None, [{"code": "MISSING_VALUE", "kind": kind, "raw": qq.raw}]
    try:
        value = float(qq.raw)
    except (TypeError, ValueError):
        return None, [{"code": "UNPARSEABLE_VALUE", "kind": kind,
                       "raw": qq.raw, "unit": qq.unit}]
    if math.isnan(value) or math.isinf(value):
        return None, [{"code": "NON_FINITE_VALUE", "kind": kind, "raw": qq.raw}]
    table = _TABLES[kind]
    unit = qq.unit
    if unit is None:
        unit = CANON[kind]  # unit-less fixture number assumes canonical unit
    factor = table.get(unit)
    if factor is None:
        return None, [{"code": "UNKNOWN_UNIT", "kind": kind,
                       "raw": qq.raw, "unit": qq.unit}]
    return value * factor, []


def as_canonical_dict(q: Any, kind: str, coord_ref: Optional[str] = None):
    """Normalise and package as {value, unit, coord_ref, diagnostics}."""
    qq = q if isinstance(q, Quantity) else quantity(q, kind, coord_ref)
    value, diag = normalize(qq, kind, coord_ref)
    out = {"value": value, "unit": CANON[kind], "kind": kind,
           "raw_value": qq.raw, "raw_unit": qq.unit}
    if qq.coord_ref:
        out["coord_ref"] = qq.coord_ref
    if diag:
        out["diagnostics"] = diag
    return out


def wrap360(deg: float) -> float:
    """Wrap any angle into [0, 360)."""
    return deg % 360.0


def wrap180(deg: float) -> float:
    """Wrap any angle into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0
