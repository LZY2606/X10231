"""Quantities: every domain number carries a unit or coordinate datum.

Raw rows keep their original representation; only the derivation layer
normalizes. A value that cannot be converted is returned as an
Unconvertible quantity carrying a diagnostic, never dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Length / velocity / angle unit conversion to canonical SI-ish internal units.
_LENGTH_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}
_VEL_TO_MPS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001}
_ANGLE_TO_DEG = {"deg": 1.0, "rad": 57.29577951308232}


@dataclass(frozen=True)
class Datum:
    kind: str                 # length | velocity | angle | vector | enum | count | text
    unit: str | None = None   # original unit for scalars
    frame: str | None = None  # beam | instrument | ship | earth for vectors
    basis: str | None = None  # extra datum: e.g. "magnetic" / "true"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Quantity:
    value: Any
    datum: Datum
    raw_repr: str | None = None  # exact original textual representation
    diagnostics: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def to_dict(self) -> dict:
        v = self.value
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                v = v.tolist()
        except Exception:
            pass
        return {
            "value": v,
            "datum": self.datum.to_dict(),
            "raw_repr": self.raw_repr,
            "diagnostics": list(self.diagnostics),
        }


def scalar(kind: str, raw: Any, unit: str | None, conv_table: dict | None,
           basis: str | None = None) -> Quantity:
    """Parse a raw scalar, retaining raw text. Unconvertible -> diagnostic kept in place."""
    raw_repr = None if raw is None else str(raw)
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return Quantity(None, Datum(kind, unit, None, basis), raw_repr,
                        (f"unparseable:{kind}:{raw!r}",))
    if unit is None:
        return Quantity(num, Datum(kind, None, None, basis), raw_repr)
    factor = (conv_table or {}).get(unit)
    if factor is None:
        return Quantity(num, Datum(kind, unit, None, basis), raw_repr,
                        (f"unknown_unit:{kind}:{unit}",))
    return Quantity(num * factor, Datum(kind, unit, None, basis), raw_repr)


def length(raw: Any, unit: str = "m") -> Quantity:
    return scalar("length", raw, unit, _LENGTH_TO_M)


def velocity(raw: Any, unit: str = "m/s") -> Quantity:
    return scalar("velocity", raw, unit, _VEL_TO_MPS)


def angle(raw: Any, unit: str = "deg", basis: str | None = None) -> Quantity:
    return scalar("angle", raw, unit, _ANGLE_TO_DEG, basis)


def vec(value: Any, frame: str, basis: str | None = None,
        diagnostics: tuple[str, ...] = ()) -> Quantity:
    import numpy as np
    arr = np.asarray(value, dtype=float)
    return Quantity(arr, Datum("vector", None, frame, basis), None, diagnostics)


def enum(value: Any) -> Quantity:
    return Quantity(value, Datum("enum"), None if value is None else str(value))


def q_to_dict(q: Quantity) -> dict:
    return q.to_dict()
