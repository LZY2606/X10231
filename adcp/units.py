"""Units and coordinate datums.

Raw records keep their original representation verbatim. Only the derived
layer asks this module to normalise a value into the working SI convention.
A value that cannot be converted is *not* dropped: the raw representation
stays in place and a structured diagnostic is attached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Linear speed factors to m/s.
SPEED_TO_MPS = {
    "m/s": 1.0,
    "m s-1": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "knot": 0.514444,
}
# Angle factors to degrees.
ANGLE_TO_DEG = {"deg": 1.0, "degree": 1.0, "rad": 57.29577951308232}
# Length factors to metres.
LENGTH_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "dm": 0.1}
# Time factors to seconds.
TIME_TO_S = {"s": 1.0, "ms": 0.001, "us": 1e-6}
# Dimensionless receiver counts; kept as recorded magnitudes.
COUNTS = {"counts": 1.0, "count": 1.0, "": 1.0}

# Coordinate frames used by the tool.
FRAMES = ("beam", "instrument", "ship", "earth")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    raw_repr: str
    unit: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "raw_repr": self.raw_repr, "unit": self.unit}


@dataclass(frozen=True)
class Quantity:
    """A raw measurement together with its SI-normalised interpretation."""
    raw: Any
    raw_repr: str
    unit: Optional[str]
    datum: Optional[str]
    value: Optional[float]
    diagnostic: Optional[Diagnostic] = None

    def as_dict(self) -> dict[str, Any]:
        out = {"raw": self.raw, "raw_repr": self.raw_repr,
               "unit": self.unit, "datum": self.datum, "value": self.value}
        if self.diagnostic is not None:
            out["diagnostic"] = self.diagnostic.as_dict()
        return out


def as_repr(raw: Any) -> str:
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if raw is None:
        return "null"
    if isinstance(raw, float):
        return repr(raw)
    return str(raw)


def _to_float(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value == value else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.upper() in ("NA", "N/A", "BAD", "NULL", "NAN"):
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def quantity(raw: Any, unit: Optional[str], datum: Optional[str] = None,
             kind: str = "speed") -> Quantity:
    table = {"speed": (SPEED_TO_MPS, "m/s"),
             "angle": (ANGLE_TO_DEG, "deg"),
             "length": (LENGTH_TO_M, "m"),
             "time": (TIME_TO_S, "s"),
             "counts": (COUNTS, "counts")}
    factors, si_unit = table[kind]
    raw_repr = as_repr(raw)
    if raw is None:
        diag = Diagnostic("missing", "value is missing/null", raw_repr, unit)
        return Quantity(raw, raw_repr, unit, datum, None, diag)
    value = _to_float(raw)
    if value is None:
        diag = Diagnostic("unconvertible",
                          f"cannot interpret {raw_repr!r} as {kind}",
                          raw_repr, unit)
        return Quantity(raw, raw_repr, unit, datum, None, diag)
    if unit is None:
        diag = Diagnostic("unit_missing",
                          f"no unit declared; kept raw, assumed {si_unit} only at derived layer",
                          raw_repr, unit)
        return Quantity(raw, raw_repr, unit, datum, value, diag)
    factor = factors.get(unit)
    if factor is None:
        diag = Diagnostic("unknown_unit",
                          f"unit {unit!r} is not in the {kind} unit table",
                          raw_repr, unit)
        return Quantity(raw, raw_repr, unit, datum, None, diag)
    return Quantity(raw, raw_repr, unit, datum, value * factor, None)


def normalize(raw: Any, unit: Optional[str], kind: str = "speed") -> tuple[Optional[float], Optional[dict[str, Any]]]:
    q = quantity(raw, unit, None, kind)
    diag = q.diagnostic.as_dict() if q.diagnostic is not None else None
    return q.value, diag


def wrap_360(deg: float) -> float:
    """Wrap an angle into [0, 360)."""
    return deg % 360.0


def nearest_angular(base: float, deg: float) -> float:
    """Represent ``deg`` on the branch closest to ``base`` (unwrap helper)."""
    while deg - base > 180.0:
        deg -= 360.0
    while deg - base < -180.0:
        deg += 360.0
    return deg
