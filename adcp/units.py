"""Units, datums and coordinate frames for ADCP quantities.

Design rules
------------
* Imported raw measurements are kept verbatim in :class:`RawValue`.
* Only the derivation layer normalises units, producing :class:`Quantity`.
* A value that cannot be converted stays at its original location together
  with a :class:`Diagnostic` -- it is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class UnconvertibleError(ValueError):
    """Raised when a raw representation cannot be normalised."""


# velocity unit -> factor to m/s
VELOCITY_TO_MPS = {
    "mm/s": 0.001,
    "cm/s": 0.01,
    "m/s": 1.0,
    "m s-1": 1.0,
    "knot": 0.5144444444444445,
}

# angle unit -> factor to degree
ANGLE_TO_DEG = {"deg": 1.0, "degree": 1.0, "rad": 57.29577951308232}

# length unit -> factor to metre
LENGTH_TO_M = {"m": 1.0, "metre": 1.0, "dm": 0.1, "cm": 0.01, "ft": 0.3048}

DIMENSIONS = {"velocity": VELOCITY_TO_MPS, "angle": ANGLE_TO_DEG, "length": LENGTH_TO_M}


@dataclass(frozen=True)
class Datum:
    """Coordinate/unit reference carried by every normalised value."""

    dimension: str
    unit: str
    frame: str = "NONE"          # NONE / BEAM / INSTRUMENT / SHIP / EARTH
    axes: tuple[str, ...] = ()   # e.g. ("east", "north", "up")
    magnetic: str = "NONE"       # NONE / MAGNETIC / TRUE
    extra: tuple[tuple[str, str], ...] = ()

    def key(self) -> str:
        tail = ",".join(f"{k}={v}" for k, v in sorted(self.extra))
        return f"{self.dimension}|{self.unit}|{self.frame}|{':'.join(self.axes)}|{self.magnetic}|{tail}"


@dataclass(frozen=True)
class Quantity:
    """A normalised numeric value carrying its datum."""

    value: Any  # numpy array or float
    datum: Datum

    def canonical(self) -> dict[str, Any]:
        v = self.value
        if hasattr(v, "tolist"):
            v = v.tolist()
        return {"v": v, "datum": self.datum.key()}


@dataclass(frozen=True)
class RawValue:
    """Original representation preserved on import."""

    raw: Any
    unit: str
    frame: str = "NONE"
    note: str = ""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "error"  # error / warn / info
    at: str = ""
    detail: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "at": self.at,
            "detail": dict(self.detail),
        }


def normalise(raw: RawValue, dimension: str, frame: str | None = None,
              axes: tuple[str, ...] = (), magnetic: str = "NONE") -> Quantity:
    """Convert a raw measurement to SI (m, m/s) or degrees."""
    table = DIMENSIONS[dimension]
    unit = raw.unit.strip()
    if unit not in table:
        raise UnconvertibleError(f"unknown {dimension} unit: {raw.unit!r}")
    factor = table[unit]
    import numpy as np

    val = np.asarray(raw.raw, dtype=float) * factor
    return Quantity(
        value=val,
        datum=Datum(
            dimension=dimension,
            unit={"velocity": "m/s", "angle": "deg", "length": "m"}[dimension],
            frame=frame or raw.frame,
            axes=axes,
            magnetic=magnetic,
        ),
    )
