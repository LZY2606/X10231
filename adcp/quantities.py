"""Domain quantities carrying a unit or coordinate datum.

Rule enforced across the project:

* Raw imports keep their *original representation* (string + declared unit);
  they are never overwritten by a normalized number.
* Only the derived layer performs normalization, and the normalized value
  keeps a reference to the raw token it came from.
* A value that cannot be converted stays at its original location together
  with a diagnostic instead of being dropped to "make the plot work".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


class ConversionError(ValueError):
    """Raised when a raw token cannot be converted to the canonical unit."""


# Canonical units used inside the derived layer.
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "depth": "m",
    "time": "s",
}

# Every accepted raw unit maps to (dimension, linear scale to canonical).
# Angles are handled separately (wrap semantics), so only deg/rad here.
UNIT_TABLE: dict[str, tuple[str, float]] = {
    "m/s": ("velocity", 1.0),
    "cm/s": ("velocity", 0.01),
    "mm/s": ("velocity", 0.001),
    "deg": ("angle", 1.0),
    "rad": ("angle", 1.0),  # converted with math, not the scale, in Q.from_raw
    "m": ("depth", 1.0),
    "cm": ("depth", 0.01),
    "ft": ("depth", 0.3048),
    "s": ("time", 1.0),
    "ms": ("time", 0.001),
}

MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-"})


@dataclass(frozen=True)
class Q:
    """A scalar quantity with explicit unit/datum.

    ``raw`` preserves the imported token verbatim; ``value`` is the normalized
    numeric value in ``unit`` (always the canonical unit for the dimension).
    ``raw_unit`` records the unit declared at import time.
    """

    value: float | None
    unit: str
    raw: str | None = None
    raw_unit: str | None = None
    datum: str | None = None
    diagnostic: str | None = None

    @staticmethod
    def from_raw(
        token: Any,
        unit: str,
        *,
        datum: str | None = None,
    ) -> "Q":
        """Parse one raw token. Never raises: failures become diagnostics."""
        text = None if token is None else str(token)
        if text is not None and text.strip().lower() in MISSING_TOKENS:
            return Q(None, UNIT_TABLE.get(unit, (unit, 1.0))[0]
                     if unit in UNIT_TABLE else unit,
                     raw=text, raw_unit=unit, datum=datum,
                     diagnostic="missing-token")
        if unit not in UNIT_TABLE:
            return Q(None, unit, raw=text, raw_unit=unit, datum=datum,
                     diagnostic=f"unknown-unit:{unit}")
        dimension, scale = UNIT_TABLE[unit]
        try:
            number = float(text)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return Q(None, dimension, raw=text, raw_unit=unit, datum=datum,
                     diagnostic=f"unparseable-token:{text!r}")
        canonical_unit = CANONICAL[dimension]
        if unit == "rad":
            import math

            number = math.degrees(number)
        else:
            number = number * scale
        return Q(number, canonical_unit, raw=text, raw_unit=unit, datum=datum)

    @property
    def ok(self) -> bool:
        return self.value is not None and self.diagnostic is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Q":
        return Q(**data)


@dataclass(frozen=True)
class Vec:
    """A labelled 3-component vector in a declared coordinate frame."""

    x: float
    y: float
    z: float
    frame: str
    components: tuple[str, str, str] = ("x", "y", "z")
    unit: str = "m/s"
    meta: dict[str, Any] = field(default_factory=dict)

    def array(self):
        import numpy as np

        return np.array([self.x, self.y, self.z], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "frame": self.frame,
            "components": list(self.components),
            "unit": self.unit,
            "meta": dict(self.meta),
        }


# Coordinate frames used in the pipeline and their human labels.
FRAMES = {
    "beam": "beam frame (along-transducer signed radial velocities)",
    "instrument": "instrument frame (x/y/z RDI transducer axes)",
    "ship": "ship frame (vessel axes, corrected for mount angle)",
    "earth": "earth frame ENU (east/north/up)",
}

# Angle datums.
ANGLE_DATUMS = {
    "magnetic": "heading relative to magnetic north",
    "true": "heading relative to true north",
}


def declination_corrected_heading(
    heading_deg: float,
    declination_deg: float,
) -> float:
    """Apply magnetic declination.

    Convention (documented on every derived fingerprint):
        true_heading = wrap360(magnetic_heading + declination)
    i.e. a positive declination means magnetic north lies east of true north.
    """

    return wrap360(heading_deg + declination_deg)


def wrap360(angle_deg: float) -> float:
    wrapped = angle_deg % 360.0
    if wrapped == 360.0:
        wrapped = 0.0
    return wrapped
