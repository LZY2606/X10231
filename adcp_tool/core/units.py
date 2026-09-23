"""Domain quantities always carry a unit or a coordinate/datum reference.

Import preserves the *raw representation*; normalisation to SI (rad, m/s ...)
only happens in the derivation layer.  A value that cannot be converted stays
where it was and gains a diagnostic instead of being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Quantity:
    """A numeric value plus the unit/frame it was measured in."""

    value: float
    unit: str

    def as_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}


@dataclass
class Measured:
    """Raw measurement: import keeps ``raw`` untouched; derivation uses ``norm``.

    ``raw_unit`` is the unit the instrument delivered (deg, cm/s, count ...).
    ``norm`` is the same quantity expressed in the canonical SI unit, or
    ``None`` when conversion is impossible (see ``diagnostic``).
    """

    raw: Any
    raw_unit: str
    norm: Optional[float] = None
    norm_unit: Optional[str] = None
    diagnostic: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.norm is not None and self.diagnostic is None

    def as_dict(self) -> dict:
        return asdict(self)


# Canonical units used below the derivation layer.
UNIT_RAD = "rad"
UNIT_DEG = "deg"
UNIT_MS = "m/s"
UNIT_CMS = "cm/s"
UNIT_COUNT = "count"
UNIT_M = "m"
UNIT_S = "s"
UNIT_DB = "dB"


def to_rad(m: Measured) -> Optional[float]:
    """Normalise an angle measurement to radians."""
    if not isinstance(m.raw, (int, float)) or isinstance(m.raw, bool):
        m.diagnostic = "non-numeric raw angle: %r" % (m.raw,)
        return None
    value = float(m.raw)
    if value != value:  # NaN
        m.diagnostic = "NaN angle"
        return None
    if m.raw_unit == UNIT_DEG:
        m.norm, m.norm_unit = value, UNIT_DEG
        import math

        return math.radians(value)
    if m.raw_unit == UNIT_RAD:
        m.norm, m.norm_unit = value, UNIT_RAD
        return value
    m.diagnostic = "unsupported angle unit: %s" % (m.raw_unit,)
    return None


def to_mps(m: Measured) -> Optional[float]:
    """Normalise a velocity measurement to m/s."""
    if not isinstance(m.raw, (int, float)) or isinstance(m.raw, bool):
        m.diagnostic = "non-numeric raw velocity: %r" % (m.raw,)
        return None
    value = float(m.raw)
    if value != value:
        m.diagnostic = "NaN velocity"
        return None
    if m.raw_unit == UNIT_CMS:
        m.norm, m.norm_unit = value, UNIT_CMS
        return value / 100.0
    if m.raw_unit == UNIT_MS:
        m.norm, m.norm_unit = value, UNIT_MS
        return value
    m.diagnostic = "unsupported velocity unit: %s" % (m.raw_unit,)
    return None


def identity(m: Measured) -> Optional[float]:
    """Pass through a dimensionless/count measurement after sanity checks."""
    if not isinstance(m.raw, (int, float)) or isinstance(m.raw, bool):
        m.diagnostic = "non-numeric raw value: %r" % (m.raw,)
        return None
    value = float(m.raw)
    if value != value:
        m.diagnostic = "NaN value"
        return None
    m.norm, m.norm_unit = value, m.raw_unit
    return value


@dataclass(frozen=True)
class Vector:
    """3-component vector pinned to a named coordinate frame."""

    x: float
    y: float
    z: float
    frame: str

    def components(self) -> tuple:
        return (self.x, self.y, self.z)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "frame": self.frame}


# Frames/datums the domain understands.
FRAME_BEAM = "beam"
FRAME_INSTRUMENT = "instrument(x=forward,y=starboard,z=down)"
FRAME_SHIP = "ship(x=forward,y=starboard,z=down)"
FRAME_EARTH = "earth(ENU)"
MAG_REL = "heading=magnetic"
MAG_TRUE = "heading=true"
