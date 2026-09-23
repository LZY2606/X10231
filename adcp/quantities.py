"""Domain quantities with explicit units / coordinate datums.

Import preserves the *raw representation*; only derived layers normalise.
A value that cannot be converted stays in place with a diagnostic instead
of being dropped for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

# Angles are normalised to **radians** in the derived layer.
ANGLE_UNITS = {"rad": 1.0, "deg": math.pi / 180.0, "centidegree": math.pi / 18000.0}
# Velocities are normalised to m/s.
VELOCITY_UNITS = {"m/s": 1.0, "mm/s": 0.001, "cm/s": 0.01}
# Lengths are normalised to m.
LENGTH_UNITS = {"m": 1.0, "dm": 0.1, "cm": 0.01, "ft": 0.3048}
# Correlation is a dimensionless count on [0, 255] for RDI-style instruments.
DIMENSIONLESS = {"count", "dB", "unitless", "1", ""}


@dataclass(frozen=True)
class Q:
    """A scalar carrying its unit (and optionally a coordinate datum)."""

    value: Optional[float]
    unit: str
    datum: Optional[str] = None
    raw: Optional[str] = None
    diagnostic: Optional[str] = None

    def with_diagnostic(self, message: str) -> "Q":
        return replace(self, diagnostic=message)


@dataclass(frozen=True)
class Vec:
    """A 3-vector carrying its coordinate frame (instrument/ship/earth)."""

    components: Any  # numpy array after normalisation, or tuple of Q on import
    frame: str
    units: str = "m/s"
    components_raw: Any = None
    diagnostics: tuple = ()

    def as_array(self):
        return self.components


def unit_family(unit: str) -> str:
    if unit in ANGLE_UNITS:
        return "angle"
    if unit in VELOCITY_UNITS:
        return "velocity"
    if unit in LENGTH_UNITS:
        return "length"
    return "dimensionless"


def normalize(q: Q, family: Optional[str] = None) -> Q:
    """Normalise a scalar to the SI canonical unit of its family.

    An unconvertible value is returned *in place* with a diagnostic rather
    than raising (so the UI can still show why it was excluded).
    """
    fam = family or unit_family(q.unit)
    if q.value is None:
        return q
    table = {
        "angle": (ANGLE_UNITS, "rad"),
        "velocity": (VELOCITY_UNITS, "m/s"),
        "length": (LENGTH_UNITS, "m"),
    }.get(fam)
    if table is None:
        return replace(q, diagnostic=q.diagnostic or f"dimensionless:{q.unit}")
    conv, canonical = table
    factor = conv.get(q.unit)
    if factor is None:
        return replace(
            q,
            diagnostic=(q.diagnostic or f"cannot convert unit '{q.unit}' to {canonical}"),
        )
    return replace(q, value=q.value * factor, unit=canonical, datum=q.datum)


def normalize_value(value: float, unit: str, family: Optional[str] = None) -> float:
    out = normalize(Q(value, unit), family)
    if out.diagnostic or out.value is None:
        raise ValueError(out.diagnostic or "unconvertible value")
    return out.value


def wrap_degrees(angle_deg: float) -> float:
    """Wrap an angle in degrees to [0, 360)."""
    return angle_deg % 360.0


def wrap_pi(angle: float) -> float:
    """Wrap a radian angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
