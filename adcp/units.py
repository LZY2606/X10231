"""Domain quantities carry an explicit unit or coordinate reference.

Raw imports are preserved verbatim in the storage layer; this module only
serves the derived layer, where normalization happens.  A value that cannot be
converted is represented by :class:`Unconvertible` and stays where it was with
a diagnostic instead of being dropped for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Velocity is normalized to m/s.  Angles are normalized to radians inside the
# derived layer, but the input representation (degrees / radian / true / mag)
# is retained on the source record.
VELOCITY_TO_MS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001}
ANGLE_TO_RAD = {"rad": 1.0, "deg": 0.017453292519943295}

COORD_FRAMES = ("beam", "instrument", "ship", "earth")
EARTH_DATUM = "ENU"  # east, north, up; ship is x-forward, y-starboard, z-down


@dataclass(frozen=True)
class Q:
    """A value with a unit and optional coordinate reference."""

    value: float | None
    unit: str
    ref: str | None = None
    ok: bool = True
    diagnostic: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "unit": self.unit, "ok": self.ok}
        if self.ref is not None:
            out["ref"] = self.ref
        if self.diagnostic is not None:
            out["diagnostic"] = self.diagnostic
        return out


def velocity_ms(raw: Any, unit: str) -> Q:
    if not _is_number(raw):
        return Q(None, unit, None, False, f"non-numeric velocity {raw!r}")
    factor = VELOCITY_TO_MS.get(unit)
    if factor is None:
        return Q(float(raw), unit, None, False, f"unknown velocity unit {unit!r}")
    return Q(float(raw) * factor, "m/s")


def angle_rad(raw: Any, unit: str) -> Q:
    if not _is_number(raw):
        return Q(None, unit, None, False, f"non-numeric angle {raw!r}")
    factor = ANGLE_TO_RAD.get(unit)
    if factor is None:
        return Q(float(raw), unit, None, False, f"unknown angle unit {unit!r}")
    return Q(float(raw) * factor, "rad")


def _is_number(raw: Any) -> bool:
    if isinstance(raw, bool):
        return False
    try:
        float(raw)
    except (TypeError, ValueError):
        return False
    return True
