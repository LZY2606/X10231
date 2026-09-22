"""Units and coordinate-datum bookkeeping.

Raw import keeps the original representation; normalization happens only in the
derived layer. Values that cannot be converted stay where they are with a
diagnostic instead of being silently coerced for the sake of a plot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Every domain quantity carries one of these canonical units/datums.
UNIT = {
    "velocity": "mm/s",           # beam and Cartesian velocity components
    "angle": "deg",               # heading/pitch/roll/mount/declination angles
    "time": "seconds since ping_epoch (UTC monotonic seconds)",
    "depth": "m",                 # bin depth
    "correlation": "counts(0-255)",
    "amplitude": "counts",
    "beam_angle": "deg_from_transducer_axis",
}

# Frames are part of the coordinate datum, not just a label.
FRAMES = ("beam", "instrument", "ship", "earth")

KNOWN_UNITS = {
    "mm/s", "m/s", "cm/s",
    "deg", "rad",
    "m",
    "counts", "counts(0-255)",
    "seconds since ping_epoch (UTC monotonic seconds)",
}


@dataclass(frozen=True)
class Quant:
    """A value tagged with the unit/frame it was measured or derived in."""
    value: Any
    unit: str
    frame: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"v": self.value, "unit": self.unit}
        if self.frame is not None:
            out["frame"] = self.frame
        return out


def normalize_velocity(value: Any, unit: str) -> float:
    """Normalize a velocity component to the canonical mm/s."""
    if unit == "mm/s":
        return float(value)
    if unit == "cm/s":
        return float(value) * 10.0
    if unit == "m/s":
        return float(value) * 1000.0
    raise ConversionError("velocity", unit, UNIT["velocity"])


def normalize_angle_deg(value: Any, unit: str) -> float:
    """Normalize an angle to degrees (does NOT wrap, keeps sign/order)."""
    if unit == "deg":
        return float(value)
    if unit == "rad":
        import math
        return math.degrees(float(value))
    raise ConversionError("angle", unit, UNIT["angle"])


class ConversionError(ValueError):
    def __init__(self, kind: str, src: str, dst: str):
        super().__init__("cannot convert %s from %r to %r" % (kind, src, dst))
        self.kind, self.src, self.dst = kind, src, dst
