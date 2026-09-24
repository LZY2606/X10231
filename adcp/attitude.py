"""Attitude rotations and gap-aware interpolation.

Heading/pitch/roll are kept as SEPARATE source facts. Interpolation unwraps
heading so a 359 deg -> 1 deg crossing is a +2 deg motion, and it NEVER bridges
an explicitly recorded time gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import shortest_arc_delta, wrap180


def rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(((1, 0, 0), (0, c, s), (0, -s, c)), dtype=float)


def ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)), dtype=float)


def rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)), dtype=float)


def mount_matrix(roll_install_deg: float, pitch_install_deg: float,
                 heading_install_deg: float) -> np.ndarray:
    """Instrument -> ship using the separately recorded mounting angles."""
    return (rz(math.radians(heading_install_deg))
            @ ry(math.radians(pitch_install_deg))
            @ rx(math.radians(roll_install_deg)))


def earth_matrix(heading_deg: float, pitch_deg: float, roll_deg: float,
                 declination_deg: float, order: str) -> np.ndarray:
    """Ship -> earth. `order` is versioned and enters the derived fingerprint."""
    h = math.radians(heading_deg + declination_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    if order == "RDI_HPR":
        # Heading (Z, geographic) applied last to the horizontal result.
        return rz(h) @ ry(p) @ rx(r)
    if order == "RPH_ZYX":
        # Roll-pitch-heading applied to the vector in that order; a genuinely
        # different rule version (kept distinct so ordering errors are visible).
        return rx(r) @ ry(p) @ rz(h)
    raise ValueError(f"unknown transform order {order!r}")


@dataclass(frozen=True)
class AttitudeSample:
    time_s: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    gap_after: bool = False  # source explicitly marks a discontinuity after it


class AttitudeGap(ValueError):
    """Interpolation would cross a recorded time gap or leave the bracket."""


def interpolate_attitude(samples: list[AttitudeSample], t: float) -> dict:
    """Linear interpolation on the shortest arc; refuses to cross gaps.

    Heading is unwrapped (handles 359 -> 1). A sample flagged gap_after
    breaks interpolation at its right edge.
    """
    ss = sorted(samples, key=lambda s: s.time_s)
    if not ss:
        raise AttitudeGap("no attitude samples")
    if t < ss[0].time_s - 1e-9 or t > ss[-1].time_s + 1e-9:
        raise AttitudeGap(f"time {t} outside attitude bracket")
    for i, s in enumerate(ss[:-1]):
        nxt = ss[i + 1]
        if s.time_s <= t <= nxt.time_s:
            if s.gap_after:
                raise AttitudeGap(
                    f"gap flagged between {s.time_s} and {nxt.time_s}")
            if abs(nxt.time_s - s.time_s) > 1e-12:
                frac = (t - s.time_s) / (nxt.time_s - s.time_s)
            else:
                frac = 0.0
            return {
                "heading_deg": wrap180(s.heading_deg + shortest_arc_delta(
                    nxt.heading_deg, s.heading_deg) * frac),
                "pitch_deg": s.pitch_deg + (nxt.pitch_deg - s.pitch_deg) * frac,
                "roll_deg": s.roll_deg + (nxt.roll_deg - s.roll_deg) * frac,
                "bracket": (s.time_s, nxt.time_s),
                "crossed_heading_wrap": shortest_arc_delta(
                    nxt.heading_deg, s.heading_deg) != (
                        nxt.heading_deg - s.heading_deg),
            }
    last = ss[-1]
    return {"heading_deg": last.heading_deg, "pitch_deg": last.pitch_deg,
            "roll_deg": last.roll_deg, "bracket": (last.time_s, last.time_s),
            "crossed_heading_wrap": False}
