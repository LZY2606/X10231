"""Attitude interpolation with heading wrap-around and protected gaps.

Policy (versioned as ``att-interp@linear-gap-v1``):

* pitch/roll are linearly interpolated between *adjacent* samples.
* heading is interpolated on the unit circle (sin/cos) so a jump across
  359.9deg -> 0.1deg is a 0.2deg move, not a 359.8deg spin.
* interpolation never crosses a gap: if adjacent attitude samples are more
  than ``max_gap_s`` apart, pings in between get no interpolated attitude and
  the derived earth velocity is marked unresolved instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ATTITUDE_RULE = "att-interp@linear-gap-v1"


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    heading_rad: float  # magnetic
    pitch_rad: float
    roll_rad: float
    unit_in: str  # "deg" or "rad" of the original representation


def _interp_angle_circle(a0: float, a1: float, f: float) -> float:
    """Shortest-path interpolation between two angles, result in (-pi, pi]."""
    d = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    return float((a0 + f * d) % (2.0 * math.pi))


@dataclass(frozen=True)
class InterpResult:
    heading_rad: float | None
    pitch_rad: float | None
    roll_rad: None | float
    interpolated: bool
    in_gap: bool
    bracket: tuple[float, float] | None
    gap_s: float | None


def interpolate_at(
    samples: list[AttitudeSample], t: float, max_gap_s: float
) -> InterpResult:
    """Interpolate attitude at time ``t``; never bridge an over-wide gap."""
    if not samples:
        return InterpResult(None, None, None, False, True, None, None)
    ts = [s.t for s in samples]
    if t <= ts[0]:
        s = samples[0]
        return InterpResult(s.heading_rad, s.pitch_rad, s.roll_rad, False, False, (s.t, s.t), 0.0)
    if t >= ts[-1]:
        s = samples[-1]
        return InterpResult(s.heading_rad, s.pitch_rad, s.roll_rad, False, False, (s.t, s.t), 0.0)
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        if a.t <= t <= b.t:
            gap = b.t - a.t
            if gap > max_gap_s:
                return InterpResult(None, None, None, False, True, (a.t, b.t), gap)
            f = 0.0 if gap == 0.0 else (t - a.t) / gap
            return InterpResult(
                _interp_angle_circle(a.heading_rad, b.heading_rad, f),
                float(a.pitch_rad + f * (b.pitch_rad - a.pitch_rad)),
                float(a.roll_rad + f * (b.roll_rad - a.roll_rad)),
                True,
                False,
                (a.t, b.t),
                gap,
            )
    return InterpResult(None, None, None, False, True, None, None)
