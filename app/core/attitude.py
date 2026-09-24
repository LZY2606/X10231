"""Attitude samples: heading wrap-around handling and gap-aware interpolation.

Rules (part of the versioned derivation contract):
  * heading angles are normalized to [0, 360); interpolation unwraps along the
    shortest arc, so a transition 359 -> 1 interpolates through 0/360.
  * interpolation NEVER bridges a time gap larger than ``max_gap_s``: a target
    time in a gap gets no interpolated attitude (diagnostic, no value).
  * linear interpolation is used for pitch/roll; extrapolation is forbidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AttitudePoint:
    t: float
    heading: float  # degrees, [0,360)
    pitch: float
    roll: float


def _short_arc_lerp(a: float, b: float, f: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    value = a + delta * f
    return value % 360.0


def interpolate(samples: list[AttitudePoint], t: float, max_gap_s: float = None):
    """Return ``(h, p, r)`` at time ``t`` or ``(None, diagnostics)``."""
    if not samples:
        return None, ["no attitude samples"]
    ordered = sorted(samples, key=lambda s: s.t)
    if t < ordered[0].t or t > ordered[-1].t:
        return None, [f"t={t} outside attitude sample range; extrapolation forbidden"]
    for left, right in zip(ordered, ordered[1:]):
        gap = right.t - left.t
        if left.t <= t <= right.t:
            if max_gap_s is not None and gap > max_gap_s:
                return None, [
                    f"attitude gap {gap:g}s > max_gap {max_gap_s:g}s at t={t}; "
                    "not interpolated across gap"
                ]
            if t == left.t:
                return (left.heading, left.pitch, left.roll), []
            if t == right.t:
                return (right.heading, right.pitch, right.roll), []
            f = (t - left.t) / gap
            return (
                _short_arc_lerp(left.heading, right.heading, f),
                left.pitch + (right.pitch - left.pitch) * f,
                left.roll + (right.roll - left.roll) * f,
            ), []
    return None, ["attitude interpolation failed"]
