"""Attitude helpers: angle wrapping and gap-aware interpolation.

Interpolation never bridges a time gap. Heading is interpolated on the unit
circle so a 359 deg -> 1 deg transition is a 2 deg move, not -358 deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap360(x: float) -> float:
    return x % 360.0


def wrapped_delta(frm: float, to: float) -> float:
    """Signed shortest angular difference to - frm, in (-180, 180]."""
    d = (to - frm + 180.0) % 360.0 - 180.0
    return d


def crosses_seam(h0: float, h1: float) -> bool:
    return abs(wrapped_delta(h0, h1)) > 90.0 and (h0 > 270 or h1 > 270) \
        and (h0 < 90 or h1 < 90)


@dataclass
class AttitudeSample:
    t: float
    heading: float
    pitch: float
    roll: float
    present: bool = True


@dataclass
class Gap:
    t0: float
    t1: float
    dt: float


@dataclass
class AttitudeResult:
    heading: float | None
    pitch: float | None
    roll: float | None
    status: str            # measured | interpolated | missing-gap
    anchors: tuple         # (t0, t1) used for interpolation, else ()
    reason: str = ""


def find_gaps(times, max_dt: float):
    gaps = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > max_dt:
            gaps.append(Gap(times[i - 1], times[i], dt))
    return gaps


def in_gap(t: float, gaps, pad=0.0) -> bool:
    return any(g.t0 + pad < t < g.t1 - pad for g in gaps)


def interpolate(samples, t: float, max_dt: float) -> AttitudeResult:
    """Linear interpolation; no extrapolation and never across a gap."""
    present = [s for s in samples if s.present]
    if not present:
        return AttitudeResult(None, None, None, "missing-gap", (),
                              "no attitude samples")
    for s in present:
        if abs(s.t - t) <= 1e-9:
            return AttitudeResult(wrap360(s.heading), s.pitch, s.roll,
                                  "measured", (s.t, s.t))
    before = [s for s in present if s.t < t]
    after = [s for s in present if s.t > t]
    if not before or not after:
        return AttitudeResult(None, None, None, "missing-gap", (),
                              "outside attitude coverage (no extrapolation)")
    a, b = before[-1], after[0]
    gaps = find_gaps([s.t for s in present], max_dt)
    if any(g.t0 == a.t and g.t1 == b.t for g in gaps) or (b.t - a.t) > max_dt:
        return AttitudeResult(None, None, None, "missing-gap", (a.t, b.t),
                              f"attitude gap {b.t - a.t:g}s not bridged")
    w = (t - a.t) / (b.t - a.t)
    dh = wrapped_delta(a.heading, b.heading)
    return AttitudeResult(wrap360(a.heading + dh * w),
                          a.pitch + (b.pitch - a.pitch) * w,
                          a.roll + (b.roll - a.roll) * w,
                          "interpolated", (a.t, b.t))
