"""Heading/pitch/roll interpolation.

Pitch and roll are plain linear.  Heading is interpolated on the shortest
arc on the circle (complex unit vector).  Interpolation never crosses a
time gap: pings falling inside a gap are emitted as a GAP status with no
attitude, so downstream frames are blocked rather than fabricated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AttitudeSample:
    t: float                       # seconds
    heading_deg: Optional[float]   # None = missing at source
    pitch_deg: Optional[float]
    roll_deg: Optional[float]


@dataclass
class AttitudeAtPing:
    t: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    status: str                    # "measured" | "interpolated" | "gap" | "missing"
    bracket: Optional[tuple] = None       # (t0, t1)
    diagnostics: List[dict] = field(default_factory=list)


def _lerp(a, b, f):
    return a + (b - a) * f


def _shortest_arc_interp(h0: float, h1: float, f: float):
    """Return interpolated heading (deg) and the naive-linear value."""
    naive = _lerp(h0, h1, f)
    z0 = complex(math.cos(math.radians(h0)), math.sin(math.radians(h0)))
    z1 = complex(math.cos(math.radians(h1)), math.sin(math.radians(h1)))
    z = z0 * (z1 / z0) ** f
    arc = math.degrees(math.atan2(z.imag, z.real)) % 360.0
    return arc, naive % 360.0


def interpolate(samples: List[AttitudeSample], t: float,
                gap_threshold_s: float) -> AttitudeAtPing:
    valid = [s for s in samples if s.heading_deg is not None
             and s.pitch_deg is not None and s.roll_deg is not None]
    if not valid:
        return AttitudeAtPing(t, None, None, None, "missing",
                              diagnostics=[{"code": "NO_ATTITUDE_SAMPLES"}])

    before = [s for s in valid if s.t <= t]
    after = [s for s in valid if s.t >= t]
    lo = max(before, key=lambda s: s.t) if before else None
    hi = min(after, key=lambda s: s.t) if after else None

    if lo is not None and abs(lo.t - t) < 1e-9:
        return AttitudeAtPing(t, lo.heading_deg, lo.pitch_deg, lo.roll_deg,
                              "measured", bracket=(lo.t, lo.t))
    if hi is not None and abs(hi.t - t) < 1e-9:
        return AttitudeAtPing(t, hi.heading_deg, hi.pitch_deg, hi.roll_deg,
                              "measured", bracket=(hi.t, hi.t))
    if lo is None or hi is None or hi.t <= lo.t:
        return AttitudeAtPing(t, None, None, None, "gap",
                              diagnostics=[{"code": "ATTITUDE_EXTRAPOLATION",
                                            "t": t}])
    span = hi.t - lo.t
    if span > gap_threshold_s:
        return AttitudeAtPing(
            t, None, None, None, "gap", bracket=(lo.t, hi.t),
            diagnostics=[{"code": "ATTITUDE_GAP", "t": t,
                          "bracket": [lo.t, hi.t], "gap_seconds": span,
                          "threshold_seconds": gap_threshold_s}])
    f = (t - lo.t) / span
    heading, naive = _shortest_arc_interp(lo.heading_deg, hi.heading_deg, f)
    diag = []
    if abs((naive - heading + 180.0) % 360.0 - 180.0) > 1e-9:
        diag.append({"code": "HEADING_WRAP_360", "t": t,
                     "bracket": [lo.t, hi.t],
                     "heading_endpoints_deg": [lo.heading_deg, hi.heading_deg],
                     "shortest_arc_deg": heading,
                     "naive_linear_deg": naive,
                     "message": "shortest-arc interpolation differs from "
                                "naive linear interpolation across 360 deg"})
    return AttitudeAtPing(
        t, heading, _lerp(lo.pitch_deg, hi.pitch_deg, f),
        _lerp(lo.roll_deg, hi.roll_deg, f),
        "interpolated", bracket=(lo.t, hi.t), diagnostics=diag)
