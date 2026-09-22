"""Attitude handling: heading wrap and gap-aware interpolation.

Raw heading is recorded exactly as imported (degrees clockwise from magnetic
north, typically 0..359.99).  Two derived services are provided:

1. ``segments``  - split a time series wherever the time gap exceeds a
   threshold.  Interpolation is **forbidden across** a segment boundary.
2. ``interpolate`` - linear interpolation inside one segment; heading is
   unwrapped before interpolating so a 359 -> 1 crossing interpolates through
   0/360 instead of sweeping the long way around.

Attitude interpolation therefore never bridges a data gap (case requirement).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AttitudePoint:
    t: float                      # seconds (relative epoch is fine)
    heading: float                # raw degrees, magnetic, 0..360
    pitch: float                  # raw degrees
    roll: float                   # raw degrees
    present: bool = True


@dataclass(frozen=True)
class InterpResult:
    t: float
    heading: Optional[float]      # always re-wrapped to [0, 360)
    pitch: Optional[float]
    roll: Optional[float]
    status: str                   # "interpolated" | "exact" | "gap" | "outside"
    segment_id: Optional[int]
    diagnostic: Optional[str] = None


def split_segments(points: Sequence[AttitudePoint],
                   max_gap_s: float) -> List[List[AttitudePoint]]:
    """Break ``points`` (time sorted) at gaps strictly greater than max_gap_s."""
    pts = sorted([p for p in points if p.present], key=lambda p: p.t)
    segments: List[List[AttitudePoint]] = []
    for p in pts:
        if not segments or p.t - segments[-1][-1].t > max_gap_s:
            segments.append([p])
        else:
            segments[-1].append(p)
    return segments


def _unwrap(angles: Sequence[float]) -> List[float]:
    out = [angles[0]]
    for a in angles[1:]:
        prev = out[-1]
        delta = (a - prev + 180.0) % 360.0 - 180.0
        out.append(prev + delta)
    return out


def _wrap01_360(a: float) -> float:
    return a % 360.0


def _lerp(x0, y0, x1, y1, x):
    if x1 == x0:
        return y0
    f = (x - x0) / (x1 - x0)
    return y0 + f * (y1 - y0)


def interpolate(points: Sequence[AttitudePoint],
                t: float,
                max_gap_s: float = 1.0,
                gap_boundary: bool = True) -> InterpResult:
    """Interpolate attitude at time ``t`` without crossing a gap.

    A requested time landing outside any segment span, or straddling a gap,
    returns a ``gap``/``outside`` result with ``None`` angles; the caller must
    keep the un-interpolated value missing rather than inventing one.
    """
    segments = split_segments(points, max_gap_s)
    for sid, seg in enumerate(segments):
        t0, t1 = seg[0].t, seg[-1].t
        if t0 - 1e-12 <= t <= t1 + 1e-12:
            # exact hit?
            for p in seg:
                if abs(p.t - t) <= 1e-12:
                    return InterpResult(t, _wrap01_360(p.heading), p.pitch,
                                        p.roll, "exact", sid)
            # find bracketing neighbours (guaranteed within this segment)
            for a, b in zip(seg[:-1], seg[1:]):
                if a.t <= t <= b.t:
                    if gap_boundary and b.t - a.t > max_gap_s:
                        return InterpResult(t, None, None, None, "gap", sid,
                                            "interpolation crosses gap "
                                            "%.3fs>%.3fs" % (b.t - a,
                                                             max_gap_s))
                    headings = _unwrap([a.heading, b.heading])
                    h = _wrap01_360(_lerp(a.t, headings[0], b.t, headings[1], t))
                    p = _lerp(a.t, a.pitch, b.t, b.pitch, t)
                    r = _lerp(a.t, a.roll, b.t, b.roll, t)
                    return InterpResult(t, h, p, r, "interpolated", sid)
            return InterpResult(t, None, None, None, "gap", sid,
                                "no bracketing samples in segment")
    # detect whether t falls between two segments (a true data gap)
    allpts = sorted(points, key=lambda p: p.t)
    if allpts and allpts[0].t <= t <= allpts[-1].t:
        return InterpResult(t, None, None, None, "gap", None,
                            "time falls inside a data gap; no interpolation "
                            "across missing attitude")
    return InterpResult(t, None, None, None, "outside", None,
                        "time outside attitude time span")


def gap_intervals(points: Sequence[AttitudePoint],
                  max_gap_s: float) -> List[Tuple[float, float]]:
    """Return (start,end) intervals of detected gaps for evidence display."""
    pts = sorted([p for p in points if p.present], key=lambda p: p.t)
    gaps = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b.t - a.t > max_gap_s:
            gaps.append((a.t, b.t))
    return gaps


def magnetic_heading_to_true(heading_deg: float, declination_deg: float) -> float:
    """Apply magnetic declination. Positive declination = east of north.

    Both the declination value and its *version* are tracked separately in the
    store and become part of the derived fingerprint (see derive/fingerprint).
    """
    return _wrap01_360(heading_deg + declination_deg)
