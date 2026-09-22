"""Heading/pitch/roll handling.

Heading is circular: interpolating 359 -> 1 must yield 0, not 180. Attitude is
never interpolated across a time gap: a query not bracketed by samples inside
the allowed gap is returned as unknown with an explicit reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def wrap_deg(a: float) -> float:
    """Wrap any angle into [0, 360)."""
    return a % 360.0


def circular_interp(a0: float, a1: float, frac: float) -> float:
    """Linear interpolation on the circle; result wrapped to [0, 360)."""
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    return wrap_deg(a0 + frac * d)


@dataclass
class AttitudeSample:
    t: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]

    @staticmethod
    def from_dict(d: dict) -> "AttitudeSample":
        return AttitudeSample(float(d["t"]), d.get("heading_deg"),
                              d.get("pitch_deg"), d.get("roll_deg"))

    def to_dict(self) -> dict:
        return {"t": self.t, "heading_deg": self.heading_deg,
                "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg}


@dataclass
class AttitudeResult:
    t: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    status: str            # "interpolated" | "exact" | "gap" | "outside"
    reason: Optional[str]
    bracket: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "heading_deg": self.heading_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "status": self.status,
            "reason": self.reason,
            "bracket": None if self.bracket is None else list(self.bracket),
        }


def _linear(v0: Optional[float], v1: Optional[float], frac: float) -> Optional[float]:
    if v0 is None or v1 is None:
        return None
    return v0 + frac * (v1 - v0)


def attitude_at(
    samples: Sequence[AttitudeSample],
    t: float,
    max_gap_s: float,
) -> AttitudeResult:
    """Bracket `t`; refuse interpolation across a gap wider than max_gap_s.

    Samples are expected sorted by time. None heading/pitch/roll values are
    propagated as missing rather than filled.
    """
    if not samples:
        return AttitudeResult(t, None, None, None, "outside", "no_samples")
    ts = [s.t for s in samples]
    if t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
        return AttitudeResult(t, None, None, None, "outside",
                              "outside_sample_window")
    # exact or bracketing sample (samples are sorted by time)
    i0 = max(i for i, s in enumerate(samples) if s.t <= t)
    i1 = min(i for i, s in enumerate(samples) if s.t >= t)
    s0, s1 = samples[i0], samples[i1]
    if s0 is s1:
        return AttitudeResult(t, s0.heading_deg, s0.pitch_deg, s0.roll_deg,
                              "exact", None, (s0.t, s1.t))
    span = s1.t - s0.t
    if span > max_gap_s + 1e-9:
        return AttitudeResult(t, None, None, None, "gap",
                              "attitude_gap %.1fs > max %.1fs" % (span, max_gap_s),
                              (s0.t, s1.t))
    frac = (t - s0.t) / span
    h = circular_interp(s0.heading_deg, s1.heading_deg, frac) \
        if s0.heading_deg is not None and s1.heading_deg is not None else None
    return AttitudeResult(
        t, h,
        _linear(s0.pitch_deg, s1.pitch_deg, frac),
        _linear(s0.roll_deg, s1.roll_deg, frac),
        "interpolated", None, (s0.t, s1.t),
    )
