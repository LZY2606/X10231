"""Attitude interpolation between heading/pitch/roll samples.

Interpolation never crosses a time gap: when the bracketing samples are more
than ``max_gap_s`` apart (or either side is missing), the ping is reported as
``gap`` with no interpolated attitude, so earth-frame velocity cannot be
produced at that ping instead of being guessed.

Headings are unwrapped before interpolation so a step 359 deg -> 1 deg across
the 360 boundary interpolates through 360/0 rather than spinning backwards.
"""
from __future__ import annotations

from dataclasses import dataclass

from .units import normalize_degrees, unwrap_headings_deg


@dataclass(frozen=True)
class AttitudeSample:
    t_epoch: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class InterpolatedAttitude:
    status: str            # "ok" | "gap" | "outside_range"
    heading_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    left_index: int | None = None
    right_index: int | None = None
    bracket_gap_s: float | None = None


def interpolate(samples: list[AttitudeSample], t_epoch: float,
                max_gap_s: float = 2.0) -> InterpolatedAttitude:
    if not samples:
        return InterpolatedAttitude("gap")
    seq = sorted(samples, key=lambda s: s.t_epoch)
    if t_epoch < seq[0].t_epoch or t_epoch > seq[-1].t_epoch:
        return InterpolatedAttitude("outside_range")

    right = next((i for i, s in enumerate(seq) if s.t_epoch >= t_epoch), len(seq) - 1)
    left = max(0, right - 1)
    a, b = seq[left], seq[right]
    gap = b.t_epoch - a.t_epoch
    if gap > max_gap_s:
        return InterpolatedAttitude("gap", None, None, None, left, right, gap)
    if gap == 0:
        frac = 0.0
    else:
        frac = (t_epoch - a.t_epoch) / gap

    # circular heading interpolation across the 0/360 seam
    unwrapped = unwrap_headings_deg([a.heading_deg, b.heading_deg])
    heading = normalize_degrees(unwrapped[0] + frac * (unwrapped[1] - unwrapped[0]))
    pitch = a.pitch_deg + frac * (b.pitch_deg - a.pitch_deg)
    roll = a.roll_deg + frac * (b.roll_deg - a.roll_deg)
    return InterpolatedAttitude("ok", heading, pitch, roll, left, right, gap)
