"""Attitude lookup and gap-safe interpolation.

Heading is circular: samples near 359deg / 1deg must interpolate across the
0/360 seam rather than swinging the long way around. Interpolation never
crosses a recorded time gap: a requested time without two in-gap bracketing
samples returns no attitude plus a diagnostic.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

ATTITUDE_RULE_VERSION = "attitude-2.0.0"


@dataclass(frozen=True)
class AttitudeSample:
    time_s: float
    heading_deg: float       # 0..360, magnetic compass reading
    pitch_deg: float
    roll_deg: float
    quality_flag: int = 0    # 0 valid


@dataclass(frozen=True)
class AttitudeResult:
    ok: bool
    heading_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    diagnostic: str | None = None
    bracket: tuple[int, int] | None = None

    def as_rad(self) -> tuple[float, float, float] | None:
        if not self.ok:
            return None
        return (math.radians(self.heading_deg),
                math.radians(self.pitch_deg),
                math.radians(self.roll_deg))


def _wrap360(angle: float) -> float:
    return angle % 360.0


def lookup(samples: list[AttitudeSample], t: float) -> AttitudeResult:
    for s in samples:
        if abs(s.time_s - t) <= 1e-9:
            if s.quality_flag:
                return AttitudeResult(False, None, None, None,
                                      f"attitude_flag:{s.quality_flag}")
            return AttitudeResult(True, s.heading_deg, s.pitch_deg, s.roll_deg)
    return AttitudeResult(False, None, None, None, "attitude_missing")


def interpolate(
    samples: list[AttitudeSample],
    t: float,
    max_gap_s: float,
) -> AttitudeResult:
    """Linear interpolation between the two enclosing valid samples.

    Fails (never extrapolates, never bridges) if ``t`` is outside the sample
    span, a bracket sample is flagged, or the bracket span exceeds
    ``max_gap_s`` (a recorded time gap).
    """
    valid = sorted((s for s in samples if not s.quality_flag),
                   key=lambda s: s.time_s)
    if len(valid) < 2:
        return AttitudeResult(False, None, None, None, "attitude_insufficient")
    times = [s.time_s for s in valid]
    if t < times[0] - 1e-9 or t > times[-1] + 1e-9:
        return AttitudeResult(False, None, None, None, "attitude_no_extrapolation")
    j = bisect.bisect_left(times, t)
    if j == 0:
        i0, i1 = 0, 1
    elif j == len(times):
        i0, i1 = len(times) - 2, len(times) - 1
    elif abs(times[j] - t) <= 1e-9:
        s = valid[j]
        return AttitudeResult(True, s.heading_deg, s.pitch_deg, s.roll_deg,
                              bracket=(j, j))
    else:
        i0, i1 = j - 1, j
    a, b = valid[i0], valid[i1]
    span = b.time_s - a.time_s
    if span > max_gap_s + 1e-9:
        return AttitudeResult(False, None, None, None,
                              f"attitude_gap:{span:.3f}s>{max_gap_s:.3f}s",
                              bracket=(i0, i1))
    f = (t - a.time_s) / span
    # Unwrap heading onto a continuous axis so 359->1 crosses zero correctly.
    delta = (b.heading_deg - a.heading_deg + 180.0) % 360.0 - 180.0
    heading = _wrap360(a.heading_deg + f * delta)
    return AttitudeResult(True, heading,
                          a.pitch_deg + f * (b.pitch_deg - a.pitch_deg),
                          a.roll_deg + f * (b.roll_deg - a.roll_deg),
                          bracket=(i0, i1))
