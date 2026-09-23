"""Attitude interpolation with heading wrap-around and gap protection.

Rules (enforced and tested):

* Heading samples are angles on the circle; interpolation is done after
  unwrapping so crossing 360/0 stays continuous.  A wrap event is recorded
  whenever two consecutive raw samples straddle the +/-180 cut.
* Interpolation never bridges a gap.  If the bracketing sample interval is
  longer than ``max_gap_s`` (configurable), the requested time stays
  indeterminate and the gap interval is reported.
* Pitch and roll are interpolated linearly inside the same brackets; a gap
  blocks all three channels together (no half-filled attitude).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    heading_deg: float        # 0..360, magnetic or true per source metadata
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class AttitudeResult:
    t: float
    ok: bool
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    bracket: Optional[tuple[int, int]]
    span: Optional[float]
    wrap_applied: bool
    reason: Optional[str]


def find_wrap_events(samples: Sequence[AttitudeSample]) -> list[dict]:
    """Record indices where consecutive raw headings cross the 360/0 cut."""
    events = []
    for i in range(1, len(samples)):
        delta = samples[i].heading_deg - samples[i - 1].heading_deg
        if abs(delta) > 180.0:
            events.append({
                "after_index": i - 1,
                "t0": samples[i - 1].t,
                "t1": samples[i].t,
                "h0": samples[i - 1].heading_deg,
                "h1": samples[i].heading_deg,
                "unwrapped_delta": ((delta + 180.0) % 360.0) - 180.0,
            })
    return events


def _unwrap_series(samples: Sequence[AttitudeSample]) -> list[float]:
    unwrapped = [samples[0].heading_deg]
    for i in range(1, len(samples)):
        delta = samples[i].heading_deg - samples[i - 1].heading_deg
        delta = ((delta + 180.0) % 360.0) - 180.0
        unwrapped.append(unwrapped[-1] + delta)
    return unwrapped


def interpolate(samples: Sequence[AttitudeSample], t: float,
                max_gap_s: float) -> AttitudeResult:
    """Linear interpolation; never extrapolates and never bridges a gap."""
    if not samples:
        return AttitudeResult(t, False, None, None, None, None, None, False,
                              "no attitude samples")
    times = [s.t for s in samples]
    if t < times[0] or t > times[-1]:
        return AttitudeResult(t, False, None, None, None, None, None, False,
                              "outside attitude sample span")
    j = 0
    while j + 1 < len(times) and times[j + 1] <= t:
        j += 1
    if times[j] == t:
        s = samples[j]
        return AttitudeResult(t, True, s.heading_deg % 360.0, s.pitch_deg,
                              s.roll_deg, (j, j), 0.0, False, None)
    k = j + 1
    if k >= len(samples):
        return AttitudeResult(t, False, None, None, None, None, None, False,
                              "outside attitude sample span")
    span = times[k] - times[j]
    if span > max_gap_s:
        return AttitudeResult(t, False, None, None, None, (j, k), span, False,
                              f"gap {span:g}s exceeds max {max_gap_s:g}s; not interpolated")
    frac = (t - times[j]) / span
    unwrapped = _unwrap_series(samples)
    wrap = abs(unwrapped[k] - unwrapped[j]) > 180.0 or \
        abs(samples[k].heading_deg - samples[j].heading_deg) > 180.0
    heading = (unwrapped[j] + frac * (unwrapped[k] - unwrapped[j])) % 360.0
    pitch = samples[j].pitch_deg + frac * (samples[k].pitch_deg - samples[j].pitch_deg)
    roll = samples[j].roll_deg + frac * (samples[k].roll_deg - samples[j].roll_deg)
    return AttitudeResult(t, True, heading, pitch, roll, (j, k), span,
                          bool(wrap), None)
