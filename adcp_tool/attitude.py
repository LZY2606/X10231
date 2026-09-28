"""Attitude interpolation with heading wrap and gap protection.

Rules kept here (and exercised by the counterexample pack):

* Heading is an angular quantity, so interpolation happens in (sin, cos)
  space. Crossing 359 -> 1 deg therefore travels 2 degrees, not 358.
* Linear attitude interpolation NEVER spans a time gap. When the samples
  bracketing a target time are separated by more than ``max_gap_s`` the
  target is marked ``gap`` and left undetermined rather than extrapolated.
* Pitch/roll interpolate linearly and share the same gap rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    source: str = "attitude"


@dataclass(frozen=True)
class InterpAttitude:
    t: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    status: str          # "ok" | "gap" | "missing" | "exact"
    bracket: Optional[tuple[int, int]] = None
    source_span: Optional[tuple[str, str]] = None


def wrap360(a: float) -> float:
    return a % 360.0


def lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def interp_heading_deg(h0: float, h1: float, f: float) -> float:
    """Angular interpolation choosing the short way across the branch cut."""
    h0, h1 = np.deg2rad(h0), np.deg2rad(h1)
    s = lerp(np.sin(h0), np.sin(h1), f)
    c = lerp(np.cos(h0), np.cos(h1), f)
    return wrap360(float(np.rad2deg(np.arctan2(s, c))))


class AttitudeSeries:
    def __init__(self, samples: list[AttitudeSample], max_gap_s: float = 2.5):
        self.samples = sorted(samples, key=lambda s: s.t)
        self.max_gap_s = max_gap_s

    def at(self, t: float) -> InterpAttitude:
        ts = [s.t for s in self.samples]
        if not self.samples:
            return InterpAttitude(t, None, None, None, "missing")
        # exact hit
        for i, s in enumerate(self.samples):
            if s.t == t:
                return InterpAttitude(t, s.heading_deg, s.pitch_deg, s.roll_deg,
                                      "exact", (i, i), (s.source, s.source))
        idx = int(np.searchsorted(ts, t))
        if idx == 0 or idx == len(self.samples):
            return InterpAttitude(t, None, None, None, "gap")
        a, b = self.samples[idx - 1], self.samples[idx]
        # never interpolate across a time gap
        if (b.t - a.t) > self.max_gap_s:
            return InterpAttitude(t, None, None, None, "gap",
                                  (idx - 1, idx), (a.source, b.source))
        if None in (a.heading_deg, b.heading_deg, a.pitch_deg, b.pitch_deg,
                    a.roll_deg, b.roll_deg):
            return InterpAttitude(t, None, None, None, "gap",
                                  (idx - 1, idx), (a.source, b.source))
        f = (t - a.t) / (b.t - a.t)
        return InterpAttitude(
            t,
            interp_heading_deg(a.heading_deg, b.heading_deg, f),
            lerp(a.pitch_deg, b.pitch_deg, f),
            lerp(a.roll_deg, b.roll_deg, f),
            "ok", (idx - 1, idx), (a.source, b.source),
        )

    def gaps(self) -> list[tuple[float, float, float]]:
        return [(a.t, b.t, b.t - a.t)
                for a, b in zip(self.samples, self.samples[1:])
                if b.t - a.t > self.max_gap_s]
