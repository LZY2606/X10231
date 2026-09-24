"""Attitude series with 360-degree unwrap and gap-aware interpolation.

Interpolation never crosses a time gap: samples separated by more than
`gap_threshold_s` are a hard boundary; queries inside a gap return status
'gap' with no fabricated values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AttitudeResult:
    t: float
    heading: float | None
    pitch: float | None
    roll: float | None
    status: str           # exact | interpolated | gap | out_of_range
    diagnostic: str | None = None

    def to_dict(self):
        return {
            "t": self.t, "heading": self.heading, "pitch": self.pitch,
            "roll": self.roll, "status": self.status,
            "diagnostic": self.diagnostic,
        }


class AttitudeSeries:
    def __init__(self, times, headings, pitches, rolls, gap_threshold_s=3.0):
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=float)[order]
        h = np.asarray(headings, dtype=float)[order]
        # unwrap across the 360/0 boundary before any interpolation
        self.headings = np.rad2deg(np.unwrap(np.deg2rad(h)))
        self.pitches = np.asarray(pitches, dtype=float)[order]
        self.rolls = np.asarray(rolls, dtype=float)[order]
        self.gap_threshold_s = float(gap_threshold_s)
        self.wrapped = bool(np.any(np.abs(np.diff(h)) > 180.0))

    def interpolate(self, t) -> AttitudeResult:
        t = float(t)
        times = self.times
        if t < times[0] or t > times[-1]:
            return AttitudeResult(t, None, None, None, "out_of_range",
                                  "no attitude samples bracket this ping")
        idx = int(np.searchsorted(times, t))
        if idx < len(times) and times[idx] == t:
            return AttitudeResult(t, float(self.headings[idx]) % 360.0,
                                  float(self.pitches[idx]), float(self.rolls[idx]),
                                  "exact")
        lo, hi = idx - 1, idx
        gap = times[hi] - times[lo]
        if gap > self.gap_threshold_s:
            return AttitudeResult(
                t, None, None, None, "gap",
                f"attitude gap {gap:.2f}s > {self.gap_threshold_s:.2f}s; "
                "interpolation across gaps is forbidden")
        w = (t - times[lo]) / gap
        h = self.headings[lo] + w * (self.headings[hi] - self.headings[lo])
        p = self.pitches[lo] + w * (self.pitches[hi] - self.pitches[lo])
        r = self.rolls[lo] + w * (self.rolls[hi] - self.rolls[lo])
        return AttitudeResult(t, float(h) % 360.0, float(p), float(r),
                              "interpolated")
