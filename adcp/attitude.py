"""Heading / pitch / roll interpolation with explicit gap handling.

Attitude samples are an independent time series (they are recorded
separately from pings). Interpolation is linear for pitch/roll and
wrap-aware for heading. It never crosses a time gap larger than the
configured threshold: the ping is then reported as ``gap`` and the two
bounding samples are attached as evidence, not silently bridged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

OK = "ok"
GAP = "gap"
EXTRAP = "extrapolated"
INVALID = "invalid"


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def unwrap_heading(values: list[float]) -> list[float]:
    """Remove 360-degree discontinuities so interpolation is monotone."""
    out: list[float] = []
    for v in values:
        if out:
            out.append(out[-1] + wrap180(v - out[-1]))
        else:
            out.append(v)
    return out


def _lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class AttitudeResult:
    ping_id: str
    t: float
    heading: float | None
    pitch: float | None
    roll: float | None
    status: str
    gap_seconds: float | None = None
    bounding: list[str] = field(default_factory=list)  # sample ids used/refused
    diagnostics: list[str] = field(default_factory=list)


class AttitudeSeries:
    """Sortable, wrap-aware attitude log keyed by sample id."""

    def __init__(self, samples: list[dict]):
        self.samples = sorted(samples, key=lambda s: s["t"])
        # Pre-compute unwrapped heading so a 359 -> 1 transition stays smooth.
        heads = [s["heading"] for s in self.samples]
        self._unwrapped = unwrap_heading(heads)

    def at(self, ping_id: str, t: float, gap_threshold_s: float = 45.0) -> AttitudeResult:
        n = len(self.samples)
        if n == 0:
            return AttitudeResult(ping_id, t, None, None, None, INVALID,
                                  diagnostics=["no attitude samples"])
        if t < self.samples[0]["t"] - 1e-9 or t > self.samples[-1]["t"] + 1e-9:
            s0 = self.samples[0] if t < self.samples[0]["t"] else self.samples[-1]
            return AttitudeResult(ping_id, t, s0["heading"], s0["pitch"], s0["roll"],
                                  EXTRAP, diagnostics=["ping outside attitude log; no extrapolation accepted"])
        for i in range(n - 1):
            a, b = self.samples[i], self.samples[i + 1]
            if a["t"] - 1e-9 <= t <= b["t"] + 1e-9:
                span = b["t"] - a["t"]
                if span > gap_threshold_s:
                    # Hard rule: attitude interpolation never crosses a gap.
                    return AttitudeResult(
                        ping_id, t, None, None, None, GAP, gap_seconds=span,
                        bounding=[a["sample_id"], b["sample_id"]],
                        diagnostics=[f"attitude gap {span:g}s exceeds threshold "
                                     f"{gap_threshold_s:g}s; values not bridged"])
                h = _lerp(a["t"], self._unwrapped[i], b["t"], self._unwrapped[i + 1], t)
                p = _lerp(a["t"], a["pitch"], b["t"], b["pitch"], t)
                r = _lerp(a["t"], a["roll"], b["t"], b["roll"], t)
                return AttitudeResult(ping_id, t, h % 360.0, p, r, OK,
                                      gap_seconds=span,
                                      bounding=[a["sample_id"], b["sample_id"]])
        last = self.samples[-1]
        return AttitudeResult(ping_id, t, last["heading"], last["pitch"], last["roll"], OK)
