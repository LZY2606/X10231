"""Attitude streams: heading wrap handling and gap-aware interpolation.

Semantics required by the case archive:

* Heading values crossing 360 deg must be handled on the circle
  (e.g. 359 deg -> 1 deg is a 2 deg change, not a 358 deg spin).
* Interpolation never crosses a declared time gap. A query inside a gap, or
  bracketed by a gap, is returned as *not estimable* with a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    heading: float  # degrees, 0..360
    pitch: float  # degrees
    roll: float  # degrees


@dataclass(frozen=True)
class Gap:
    t0: float
    t1: float

    def contains(self, t: float) -> bool:
        return self.t0 < t < self.t1


def unwrap_heading_deg(headings: np.ndarray) -> np.ndarray:
    """Remove 360 deg jumps, preserving the first value's absolute level."""
    h = np.asarray(headings, dtype=float)
    return np.unwrap(np.deg2rad(h), period=2 * np.pi) * 180.0 / np.pi


def interpolate_attitude(
    samples: list[AttitudeSample],
    gaps: list[Gap],
    t: float,
    *,
    max_spacing_s: float | None = None,
) -> dict:
    """Linearly interpolate attitude at time ``t``.

    Returns ``{"ok": bool, "heading": float|None, "pitch": ..., "roll": ...,
    "diagnostic": str|None, "bracket": (i0, i1)|None}``.

    Rules:

    * Exact sample hit -> that sample.
    * Any declared gap overlapping the bracketing interval -> refused.
    * ``max_spacing_s`` optionally refuses interpolation over sparse data;
      the case pack uses it to mark missing attitude as not-estimable rather
      than guessing.
    """

    if not samples:
        return {"ok": False, "heading": None, "pitch": None, "roll": None,
                "diagnostic": "no-attitude-samples", "bracket": None}
    ts = np.array([s.t for s in samples], dtype=float)
    order = np.argsort(ts)
    ts = ts[order]
    if t < ts[0] or t > ts[-1]:
        return {"ok": False, "heading": None, "pitch": None, "roll": None,
                "diagnostic": "outside-attitude-range", "bracket": None}
    idx = int(np.searchsorted(ts, t))
    if idx < len(ts) and abs(ts[idx] - t) < 1e-12:
        s = samples[int(order[idx])]
        return {"ok": True, "heading": s.heading, "pitch": s.pitch,
                "roll": s.roll, "diagnostic": None,
                "bracket": (int(order[idx]), int(order[idx]))}
    i1 = idx
    i0 = idx - 1
    t0, t1 = ts[i0], ts[i1]
    for gap in gaps:
        if not (t1 <= gap.t0 or t0 >= gap.t1):
            return {"ok": False, "heading": None, "pitch": None,
                    "roll": None,
                    "diagnostic": f"attitude-gap[{gap.t0},{gap.t1}]",
                    "bracket": (int(order[i0]), int(order[i1]))}
    if max_spacing_s is not None and (t1 - t0) > max_spacing_s:
        return {"ok": False, "heading": None, "pitch": None, "roll": None,
                "diagnostic": f"spacing>{max_spacing_s}s",
                "bracket": (int(order[i0]), int(order[i1]))}
    f = (t - t0) / (t1 - t0)
    s0 = samples[int(order[i0])]
    s1 = samples[int(order[i1])]
    # Circular interpolation for heading via unwrap; plain for pitch/roll.
    h0, h1 = s0.heading, s1.heading
    dh = (h1 - h0 + 180.0) % 360.0 - 180.0
    heading = (h0 + f * dh) % 360.0
    return {
        "ok": True,
        "heading": float(heading),
        "pitch": float(s0.pitch + f * (s1.pitch - s0.pitch)),
        "roll": float(s0.roll + f * (s1.roll - s0.roll)),
        "diagnostic": None,
        "bracket": (int(order[i0]), int(order[i1])),
    }
