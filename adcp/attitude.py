"""Attitude series handling: 360-degree wrap and gap-aware interpolation.

Two invariants are enforced here and surfaced as machine-readable events:

* headings are unwrapped before interpolation, so a series crossing
  360 -> 0 interpolates through 0, never through 180;
* interpolation never crosses a time gap larger than ``max_gap_s`` -- pings
  inside such a gap get status ``"gap"`` and no attitude value at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def unwrap_headings(headings_deg: Sequence[float]) -> List[float]:
    return np.rad2deg(np.unwrap(np.deg2rad(list(headings_deg)))).tolist()


def wrap_heading(heading_deg: float) -> float:
    return float(heading_deg) % 360.0


def interpolate_attitude(
    times_s: Sequence[float],
    headings_deg: Sequence[float],
    pitches_deg: Sequence[float],
    rolls_deg: Sequence[float],
    target_times_s: Sequence[float],
    max_gap_s: float,
) -> List[Dict[str, object]]:
    """Interpolate attitude onto ping times without crossing gaps.

    Returns one record per target time:
    ``{"time", "status": "ok"|"gap"|"out_of_range",
       "heading"|"pitch"|"roll" (deg) or None, "gap_after_s"}``
    """
    times = [float(t) for t in times_s]
    if len(times) < 2:
        raise ValueError("need at least two attitude samples")
    order = sorted(range(len(times)), key=lambda i: times[i])
    t = [times[i] for i in order]
    h = unwrap_headings([headings_deg[i] for i in order])
    p = [float(pitches_deg[i]) for i in order]
    r = [float(rolls_deg[i]) for i in order]

    out: List[Dict[str, object]] = []
    for target in target_times_s:
        tt = float(target)
        rec: Dict[str, object] = {
            "time": tt, "status": "ok", "heading": None,
            "pitch": None, "roll": None, "gap_after_s": None,
        }
        if tt < t[0] or tt > t[-1]:
            rec["status"] = "out_of_range"
            out.append(rec)
            continue
        idx = int(np.searchsorted(t, tt, side="right")) - 1
        idx = min(max(idx, 0), len(t) - 2)
        gap = t[idx + 1] - t[idx]
        rec["gap_after_s"] = gap
        if gap > max_gap_s and not (t[idx] <= tt <= t[idx] and gap == 0.0):
            # Inside (or at the edge of) a gap wider than allowed: do not
            # interpolate across it.
            if tt > t[idx] or gap > max_gap_s:
                rec["status"] = "gap"
                out.append(rec)
                continue
        frac = 0.0 if gap == 0.0 else (tt - t[idx]) / gap
        rec["heading"] = wrap_heading(h[idx] + frac * (h[idx + 1] - h[idx]))
        rec["pitch"] = p[idx] + frac * (p[idx + 1] - p[idx])
        rec["roll"] = r[idx] + frac * (r[idx + 1] - r[idx])
        out.append(rec)
    return out
