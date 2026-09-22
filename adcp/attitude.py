"""Attitude interpolation across ping times.

Rules that must hold:
  * Heading interpolations unwrap the 0/360 degree boundary per continuous run.
  * Interpolation NEVER crosses a gap: a run is broken when spacing exceeds
    gap_factor * median spacing. Samples inside a gap are marked missing.
  * Source records remain separate; only the derived layer interpolates.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _unwrap_deg(headings: List[float]) -> List[float]:
    out = [headings[0]]
    for h in headings[1:]:
        prev = out[-1]
        delta = (h - prev + 180.0) % 360.0 - 180.0
        out.append(prev + delta)
    return out


def _runs(times: Sequence[float], gap_factor: float) -> List[Tuple[int, int]]:
    times = list(times)
    if len(times) < 2:
        return [(0, len(times) - 1)] if times else []
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    median = float(np.median(diffs))
    limit = gap_factor * median if median > 0 else float("inf")
    runs, start = [], 0
    for i, d in enumerate(diffs):
        if d > limit:
            runs.append((start, i))
            start = i + 1
    runs.append((start, len(times) - 1))
    return runs


def interpolate_attitude(ping_times: Sequence[float],
                         sample_times: Sequence[float],
                         headings_deg: Sequence[float],
                         pitch_deg: Sequence[float],
                         roll_deg: Sequence[float],
                         gap_factor: float = 2.5) -> List[dict]:
    """Return one attitude result per ping time; gap samples are missing."""
    st = list(sample_times)
    runs = _runs(st, gap_factor)
    h_un = _unwrap_deg([float(x) for x in headings_deg])
    results: List[dict] = []
    for t in ping_times:
        placed = None
        for a, z in runs:
            if st[a] <= t <= st[z]:
                placed = (a, z)
                break
        if placed is None:
            results.append({
                "time": float(t), "available": False,
                "diagnostic": "attitude-gap" if st else "no-attitude",
                "run": None,
            })
            continue
        a, z = placed
        if a == z:
            hu, p, r = h_un[a], float(pitch_deg[a]), float(roll_deg[a])
            frac, lo, hi = 0.0, a, a
        else:
            span = st[z] - st[a]
            frac = 0.0 if span == 0 else (t - st[a]) / span
            lo = max(a, next((i for i in range(z, a - 1, -1) if st[i] <= t), a))
            hi = min(z, next((i for i in range(a, z + 1) if st[i] >= t), z))
            def lerp(y):
                if lo == hi:
                    return float(y[lo])
                f = (t - st[lo]) / (st[hi] - st[lo])
                return float(y[lo]) + f * (float(y[hi]) - float(y[lo]))
            hu = lerp(h_un)
            p = lerp(pitch_deg)
            r = lerp(roll_deg)
        heading = hu % 360.0
        results.append({
            "time": float(t), "available": True,
            "heading_deg": heading,
            "heading_unwrapped_deg": hu,
            "pitch_deg": p, "roll_deg": r,
            "fraction": frac, "run": [a, z],
            "support_samples": [lo, hi],
            "crossed_boundary": not (0.0 <= hu < 360.0),
        })
    return results
