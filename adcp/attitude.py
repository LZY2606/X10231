"""Heading/pitch/roll handling: wrap crossing, gaps, no interpolation over gaps.

Attitude is its own time series, separate from pings. Values are kept in their
original representation; derived alignment normalises to degrees/radians only
when building rotations.
"""
from __future__ import annotations

from typing import Dict, List, Optional

MAX_GAP = 0.75  # seconds of attitude sample spacing before gap is declared


def wrap_360(h: float) -> float:
    return h % 360.0


def unwrap_headings(headings_deg: List[float]) -> List[float]:
    """Continuous heading series, unwrapping jumps at the 359.999/0 seam."""
    out: List[float] = []
    prev: Optional[float] = None
    for h in headings_deg:
        if prev is None:
            out.append(float(h))
        else:
            d = (h - prev + 180.0) % 360.0 - 180.0
            out.append(prev + d)
        prev = out[-1]
    return out


def detect_wrap_crossings(headings_deg: List[float]) -> List[int]:
    """Indices i where the seam between i-1 and i crosses the 360/0 boundary."""
    return [i for i in range(1, len(headings_deg))
            if (headings_deg[i - 1] > 270.0 and headings_deg[i] < 90.0)
            or (headings_deg[i - 1] < 90.0 and headings_deg[i] > 270.0)]


def find_gaps(times: List[float], max_gap: float = MAX_GAP) -> List[dict]:
    return [{"after_index": i, "after_time": times[i],
             "before_time": times[i + 1], "gap": times[i + 1] - times[i]}
            for i in range(len(times) - 1)
            if times[i + 1] - times[i] > max_gap]


def align(times: List[float], raw_headings: List[float],
          pitchs: List[float], rolls: List[float], ping_time: float,
          max_gap: float = MAX_GAP) -> Dict[str, object]:
    """Linearly interpolate attitude at ping_time; refuse to cross a gap."""
    if not times or ping_time < times[0] or ping_time > times[-1]:
        return {"aligned": False, "reason": "ping_outside_attitude_span"}
    after = next((i for i in range(len(times))
                  if times[i] >= ping_time), len(times) - 1)
    if times[after] == ping_time:
        i0 = i1 = after
    else:
        i0, i1 = after - 1, after
    if i0 == i1:
        return {"aligned": True, "interpolated": False, "bracket": [i0, i1],
                "heading_deg": wrap_360(raw_headings[i0]),
                "pitch_deg": float(pitchs[i0]), "roll_deg": float(rolls[i0])}
    gap = times[i1] - times[i0]
    if gap > max_gap:
        return {"aligned": False, "reason": "attitude_gap",
                "gap": gap, "bracket": [i0, i1]}
    cont = unwrap_headings(raw_headings)  # seam-safe interpolation
    f = (ping_time - times[i0]) / gap
    h = cont[i0] + f * (cont[i1] - cont[i0])
    return {"aligned": True, "interpolated": True, "bracket": [i0, i1],
            "heading_deg": wrap_360(h),
            "pitch_deg": pitchs[i0] + f * (pitchs[i1] - pitchs[i0]),
            "roll_deg": rolls[i0] + f * (rolls[i1] - rolls[i0])}
