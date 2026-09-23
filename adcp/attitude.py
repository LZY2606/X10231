"""Attitude handling: heading unwrap and interpolation that never crosses gaps."""

from __future__ import annotations

import numpy as np


def unwrap_headings(headings_deg):
    """Unwrap a heading sequence so 359.5 -> 0.5 interpolates through 360."""
    return np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, dtype=float))))


def interpolate_attitude(samples, query_time: float, max_gap_s: float):
    """Interpolate heading/pitch/roll at query_time.

    samples: list of dicts with keys t, heading, pitch, roll (deg), sorted by t.
    Returns a dict with status:
      "ok"           - interpolated (heading unwrapped first, result in [0,360))
      "gap"          - bracketing samples are farther apart than max_gap_s;
                       no interpolation is performed across the gap
      "out_of_range" - query time outside the sample range
    """
    if not samples:
        return {"status": "out_of_range", "diagnostic": "no attitude samples"}
    times = [s["t"] for s in samples]
    if query_time < times[0] or query_time > times[-1]:
        return {"status": "out_of_range",
                "diagnostic": f"t={query_time} outside [{times[0]}, {times[-1]}]"}
    idx = int(np.searchsorted(times, query_time, side="right")) - 1
    if times[idx] == query_time or idx == len(times) - 1:
        s = samples[idx]
        return {"status": "ok", "heading": s["heading"] % 360.0,
                "pitch": s["pitch"], "roll": s["roll"],
                "bracket": [s["t"], s["t"]]}
    lo, hi = samples[idx], samples[idx + 1]
    gap = hi["t"] - lo["t"]
    if gap > max_gap_s:
        return {"status": "gap",
                "diagnostic": (f"attitude gap {gap:.3f}s > {max_gap_s:.3f}s "
                               f"between t={lo['t']} and t={hi['t']}; "
                               "interpolation refused"),
                "bracket": [lo["t"], hi["t"]]}
    frac = (query_time - lo["t"]) / gap
    headings = unwrap_headings([lo["heading"], hi["heading"]])
    heading = (headings[0] + frac * (headings[1] - headings[0])) % 360.0
    pitch = lo["pitch"] + frac * (hi["pitch"] - lo["pitch"])
    roll = lo["roll"] + frac * (hi["roll"] - lo["roll"])
    return {"status": "ok", "heading": heading, "pitch": pitch, "roll": roll,
            "bracket": [lo["t"], hi["t"]]}
