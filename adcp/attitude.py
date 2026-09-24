"""Attitude (heading/pitch/roll) handling.

Rules enforced here (each becomes a machine-readable event upstream):
- Headings are angles on a circle: interpolation unwraps across the 360/0
  seam, so 359.5 -> 0.5 interpolates through 0, not through 180.
- Interpolation NEVER crosses a time gap: if two consecutive attitude samples
  are separated by more than `max_gap_s`, pings inside the gap get status
  "attitude_gap" instead of interpolated values.
"""
from __future__ import annotations


def unwrap_headings(headings_deg):
    """Unwrap a heading sequence so consecutive deltas are within +/-180."""
    out = []
    prev = None
    for h in headings_deg:
        h = float(h) % 360.0
        if prev is not None:
            while h - prev > 180.0:
                h -= 360.0
            while h - prev < -180.0:
                h += 360.0
        out.append(h)
        prev = h
    return out


def interpolate_attitude(query_times, sample_times, samples, max_gap_s):
    """Interpolate heading/pitch/roll onto query_times.

    sample_times: sorted epoch seconds. samples: list of dicts with keys
    heading_deg, pitch_deg, roll_deg. Returns a list per query time:
      {"status": "ok", "heading_deg":..., "pitch_deg":..., "roll_deg":...,
       "interpolated": bool}
    or {"status": "attitude_gap", ...} / {"status": "out_of_range"}.
    Headings are unwrapped before interpolation and re-wrapped after.
    """
    results = []
    heads = unwrap_headings([s["heading_deg"] for s in samples])
    pitches = [float(s["pitch_deg"]) for s in samples]
    rolls = [float(s["roll_deg"]) for s in samples]
    n = len(sample_times)

    for t in query_times:
        t = float(t)
        if t < sample_times[0] or t > sample_times[-1]:
            results.append({"status": "out_of_range", "time": t})
            continue
        # find bracketing samples
        j = 0
        while j < n - 1 and sample_times[j + 1] < t:
            j += 1
        if sample_times[j] == t:
            results.append({
                "status": "ok", "time": t, "interpolated": False,
                "heading_deg": heads[j] % 360.0,
                "pitch_deg": pitches[j], "roll_deg": rolls[j],
            })
            continue
        t0, t1 = sample_times[j], sample_times[j + 1]
        if t1 - t0 > max_gap_s:
            results.append({
                "status": "attitude_gap", "time": t,
                "gap_before": t0, "gap_after": t1, "gap_s": t1 - t0,
                "reason": "attitude interpolation must not cross a time gap",
            })
            continue
        f = (t - t0) / (t1 - t0)
        results.append({
            "status": "ok", "time": t, "interpolated": True,
            "heading_deg": (heads[j] + f * (heads[j + 1] - heads[j])) % 360.0,
            "pitch_deg": pitches[j] + f * (pitches[j + 1] - pitches[j]),
            "roll_deg": rolls[j] + f * (rolls[j + 1] - rolls[j]),
        })
    return results
