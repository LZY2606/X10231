"""Attitude segmentation, heading unwrap and bounded interpolation.

Rules (``attitude-segment-v1``):

* Pings split into segments at explicit gap flags or where
  ``dt > gap_factor * median(dt)``.
* Attitude is only interpolated between two samples inside the SAME segment.
  An interpolation whose bracket straddles a gap is blocked and a diagnostic
  is attached; nothing is invented across the hole.
* Headings wrap at 360 deg: samples are unwrapped linearly within a segment
  before interpolation, then wrapped back.  Evidence keeps the raw endpoint
  headings so the choice is auditable.
"""
from __future__ import annotations

import math

ATTITUDE_RULE = "attitude-segment-v1"


def build_segments(ping_times, gap_flags=None, gap_factor=1.5):
    """Return ``(segments, gaps)``.

    ``segments`` is a list of lists of ping indexes; ``gaps`` lists each break
    as ``(after_ping_index, dt, median_dt, reason)``.
    """
    times = list(ping_times)
    n = len(times)
    if n == 0:
        return [], []
    dts = [times[i + 1] - times[i] for i in range(n - 1)]
    if dts:
        s = sorted(dts)
        mid = len(s) // 2
        median_dt = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    else:
        median_dt = None

    gaps = []
    breaks = set()
    for i, dt in enumerate(dts):
        flagged = gap_flags is not None and i < len(gap_flags) and bool(gap_flags[i])
        oversized = median_dt is not None and dt > gap_factor * median_dt
        if flagged or oversized:
            breaks.add(i)
            gaps.append(
                {
                    "after_ping_index": i,
                    "dt": dt,
                    "median_dt": median_dt,
                    "reason": "explicit-gap-flag" if flagged else "dt>%.2f*median" % gap_factor,
                }
            )

    segments, cur = [], [0]
    for i in range(n - 1):
        if i in breaks:
            segments.append(cur)
            cur = [i + 1]
        else:
            cur.append(i + 1)
    segments.append(cur)
    return segments, gaps


def segment_of_index(index, segments):
    for k, seg in enumerate(segments):
        if index in seg:
            return k
    return None


def _same_segment(i0, i1, segments):
    return segment_of_index(i0, segments) is not None and (
        segment_of_index(i0, segments) == segment_of_index(i1, segments)
    )


def unwrap_deg(values_deg):
    """Unwrap a within-segment angle series in degrees."""
    out = list(values_deg)
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        out[i] = out[i] - 360.0 * round(d / 360.0)
    return out


def interpolate(samples, query_time, gap_flags=None, gap_factor=1.5):
    """Interpolate heading/pitch/roll at ``query_time`` (seconds).

    ``samples`` is a time-ordered list of dicts with ``time`` (s),
    ``heading_deg``, ``pitch_deg``, ``roll_deg`` and optional
    ``usable``/``diagnostic``.

    Returns an evidence dict whose ``status`` is one of ``exact``,
    ``interpolated``, ``blocked-gap``, ``blocked-range`` or ``blocked-sample``.
    Interpolation never crosses a segment gap.
    """
    times = [s["time"] for s in samples]
    segments, gaps = build_segments(times, gap_flags, gap_factor)
    base = {
        "rule": ATTITUDE_RULE,
        "query_time": query_time,
        "gaps": gaps,
        "segments": segments,
    }

    if not times:
        base.update(status="blocked-range", diagnostic="no attitude samples")
        return base
    if query_time < times[0] or query_time > times[-1]:
        base.update(status="blocked-range", diagnostic="query time outside attitude sample range")
        return base

    for i, s in enumerate(samples):
        if s["time"] == query_time:
            if not s.get("usable", True):
                base.update(
                    status="blocked-sample",
                    ping_index=i,
                    diagnostic="attitude sample unusable: " + str(s.get("diagnostic")),
                )
                return base
            base.update(
                status="exact",
                ping_index=i,
                heading_deg=s["heading_deg"],
                pitch_deg=s["pitch_deg"],
                roll_deg=s["roll_deg"],
            )
            return _radians(base)

    i1 = next(k for k, t in enumerate(times) if t > query_time)
    i0 = i1 - 1
    s0, s1 = samples[i0], samples[i1]
    base["bracket_ping_indexes"] = [i0, i1]
    base["bracket_times"] = [s0["time"], s1["time"]]

    if not _same_segment(i0, i1, segments):
        base.update(
            status="blocked-gap",
            diagnostic="attitude interpolation would cross a time gap; refused",
        )
        return base
    if not s0.get("usable", True) or not s1.get("usable", True):
        base.update(
            status="blocked-sample",
            diagnostic="bracketing attitude sample carries a conversion diagnostic",
        )
        return base

    f = (query_time - s0["time"]) / (s1["time"] - s0["time"])
    h0, h1 = s0["heading_deg"], s1["heading_deg"]
    u0, u1 = unwrap_deg([h0, h1])
    head_deg = (u0 + f * (u1 - u0)) % 360.0
    pitch_deg = s0["pitch_deg"] + f * (s1["pitch_deg"] - s0["pitch_deg"])
    roll_deg = s0["roll_deg"] + f * (s1["roll_deg"] - s0["roll_deg"])

    base.update(
        status="interpolated",
        fraction=f,
        evidence={
            "endpoint_heading_deg": [h0, h1],
            "unwrapped_heading_deg": [u0, u1],
            "naive_linear_heading_deg": h0 + f * (h1 - h0),
            "unwrap_applied": abs(u1 - u0) != abs(h1 - h0),
            "endpoint_pitch_deg": [s0["pitch_deg"], s1["pitch_deg"]],
            "endpoint_roll_deg": [s0["roll_deg"], s1["roll_deg"]],
        },
        heading_deg=head_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    return _radians(base)


def _radians(base):
    base.update(
        heading_rad=math.radians(base["heading_deg"]),
        pitch_rad=math.radians(base["pitch_deg"]),
        roll_rad=math.radians(base["roll_deg"]),
    )
    return base
