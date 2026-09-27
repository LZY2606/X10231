"""Heading/pitch/roll resolution.

Rules:
- Samples are sorted by time and split into segments when a gap exceeds
  gap_seconds. Interpolation NEVER crosses a gap and NEVER extrapolates.
- Heading interpolation is done after unwrapping 360<->0 crossings.
- True heading = (magnetic heading + declination) wrapped to [0,360).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AttitudeSample:
    ping_id: str
    t: float
    heading: float | None      # deg, magnetic, [0,360)
    pitch: float | None
    roll: float | None


@dataclass
class AttitudeResult:
    ping_id: str
    t: float
    heading: float | None
    pitch: float | None
    roll: float | None
    true_heading: float | None
    status: str                 # measured | interpolated | missing
    segment_id: str | None
    wrap_crossed: bool
    diagnostics: tuple[str, ...]


def _interp(t, t0, v0, t1, v1):
    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def resolve_attitude(samples: list[AttitudeSample], query_times: dict[str, float],
                     declination: float, gap_seconds: float
                     ) -> dict[str, AttitudeResult]:
    samples = sorted(samples, key=lambda s: s.t)
    # split into segments by gap
    segments: list[list[AttitudeSample]] = []
    cur: list[AttitudeSample] = []
    for s in samples:
        if cur and (s.t - cur[-1].t) > gap_seconds:
            segments.append(cur)
            cur = []
        cur.append(s)
    if cur:
        segments.append(cur)

    # unwrap heading within each segment, recording a wrap crossing
    seg_info = []
    for si, seg in enumerate(segments):
        hs = [s.heading if s.heading is not None else math.nan for s in seg]
        arr = hs[:]
        wrap = False
        prev = None
        for i in range(1, len(arr)):
            if arr[i] is None or math.isnan(arr[i]) or arr[i - 1] is None:
                continue
            # cumulative unwrap relative to first valid
        # numpy-free unwrap:
        raw = [s.heading for s in seg]
        unwrapped: list[float | None] = []
        last: float | None = None
        for h in raw:
            if h is None:
                unwrapped.append(None)
                continue
            if last is None:
                unwrapped.append(float(h))
            else:
                u = float(h)
                while u - last > 180.0:
                    u -= 360.0
                while u - last < -180.0:
                    u += 360.0
                if abs(u - last) > 180.0 - 1e-9 and abs((u - last) - 0) > 1e-9:
                    wrap = True
                unwrapped.append(u)
            last = unwrapped[-1]
        seg_info.append((seg, unwrapped, wrap))

    out: dict[str, AttitudeResult] = {}
    for ping_id, t in query_times.items():
        placed = None
        for si, (seg, unwrapped, wrap) in enumerate(seg_info):
            if seg[0].t <= t <= seg[-1].t:
                placed = (si, seg, unwrapped, wrap)
                break
        if placed is None:
            out[ping_id] = AttitudeResult(ping_id, t, None, None, None, None,
                                          "missing", None, False,
                                          ("attitude_outside_any_segment",))
            continue
        si, seg, unwrapped, wrap = placed
        # find exact measured sample
        exact = next((k for k, s in enumerate(seg) if s.t == t
                      and s.heading is not None and s.pitch is not None
                      and s.roll is not None), None)
        if exact is not None:
            s = seg[exact]
            out[ping_id] = AttitudeResult(
                ping_id, t, s.heading, s.pitch, s.roll,
                _true(s.heading, declination), "measured", f"seg{si}", False, ())
            continue
        # need bracketing valid samples inside this segment for interpolation
        h = _bracket_interp(t, [s.t for s in seg], unwrapped)
        p = _bracket_interp(t, [s.t for s in seg],
                            [s.pitch if s.pitch is not None else None for s in seg])
        r = _bracket_interp(t, [s.t for s in seg],
                            [s.roll if s.roll is not None else None for s in seg])
        if h is None or p is None or r is None:
            out[ping_id] = AttitudeResult(ping_id, t, None, None, None, None,
                                          "missing", f"seg{si}", False,
                                          ("no_bracketing_attitude_sample",))
            continue
        h_mag = h % 360.0
        out[ping_id] = AttitudeResult(
            ping_id, t, h_mag, p, r, _true(h_mag, declination),
            "interpolated", f"seg{si}", wrap,
            ("attitude_interpolated",))
    return out


def _true(heading_mag: float | None, declination: float) -> float | None:
    if heading_mag is None:
        return None
    return (heading_mag + declination) % 360.0


def _bracket_interp(t, ts, vs):
    left = None
    for i, (tt, vv) in enumerate(zip(ts, vs)):
        if vv is None:
            continue
        if abs(tt - t) < 1e-9:
            return vv
        if tt < t:
            left = (tt, vv)
        elif tt > t and left is not None:
            return _interp(t, left[0], left[1], tt, vv)
    return None
