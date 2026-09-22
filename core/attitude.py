"""姿态序列与插值。

规则
----
- heading/pitch/roll 分开成序列记录，可携带各自单位。
- heading 跨越 360° 时先 unwrap 再线性插值（359 -> 1 正确取 0）。
- 若相邻实测锚点间隔超过 ``max_gap_s``（时间缺口），绝不跨缺口插值，
  返回不可用状态与诊断，派生层不得用最近邻/外推偷偷补齐。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    heading: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]


@dataclass(frozen=True)
class AttitudeResult:
    heading: float
    pitch: float
    roll: float
    status: str               # "measured" | "interpolated" | "unavailable"
    anchor_before: Optional[float]
    anchor_after: Optional[float]
    span_s: Optional[float]
    diagnostic: Optional[str] = None


def _interp_channel(t0, t1, v0, v1, t, wrap: bool):
    if v0 is None or v1 is None:
        return None
    if wrap:
        rad = np.unwrap(np.deg2rad([float(v0), float(v1)]))
        d0, d1 = np.rad2deg(rad)
    else:
        d0, d1 = float(v0), float(v1)
    return d0 + (d1 - d0) * (t - t0) / (t1 - t0)


def attitude_at(samples: Sequence[AttitudeSample], t: float,
                max_gap_s: float = 5.0) -> AttitudeResult:
    meas = [s for s in samples if s.heading is not None
            and s.pitch is not None and s.roll is not None]
    ts = [s.t for s in meas]

    for s in meas:
        if s.t == t:
            return AttitudeResult(s.heading, s.pitch, s.roll, "measured",
                                  t, t, 0.0)

    before = [s for s in meas if s.t < t]
    after = [s for s in meas if s.t > t]
    if not before or not after:
        return AttitudeResult(
            float("nan"), float("nan"), float("nan"), "unavailable",
            None, None, None, "outside measured attitude range; no extrapolation")

    s0, s1 = before[-1], after[0]
    span = s1.t - s0.t
    if span > max_gap_s:
        return AttitudeResult(
            float("nan"), float("nan"), float("nan"), "unavailable",
            s0.t, s1.t, span,
            f"attitude gap {span:.1f}s exceeds max {max_gap_s:.1f}s; interpolation across gap refused")

    h = _interp_channel(s0.t, s1.t, s0.heading, s1.heading, t, wrap=True)
    p = _interp_channel(s0.t, s1.t, s0.pitch, s1.pitch, t, wrap=False)
    r = _interp_channel(s0.t, s1.t, s0.roll, s1.roll, t, wrap=False)
    return AttitudeResult(h, p, r, "interpolated", s0.t, s1.t, span)
