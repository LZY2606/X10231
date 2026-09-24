"""姿态处理：heading 跨越 360° 的展开，以及不跨时间缺口的线性插值。

规则（案例硬约束）：
- heading 以 0..360° 记录，359 -> 1 这样的真实连续转动必须被正确识别，
  插值不能反向绕大圈。
- 相邻姿态样本时间差超过 ``max_gap_s`` 即产生“缺口”。缺口两侧属于不同分段，
  姿态插值绝不跨越缺口；落在缺口内（或分段外）的 ping 得不到姿态，状态为
  ``unresolved``，对应 earth 流速被阻断而不是静默丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AttitudeSeries:
    times: np.ndarray          # 秒，严格递增
    heading_deg: np.ndarray    # 0..360
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    max_gap_s: float

    def __post_init__(self) -> None:
        if not (np.all(np.diff(self.times) > 0)):
            raise ValueError("attitude times must be strictly increasing")


@dataclass(frozen=True)
class AttitudePoint:
    time_s: float
    heading_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    status: str                # "ok" | "unresolved"
    segment: int
    evidence: dict[str, Any]


def segments(times: np.ndarray, max_gap_s: float) -> list[tuple[int, int, list[int]]]:
    """按缺口切分连续段。返回 [(seg_index, [sample indices])]。"""
    out: list[tuple[int, list[int]]] = []
    seg = [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap_s:
            out.append((len(out), seg))
            seg = []
        seg.append(i)
    out.append((len(out), seg))
    return [(idx, idx_list) for idx, idx_list in out]  # type: ignore[misc]


def _unwrap_heading(h: np.ndarray) -> np.ndarray:
    """把 0..360 航向展开为连续度数（处理 359 -> 1 的跨圈）。"""
    rad = np.deg2rad(h)
    return np.rad2deg(np.unwrap(rad, period=2 * np.pi))


def interpolate(series: AttitudeSeries, ping_time_s: float) -> AttitudePoint:
    """在不跨缺口的前提下，对单个 ping 时刻做姿态线性插值。"""
    segs = segments(series.times, series.max_gap_s)
    t = float(ping_time_s)
    for seg_id, idxs in segs:
        t0, t1 = series.times[idxs[0]], series.times[idxs[-1]]
        # 允许与端点精确相等；不外推到分段范围之外。
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            continue
        seg_times = series.times[idxs]
        hu = _unwrap_heading(series.heading_deg[idxs])
        hv = float(np.interp(t, seg_times, hu))
        pv = float(np.interp(t, seg_times, series.pitch_deg[idxs]))
        rv = float(np.interp(t, seg_times, series.roll_deg[idxs]))
        hv = hv % 360.0
        # 记录用于证据：实际使用了哪两个样本
        j = int(np.searchsorted(seg_times, t))
        if j == 0:
            used = [int(idxs[0]), int(idxs[0])]
        elif j >= len(idxs):
            used = [int(idxs[-1]), int(idxs[-1])]
        else:
            used = [int(idxs[j - 1]), int(idxs[j])]
        return AttitudePoint(
            time_s=t, heading_deg=hv, pitch_deg=pv, roll_deg=rv,
            status="ok", segment=seg_id,
            evidence={"used_sample_indices": used, "segment_span_s": [float(t0), float(t1)]},
        )
    return AttitudePoint(
        time_s=t, heading_deg=None, pitch_deg=None, roll_deg=None,
        status="unresolved", segment=-1,
        evidence={"reason": "attitude-gap-or-outside-segment", "max_gap_s": series.max_gap_s},
    )
