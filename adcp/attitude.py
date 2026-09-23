"""姿态采样与插值。

红线语义:
1. heading 跨越 360 度必须先 unwrap 再插值, 359deg 与 1deg 之间的
   正确结果是 0deg, 而不是 180deg。
2. **姿态插值不跨时间缺口**: 只有当 ping 时刻两侧姿态样本都存在,
   且相邻样本间隔不超过 gap_tolerance 时才插值; 否则标记 GAP,
   ship/earth 层对该 ping 保持不可用 (instrument 层不受影响)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

ATTITUDE_RULE = "linear-unwrap-nogap/v1"


@dataclass(frozen=True)
class AttitudeSample:
    """原始姿态记录 (deg); 磁/真基准在磁偏角修订层处理, 这里只存原始表示。"""

    t: float
    heading: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]


@dataclass(frozen=True)
class InterpolatedAttitude:
    t: float
    heading: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    status: str          # "ok" | "gap" | "missing"
    source_times: Optional[Tuple[float, float]] = None
    reason: Optional[str] = None


def _wrap_0_360(x: float) -> float:
    return x % 360.0


def unwrap_headings(headings: Sequence[Optional[float]]) -> List[Optional[float]]:
    """把跨越 360 的航向展开成连续序列; None 保持为缺口。"""
    out: List[Optional[float]] = []
    last: Optional[float] = None
    for h in headings:
        if h is None:
            out.append(None)
            continue
        if last is None:
            out.append(float(h))
        else:
            step = (h - last + 180.0) % 360.0 - 180.0
            out.append(last + step)
        last = out[-1]
    return out


def _linear(samples: Sequence[AttitudeSample], times: Sequence[float],
            values: Sequence[Optional[float]], t: float,
            gap_tolerance: float) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str]]:
    """在 (times, values) 上线性插值; 不跨缺口。

    - t 早于第一个/晚于最后一个有效样本 -> 不外推;
    - 两侧样本间隔 > gap_tolerance -> 判为时间缺口, 不插值;
    - 任一侧为 None -> 不跨 None 插值。
    """
    idx = 0
    while idx < len(times) and times[idx] < t:
        idx += 1
    if idx == 0 or idx == len(times):
        return None, None, "extrapolation-forbidden"
    t0, t1 = times[idx - 1], times[idx]
    v0, v1 = values[idx - 1], values[idx]
    if v0 is None or v1 is None:
        return None, (t0, t1), "missing-sample-adjacent"
    if (t1 - t0) > gap_tolerance + 1e-12:
        return None, (t0, t1), "time-gap-exceeds-tolerance"
    frac = (t - t0) / (t1 - t0)
    return v0 + frac * (v1 - v0), (t0, t1), None


def interpolate_attitude(ping_times: Sequence[float],
                         samples: Sequence[AttitudeSample],
                         gap_tolerance: float = 2.0
                         ) -> List[InterpolatedAttitude]:
    """逐 ping 给出姿态。heading 走 unwrap 通道, pitch/roll 直接线性。

    三个分量共用同一条"不跨缺口"判定: 只要该 ping 的包围样本跨越缺口,
    heading/pitch/roll 一起标 gap, 避免出现半套姿态被下游误用。
    """
    ordered = sorted(samples, key=lambda s: s.t)
    times = [s.t for s in ordered]
    raw_headings = [s.heading for s in ordered]
    headings = unwrap_headings(raw_headings)
    pitches = [s.pitch for s in ordered]
    rolls = [s.roll for s in ordered]

    result: List[InterpolatedAttitude] = []
    for t in ping_times:
        h, span, reason = _linear(ordered, times, headings, t, gap_tolerance)
        p, span2, _ = _linear(ordered, times, pitches, t, gap_tolerance)
        r, span3, _ = _linear(ordered, times, rolls, t, gap_tolerance)
        span = span or span2 or span3
        if reason == "extrapolation-forbidden":
            result.append(InterpolatedAttitude(t, None, None, None,
                                               "missing", None, reason))
            continue
        if h is None or p is None or r is None:
            result.append(InterpolatedAttitude(t, None, None, None,
                                               "gap", span, reason or "gap"))
            continue
        result.append(InterpolatedAttitude(
            t=t,
            heading=_wrap_0_360(h),
            pitch=p,
            roll=r,
            status="ok",
            source_times=span,
        ))
    return result


def apply_mount_angles(heading_deg: float, pitch_deg: float, roll_deg: float,
                       mount_heading_deg: float = 0.0,
                       mount_pitch_deg: float = 0.0,
                       mount_roll_deg: float = 0.0
                       ) -> Tuple[float, float, float]:
    """安装角修订: 安装角是独立来源, 修订产生新版本并进入指纹。"""
    return (
        _wrap_0_360(heading_deg + mount_heading_deg),
        pitch_deg + mount_pitch_deg,
        roll_deg + mount_roll_deg,
    )


def magnetic_to_true(heading_mag_deg: float, declination_deg: float) -> float:
    """磁偏角版本独立记录; 真航向 = 磁航向 + 磁偏角(东正)。"""
    return _wrap_0_360(heading_mag_deg + declination_deg)
