"""姿态插值：pitch/roll 线性，heading 在单位圆上插值（正确处理跨越 360）。

硬性语义：姿态插值不跨缺口。相邻姿态样本时间差超过 gap_s 即视为缺口，
缺口内（含边界判断）任何 ping 都不得插值，而是返回 None + 诊断事件。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .transforms import anglediff, wrap_360


@dataclass
class AttitudeSample:
    t: float
    heading_deg: float   # 磁艏向，可在任意范围（含 359 -> 1 这种跨 360）
    pitch_deg: float
    roll_deg: float


@dataclass
class AttitudeResult:
    t: float
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    interpolated: bool = False
    wrapped: bool = False
    diagnostics: List[str] = field(default_factory=list)


class AttitudeTable:
    def __init__(self, samples: Sequence[AttitudeSample], gap_s: float):
        self.samples = sorted(samples, key=lambda s: s.t)
        self.gap_s = gap_s
        self.gaps = self._find_gaps()

    def _find_gaps(self) -> List[dict]:
        gaps = []
        for a, b in zip(self.samples, self.samples[1:]):
            if b.t - a.t > self.gap_s:
                gaps.append({"t_start": a.t, "t_end": b.t,
                             "duration_s": b.t - a.t})
        return gaps

    def in_gap(self, t: float) -> Optional[dict]:
        if t < self.samples[0].t or t > self.samples[-1].t:
            return {"t_start": None, "t_end": None, "duration_s": None,
                    "reason": "E_ATTITUDE_OUT_OF_RANGE"}
        for g in self.gaps:
            if g["t_start"] < t < g["t_end"]:
                return g
        return None

    def lookup(self, t: float) -> AttitudeResult:
        gap = self.in_gap(t)
        if gap is not None:
            why = gap.get("reason", "E_ATTITUDE_GAP")
            return AttitudeResult(t=t, diagnostics=[
                "%s: t=%s 落在姿态缺口 %s..%s，禁止跨缺口插值"
                % (why, t, gap.get("t_start"), gap.get("t_end"))])
        for s in self.samples:
            if s.t == t:
                return AttitudeResult(t, s.heading_deg, s.pitch_deg, s.roll_deg,
                                      interpolated=False)
        a, b = self._bracket(t)
        frac = (t - a.t) / (b.t - a.t)
        dh = anglediff(b.heading_deg, a.heading_deg)
        wrapped = abs(dh) > 1e-9 and abs(b.heading_deg - a.heading_deg) > 180.0
        heading = wrap_360(a.heading_deg + frac * dh)
        return AttitudeResult(
            t=t,
            heading_deg=heading,
            pitch_deg=a.pitch_deg + frac * (b.pitch_deg - a.pitch_deg),
            roll_deg=a.roll_deg + frac * (b.roll_deg - a.roll_deg),
            interpolated=True,
            wrapped=wrapped,
            diagnostics=(["I_HEADING_WRAP: 艏向跨越 0/360，按单位圆插值"] if wrapped else []),
        )

    def _bracket(self, t: float) -> tuple:
        prev = self.samples[0]
        for s in self.samples[1:]:
            if prev.t <= t <= s.t:
                return prev, s
            prev = s
        raise LookupError(t)
