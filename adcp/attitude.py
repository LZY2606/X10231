"""姿态序列处理：360 度绕卷与缺口感知插值。

铁律：姿态插值不跨时间缺口。时间戳是独立直接来源。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def unwrap_deg(values: Sequence[float]) -> List[float]:
    """把跨越 360 度的序列展开到连续表示（例如 358, 359, 361, 363）。"""
    out: List[float] = []
    prev: Optional[float] = None
    for v in values:
        if v is None:
            out.append(None)  # type: ignore[arg-type]
            continue
        if prev is None:
            out.append(float(v))
        else:
            d = (v - prev + 180.0) % 360.0 - 180.0
            out.append(prev + d)
        prev = out[-1]
    return out


def interp_attitude(times: Sequence[float], observed: Sequence[Optional[float]],
                    record_gaps: Sequence[Tuple[float, float]],
                    max_gap_s: float) -> Tuple[List[Optional[float]], List[Optional[Dict]]]:
    """对每个采样时刻返回插值姿态与诊断。

    observed: 与 times 等长，None 表示该 ping 缺失姿态观测。
    record_gaps: 观测记录层面的缺口 (t_prev, t_next)；插值绝不跨越它们。
    只有两侧观测均存在、该区间没有 record gap、且时间间隔 <= max_gap_s 时才插值。
    """
    values: List[Optional[float]] = [None] * len(times)
    diag: List[Optional[Dict]] = [None] * len(times)
    obs_idx = [i for i, v in enumerate(observed) if v is not None]
    unwrapped = unwrap_deg([observed[i] for i in obs_idx])

    def blocked(i: int, code: str, gap: Optional[float] = None, msg: str = ""):
        values[i] = None
        d = {"severity": "error", "code": code, "message": msg}
        if gap is not None:
            d["gap_s"] = gap
        diag[i] = d

    for k, i in enumerate(obs_idx):
        values[i] = observed[i]

    for i, v in enumerate(observed):
        if v is not None:
            continue
        left = [j for j in obs_idx if j < i]
        right = [j for j in obs_idx if j > i]
        if not left or not right:
            blocked(i, "attitude_unbounded", msg="no observation on both sides; not interpolated")
            continue
        li, ri = left[-1], right[0]
        t0, t1 = times[li], times[ri]
        gap = t1 - t0
        crossed = any(t0 < a and b < t1 for a, b in record_gaps)
        if crossed:
            blocked(i, "attitude_gap_cross", gap=gap,
                    msg="interpolation refused: source record gap present")
        elif gap > max_gap_s:
            blocked(i, "attitude_gap_too_large", gap=gap,
                    msg=f"gap {gap:g}s exceeds limit {max_gap_s:g}s; not interpolated")
        else:
            u0 = unwrapped[obs_idx.index(li)]
            u1 = unwrapped[obs_idx.index(ri)]
            ui = u0 + (u1 - u0) * (times[i] - t0) / (t1 - t0)
            values[i] = ui % 360.0
            diag[i] = {"severity": "info", "code": "attitude_interpolated",
                       "message": "linear interpolation on unwrapped series",
                       "brackets": [li, ri], "gap_s": gap}
    return values, diag
