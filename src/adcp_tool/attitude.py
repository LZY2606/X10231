"""姿态（heading/pitch/roll）处理。

姿态是独立时间序列，不与 ping 时刻混为一条记录：

* heading 以度、磁北基准（MAGNETIC）记录，先做跨 360° 解绕再线性插值；
* pitch/roll 以度记录后插值；
* 当 ping 时刻两侧样本间隔超过 ``max_gap_s``（或任一侧缺样本），
  视为时间缺口 —— 姿态插值 **不跨缺口**，earth 坐标结论标记 ATTITUDE_GAP，
  但 ship/instrument 结论仍然保留有效。

插值方式（线性）与最大缺口进入派生指纹。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import to_quantity


@dataclass(frozen=True)
class AttitudeInterpConfig:
    method: str = "linear"
    max_gap_s: float = 2.0  # 超过该间隔不插值

    def fingerprint_input(self) -> dict:
        return {"method": self.method, "max_gap_s": self.max_gap_s}


def unwrap_headings(headings_deg: Sequence[float]) -> np.ndarray:
    """跨 360° 解绕（度），例如 359 -> 1 解绕为 359 -> 361。"""
    return np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, dtype=float))))


@dataclass
class AttitudeAtTime:
    t: float
    heading_deg: Optional[float]      # 已折回 [0,360) 的磁航向（成功时）
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    status: str                       # OK / ATTITUDE_GAP / NO_EVIDENCE
    reasons: List[str]

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "heading_deg": self.heading_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def _linear(t0, v0, t1, v1, t):
    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def interpolate_at(samples: List[dict], t: float, cfg: AttitudeInterpConfig
                   ) -> AttitudeAtTime:
    """samples: [{'t':秒,'heading_deg','pitch_deg','roll_deg'}]（原始、按 t 排序）。

    任何参与分量缺失则该时刻 NO_EVIDENCE；跨缺口则 ATTITUDE_GAP。
    """
    ts = [s["t"] for s in samples]
    if not ts or t < min(ts) or t > max(ts):
        return AttitudeAtTime(t, None, None, None, "NO_EVIDENCE",
                              ["OUT_OF_ATTITUDE_RANGE"])

    # 定位两侧
    left_i = max(i for i, st in enumerate(ts) if st <= t)
    right_i = min(i for i, st in enumerate(ts) if st >= t)
    l, r = samples[left_i], samples[right_i]
    gap = r["t"] - l["t"]
    if gap > cfg.max_gap_s + 1e-9:
        return AttitudeAtTime(t, None, None, None, "ATTITUDE_GAP",
                              [f"GAP_{gap:.3f}s>{cfg.max_gap_s}s"])

    for key in ("heading_deg", "pitch_deg", "roll_deg"):
        if l.get(key) is None or r.get(key) is None:
            return AttitudeAtTime(t, None, None, None, "NO_EVIDENCE",
                                  [f"{key}_EVIDENCE_MISSING"])

    # heading 解绕后插值；解绕必须在“连续的两侧样本”上做，缺口已在上面拦截
    h_l, h_r = _aligned_pair_headings(l["heading_deg"], r["heading_deg"])
    h = _linear(l["t"], h_l, r["t"], h_r, t) % 360.0
    p = _linear(l["t"], l["pitch_deg"], r["t"], r["pitch_deg"], t)
    ro = _linear(l["t"], l["roll_deg"], r["t"], r["roll_deg"], t)
    return AttitudeAtTime(t, h, p, ro, "OK", [])


def _aligned_pair_headings(a: float, b: float) -> Tuple[float, float]:
    u = unwrap_headings([float(a), float(b)])
    return float(u[0]), float(u[1])
