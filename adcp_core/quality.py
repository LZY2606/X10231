"""波束质量层级与深度 cell 校验。

坏 beam 标志、correlation、amplitude 是三条独立来源，判定时分别留痕：
  explicit 坏标志 -> bad(HARD_FLAG)
  correlation 低于下限 -> bad(LOW_CORR)；低于疑似带 -> suspect
  amplitude   低于下限 -> bad(LOW_AMP) ；低于疑似带 -> suspect
  波束速度本身无法单位规范化 -> bad(UNCONVERTIBLE)，原值原位保留
多解来自 suspect：保留全部合法候选（4 波束 + 每个 suspect 单波束剔除的 3 波束解）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .units import Q

GOOD = "GOOD"
SUSPECT = "SUSPECT"
BAD = "BAD"

RULE_VERSION = "adcp-rules-v1"


@dataclass
class Thresholds:
    corr_min: float = 60.0
    corr_suspect: float = 90.0
    amp_min_db: float = 20.0
    amp_suspect_db: float = 40.0
    version: str = "thr-default-v1"

    def fingerprint_basis(self) -> Dict[str, Any]:
        return {
            "version": self.version, "corr_min": self.corr_min,
            "corr_suspect": self.corr_suspect, "amp_min_db": self.amp_min_db,
            "amp_suspect_db": self.amp_suspect_db,
        }


@dataclass
class BeamState:
    beam: int
    state: str
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.state != BAD

    @property
    def firm(self) -> bool:
        return self.state == GOOD


def classify_beam(beam: int, velocity: Q, corr: Optional[Q], amp: Optional[Q],
                  flagged_bad: Optional[bool], flag_raw: Any,
                  thr: Thresholds) -> BeamState:
    reasons: List[str] = []
    evidence: Dict[str, Any] = {
        "velocity": velocity.to_dict(),
        "correlation": corr.to_dict() if corr is not None else None,
        "amplitude": amp.to_dict() if amp is not None else None,
        "bad_flag_raw": flag_raw,
    }
    state = GOOD
    if flagged_bad is True:
        state = BAD
        reasons.append("HARD_FLAG: 坏 beam 标志在来源中置位")
    if not velocity.ok():
        state = BAD
        reasons.append("UNCONVERTIBLE: 波束速度未规范化: %s"
                       % ";".join(velocity.diagnostics))
    if corr is not None:
        if corr.ok() and corr.value < thr.corr_min:
            state = BAD
            reasons.append("LOW_CORR: correlation %.1f < %.1f"
                           % (corr.value, thr.corr_min))
        elif corr.ok() and corr.value < thr.corr_suspect and state != BAD:
            state = SUSPECT
            reasons.append("SUSPECT_CORR: correlation %.1f 位于疑似带 [%.1f, %.1f)"
                           % (corr.value, thr.corr_min, thr.corr_suspect))
        elif not corr.ok():
            reasons.append("CORR_UNCONVERTIBLE: %s" % ";".join(corr.diagnostics))
    if amp is not None:
        if amp.ok() and amp.value < thr.amp_min_db:
            state = BAD
            reasons.append("LOW_AMP: amplitude %.1f < %.1f"
                           % (amp.value, thr.amp_min_db))
        elif amp.ok() and amp.value < thr.amp_suspect_db and state != BAD:
            state = SUSPECT
            reasons.append("SUSPECT_AMP: amplitude %.1f 位于疑似带 [%.1f, %.1f)"
                           % (amp.value, thr.amp_min_db, thr.amp_suspect_db))
        elif not amp.ok():
            reasons.append("AMP_UNCONVERTIBLE: %s" % ";".join(amp.diagnostics))
    if not reasons:
        reasons.append("OK: 通过全部质量门限")
    return BeamState(beam, state, reasons, evidence)


def cell_depth_m(blanking_q: Q, cell_size_q: Q, cell_index: int) -> Q:
    """cell 中心深度 = blanking + (index+0.5)*cell_size，单位不合法则保留诊断。"""
    if blanking_q.ok() and cell_size_q.ok():
        d = blanking_q.value + (cell_index + 0.5) * cell_size_q.value
        return Q(d, "m", d, "m", "")
    diag = "E_INVALID_DEPTH: blanking/cell_size 不可转换 (%r %r, %r %r)" % (
        blanking_q.raw_value, blanking_q.raw_unit,
        cell_size_q.raw_value, cell_size_q.raw_unit)
    return Q.raw(None, "m", diag)


def depth_cell_valid(depth_q: Q, bottom_range_q: Optional[Q],
                     beam_angle_deg: float) -> dict:
    """返回 {valid, reason, evidence}。超越底跟踪声束覆盖（旁瓣）即判无效。"""
    if not depth_q.ok():
        return {"valid": False, "reason": "INVALID_DEPTH_CELL",
                "evidence": {"depth": depth_q.to_dict()}}
    if bottom_range_q is not None and bottom_range_q.ok():
        limit = bottom_range_q.value * math.cos(math.radians(beam_angle_deg))
        if depth_q.value > limit:
            return {"valid": False, "reason": "CELL_BEYOND_BOTTOM",
                    "evidence": {"depth_m": depth_q.value,
                                 "sidelobe_limit_m": limit,
                                 "bottom_range_m": bottom_range_q.value}}
    return {"valid": True, "reason": "",
            "evidence": {"depth_m": depth_q.value}}
