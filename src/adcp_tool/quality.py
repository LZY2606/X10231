"""波束质量层级。

证据（测量、correlation、amplitude、坏 beam 标志）分开判定，结论分三态：

* ``VALID`` —— 证据支持该波束可用；
* ``INVALID`` —— 存在确定性的否决证据（坏标志、相关度/幅度确定偏低、
  波束速度无法换算）；
* ``INDETERMINATE`` —— 关键证据缺失或无法换算，无法唯一决定。

INDETERMINATE 不会被静默当作好或坏：枚举三波束/四波束候选时，包含与不包含
该波束的合法解都会保留。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VALID = "VALID"
INVALID = "INVALID"
INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class QCThresholds:
    min_correlation_count: float = 64.0   # correlation 原始计数下限
    min_amplitude_db: float = 40.0        # amplitude(dB) 下限
    corr_unit: str = "count"
    amp_unit: str = "dB"

    def fingerprint_input(self) -> dict:
        return {
            "min_correlation_count": self.min_correlation_count,
            "min_amplitude_db": self.min_amplitude_db,
            "corr_unit": self.corr_unit,
            "amp_unit": self.amp_unit,
        }


@dataclass
class BeamVerdict:
    beam: int
    status: str
    reasons: List[str] = field(default_factory=list)
    diagnostics: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "beam": self.beam,
            "status": self.status,
            "reasons": list(self.reasons),
            "diagnostics": list(self.diagnostics),
        }


def _norm(value, raw_unit, unit):
    from .units import to_quantity
    if value is None and raw_unit is None:
        return None, None  # 该项证据完全不存在
    return to_quantity(value, raw_unit, unit)


def judge_beam(beam: int, *, velocity: Any, velocity_raw_unit: Optional[str],
               correlation: Any, correlation_raw_unit: Optional[str],
               amplitude: Any, amplitude_raw_unit: Optional[str],
               bad_flag: Optional[bool], thresholds: QCThresholds) -> BeamVerdict:
    """对一条波束做三态判定。velocity/correlation/amplitude 为原始层数值。"""
    reasons: List[str] = []
    diags: List[dict] = []
    status = VALID

    # 1) 坏 beam 标志（确定性否决）
    if bad_flag is True:
        reasons.append("BAD_BEAM_FLAG")
        status = INVALID

    # 2) 波束速度能否规范化（无法换算 -> 确定性否决；值缺失但单位在 -> 否决）
    from .units import to_quantity, UnitDiagnostic
    if velocity_raw_unit is not None or velocity is not None:
        vq, vdiag = to_quantity(velocity, velocity_raw_unit, "m/s")
        if vdiag is not None:
            reasons.append(f"VELOCITY_{vdiag.code}")
            diags.append(vdiag.as_dict())
            status = INVALID

    # 3) correlation
    cq, cdiag = _norm(correlation, correlation_raw_unit, thresholds.corr_unit)
    if correlation_raw_unit is None and correlation is None:
        reasons.append("CORRELATION_EVIDENCE_MISSING")
        if status == VALID:
            status = INDETERMINATE
    elif cdiag is not None:
        reasons.append(f"CORRELATION_{cdiag.code}")
        diags.append(cdiag.as_dict())
        if status == VALID:
            status = INDETERMINATE
    elif cq is not None and cq.value < thresholds.min_correlation_count:
        reasons.append("LOW_CORRELATION")
        status = INVALID

    # 4) amplitude
    aq, adiag = _norm(amplitude, amplitude_raw_unit, thresholds.amp_unit)
    if amplitude_raw_unit is None and amplitude is None:
        reasons.append("AMPLITUDE_EVIDENCE_MISSING")
        if status == VALID:
            status = INDETERMINATE
    elif adiag is not None:
        reasons.append(f"AMPLITUDE_{adiag.code}")
        diags.append(adiag.as_dict())
        if status == VALID:
            status = INDETERMINATE
    elif aq is not None and aq.value < thresholds.min_amplitude_db:
        reasons.append("LOW_AMPLITUDE")
        status = INVALID

    return BeamVerdict(beam=beam, status=status, reasons=reasons, diagnostics=diags)


def candidate_beam_sets(verdicts: List[BeamVerdict], n_beams: int
                        ) -> List[Dict[str, Any]]:
    """枚举合法求解波束集合。

    * 至少 3 条波束；
    * INVALID 永不入集合；
    * INDETERMINATE 波束使“包含它 / 排除它”的集合都被保留（多解）；
    * 优先四波束，其次三波束；同一枚举结果不重复。
    """
    invalid = {v.beam for v in verdicts if v.status == INVALID}
    indeterminate = {v.beam for v in verdicts if v.status == INDETERMINATE}
    all_beams = set(range(n_beams))
    usable = all_beams - invalid

    candidates: List[Dict[str, Any]] = []
    seen = set()
    # 对每条不确定波束枚举 in/out
    ind_list = sorted(indeterminate)
    for mask in range(1 << len(ind_list)):
        chosen = set(usable)
        for i, b in enumerate(ind_list):
            if not ((mask >> i) & 1):
                chosen.discard(b)
        chosen_s = frozenset(chosen)
        if len(chosen_s) < 3 or chosen_s in seen:
            continue
        seen.add(chosen_s)
        beams = tuple(sorted(chosen_s))
        candidates.append({
            "beams": beams,
            "rank": 4 if len(beams) == 4 else 3,
            "indeterminate_included": tuple(sorted(set(beams) & indeterminate)),
        })
    # 四波束优先，其次三波束；同 rank 按波束元组
    candidates.sort(key=lambda c: (-c["rank"], c["beams"]))
    return candidates
