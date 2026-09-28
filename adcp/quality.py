"""质量层级：correlation / amplitude / 坏 beam 标志决定每条波束可用性。

坏 beam 标志是独立直接来源，不与 correlation/amplitude 混存。
status: good | suspect | bad；bad 不参与任何解。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

CORR_MIN = 64.0        # counts, 典型 RDI 阈值
AMP_MIN = 40.0         # counts
CORR_SUSPECT = 90.0    # 边缘：进入 suspect，可产生多解候选
AMP_SUSPECT = 60.0


def beam_status(correlation_counts: float, amplitude_counts: float,
                bad_flag: Any, thresholds: Optional[Dict[str, float]] = None) -> Dict:
    th = {"corr_min": CORR_MIN, "amp_min": AMP_MIN,
          "corr_suspect": CORR_SUSPECT, "amp_suspect": AMP_SUSPECT}
    if thresholds:
        th.update(thresholds)
    reasons = []
    status = "good"
    if bad_flag in (1, True, "1", "bad"):
        return {"status": "bad", "reasons": ["bad_beam_flag"], "thresholds": th}
    if correlation_counts is None or amplitude_counts is None:
        return {"status": "bad", "reasons": ["missing_quality"], "thresholds": th}
    if correlation_counts < th["corr_min"]:
        status = "bad"
        reasons.append("correlation_low")
    if amplitude_counts < th["amp_min"]:
        status = "bad"
        reasons.append("amplitude_low")
    if status == "good":
        if correlation_counts < th["corr_suspect"]:
            status = "suspect"
            reasons.append("correlation_marginal")
        if amplitude_counts < th["amp_suspect"]:
            status = "suspect"
            reasons.append("amplitude_marginal")
    return {"status": status, "reasons": reasons, "thresholds": th}


def classify_beams(corr: Sequence[float], amp: Sequence[float],
                   bad_flags: Sequence[Any], thresholds: Optional[Dict[str, float]] = None) -> List[Dict]:
    return [beam_status(corr[i], amp[i], bad_flags[i], thresholds)
            for i in range(len(corr))]


def depth_cell_valid(cell: Dict) -> Dict:
    """无效深度 cell：深度/厚度缺失或非正、越界即排除，证据就地保留。"""
    depth = cell.get("depth_m")
    thickness = cell.get("thickness_m")
    reasons = []
    if depth is None or thickness is None:
        reasons.append("depth_cell_missing_geometry")
    else:
        if depth <= 0 or thickness <= 0:
            reasons.append("depth_cell_nonpositive")
        if depth < thickness / 2.0:
            reasons.append("depth_cell_above_transducer")
    return {"valid": not reasons, "reasons": reasons, "cell": cell}
