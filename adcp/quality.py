"""波束质量层级与无效深度 cell 判定。

每个波束单独留下测量、correlation、amplitude、坏 beam 标志；
质量分级是派生层，不回写覆盖原始测量。
"""
from __future__ import annotations

from dataclasses import dataclass

BEAM_STATES = ("trusted", "marginal", "excluded")


@dataclass(frozen=True)
class QualityRules:
    corr_trusted: float = 80.0
    corr_marginal: float = 40.0
    amp_floor: float = 30.0
    error_velocity_limit_mps: float = 0.20
    cell_depth_min_m: float = 0.5
    cell_depth_max_m: float = 60.0

    def fingerprint(self) -> str:
        import hashlib
        import json
        blob = json.dumps(self.__dict__, sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def classify_beam(bad_flag: bool, correlation, amplitude, radial,
                  rules: QualityRules):
    """返回 (state, reasons)。state ∈ trusted/marginal/excluded。"""
    reasons = []
    if bad_flag:
        return "excluded", ["bad_beam_flag"]
    corr = None
    amp = None
    try:
        corr = float(correlation)
    except (TypeError, ValueError):
        reasons.append("correlation_unreadable")
    try:
        amp = float(amplitude)
    except (TypeError, ValueError):
        reasons.append("amplitude_unreadable")
    if radial is None:
        return "excluded", reasons + ["missing_radial"]
    if isinstance(radial, float) and radial != radial:
        return "excluded", reasons + ["radial_nan"]
    if corr is None:
        return "excluded", reasons + ["correlation_missing"]
    if amp is None:
        return "excluded", reasons + ["amplitude_missing"]
    if corr < rules.corr_marginal:
        return "excluded", reasons + [f"correlation<{rules.corr_marginal}"]
    if amp < rules.amp_floor:
        return "excluded", reasons + [f"amplitude<{rules.amp_floor}"]
    if corr < rules.corr_trusted:
        return "marginal", reasons + [
            f"correlation<{rules.corr_trusted}", f"corr>={rules.corr_marginal}"]
    if amp < rules.amp_floor + 20:
        return "marginal", reasons + ["amplitude_low"]
    return "trusted", reasons


def beam_gate(states):
    """4 个波束状态 -> (gate, usable count, detail)。"""
    n_trusted = sum(s == "trusted" for s in states)
    n_ok = sum(s != "excluded" for s in states)
    if n_ok == 4 and n_trusted == 4:
        return "four_beam_unique", 4, "all beams trusted"
    if n_ok == 4:
        return "four_beam_ambiguous", 4, f"{4 - n_trusted} marginal; 4-beam LS plus 3-beam candidates retained"
    if n_ok == 3:
        return "three_beam_unique", 3, "exactly one beam excluded"
    if n_ok >= 3:
        return "three_beam", n_ok, "multiple 3-beam candidates"
    return "unsolvable", n_ok, "fewer than 3 usable beams"


def valid_cell(depth_m, blank_m, rules: QualityRules):
    """无效深度 cell 判定：物理不可能/超量程/落入盲区。"""
    if depth_m is None or depth_m != depth_m:
        return False, "cell depth missing/NaN"
    if depth_m < rules.cell_depth_min_m:
        return False, f"depth {depth_m:g}m shallower than min"
    if depth_m > rules.cell_depth_max_m:
        return False, f"depth {depth_m:g}m deeper than max"
    if blank_m is not None and depth_m < blank_m:
        return False, f"depth {depth_m:g}m within blanking {blank_m:g}m"
    return True, "ok"
