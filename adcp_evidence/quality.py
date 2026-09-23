"""Quality hierarchy: beam -> ping -> depth cell, plus bottom-track checks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class QualityThresholds:
    correlation_min: float
    amplitude_min: float
    # Maximum allowed attitude interpolation span across a missing sample.
    attitude_max_gap_s: float
    # Extra upward margin applied to the sidelobe exclusion depth.
    sidelobe_margin_m: float = 0.5

    def fingerprint_parts(self) -> dict:
        return {
            "correlation_min": self.correlation_min,
            "amplitude_min": self.amplitude_min,
            "attitude_max_gap_s": self.attitude_max_gap_s,
            "sidelobe_margin_m": self.sidelobe_margin_m,
        }


def judge_beam(
    beam_index: int,
    velocity_ok: bool,
    correlation: Optional[float],
    amplitude: Optional[float],
    bad_flag: bool,
    thresholds: QualityThresholds,
) -> dict:
    """Return the quality verdict for one beam with explicit reasons."""
    reasons = []
    marginal_reasons = []
    if bad_flag:
        reasons.append("bad_beam_flag")
    if not velocity_ok:
        reasons.append("velocity_unusable")
    if correlation is None:
        reasons.append("correlation_missing")
    elif correlation < thresholds.correlation_min:
        marginal_reasons.append("correlation_low")
    if amplitude is None:
        marginal_reasons.append("amplitude_missing")
    elif amplitude < thresholds.amplitude_min:
        marginal_reasons.append("amplitude_low")
    return {
        "beam": beam_index,
        "valid": not reasons,
        "marginal": bool(marginal_reasons),
        "reasons": reasons,
        "marginal_reasons": marginal_reasons,
        "correlation": correlation,
        "amplitude": amplitude,
        "flagged": bad_flag,
    }


def judge_ping(beam_verdicts: Sequence[dict]) -> dict:
    valid_beams = [v["beam"] for v in beam_verdicts if v["valid"]]
    marginal_beams = [v["beam"] for v in beam_verdicts
                      if v["valid"] and v["marginal"]]
    if len(valid_beams) >= 4:
        state = "four_beam_marginal" if marginal_beams else "four_beam"
    elif len(valid_beams) == 3:
        state = "three_beam"
    else:
        state = "insufficient"
    return {
        "state": state,
        "valid_beams": valid_beams,
        "marginal_beams": marginal_beams,
        "invalid_beams": [v["beam"] for v in beam_verdicts if not v["valid"]],
        "verdicts": beam_verdicts,
    }


def sidelobe_safe_depth_m(
    bottom_track_depth_m: Optional[float], beam_angle_deg: float
) -> Optional[float]:
    """Deepest bin center that avoids the four-beam sidelobe annulus."""
    if bottom_track_depth_m is None:
        return None
    return bottom_track_depth_m * math.cos(math.radians(beam_angle_deg))


def judge_cell(
    ping_id: str,
    cell_index: int,
    cell_center_m: float,
    configured: bool,
    bottom_track_depth_m: Optional[float],
    beam_angle_deg: float,
    thresholds: QualityThresholds,
) -> dict:
    """Decide whether a depth cell may contribute to the vertical profile."""
    reasons = []
    if not configured:
        reasons.append("cell_not_configured")
    if cell_center_m <= 0:
        reasons.append("cell_depth_nonpositive")
    safe = sidelobe_safe_depth_m(bottom_track_depth_m, beam_angle_deg)
    if safe is not None and cell_center_m > safe - thresholds.sidelobe_margin_m:
        reasons.append("sidelobe_contaminated")
    if (
        bottom_track_depth_m is not None
        and cell_center_m > bottom_track_depth_m
    ):
        reasons.append("below_bottom_track")
    return {
        "ping": ping_id,
        "cell": cell_index,
        "cell_center_m": cell_center_m,
        "included": not reasons,
        "reasons": reasons,
        "sidelobe_safe_depth_m": safe,
    }
