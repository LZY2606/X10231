"""Quality hierarchy: beam measurements -> beam verdict -> cell solution.

Beam velocity is never discarded: a failed beam stays visible as an excluded
input with the reasons that excluded it. Correlation/amplitude thresholds and
the unconvertible diagnostic are explicit, versioned policy inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .geometry import solve_beams, Geometry


@dataclass(frozen=True)
class QualityPolicy:
    min_correlation: float = 60.0       # normalized "count" scale
    min_amplitude: float = 30.0
    error_tolerance: float = 0.05       # m/s
    policy_version: str = "qpol-1"

    def to_json(self):
        return {
            "min_correlation": self.min_correlation,
            "min_amplitude": self.min_amplitude,
            "error_tolerance": self.error_tolerance,
            "policy_version": self.policy_version,
        }


def judge_beam(velocity_q, correlation_q, amplitude_q, bad_flag: bool,
               policy: QualityPolicy) -> dict:
    reasons: List[str] = []
    if bad_flag:
        reasons.append("bad-beam-flag")
    if not getattr(velocity_q, "convertible", True):
        reasons.append("unconvertible-velocity")
    corr = getattr(correlation_q, "si_value", None)
    if corr is None:
        reasons.append("unconvertible-correlation")
    elif corr < policy.min_correlation:
        reasons.append("low-correlation")
    amp = getattr(amplitude_q, "si_value", None)
    if amp is None:
        reasons.append("unconvertible-amplitude")
    elif amp < policy.min_amplitude:
        reasons.append("low-amplitude")
    return {
        "usable": not reasons,
        "reasons": reasons,
        "correlation": corr,
        "amplitude": amp,
    }


def evaluate_cell(verdicts: Sequence[dict], beam_velocities_si: Sequence[Optional[float]],
                  geom: Geometry, policy: QualityPolicy) -> dict:
    usable_velocities = [
        beam_velocities_si[i] if verdicts[i]["usable"] else None
        for i in range(len(verdicts))
    ]
    solution = solve_beams(usable_velocities, geom, policy.error_tolerance)
    solution["verdicts"] = verdicts
    excluded = [i for i, vd in enumerate(verdicts) if not vd["usable"]]
    solution["excluded_beams"] = excluded
    if len(solution["usable"]) <= 2:
        solution["status"] = "insufficient"
    return solution
