"""Beam / cell quality hierarchy and exclusion reasons.

Tiering is explicit and every excluded beam/cell keeps its evidence in place
instead of being deleted so a plot can be drawn:

* ``good``            participates in the normal 4-beam solution
* ``marginal``        below correlation floor but above hard floor; it is
                      excluded from the 4-beam solution yet remains a legal
                      member of a 3-beam candidate set -> multi-valued output
* ``bad``             explicitly flagged bad beam / below hard floor / NaN;
                      never used in any solution
* ``cell-invalid``    depth cell rejected for that ping (range / side-lobe)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class QualityPolicy:
    correlation_floor: int = 90       # counts; below -> marginal
    correlation_hard: int = 64        # below -> bad
    amplitude_min: int = 40           # counts; below -> marginal
    amplitude_bad: int = 20
    id: str = "qpolicy-v1"

    def payload(self) -> dict:
        return {"id": self.id, "correlation_floor": self.correlation_floor,
                "correlation_hard": self.correlation_hard,
                "amplitude_min": self.amplitude_min,
                "amplitude_bad": self.amplitude_bad}


@dataclass
class BeamVerdict:
    beam: int
    tier: str
    reasons: list[str] = field(default_factory=list)
    correlation: Optional[int] = None
    amplitude: Optional[int] = None
    flagged_bad: bool = False
    velocity_missing: bool = False


def classify_beam(beam: int, velocity, correlation: Optional[int],
                  amplitude: Optional[int], flagged_bad: bool,
                  policy: QualityPolicy) -> BeamVerdict:
    reasons = []
    missing = velocity is None or (isinstance(velocity, float)
                                   and velocity != velocity)
    if flagged_bad:
        reasons.append("bad-beam-flag")
    if missing:
        reasons.append("velocity-missing")
    if correlation is not None and correlation < policy.correlation_hard:
        reasons.append("correlation-below-hard")
    if amplitude is not None and amplitude < policy.amplitude_bad:
        reasons.append("amplitude-below-bad")
    hard = any(k in reasons for k in
               ("bad-beam-flag", "velocity-missing",
                "correlation-below-hard", "amplitude-below-bad"))
    if hard:
        tier = "bad"
    else:
        soft = []
        if correlation is not None and correlation < policy.correlation_floor:
            soft.append("correlation-low")
        if amplitude is not None and amplitude < policy.amplitude_min:
            soft.append("amplitude-low")
        if soft:
            tier, reasons = "marginal", reasons + soft
        else:
            tier = "good"
    return BeamVerdict(beam, tier, reasons, correlation, amplitude,
                       flagged_bad, missing)


def valid_for_4beam(verdicts: list[BeamVerdict]) -> list[bool]:
    return [v.tier == "good" for v in verdicts]


def valid_for_3beam(verdicts: list[BeamVerdict]) -> list[bool]:
    # good and marginal beams may anchor an exact 3-beam solution; bad may not
    return [v.tier in ("good", "marginal") for v in verdicts]


def cell_status(ping_idx: int, cell_idx: int, depth_m: Optional[float],
                invalid_cells: set[tuple[int, int]],
                min_depth_m: float, blank_m: float) -> tuple[str, list[str]]:
    reasons = []
    if (ping_idx, cell_idx) in invalid_cells:
        return "cell-invalid", ["marked-invalid-depth-cell"]
    if depth_m is None:
        return "cell-invalid", ["no-depth"]
    if depth_m < blank_m:
        reasons.append("within-blank-distance")
    if depth_m < min_depth_m:
        reasons.append("below-min-valid-depth")
    return ("cell-invalid" if reasons else "valid"), reasons
