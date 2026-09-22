"""Quality hierarchy: beam quality -> beam flags -> cell validity -> solutions.

Every exclusion carries a machine-readable reason plus the evidence (correlation
count, amplitude counts, bad-beam flag, bottom track depth, cell depth).
Nothing is dropped "to make the plot work".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

QUALITY_RULE_VERSION = "quality/1"


@dataclass(frozen=True)
class QualityThresholds:
    correlation_min: int = 64      # RDI-style correlation counts 0..255
    amplitude_min: int = 30        # dB-ish counts
    amplitude_max: int = 250
    # A beam is "suspect" when one (not all) indicator is marginal.
    correlation_marginal: int = 80

    def to_fingerprint(self) -> Dict[str, Any]:
        return {
            "rule": QUALITY_RULE_VERSION,
            "correlation_min": self.correlation_min,
            "correlation_marginal": self.correlation_marginal,
            "amplitude_min": self.amplitude_min,
            "amplitude_max": self.amplitude_max,
        }


@dataclass(frozen=True)
class BeamQuality:
    beam: int
    hard_bad: bool
    suspect: bool
    reasons: List[str] = field(default_factory=list)
    correlation: Optional[int] = None
    amplitude: Optional[int] = None
    flagged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam": self.beam,
            "hard_bad": self.hard_bad,
            "suspect": self.suspect,
            "reasons": list(self.reasons),
            "correlation": self.correlation,
            "amplitude": self.amplitude,
            "flagged": self.flagged,
        }


def assess_beam(
    beam: int,
    correlation: Optional[int],
    amplitude: Optional[int],
    flagged_bad: bool,
    thresholds: QualityThresholds,
) -> BeamQuality:
    """Combine the *separately stored* quality evidence for one beam."""
    reasons: List[str] = []
    hard = False
    suspect = False

    if flagged_bad:
        hard = True
        reasons.append("bad_beam_flag")
    if correlation is None:
        hard = True
        reasons.append("correlation_missing")
    elif correlation < thresholds.correlation_min:
        hard = True
        reasons.append("correlation_low")
    elif correlation < thresholds.correlation_marginal:
        suspect = True
        reasons.append("correlation_marginal")
    if amplitude is None:
        suspect = True
        reasons.append("amplitude_missing")
    elif amplitude < thresholds.amplitude_min:
        hard = True
        reasons.append("amplitude_low")
    elif amplitude > thresholds.amplitude_max:
        suspect = True
        reasons.append("amplitude_saturated")

    return BeamQuality(
        beam=beam,
        hard_bad=hard,
        suspect=suspect,
        reasons=reasons,
        correlation=correlation,
        amplitude=amplitude,
        flagged=flagged_bad,
    )


def assess_ping_beams(
    correlations: Sequence[Optional[int]],
    amplitudes: Sequence[Optional[int]],
    bad_flags: Sequence[bool],
    thresholds: QualityThresholds,
) -> List[BeamQuality]:
    return [
        assess_beam(i, correlations[i], amplitudes[i], bool(bad_flags[i]), thresholds)
        for i in range(len(correlations))
    ]


def cell_exclusion(
    cell_depth_m: Optional[float],
    bottom_depth_m: Optional[float],
    sidelobe_fraction: float = 0.06,
    blanking_m: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Return a diagnostic when a depth cell must be excluded, else None.

    Invalid cells: no depth, above blanking, or inside the side-lobe layer
    ``bottom_depth * (1 - sin(beam_angle))`` approximated by a configurable
    fraction.
    """
    if cell_depth_m is None:
        return {"code": "cell_depth_missing", "message": "cell has no depth"}
    if cell_depth_m < blanking_m:
        return {
            "code": "inside_blanking",
            "message": "cell inside blanking distance",
            "cell_depth_m": cell_depth_m,
            "blanking_m": blanking_m,
        }
    if bottom_depth_m is None:
        return None  # open water / no bottom track is not itself invalid
    cutoff = bottom_depth_m * (1.0 - sidelobe_fraction)
    if cell_depth_m > cutoff:
        return {
            "code": "sidelobe_contamination",
            "message": "cell inside bottom side-lobe layer",
            "cell_depth_m": cell_depth_m,
            "bottom_depth_m": bottom_depth_m,
            "valid_cutoff_m": cutoff,
        }
    return None


def solution_decision(beam_quality: Sequence[BeamQuality]) -> Dict[str, Any]:
    """Classify a ping/cell into four vs three beam vs unusable.

    ``ambiguous`` is set whenever >=1 beam is *suspect* but there are not two
    hard-bad beams: in that case reviewers must see all legal 3-beam
    candidates, not a single auto-picked value.
    """
    hard = [q.beam for q in beam_quality if q.hard_bad]
    suspect = [q.beam for q in beam_quality if q.suspect and not q.hard_bad]
    if len(hard) >= 2:
        return {
            "kind": "unusable",
            "hard_bad_beams": hard,
            "suspect_beams": suspect,
            "ambiguous": False,
            "reason": "two_or_more_hard_bad_beams",
        }
    if len(hard) == 1:
        return {
            "kind": "three_beam",
            "hard_bad_beams": hard,
            "suspect_beams": suspect,
            "ambiguous": bool(suspect),
            "forced_drop": hard[0],
            "reason": "one_hard_bad_beam",
        }
    if suspect:
        return {
            "kind": "three_beam_ambiguous",
            "hard_bad_beams": hard,
            "suspect_beams": suspect,
            "ambiguous": True,
            "reason": "suspect_beam_quality",
        }
    return {
        "kind": "four_beam",
        "hard_bad_beams": hard,
        "suspect_beams": suspect,
        "ambiguous": False,
        "reason": "all_beams_good",
    }
