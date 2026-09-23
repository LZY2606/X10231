"""Beam quality hierarchy and depth-cell validity.

Layers (a beam in a depth cell):
    raw  -> measured
         -> quality_checked (correlation / amplitude thresholds, bad-bit flag)
         -> geometry_solved (3- or 4-beam candidate)
         -> instrument / ship / earth vectors
         -> profiled (admitted to the vertical profile) or excluded.

Evidence for every rejection is retained, never dropped "to draw a picture".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class QualityThresholds:
    min_correlation_counts: float = 64.0
    min_amplitude_counts: float = 40.0
    # A beam is "hard bad" when the bad-bit is set or it is unparseable.
    # A beam below correlation but not flagged is "suspect" and keeps both
    # legal solutions (see adcp.beam.solve_beam_velocities).


def classify_beams(
    correlations: Sequence[Optional[float]],
    amplitudes: Sequence[Optional[float]],
    bad_flags: Sequence[bool],
    thresholds: QualityThresholds,
) -> Dict[str, object]:
    """Return per-beam good/suspect/hard-bad classification with reasons."""
    n = len(bad_flags)
    good: list = [True] * n
    suspect: list = [False] * n
    hard_bad: list = [False] * n
    reasons: Dict[int, list] = {}

    def mark(beam: int, level: str, reason: str) -> None:
        reasons.setdefault(beam, []).append(reason)
        if level == "hard_bad":
            hard_bad[beam] = True
            good[beam] = False
            suspect[beam] = False
        elif level == "suspect" and not hard_bad[beam]:
            suspect[beam] = True
            good[beam] = False

    for beam in range(n):
        if bad_flags[beam]:
            mark(beam, "hard_bad", "bad_beam_flag_set")
        correlation = correlations[beam]
        amplitude = amplitudes[beam]
        if correlation is None:
            mark(beam, "hard_bad", "correlation_missing_or_unparseable")
        elif correlation < thresholds.min_correlation_counts:
            mark(beam, "suspect", f"correlation_below_{thresholds.min_correlation_counts:g}")
        if amplitude is None:
            mark(beam, "suspect", "amplitude_missing_or_unparseable")
        elif amplitude < thresholds.min_amplitude_counts:
            mark(beam, "suspect", f"amplitude_below_{thresholds.min_amplitude_counts:g}")

    return {"good": good, "suspect": suspect, "hard_bad": hard_bad, "reasons": reasons}


def validate_depth_cell(
    cell_index: int,
    bin_distance_m: Optional[float],
    bottom_distance_m: Optional[float],
    min_valid_distance_m: float,
) -> Dict[str, object]:
    """A depth cell must sit in water: above bottom and past blanking."""
    if bin_distance_m is None:
        return {"valid": False, "reason": "cell_distance_unknown", "cell_index": cell_index}
    if bin_distance_m < min_valid_distance_m:
        return {
            "valid": False,
            "reason": "inside_blanking_distance",
            "cell_index": cell_index,
            "bin_distance_m": bin_distance_m,
            "limit_m": min_valid_distance_m,
        }
    if bottom_distance_m is not None and bin_distance_m >= bottom_distance_m:
        return {
            "valid": False,
            "reason": "at_or_below_bottom_track",
            "cell_index": cell_index,
            "bin_distance_m": bin_distance_m,
            "bottom_distance_m": bottom_distance_m,
        }
    return {"valid": True, "reason": "in_water_column", "cell_index": cell_index, "bin_distance_m": bin_distance_m}
