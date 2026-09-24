"""Quality hierarchy for beams and depth cells.

Inputs are kept as separate raw records upstream (correlation, amplitude,
bad-beam flag); this layer only judges. A cell is excluded when it is
geometrically invalid (at/above the bottom detected by bottom track, or
outside the configured cell range). Judgements carry reasons; nothing is
dropped silently.
"""
from __future__ import annotations


def judge_beam_cell(correlation, amplitude, bad_flag, thresholds):
    """Return (status, reasons). status in {"ok", "bad"}."""
    reasons = []
    if bad_flag:
        reasons.append("bad_beam_flag")
    if correlation is not None and correlation < thresholds["correlation_min"]:
        reasons.append("correlation_below_min")
    if amplitude is not None and amplitude < thresholds["amplitude_min"]:
        reasons.append("amplitude_below_min")
    return ("bad" if reasons else "ok", reasons)


def judge_cell_validity(cell_index, cell_depth_m, bottom_depth_m, blanking_m, max_cells):
    """Return (status, reasons). status in {"valid", "excluded"}."""
    reasons = []
    if cell_index < 0 or cell_index >= max_cells:
        reasons.append("cell_out_of_range")
    if cell_depth_m is not None and cell_depth_m < blanking_m:
        reasons.append("within_blanking_distance")
    if bottom_depth_m is not None and cell_depth_m is not None \
            and cell_depth_m >= bottom_depth_m:
        reasons.append("at_or_below_detected_bottom")
    return ("excluded" if reasons else "valid", reasons)
