"""Quality hierarchy and exclusion reasons.

No measured value is deleted to make a plot: an unusable cell is marked with
an exclusion reason and still displayed in the excluded-cell evidence lane.
"""

from __future__ import annotations

QUALITY_RULE_VERSION = "quality-1.1.0"

TIER_FOUR = "four_beam"
TIER_THREE = "three_beam"
TIER_EXCLUDED = "excluded"
TIER_UNRESOLVED = "unresolved"

# Stable exclusion reason codes (machine-readable events reuse these).
EX_LOW_CORRELATION = "low_correlation"
EX_BAD_BEAM_FLAG = "bad_beam_flag"
EX_INSUFFICIENT_BEAMS = "insufficient_beams"
EX_INVALID_CELL = "invalid_depth_cell"
EX_UNCONVERTIBLE = "unconvertible_input"
EX_ATTITUDE_GAP = "attitude_gap"
EX_ATTITUDE_MISSING = "attitude_missing"
EX_BOTTOM_BELOW_CELL = "bottom_above_cell"


def correlation_ok(corr_percent: float, threshold_percent: float = 64.0) -> bool:
    return corr_percent >= threshold_percent


def valid_cell(cell_depth_m: float, blanking_m: float, cell_size_m: float,
               n_cells: int, bin_distance_m: float) -> bool:
    """A depth cell is valid when it lies in the configured profiling range."""
    if not (cell_depth_m == cell_depth_m):  # NaN
        return False
    if cell_depth_m < blanking_m - 1e-9:
        return False
    if cell_depth_m > bin_distance_m + cell_size_m * n_cells + 1e-9:
        return False
    return True
