"""Per-cell quality classification.

Raw correlation / amplitude / bad-beam flags live in their own tables; this
module only interprets them.  An invalid cell is *excluded* from derived
profiles but kept in place with its diagnostic -- nothing is deleted to make
plots prettier.
"""
from __future__ import annotations

from typing import Dict, Optional

QC_VERSION = "qc-2026.09.1"


def default_thresholds() -> Dict[str, float]:
    return {
        "correlation_min": 64.0,      # counts, 0-255
        "amplitude_min": 10.0,        # counts
        "amplitude_max": 160.0,       # counts; above -> saturated
        "error_velocity_max": 0.20,   # m/s, four-beam sanity bound
    }


def classify_cell(
    correlation: Optional[float],
    amplitude: Optional[float],
    bad_beam: bool,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    th = thresholds or default_thresholds()
    reasons = []
    if bad_beam:
        reasons.append("bad_beam_flag")
    if correlation is None:
        reasons.append("correlation_missing")
    elif correlation < th["correlation_min"]:
        reasons.append("correlation_below_min")
    if amplitude is None:
        reasons.append("amplitude_missing")
    else:
        if amplitude < th["amplitude_min"]:
            reasons.append("amplitude_below_min")
        if amplitude > th["amplitude_max"]:
            reasons.append("amplitude_saturated")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "qc_version": QC_VERSION,
    }
