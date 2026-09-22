"""Beam quality hierarchy: explicit bad flag wins, then correlation/amplitude."""
from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_THRESHOLDS = {
    "min_correlation": 64.0,   # counts (0..255)
    "min_amplitude": 30.0,     # counts
}


def judge_beam(beam_index: int, cell: dict,
               thresholds: Optional[dict] = None) -> dict:
    """Decide if one beam of one depth cell is usable.

    Raw values stay in the record; thresholds apply to the canonical numeric
    form. A flagged / unconverted / low-quality beam is never silently dropped,
    it is reported with reasons.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    reasons: List[str] = []
    bad = bool(cell.get("bad_flags", [False] * 4)[beam_index])
    if bad:
        reasons.append("bad_beam_flag")
    corr = cell["correlation"][beam_index]
    amp = cell["amplitude"][beam_index]
    if corr.get("value") is None or corr.get("canonical") is None:
        reasons.append("correlation_unreadable")
    elif float(corr["canonical"]) < th["min_correlation"]:
        reasons.append("low_correlation")
    if amp.get("value") is None or amp.get("canonical") is None:
        reasons.append("amplitude_unreadable")
    elif float(amp["canonical"]) < th["min_amplitude"]:
        reasons.append("low_amplitude")
    bv = cell["beam_velocity"][beam_index]
    if not bv.get("ok", False):
        reasons.append("beam_velocity_unconvertible")
    return {"beam": beam_index, "valid": not reasons, "reasons": reasons}


def judge_cell(cell: dict, thresholds: Optional[dict] = None) -> dict:
    beams = [judge_beam(k, cell, thresholds) for k in range(4)]
    valid = [q["beam"] for q in beams if q["valid"]]
    if len(valid) == 4:
        status = "four_beam_ok"
    elif len(valid) == 3:
        status = "three_beam"
    else:
        status = "rejected"
    return {"status": status, "valid_beams": valid, "beam_quality": beams}
