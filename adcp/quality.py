"""Beam-quality hierarchy and depth-cell exclusion.

Hierarchy (hard bad flag wins, then correlation, then amplitude):
  bad-flag      -> bad       (hard)
  correlation   -> bad / suspect / good   (bad<=corr_bad, suspect<=corr_suspect)
  amplitude     -> bad / suspect / good

Cells are excluded independently of the per-beam radial quality:
  invalid-cell-geometry : center missing/<= blanking/non-finite/unconvertible
  bottom-interference   : bottom-track beam range <= cell center
Excluded cells stay in place with diagnostics; they are never silently
removed for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GOOD, SUSPECT, BAD = "good", "suspect", "bad"


@dataclass
class QualityConfig:
    corr_bad: float = 0.50
    corr_suspect: float = 0.70
    amp_bad_db: float = 30.0
    amp_suspect_db: float = 40.0


@dataclass
class BeamQuality:
    beam: int
    grade: str
    reasons: list = field(default_factory=list)
    usable: bool = True        # only good/suspect usable; suspect flagged
    suspect: bool = False
    correlation: float | None = None
    amplitude: float | None = None
    bad_flag: bool = False


def grade_beam(beam: int, bad_flag: bool, correlation, amplitude,
               cfg: QualityConfig, corr_diag=None, amp_diag=None) -> BeamQuality:
    reasons = []
    grade = GOOD
    usable, suspect = True, False
    cval, aval = correlation, amplitude
    if bad_flag:
        grade, usable = BAD, False
        reasons.append("bad-flag")
    # unconvertible correlation/amplitude are not treated as proof of failure
    if cval is None:
        if corr_diag:
            reasons.append("correlation-unconvertible")
            suspect = True
    else:
        if cval <= cfg.corr_bad:
            grade, usable = BAD, False
            reasons.append(f"correlation<={cfg.corr_bad:g}")
        elif cval <= cfg.corr_suspect:
            suspect = True
            reasons.append(f"correlation<={cfg.corr_suspect:g}")
    if aval is None:
        if amp_diag:
            reasons.append("amplitude-unconvertible")
            suspect = True
    else:
        if aval <= cfg.amp_bad_db:
            grade, usable = BAD, False
            reasons.append(f"amplitude<={cfg.amp_bad_db:g}dB")
        elif aval <= cfg.amp_suspect_db:
            suspect = True
            reasons.append(f"amplitude<={cfg.amp_suspect_db:g}dB")
    if grade != BAD and suspect:
        grade = SUSPECT
    return BeamQuality(beam, grade, reasons, usable, suspect, cval, aval, bad_flag)


def exclude_cell(cell_center, blank_m: float,
                 bt_ranges, cell_index: int) -> tuple[bool, str]:
    """Return (excluded, code). bt_ranges: per-beam bottom range in m or None."""
    if cell_center is None:
        return True, "invalid-cell-geometry:missing-center"
    try:
        center = float(cell_center)
    except (TypeError, ValueError):
        return True, "invalid-cell-geometry:non-numeric-center"
    if not (center == center) or center <= 0:
        return True, "invalid-cell-geometry:non-finite-center"
    if blank_m and center <= blank_m:
        return True, "invalid-cell-geometry:within-blanking"
    if bt_ranges:
        for b, rng in enumerate(bt_ranges):
            if rng is None:
                continue
            try:
                r = float(rng)
            except (TypeError, ValueError):
                continue
            if r == r and r > 0 and r <= center:
                return True, f"bottom-interference:beam{b+1}:range{r:g}m<=center{center:g}m"
    return False, ""
