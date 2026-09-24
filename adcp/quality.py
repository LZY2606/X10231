"""Quality hierarchy and cell exclusion diagnostics.

A depth cell for a ping is excluded only for an explicit reason. Excluded
cells are retained (with evidence), never silently dropped for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Ordered from strongest to weakest exclusion reason.
EXCLUSION_REASONS = (
    "below_bottom",
    "bad_depth",
    "insufficient_correlation",
    "low_amplitude",
    "too_many_bad_beams",
    "attitude_gap",
    "ambiguous_three_beam",
)


@dataclass
class CellQuality:
    ping: str
    cell: int
    good_beams: list[int] = field(default_factory=list)
    correlations: dict[int, float] = field(default_factory=dict)
    amplitudes: dict[int, float] = field(default_factory=dict)
    excluded: bool = False
    reason: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "ping": self.ping, "cell": self.cell,
            "good_beams": self.good_beams,
            "correlations": {str(k): v for k, v in self.correlations.items()},
            "amplitudes": {str(k): v for k, v in self.amplitudes.items()},
            "excluded": self.excluded, "reason": self.reason,
            "detail": self.detail,
        }


def judge_cell(ping: str, cell: int, depth_m: float | None,
               bottom_range_m: float | None, bad_beams: list[int],
               correlations: dict[int, float], amplitudes: dict[int, float],
               corr_threshold: int, amp_threshold_db: int,
               attitude_blocked: bool = False) -> CellQuality:
    good = [b for b in range(4) if b not in bad_beams]
    q = CellQuality(ping=ping, cell=cell, good_beams=good,
                    correlations=dict(correlations), amplitudes=dict(amplitudes))
    if depth_m is None or depth_m <= 0:
        q.excluded, q.reason = True, "bad_depth"
        q.detail = f"cell depth {depth_m!r} is not a positive depth"
        return q
    if bottom_range_m is not None and depth_m >= bottom_range_m:
        q.excluded, q.reason = True, "below_bottom"
        q.detail = f"depth {depth_m:.2f} m >= bottom {bottom_range_m:.2f} m"
        return q
    low_corr = [b for b in good if correlations.get(b, 0) < corr_threshold]
    if low_corr:
        q.excluded, q.reason = True, "insufficient_correlation"
        q.detail = f"beams {low_corr} below correlation {corr_threshold}"
        return q
    low_amp = [b for b in good if amplitudes.get(b, 0) < amp_threshold_db]
    if low_amp:
        q.excluded, q.reason = True, "low_amplitude"
        q.detail = f"beams {low_amp} below amplitude {amp_threshold_db} dB"
        return q
    if len(good) < 3:
        q.excluded, q.reason = True, "too_many_bad_beams"
        q.detail = f"only {len(good)} good beams ({good}), need >=3"
        return q
    if attitude_blocked:
        q.excluded, q.reason = True, "attitude_gap"
        q.detail = "attitude interpolation would cross a time gap"
        return q
    if len(good) == 3:
        # Not excluded, but flagged: two legal candidates remain.
        q.detail = "three-beam: multiple legal candidates retained"
    return q
