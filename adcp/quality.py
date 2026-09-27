"""Beam quality tiers and depth-cell exclusion rules.

Quality tiers per beam at a ping:

* ``hard_bad``   : explicit bad-beam flag set, never usable
* ``bad``        : correlation below ``corr_hard`` or amplitude below
                   ``amp_hard`` -- never usable
* ``marginal``   : correlation below ``corr_good`` / amplitude below
                   ``amp_good`` -- usable only inside 3-beam fallback
* ``good``       : usable in the 4-beam solution
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .units import UnitError, convert


@dataclass(frozen=True)
class QualityThresholds:
    corr_good: int = 100
    corr_hard: int = 60
    amp_good: int = 80
    amp_hard: int = 40

    def fingerprint_dict(self) -> dict:
        return {
            "corr_good": self.corr_good,
            "corr_hard": self.corr_hard,
            "amp_good": self.amp_good,
            "amp_hard": self.amp_hard,
        }


def classify_beam(correlation, amplitude, bad_flag: bool,
                  thresholds: QualityThresholds) -> str:
    if bad_flag:
        return "hard_bad"
    corr = correlation["value"]
    amp = amplitude["value"]
    # only counts/percent correlations are meaningful to compare; a value we
    # cannot interpret stays at its location as a diagnostic rather than
    # being forced into a tier.
    if correlation.get("unit") not in ("count", "%") or amplitude.get("unit") != "dB":
        raise UnitError(
            f"quality inputs carry unsupported units "
            f"corr={correlation.get('unit')!r} amp={amplitude.get('unit')!r}")
    if corr < thresholds.corr_hard or amp < thresholds.amp_hard:
        return "bad"
    if corr < thresholds.corr_good or amp < thresholds.amp_good:
        return "marginal"
    return "good"


USABLE = {"good", "marginal"}
FOUR_BEAM = {"good"}


def beam_masks(tiers: list[str]) -> dict:
    return {
        "hard_bad": [t == "hard_bad" for t in tiers],
        "usable": [t in USABLE for t in tiers],
        "four_beam_ok": [t in FOUR_BEAM for t in tiers],
    }


def cell_status(bin_depth_m: float, bin_size_m: float,
                bottom_range_m: float | None, beam_angle_deg: float) -> str:
    """Return ``valid`` / ``invalid_depth`` / ``below_bottom_track``.

    Side-lobe exclusion (convex Janus): a vertical cell is contaminated when
    its far edge is shallower than bottom_range * cos(beam_angle).
    """
    if bin_depth_m is None or bin_size_m is None:
        return "invalid_depth"
    if not (math.isfinite(bin_depth_m) and math.isfinite(bin_size_m)):
        return "invalid_depth"
    if bin_depth_m <= 0.0 or bin_size_m <= 0.0:
        return "invalid_depth"
    if bottom_range_m is None or not math.isfinite(bottom_range_m):
        return "valid"
    vertical_keep = bottom_range_m * math.cos(math.radians(beam_angle_deg))
    if bin_depth_m + bin_size_m / 2.0 > vertical_keep:
        return "below_bottom_track"
    return "valid"


def to_metres(quantity: dict) -> float:
    return convert(float(quantity["value"]), quantity["unit"], "m")
