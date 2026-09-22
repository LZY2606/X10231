"""Beam-quality tiers and cell validity gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QualityRules:
    min_correlation: float = 64.0
    min_amplitude: float = 30.0
    max_error_velocity: float = 0.5
    min_depth_m: float = 0.5
    max_depth_m: float = 120.0
    gap_seconds: float = 5.0

    def fingerprint(self) -> dict:
        return {
            "min_correlation": self.min_correlation,
            "min_amplitude": self.min_amplitude,
            "max_error_velocity": self.max_error_velocity,
            "depth_range_m": [self.min_depth_m, self.max_depth_m],
            "gap_seconds": self.gap_seconds,
        }


def grade_beam(correlation: Optional[float], amplitude: Optional[float],
               bad_flag: Optional[bool], rules: QualityRules) -> str:
    """Return good | suspect | bad with the reason recorded separately."""
    if bad_flag is True:
        return "bad"
    if correlation is None or amplitude is None:
        return "bad"
    if correlation < rules.min_correlation:
        return "bad"
    if amplitude < rules.min_amplitude:
        return "suspect"
    return "good"


def cell_valid(depth_m: Optional[float], rules: QualityRules) -> bool:
    if depth_m is None:
        return False
    return rules.min_depth_m <= depth_m <= rules.max_depth_m
