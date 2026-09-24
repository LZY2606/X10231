"""Beam/cell quality hierarchy.

Layers, from raw toward derived:
  L0 raw flag/value   - explicit bad-beam flag, missing/unconvertible radial
  L1 signal quality   - correlation / amplitude thresholds
  L2 geometry range   - depth bin valid, inside measured bottom range
A cell excluded at L2 is kept with an exclusion reason; it is never dropped
silently just to keep a plot clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class QualityThresholds:
    min_correlation_percent: float = 64.0
    min_amplitude_db: float = 6.0
    version: str = "quality-v1"


@dataclass
class BeamGate:
    beam: int
    accepted: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class CellGate:
    ping_idx: int
    cell_idx: int
    depth_m: Optional[float]
    accepted: bool
    beam_gates: List[BeamGate]
    exclusion_reasons: List[str] = field(default_factory=list)


def gate_cell(ping_idx: int, cell_idx: int,
              radials_mps: Sequence[Optional[float]],
              bad_flags: Sequence[Optional[bool]],
              correlations_pct: Sequence[Optional[float]],
              amplitudes_db: Sequence[Optional[float]],
              depth_m: Optional[float],
              bottom_altitude_m: Optional[float],
              thresholds: QualityThresholds) -> CellGate:
    gates: List[BeamGate] = []
    for i in range(len(radials_mps)):
        reasons = []
        flag = bad_flags[i] if i < len(bad_flags) else None
        if flag is True:
            reasons.append("BAD_BEAM_FLAG")
        r = radials_mps[i]
        if r is None:
            reasons.append("MISSING_OR_UNCONVERTIBLE_RADIAL")
        corr = correlations_pct[i] if i < len(correlations_pct) else None
        if corr is None:
            reasons.append("MISSING_CORRELATION")
        elif corr < thresholds.min_correlation_percent:
            reasons.append("LOW_CORRELATION")
        amp = amplitudes_db[i] if i < len(amplitudes_db) else None
        if amp is None:
            reasons.append("MISSING_AMPLITUDE")
        elif amp < thresholds.min_amplitude_db:
            reasons.append("LOW_AMPLITUDE")
        gates.append(BeamGate(i + 1, len(reasons) == 0, reasons))

    exclusion = []
    if depth_m is None:
        exclusion.append("UNCONVERTIBLE_DEPTH")
    elif depth_m <= 0:
        exclusion.append("NON_POSITIVE_DEPTH")
    if bottom_altitude_m is not None and depth_m is not None and depth_m > 0:
        if depth_m > bottom_altitude_m:
            exclusion.append("CELL_BELOW_BOTTOM_TRACK")
    accepted = len(exclusion) == 0
    return CellGate(ping_idx, cell_idx, depth_m, accepted, gates, exclusion)
