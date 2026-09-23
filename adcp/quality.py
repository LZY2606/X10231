"""Quality hierarchy.

Layer 1 - per beam: bad flag beats everything; otherwise correlation then
amplitude decide good vs marginal vs bad.  Every rejection records the
failing measurement so excluded beams/cells keep their evidence.

Layer 2 - per depth cell: valid depth window (blanking, number of cells,
bottom proximity); cells outside the window are excluded with a reason,
never silently removed from the profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class BeamThresholds:
    corr_floor: float        # below -> bad
    corr_marginal: float     # floor..marginal -> marginal
    amp_floor: float
    amp_marginal: float


DEFAULT_THRESHOLDS = BeamThresholds(
    corr_floor=64.0, corr_marginal=90.0, amp_floor=70.0, amp_marginal=80.0
)


@dataclass
class BeamVerdict:
    beam: int
    state: str               # good | marginal | bad
    reasons: list[str] = field(default_factory=list)


def classify_beam(beam: int, bad_flag: bool, correlation: float | None,
                  amplitude: float | None,
                  thresholds: BeamThresholds = DEFAULT_THRESHOLDS) -> BeamVerdict:
    reasons: list[str] = []
    if bad_flag:
        reasons.append("bad-beam flag set")
        state = "bad"
    else:
        state = "good"
        if correlation is None:
            reasons.append("correlation missing")
            state = "bad"
        elif correlation < thresholds.corr_floor:
            reasons.append(f"correlation {correlation:g} below floor {thresholds.corr_floor:g}")
            state = "bad"
        elif correlation < thresholds.corr_marginal:
            reasons.append(f"correlation {correlation:g} below marginal {thresholds.corr_marginal:g}")
            state = "marginal"
        if amplitude is None:
            reasons.append("amplitude missing")
            state = "bad"
        elif amplitude < thresholds.amp_floor:
            reasons.append(f"amplitude {amplitude:g} below floor {thresholds.amp_floor:g}")
            state = "bad"
        elif amplitude < thresholds.amp_marginal:
            reasons.append(f"amplitude {amplitude:g} below marginal {thresholds.amp_marginal:g}")
            state = "marginal" if state == "good" else state
    return BeamVerdict(beam=beam, state=state, reasons=reasons)


@dataclass(frozen=True)
class CellWindow:
    blanking_m: float
    cell_size_m: float
    n_cells: int
    bottom_m: float | None = None
    bottom_guard_m: float = 0.5

    def cell_range_m(self, cell: int) -> tuple[float, float]:
        near = self.blanking_m + cell * self.cell_size_m
        return near, near + self.cell_size_m


def classify_cell(cell: int, window: CellWindow) -> dict:
    """Return {state, reason, range_m}; state valid | excluded."""
    if cell < 0 or cell >= window.n_cells:
        return {"state": "excluded", "reason": "cell index outside configured profile",
                "range_m": None}
    near, far = window.cell_range_m(cell)
    if window.bottom_m is not None and near >= window.bottom_m - window.bottom_guard_m:
        return {"state": "excluded",
                "reason": f"cell near/far {near:.2f}-{far:.2f} m enters bottom "
                          f"{window.bottom_m:g} m (guard {window.bottom_guard_m:g} m)",
                "range_m": [near, far]}
    return {"state": "valid", "reason": None, "range_m": [near, far]}


def verdict_for_cell(correlations: Sequence[float], amplitudes: Sequence[float],
                     bad_flags: Sequence[bool],
                     thresholds: BeamThresholds = DEFAULT_THRESHOLDS) -> list[BeamVerdict]:
    return [
        classify_beam(i, bool(bad_flags[i]), correlations[i], amplitudes[i], thresholds)
        for i in range(4)
    ]
