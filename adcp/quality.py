"""Beam-quality hierarchy and depth-cell exclusion.

The hierarchy (kept separate from the raw measurements, never collapsed into
a single "good/bad" bit before the evidence is stored):

1. ``bad_flag``        - hardware bad-beam flag. A set beam is a *hard* fail.
2. ``correlation``     - below ``corr_min`` is a hard fail; in the marginal
                         band ``[corr_min, corr_marginal)`` is a *suspect*
                         beam that must remain a legal alternative candidate.
3. ``amplitude`` (dB)  - below ``amp_min`` is a hard fail.
4. cell placement      - a cell whose centre plus half-size lies beyond the
                         measured bottom is an *invalid depth cell*.

Every decision is recorded with the threshold version that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    threshold_id: str
    corr_min: int = 64
    corr_marginal: int = 80
    amp_min_db: float = 6.0

    def to_dict(self) -> dict:
        return {
            "threshold_id": self.threshold_id,
            "corr_min": self.corr_min,
            "corr_marginal": self.corr_marginal,
            "amp_min_db": self.amp_min_db,
        }


def classify_beam(
    *,
    bad_flag: bool,
    correlation: int | None,
    amplitude_db: float | None,
    thresholds: QualityThresholds,
) -> dict:
    """Classify one beam measurement; returns status + structured reasons."""
    reasons: list[str] = []
    if bad_flag:
        reasons.append("bad-flag")
    if correlation is None:
        reasons.append("correlation-missing")
    elif correlation < thresholds.corr_min:
        reasons.append(f"correlation<{thresholds.corr_min}")
    if amplitude_db is None:
        reasons.append("amplitude-missing")
    elif amplitude_db < thresholds.amp_min_db:
        reasons.append(f"amplitude<{thresholds.amp_min_db}dB")
    hard = bool(reasons)
    suspect = (
        not hard
        and correlation is not None
        and thresholds.corr_min <= correlation < thresholds.corr_marginal
    )
    status = "invalid" if hard else ("suspect" if suspect else "ok")
    if suspect:
        reasons.append(f"correlation<{thresholds.corr_marginal}")
    return {"status": status, "reasons": reasons, "suspect": suspect,
            "hard_invalid": hard}


def beam_candidates(
    classifications: list[dict],
) -> dict:
    """Enumerate legal beam subsets for one (ping, cell).

    Returns ``{"hard_invalid": [...], "suspect": [...], "subsets": [...]}``
    where each subset is a 4-element bool tuple.

    * Hard-invalid beams are excluded everywhere.
    * With >=3 surviving beams the "all surviving beams" subset is a candidate.
    * Every single-beam drop of a *suspect* beam is also a legal candidate,
      so ambiguous three-beam solutions are never collapsed into one value.
    """

    hard = [i for i, c in enumerate(classifications) if c["hard_invalid"]]
    suspect = [i for i, c in enumerate(classifications) if c["suspect"]]
    base = tuple(i not in hard for i in range(4))
    subsets: list[tuple[bool, bool, bool, bool]] = []

    def add(mask: tuple[bool, ...]) -> None:
        if sum(mask) >= 3 and mask not in subsets:
            subsets.append(mask)  # type: ignore[arg-type]

    add(base)
    if sum(base) == 4:
        # Four beams: the 4-beam solution plus one 3-beam solution per
        # suspect beam (none when all four beams are clean).
        for i in suspect:
            dropped = list(base)
            dropped[i] = False
            add(tuple(dropped))
    elif sum(base) == 3 and suspect:
        # The surviving set of three may itself contain a suspect beam.
        # Dropping it would leave only two beams, so the unique legal choice
        # is the three-beam solution - but it is marked suspect downstream.
        pass
    return {"hard_invalid": hard, "suspect": suspect, "subsets": subsets}


def cell_in_water(
    *,
    cell_centre_m: float,
    cell_size_m: float,
    blanking_m: float,
    bottom_depth_m: float | None,
) -> dict:
    """Decide whether a depth cell is a valid in-water cell.

    A cell is excluded when its far edge reaches the bottom
    (``centre + size/2 > depth``) or it lies in the blanking zone.
    Missing bottom depth is itself a diagnostic, never silently treated as
    infinitely deep.
    """

    near = cell_centre_m - cell_size_m / 2.0
    far = cell_centre_m + cell_size_m / 2.0
    reasons = []
    if near < blanking_m:
        reasons.append("in-blanking-zone")
    if bottom_depth_m is None:
        reasons.append("bottom-depth-unknown")
        return {"valid": False, "reasons": reasons}
    if far > bottom_depth_m:
        reasons.append("beyond-bottom")
    return {"valid": not reasons, "reasons": reasons}
