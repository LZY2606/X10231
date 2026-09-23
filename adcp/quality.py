"""Beam quality hierarchy.

Quality is deliberately layered so a single bad beam does not force a whole
ping/cell to be discarded, and so ambiguous failures yield more than one legal
candidate instead of one forced answer:

* ``hard_bad``  - flagged bad, missing velocity, or correlation under the hard
  floor. The beam cannot participate.
* ``marginal``  - correlation under the marginal floor: the beam may participate
  in the primary solution but excluding it also produces a recorded candidate.
* ``good``      - usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CORR_HARD_FLOOR = 64.0
CORR_MARGINAL_FLOOR = 96.0
ERR_VELOCITY_LIMIT = 0.15  # m/s Janus inconsistency limit for the four-beam err


@dataclass(frozen=True)
class BeamQuality:
    beam: int
    velocity_present: bool
    correlation: Optional[float]
    amplitude: Optional[float]
    flagged_bad: bool
    hard_bad: bool
    marginal: bool
    reason: str

    @property
    def tier(self) -> str:
        if self.hard_bad:
            return "hard_bad"
        if self.marginal:
            return "marginal"
        return "good"


def classify_beam(
    beam: int,
    velocity,
    correlation: Optional[float],
    amplitude: Optional[float],
    flagged_bad: bool,
    *,
    hard_floor: float = CORR_HARD_FLOOR,
    marginal_floor: float = CORR_MARGINAL_FLOOR,
) -> BeamQuality:
    """Classify one beam. ``velocity`` is a normalized Quantity-like value."""
    vnorm = getattr(velocity, "normalized", velocity)
    reasons = []
    if flagged_bad:
        reasons.append("bad_flag")
    if vnorm is None:
        reasons.append("missing_velocity")
    if correlation is None:
        reasons.append("missing_correlation")
    hard_bad = bool(reasons) or (
        correlation is not None and correlation < hard_floor
    )
    if correlation is not None and correlation < hard_floor and "low_correlation" not in reasons:
        reasons.append("correlation_below_hard_floor")
    marginal = (
        not hard_bad
        and correlation is not None
        and hard_floor <= correlation < marginal_floor
    )
    if marginal:
        reasons.append("correlation_marginal")
    return BeamQuality(
        beam=beam,
        velocity_present=vnorm is not None,
        correlation=correlation,
        amplitude=amplitude,
        flagged_bad=flagged_bad,
        hard_bad=hard_bad,
        marginal=marginal,
        reason=";".join(reasons) if reasons else "ok",
    )


def candidate_beam_sets(qualities: list[BeamQuality]) -> tuple[tuple[int, ...], ...]:
    """Enumerate legal participating beam sets without collapsing ambiguity.

    Returns every distinct set that can produce a solution. With four good
    beams that is just the full set plus, for each marginal beam, the three-beam
    set that excludes it. Ambiguous marginal identity therefore yields several
    recorded three-beam candidates rather than one forced value.
    """
    good = [q.beam for q in qualities if not q.hard_bad]
    marginals = [q.beam for q in qualities if q.marginal]
    if len(good) < 3:
        return tuple()
    sets: list[tuple[int, ...]] = []
    if len(good) == 4:
        sets.append(tuple(sorted(good)))
    if len(good) >= 3:
        # every 3-beam subset whose excluded beam is the failed/marginal one
        if len(good) == 3:
            sets.append(tuple(sorted(good)))
        for m in marginals:
            sub = tuple(sorted(b for b in good if b != m))
            if len(sub) == 3 and sub not in sets:
                sets.append(sub)
    return tuple(sets)
