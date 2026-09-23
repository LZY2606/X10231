"""Beam geometry and beam-velocity -> instrument-velocity conversion.

Convention used throughout this repository (documented, not magic):

* Instrument frame X = forward (beam-1/beam-2 pair axis), Y = starboard
  (beam-3/beam-4 pair axis), Z = up.
* Beam geometry id ``rdi_convex``: the four transducers point outward in
  the downward-looking convex arrangement; beam 1 is forward, beam 2 aft,
  beam 3 starboard, beam 4 port (beams 1<->2 and 3<->4 are opposite pairs).
* Beam radial velocity b_i is positive for water moving *toward* transducer
  i (the RDI sign convention).  With k_i the unit vector pointing *out* of
  the transducer, b_i = -k_i . v_instrument.
* theta is the beam angle from vertical (20 degrees for classic RDI heads).

The 4-beam solution is a least-squares solve of the four radial equations;
the redundant fourth equation defines an error velocity.  A 3-beam solve
drops one equation and imposes zero error velocity; each different dropped
beam is a distinct legal candidate and is never merged into the 4-beam
value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

BEAM_COUNT = 4


@dataclass(frozen=True)
class BeamGeometry:
    geometry_id: str
    theta_rad: float
    azimuths_rad: tuple[float, float, float, float]

    def directions(self) -> np.ndarray:
        """Return the 4x3 matrix of outward unit vectors k_i (X, Y, Z)."""
        st = math.sin(self.theta_rad)
        ct = math.cos(self.theta_rad)
        return np.array(
            [
                [st * math.cos(a), st * math.sin(a), -ct]
                for a in self.azimuths_rad
            ],
            dtype=float,
        )

    def radial_matrix(self) -> np.ndarray:
        """Matrix G with b = G v (b positive = toward transducer)."""
        return -self.directions()


RDI_CONVEX = BeamGeometry(
    geometry_id="rdi_convex",
    theta_rad=math.radians(20.0),
    azimuths_rad=(
        math.radians(45.0),    # beam 1: forward
        math.radians(225.0),   # beam 2: aft
        math.radians(315.0),   # beam 3: starboard
        math.radians(135.0),   # beam 4: port
    ),
)

GEOMETRIES = {"rdi_convex": RDI_CONVEX}


def get_geometry(geometry_id: str) -> BeamGeometry:
    try:
        return GEOMETRIES[geometry_id]
    except KeyError:
        raise KeyError(f"unknown beam geometry {geometry_id!r}") from None


@dataclass(frozen=True)
class BeamSolution:
    kind: str               # "4beam" or "3beam"
    dropped: Optional[int]  # 0-based dropped beam index for 3beam
    velocity: np.ndarray    # [u, v, w] in instrument frame (m/s)
    error_velocity: Optional[float]
    beams_used: tuple[int, ...]


def four_beam_solution(geom: BeamGeometry,
                       beams: Sequence[float]) -> Optional[BeamSolution]:
    """Least-squares 4-beam solution plus symmetric error velocity."""
    b = np.asarray(list(beams), dtype=float)
    if b.shape != (4,) or not np.all(np.isfinite(b)):
        return None
    g = geom.radial_matrix()
    velocity, *_ = np.linalg.lstsq(g, b, rcond=None)
    residual = b - g @ velocity
    error_velocity = float((residual[0] - residual[1] + residual[2] - residual[3]) / 4.0)
    return BeamSolution(
        kind="4beam",
        dropped=None,
        velocity=velocity,
        error_velocity=error_velocity,
        beams_used=(0, 1, 2, 3),
    )


def three_beam_candidates(geom: BeamGeometry,
                          beams: Sequence[float]) -> list[BeamSolution]:
    """All 3-beam (zero-error) solutions, each labelled by dropped beam."""
    b = np.asarray(list(beams), dtype=float)
    candidates: list[BeamSolution] = []
    if b.shape != (4,) or not np.all(np.isfinite(b)):
        return candidates
    for dropped in range(BEAM_COUNT):
        used = tuple(i for i in range(BEAM_COUNT) if i != dropped)
        g = geom.radial_matrix()[list(used), :]
        v = np.linalg.solve(g, b[list(used)])
        candidates.append(
            BeamSolution(
                kind="3beam",
                dropped=dropped,
                velocity=v,
                error_velocity=0.0,
                beams_used=used,
            )
        )
    return candidates


def solve(geom: BeamGeometry, beams: Sequence[float],
          beam_state: Sequence[str]) -> dict:
    """Build the non-merged candidate set for one depth cell.

    ``beam_state`` is the per-beam verdict of the quality hierarchy:
    "good" | "marginal" | "bad".  A marginal beam may participate in a
    4-beam least-squares answer, but a 3-beam candidate that stands without
    it is also legal: that is the unresolved ambiguity case.  The function
    never collapses two distinct candidates into one value.

    Returns:
      status: "solved4" | "solved3" | "ambiguous" | "insufficient"
      four: BeamSolution | None
      three: list[BeamSolution]
      states: tuple[str, ...]
    """
    states = tuple(beam_state)
    b = np.array([
        float(x) if (isinstance(x, (int, float)) and not isinstance(x, bool)
                     and math.isfinite(float(x))) else np.nan
        for x in beams
    ], dtype=float)

    good_idx = {i for i, s in enumerate(states) if s == "good"}
    lenient_idx = {i for i, s in enumerate(states) if s in ("good", "marginal")}
    bad_count = sum(s == "bad" for s in states)
    marginal_count = sum(s == "marginal" for s in states)

    four = None
    if len(lenient_idx) == 4 and np.all(np.isfinite(b)):
        four = four_beam_solution(geom, b)

    # A 3-beam candidate is trustworthy only when its three used beams are
    # all strictly good (no reliance on a marginal measurement).
    three: list[BeamSolution] = []
    if np.all(np.isfinite(b)):
        for cand in three_beam_candidates(geom, b):
            if set(cand.beams_used) <= good_idx:
                three.append(cand)

    if bad_count == 0 and marginal_count == 0 and four is not None:
        status = "solved4"
        three = []
    elif bad_count >= 2 or len(good_idx) < 3:
        status = "insufficient"
        four = None if bad_count else four
    elif bad_count == 1 and marginal_count == 0 and len(three) == 1:
        status = "solved3"
        four = None
    elif bad_count == 0 and marginal_count == 1 and four is not None and three:
        status = "ambiguous"
    else:
        status = "insufficient"
    return {"status": status, "four": four, "three": three, "states": states}
