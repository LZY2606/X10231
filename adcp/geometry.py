"""Beam geometry and beam->instrument-frame velocity solutions.

Coordinate/datum conventions (documented, asserted by tests):
  * Instrument frame X = ship-forward, Y = starboard, Z = up.
  * Beam radial velocity b_i is positive for a reflector moving TOWARD the
    transducer (RDI convention).  For a cartesian instrument velocity v,
    b_i = -u_i . v  where u_i is the beam's downward unit direction.
  * Four beams (RDI convex Janus, heads 45/135/225/315 deg, angle 20 deg from
    vertical) are over-determined: a least-squares v is returned together with
    the Janus error velocity.  Error velocity is the mismatch of the two
    redundant vertical estimates, i.e. (v1-v2)/2.
  * With exactly three usable beams the solution is exact but error velocity is
    undetermined: every legal beam triple is returned as a separate candidate.
    Three-beam and four-beam solutions are NEVER collapsed into one value.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    geometry_id: str
    beam_count: int
    beam_angle_deg: float
    beam_headings_deg: tuple[float, ...]   # clockwise from instrument +X
    unit_vectors: tuple[tuple[float, float, float], ...]

    def matrix(self, beam_idx: list[int]) -> np.ndarray:
        # b = B v  with B rows = -u_i  (toward-transducer positive)
        return np.array([[-self.unit_vectors[i][k] for k in range(3)]
                         for i in beam_idx], dtype=float)


def rdi_convex(beam_angle_deg: float = 20.0,
               geometry_id: str = "rdi-convex-20-v1") -> BeamGeometry:
    a = math.radians(beam_angle_deg)
    sin_a, cos_a = math.sin(a), math.cos(a)
    headings = (45.0, 135.0, 225.0, 315.0)
    u: list[tuple[float, float, float]] = []
    for hdg in headings:
        h = math.radians(hdg)
        # downward pointing beams: horizontal projection at beam heading, z<0
        u.append((sin_a * math.cos(h), sin_a * math.sin(h), -cos_a))
    return BeamGeometry(geometry_id, 4, beam_angle_deg, headings,
                        tuple((round(x, 12), round(y, 12), round(z, 12))
                              for x, y, z in u))


@dataclass
class Candidate:
    kind: str                       # "four_beam" | "three_beam"
    beams: tuple[int, ...]          # 0-based usable beams
    v: tuple[float, float, float] | None   # instrument x,y,z canonical m/s
    error_velocity: float | None
    residual: float | None
    rejected: bool = False
    rejection: str | None = None    # machine code
    notes: str | None = None

    def label(self) -> str:
        beams = "+".join(str(i + 1) for i in self.beams)
        prefix = "4B" if self.kind == "four_beam" else "3B"
        return f"{prefix}({beams})"

    def candidate_id(self) -> str:
        return f"cand-{self.kind}-" + "-".join(str(i + 1) for i in self.beams)


def solve(geometry: BeamGeometry, beams: tuple[np.ndarray, ...],
          values: dict[int, float], quality: dict[int, dict],
          correlation_threshold: float = 64.0,
          amplitude_threshold: float = 30.0) -> list[Candidate]:
    """Enumerate every legal beam solution for one ping/cell.

    ``beams`` maps beam index -> unit vector (kept generic; geometry supplies
    them), ``values`` maps beam index -> canonical m/s radial velocity and
    ``quality`` maps beam index -> {"correlation":..,"amplitude":..,"bad":bool}.
    Beams flagged bad or below threshold are excluded.  A beam that is present
    but below the correlation threshold still participates in a candidate as an
    explicitly rejected one (evidence retained), per callers that pass it.
    """
    usable = sorted(i for i, q in quality.items() if not q.get("bad"))
    candidates: list[Candidate] = []
    if len(usable) >= 4:
        idx = usable[:4]
        candidates.append(_exact_or_ls(geometry, idx, values, "four_beam"))
    if len(usable) == 3 or len(usable) > 4:
        for combo in itertools.combinations(usable, 3):
            candidates.append(_exact_or_ls(geometry, list(combo), values,
                                           "three_beam"))
    if len(usable) == 3 and not candidates:
        pass
    # Enumerate rejected triples too when fewer than 3 good beams so the
    # evidence of why no solution exists is preserved.
    if len(usable) < 3:
        present = sorted(values)
        for combo in itertools.combinations(present, 3):
            cand = _exact_or_ls(geometry, list(combo), values, "three_beam")
            cand.rejected = True
            cand.rejection = "insufficient_good_beams"
            cand.notes = f"only {len(usable)} usable beam(s)"
            candidates.append(cand)
    return candidates


def _exact_or_ls(geometry: BeamGeometry, idx: list[int],
                 values: dict[int, float], kind: str) -> Candidate:
    b = np.array([values[i] for i in idx], dtype=float)
    B = geometry.matrix(idx)
    try:
        if len(idx) == 3:
            v = np.linalg.solve(B, b)
            residual = 0.0
            err = None
        else:
            v, *_ = np.linalg.lstsq(B, b, rcond=None)
            residual = float(np.linalg.norm(B @ v - b))
            err = _janus_error(geometry, idx, values)
    except np.linalg.LinAlgError as exc:
        return Candidate(kind, tuple(idx), None, None, None, True,
                         "singular_geometry", str(exc))
    return Candidate(kind, tuple(idx),
                     (float(v[0]), float(v[1]), float(v[2])), err, residual)


def _janus_error(geometry: BeamGeometry, idx: list[int],
                 values: dict[int, float]) -> float | None:
    """Error velocity from two opposite-beam pairs (v1-v2)/2.

    Uses the explicit opposite pairing of the RDI convex headings.  For an
    arbitrary geometry returns None (error velocity is geometry-specific).
    """
    heads = list(geometry.beam_headings_deg)
    if heads[:4] != [45.0, 135.0, 225.0, 315.0] or len(idx) < 4:
        return None
    a = math.radians(geometry.beam_angle_deg)
    i1, i2, i3, i4 = idx[:4]
    v1 = (values[i2] - values[i1]) / (2.0 * math.cos(a))
    v2 = (values[i4] - values[i3]) / (2.0 * math.cos(a))
    return float((v1 - v2) / 2.0)


def instrument_from_pair(geometry: BeamGeometry, b_a: float, b_b: float
                         ) -> tuple[float, float]:
    """Helper: along-axis horizontal & vertical from an opposite beam pair."""
    a = math.radians(geometry.beam_angle_deg)
    horizontal = -(b_a + b_b) / (2.0 * math.sin(a))
    vertical = -(b_a - b_b) / (2.0 * math.cos(a))
    return horizontal, vertical
