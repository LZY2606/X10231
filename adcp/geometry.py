"""Beam geometry and beam-velocity -> instrument-frame velocity.

Convention used by the bundled fixture (explicit and versioned):

* Right-handed instrument frame: +x forward (beam 1), +y starboard
  (beam 3), +z upward.
* Convex 4-beam Janus, beam azimuths measured clockwise from +x:
  beam1=0deg (forward), beam2=180deg (aft), beam3=90deg (starboard),
  beam4=270deg (port).
* Beam rows point *away* from the transducer, depressed by ``beam_angle``
  below the horizontal; the vertical component is therefore negative.
* Beam velocity convention: positive when the target moves *toward* the
  transducer (RDI-style), so radial = -unit(dot)flow. ``radial_sign`` in the
  transducer configuration can flip this without changing the geometry.

Only three independent radial measurements exist for (u, v, w); with four
beams the over-determined least-squares solution gets an "error velocity".
Exactly three good beams yield an exact solution; two cannot be solved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numpy.linalg import matrix_rank

import numpy as np

GEOMETRY_RULE_VERSION = "geometry-3.0.0"


@dataclass(frozen=True)
class BeamGeometry:
    n_beams: int
    beam_angle_deg: float
    azimuths_deg: tuple[float, ...]

    @staticmethod
    def janus4(beam_angle_deg: float = 20.0) -> "BeamGeometry":
        return BeamGeometry(4, beam_angle_deg, (0.0, 180.0, 90.0, 270.0))

    @staticmethod
    def from_config(cfg: dict) -> "BeamGeometry":
        az = tuple(cfg["azimuths_deg"])
        return BeamGeometry(n_beams=int(cfg["n_beams"]),
                            beam_angle_deg=float(cfg["beam_angle_deg"]),
                            azimuths_deg=az)

    def config(self) -> dict:
        return {
            "n_beams": self.n_beams,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
        }

    def unit_rows(self) -> np.ndarray:
        """(n_beams,3) unit vectors along each beam, pointing away."""
        theta = math.radians(self.beam_angle_deg)
        rows = []
        for az_deg in self.azimuths_deg:
            az = math.radians(az_deg)
            rows.append([
                math.cos(theta) * math.cos(az),
                math.cos(theta) * math.sin(az),
                -math.sin(theta),
            ])
        return np.asarray(rows, dtype=float)

    def radial_matrix(self, radial_sign: float = -1.0) -> np.ndarray:
        """R so that ``radial = R @ velocity`` (shape n_beams x 3)."""
        return float(radial_sign) * self.unit_rows()


@dataclass(frozen=True)
class BeamSolution:
    kind: str                 # "four", "three", "insufficient"
    velocity: tuple[float, float, float] | None   # instrument u,v,w (m/s)
    error_velocity: float | None                  # m/s, only for kind=="four"
    used_beams: tuple[int, ...]
    dropped_beam: int | None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "velocity": list(self.velocity) if self.velocity is not None else None,
            "error_velocity": self.error_velocity,
            "used_beams": list(self.used_beams),
            "dropped_beam": self.dropped_beam,
        }


def solve_beams(
    geometry: BeamGeometry,
    radial: np.ndarray,
    good: np.ndarray,
    radial_sign: float = -1.0,
) -> BeamSolution:
    """Solve instrument-frame velocity from one ping / cell.

    ``radial`` and ``good`` are aligned to the geometry beam order. With
    exactly one bad beam an exact 3-beam solution is returned (``kind``
    "three"). With four good beams an over-determined least-squares solution
    and its error velocity are returned (``kind`` "four"). Fewer than three
    good beams give ``kind`` "insufficient" and no velocity. Callers needing
    all legal 3-beam *candidates* from a 4-beam ping use
    :func:`three_beam_candidates`.
    """
    radial = np.asarray(radial, dtype=float)
    good = np.asarray(good, dtype=bool)
    if good.sum() >= 4:
        r = geometry.radial_matrix(radial_sign)
        vel, *_ = np.linalg.lstsq(r, radial, rcond=None)
        null = _null_vector(r)
        err = float(null @ radial) if null is not None else None
        return BeamSolution("four", tuple(float(x) for x in vel), err,
                            tuple(int(i) for i in np.nonzero(good)[0]), None)
    if good.sum() == 3:
        idx = np.nonzero(good)[0]
        r = geometry.radial_matrix(radial_sign)[idx]
        vel = np.linalg.solve(r, radial[idx])
        return BeamSolution("three", tuple(float(x) for x in vel), None,
                            tuple(int(i) for i in idx),
                            int(np.nonzero(~good)[0][0]))
    return BeamSolution("insufficient", None, None,
                        tuple(int(i) for i in np.nonzero(good)[0]), None)


def three_beam_candidates(
    geometry: BeamGeometry,
    radial: np.ndarray,
    allowed_drops: tuple[int, ...] | None = None,
    radial_sign: float = -1.0,
) -> list[BeamSolution]:
    """Every legal exact 3-beam solution, one per dropped beam.

    ``allowed_drops`` restricts which beams may be removed (for example when
    a specific beam is flagged). Candidates are ordered by dropped-beam index
    so their identity is stable across replays.
    """
    radial = np.asarray(radial, dtype=float)
    r_all = geometry.radial_matrix(radial_sign)
    drops = list(allowed_drops) if allowed_drops is not None else list(range(geometry.n_beams))
    out: list[BeamSolution] = []
    for drop in sorted(drops):
        idx = [i for i in range(geometry.n_beams) if i != drop]
        r = r_all[idx]
        if matrix_rank(r) < 3:
            continue
        vel = np.linalg.solve(r, radial[idx])
        out.append(BeamSolution("three", tuple(float(x) for x in vel), None,
                                tuple(idx), int(drop)))
    return out


def _null_vector(matrix: np.ndarray) -> np.ndarray | None:
    """Unit row-vector spanning the 1-D left null space of an (n x 3) matrix.

    Sign-fixed so the largest component is positive, giving a stable,
    reproducible error-velocity definition. For symmetric Janus geometry
    this equals the documented (b1+b2-b3-b4)/2 family up to scale.
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    null_cols = u[:, matrix.shape[1]:]
    if null_cols.shape[1] == 0:
        return None
    vec = null_cols[:, 0]
    k = int(np.argmax(np.abs(vec)))
    if vec[k] < 0:
        vec = -vec
    return vec / np.linalg.norm(vec)
