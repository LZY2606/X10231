"""Transducer beam geometry and beam-radial -> instrument-frame solutions.

Conventions (documented and tested, not assumed):
- instrument / ship frame: X forward, Y starboard, Z *down* (right-handed
  marine frame; earth output is ENU, with the Z flip applied at the heading
  step).
- beams point downward; beam radial velocity is positive when the target
  moves *toward* the transducer (ADCP convention), so  r_i = -u_i . v.
- beam unit vectors are explicit fixture data (azimuth measured clockwise
  from instrument +X); the Janus 4-beam layout is the default.

The 4-beam solution (least squares plus the homogeneous-combination error
velocity) and every valid 3-beam solution get separate stable ids; they are
never merged into a single value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .ids import digest


@dataclass
class BeamGeometry:
    name: str
    beam_angle_deg: float                 # angle from vertical
    azimuths_deg: list                    # clockwise from instrument +X
    n_beams: int = 4
    version: str = "geo-v1"
    # depth-cell geometry (binning) — part of geometry fingerprint
    blank_m: float = 0.0
    bin_size_m: float = 1.0
    n_cells: int = 8
    first_cell_center_m: float = 1.0

    @property
    def beam_angle_rad(self) -> float:
        return math.radians(self.beam_angle_deg)

    def unit_vectors(self) -> np.ndarray:
        """u_i point *from transducer toward the cell* (n_beams x 3)."""
        theta = self.beam_angle_rad
        rows = []
        for az in self.azimuths_deg[: self.n_beams]:
            a = math.radians(az)
            rows.append([
                math.sin(theta) * math.cos(a),
                math.sin(theta) * math.sin(a),
                -math.cos(theta),
            ])
        return np.array(rows, dtype=float)

    def design_matrix(self) -> np.ndarray:
        """A so that radial r (toward positive) = A v : A = -[u_i]."""
        return -self.unit_vectors()

    def fingerprint_payload(self) -> dict:
        return {
            "name": self.name, "version": self.version,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg), "n_beams": self.n_beams,
            "blank_m": self.blank_m, "bin_size_m": self.bin_size_m,
            "n_cells": self.n_cells, "first_cell_center_m": self.first_cell_center_m,
        }

    @property
    def fingerprint(self) -> str:
        return digest("geo", self.fingerprint_payload())

    def cell_center(self, index0: int) -> float:
        return self.first_cell_center_m + index0 * self.bin_size_m


@dataclass
class BeamSolution:
    kind: str                  # "four-beam-lsq" | "three-beam"
    beams: tuple               # 0-based beam indices used
    velocity: list             # [uX,uY,uZ] m/s instrument frame (Z down)
    error_velocity: float | None
    residuals: list            # per used beam, m/s radial
    condition: float
    candidate_id: str
    geometry_fingerprint: str


def _lsq(A: np.ndarray, r: np.ndarray):
    v, *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = list(np.asarray(A @ v - r, dtype=float))
    try:
        cond = float(np.linalg.cond(A))
    except Exception:  # pragma: no cover - numpy platform variance
        cond = float("inf")
    return v, resid, cond


def solve_beams(geometry: BeamGeometry, radials, usable) -> list[BeamSolution]:
    """All legal candidates for the usable set.

    - 4 usable -> one four-beam LSQ candidate.
    - 3 usable -> one three-beam candidate.
    - <3 usable -> no candidate (underdetermined).
    Use `solve_beams_with_triples` when a beam is merely suspect: it keeps
    the four-beam value and additionally every valid three-beam triple under
    distinct ids (multi-solution, never collapsed).
    """
    return _solve(geometry, radials, usable, include_triples=False)


def solve_beams_with_triples(geometry: BeamGeometry, radials, usable) -> list[BeamSolution]:
    return _solve(geometry, radials, usable, include_triples=True)


def _solve(geometry, radials, usable, include_triples) -> list[BeamSolution]:
    A_full = geometry.design_matrix()
    r_full = np.asarray(radials, dtype=float)
    ok = [i for i in range(geometry.n_beams) if usable[i]]
    out: list[BeamSolution] = []

    if len(ok) == 4:
        v, resid4, cond4 = _lsq(A_full, r_full)
        ev = error_velocity(geometry, r_full)
        out.append(BeamSolution(
            "four-beam-lsq", (0, 1, 2, 3), [float(x) for x in v],
            None if ev is None else float(ev),
            [float(x) for x in resid4], cond4,
            digest("ncand4", {"beams": [0, 1, 2, 3],
                              "radials": [float(x) for x in r_full],
                              "geo": geometry.fingerprint}),
            geometry.fingerprint))
        if include_triples:
            out.extend(_all_triples(geometry, A_full, r_full, list(range(4))))
    elif len(ok) == 3:
        out.extend(_all_triples(geometry, A_full, r_full, ok))
    return out


def _all_triples(geometry, A_full, r_full, indices) -> list[BeamSolution]:
    out = []
    for beams in combinations(indices, 3):
        A = A_full[list(beams), :]
        r = r_full[list(beams)]
        try:
            v = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            continue
        resid = list(np.asarray(A @ v - r, dtype=float))
        try:
            cond = float(np.linalg.cond(A))
        except Exception:  # pragma: no cover
            cond = float("inf")
        out.append(BeamSolution(
            "three-beam", tuple(beams), [float(x) for x in v], None,
            [float(x) for x in resid], cond,
            digest("ncand3", {"beams": list(beams),
                              "radials": [float(r_full[b]) for b in beams],
                              "geo": geometry.fingerprint}),
            geometry.fingerprint))
    return out


def error_velocity(geometry: BeamGeometry, radials) -> float | None:
    """Homogeneous beam-combination error velocity (m/s).

    For the azimuth-symmetric Janus layout:
        e = (r1 + r2 - r3 - r4) / (4 cos theta)
    (beam pairs (1,3) and (2,4) opposing). For arbitrary geometry the null
    vector of A^T is used; no null vector exists for a full-column-rank
    rectangular nullspace only when n_beams > 3, which holds here.
    """
    radials = np.asarray(radials, dtype=float)
    if geometry.n_beams != 4 or len(radials) != 4:
        return None
    az = [round((a % 360) / 90) * 90 % 360 for a in geometry.azimuths_deg[:4]]
    if sorted(az) == [0, 90, 180, 270]:
        return float((radials[0] + radials[1] - radials[2] - radials[3])
                     / (4.0 * math.cos(geometry.beam_angle_rad)))
    # generic null vector n : A^T n = 0, normalized so sum(abs)=4-ish
    u, s, vh = np.linalg.svd(geometry.design_matrix().T)
    n = u[:, -1]
    if abs(np.linalg.norm(n)) < 1e-12:
        return None
    return float(np.dot(n, radials) * (4.0 / np.sum(np.abs(n))))
