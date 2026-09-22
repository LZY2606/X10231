"""Janus beam geometry and beam-frame solutions.

Sign convention (documented and fingerprinted, not implicit):

* instrument frame: +x forward (convex beam 3), +y to port (convex beam 4),
  +z upward.
* Beam axis unit vector ``a_k`` points *away* from the transducer (downward).
* Beam velocity ``b_k`` is positive when the water moves *toward* the
  transducer, i.e. ``b = G v`` with ``G = -A`` (rows ``-a_k``).

Two legitimate 3-beam solutions are always kept as separate candidates; they
are never averaged into one "answer":

* ``exact_inverse``     - invert the 3x3 sub-matrix (trust the three beams);
* ``zero_error_velocity``- assume homogeneous flow so RDI error velocity is 0,
                          reconstruct the missing beam, apply the 4-beam rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Default Janus azimuths (degrees, clockwise from +x) for beams 1..4.
# Beam 3 looks forward, beam 1 aft, beam 2 starboard, beam 4 port.
DEFAULT_AZIMUTHS = (180.0, 270.0, 0.0, 90.0)


@dataclass(frozen=True)
class BeamGeometry:
    """Transducer beam configuration.

    ``angle_deg`` is the beam angle from vertical; ``azimuths_deg`` fixes the
    beam numbering around the instrument.
    """

    angle_deg: float = 20.0
    azimuths_deg: Sequence[float] = DEFAULT_AZIMUTHS
    n_beams: int = 4

    def axes(self) -> np.ndarray:
        """Unit vectors along each beam, pointing away from transducer."""
        theta = math.radians(self.angle_deg)
        out = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            out.append(
                [
                    math.sin(theta) * math.cos(a),
                    math.sin(theta) * math.sin(a),
                    -math.cos(theta),
                ]
            )
        return np.asarray(out, dtype=float)

    def matrix(self) -> np.ndarray:
        """``G`` such that ``b = G v`` (positive beam velocity = toward head)."""
        return -self.axes()

    def config_id(self) -> str:
        az = ",".join("%.6g" % a for a in self.azimuths_deg)
        return "janus-%gdeg-[%s]" % (self.angle_deg, az)


def _mask_valid(b: np.ndarray, valid_mask: Sequence[bool]) -> List[int]:
    return [
        k
        for k, ok in enumerate(valid_mask)
        if ok and k < len(b) and np.isfinite(b[k])
    ]


def solve_four(
    geom: BeamGeometry, b: np.ndarray, valid_mask: Sequence[bool]
) -> Optional[dict]:
    """Four-beam least-squares instrument-frame solution.

    Returns vx, vy, vz (m/s), RDI error velocity and the beams used; None when
    fewer than four beams are valid.
    """
    g = geom.matrix()
    used = _mask_valid(b, valid_mask)
    if len(used) != geom.n_beams:
        return None
    bv = np.asarray(b, dtype=float)
    gtgi = np.linalg.inv(g.T @ g)
    v = gtgi @ (g.T @ bv)
    # normalized left-null covector of the symmetric Janus layout
    q = np.array([0.5, -0.5, 0.5, -0.5])
    err = float(q @ bv)
    return {
        "v": [float(x) for x in v],
        "error_velocity": err,
        "used_beams": used,
        "method": "four_beam_lsq",
    }


def solve_three_candidates(
    geom: BeamGeometry, b: np.ndarray, valid_mask: Sequence[bool]
) -> List[dict]:
    """All legitimate 3-beam solutions; two distinct assumptions, both kept."""
    g = geom.matrix()
    used = _mask_valid(b, valid_mask)
    bv = np.asarray(b, dtype=float)
    if len(used) != 3:
        return []
    missing = [k for k in range(geom.n_beams) if k not in used][0]

    # Candidate A: exact inverse of the surviving 3x3 block.
    g3 = g[used, :]
    v_exact = np.linalg.solve(g3, bv[used])

    # Candidate B: zero error velocity -> reconstruct missing beam.
    # e = (b1-b2+b3-b4)/2 = 0 => missing radial from the other three.
    b_full = bv.copy()
    # null covector q with +/- 1/2 signs: missing = sum over q_known/q_missing
    q = np.array([0.5, -0.5, 0.5, -0.5])
    b_full[missing] = -sum(q[k] * bv[k] for k in used) / q[missing]
    gtgi = np.linalg.inv(g.T @ g)
    v_zero = gtgi @ (g.T @ b_full)
    err_zero = float(q @ b_full)

    return [
        {
            "assumption": "exact_inverse",
            "v": [float(x) for x in v_exact],
            "error_velocity": None,
            "used_beams": list(used),
            "missing_beam": missing,
            "reconstructed_beams": None,
        },
        {
            "assumption": "zero_error_velocity",
            "v": [float(x) for x in v_zero],
            "error_velocity": err_zero,
            "used_beams": list(used),
            "missing_beam": missing,
            "reconstructed_beams": [float(x) for x in b_full],
        },
    ]


def count_valid(b: np.ndarray, valid_mask: Sequence[bool]) -> int:
    return len(_mask_valid(b, valid_mask))
