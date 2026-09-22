"""Beam geometry.

A convex Janus ADCP has four beams splayed outward at angle theta from the
instrument vertical.  Each beam has a horizontal azimuth (measured
counter-clockwise from +x in the instrument xy plane) and points downward
(z = -cos theta in a right-handed x,y,z frame with z up).

Sign convention used throughout: radial velocity is positive when the
scatterer moves *along* the beam pointing direction (away from the
transducer).  The projection matrix A therefore maps an instrument-frame
velocity v to beam radials b:  b = A v,  v = A^+ b.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_AZIMUTHS_DEG: Tuple[float, ...] = (45.0, 135.0, 225.0, 315.0)
DEFAULT_BEAM_ANGLE_DEG = 20.0


@dataclass(frozen=True)
class BeamGeometry:
    azimuths_deg: Tuple[float, ...]
    beam_angle_deg: float
    pattern: str = "convex_janus"

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def matrix(self) -> np.ndarray:
        """A with shape (n_beams, 3): radial = A @ velocity."""
        theta = np.deg2rad(self.beam_angle_deg)
        rows = []
        for az in self.azimuths_deg:
            a = np.deg2rad(az)
            rows.append([np.sin(theta) * np.cos(a),
                         np.sin(theta) * np.sin(a),
                         -np.cos(theta)])
        return np.asarray(rows, dtype=float)

    def solve(self, radials: Dict[int, float]) -> Tuple[np.ndarray, float]:
        """Least-squares instrument velocity plus a redundancy measure.

        Beam keys are 1-based (beam 1 is the first azimuth).
        Returns (velocity_xyz, residual_norm).  With exactly three beams the
        residual is always zero (the system is exactly determined, with no
        error-velocity redundancy); four beams give a non-zero residual only
        when the four radials are geometrically inconsistent.
        """
        beams = sorted(b - 1 for b in radials)
        A = self.matrix()[beams]
        b = np.array([radials[k + 1] for k in beams], dtype=float)
        if len(beams) >= 4:
            v, _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
        else:
            v = np.linalg.solve(A, b)
        residual = float(np.linalg.norm(A @ v - b))
        return v, residual

    def error_velocity(self, radials: Sequence[float]) -> float:
        """RDI-style error velocity for a consistent four-beam Janus rig.

        For the 45/135/225/315 layout, the difference between the two
        three-beam vertical solutions is proportional to (b1 - b2 + b3 - b4).
        We instead compute the least-squares residual projected onto the
        null-space of A, which is the geometry-general definition of error
        velocity.
        """
        A = self.matrix()
        b = np.asarray(radials, dtype=float)
        v, _ = self.solve({i: float(x) for i, x in enumerate(b)})
       null_vec = (b - A @ v)
       # null-space of a (4x3) Janus matrix is 1-D
        u, s, vt = np.linalg.svd(A, full_matrices=True)
        null = u[:, -1]
        return float(np.dot(null_vec, null))

    def fingerprint_inputs(self) -> dict:
        return {
            "pattern": self.pattern,
            "azimuths_deg": list(self.azimuths_deg),
            "beam_angle_deg": self.beam_angle_deg,
        }


def geometry_from_config(cfg: dict) -> BeamGeometry:
    return BeamGeometry(
        azimuths_deg=tuple(cfg.get("azimuths_deg", DEFAULT_AZIMUTHS_DEG)),
        beam_angle_deg=float(cfg.get("beam_angle_deg", DEFAULT_BEAM_ANGLE_DEG)),
        pattern=str(cfg.get("pattern", "convex_janus")),
    )
