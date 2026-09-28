"""Beam geometry and beam -> instrument frame solutions.

Supports the classic 4-beam RDI-style Janus head. A solution is produced in
either frame mode:

* 4-beam: one over-determined instrument velocity + an *error velocity*
  (projection onto the Janus null direction). This is a distinct value; it is
  never merged with a 3-beam solution.
* 3-beam: with one beam missing/invalid every choice of the 3 surviving beams
  is an exact but frame-biased solution. We keep ALL legal candidates rather
  than collapsing them into one answer.

Beam radial direction convention (documented and fingerprinted, so a different
convention is a different derivation):

    r_k = [sin(theta) sin(az_k), sin(theta) cos(az_k), -cos(theta)]

i.e. positive beam velocity is flow *away from the transducer* along the
outward radial; instrument axes are x forward, y starboard, z up.
"""
from __future__ import annotations

import hashlib
import json
import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    """Transducer beam configuration."""

    beam_angle_deg: float          # theta: angle from vertical
    azimuths_deg: tuple[float, ...]  # azimuth of each beam, CW from +x
    convention: str = "positive-away"
    coord_frame: str = "x-forward,y-starboard,z-up"
    id: str = "geom-rdi-janus-20"

    @property
    def nbeams(self) -> int:
        return len(self.azimuths_deg)

    def radial_matrix(self) -> np.ndarray:
        theta = np.deg2rad(self.beam_angle_deg)
        rows = []
        for az in self.azimuths_deg:
            a = np.deg2rad(az)
            rows.append([np.sin(theta) * np.sin(a),
                         np.sin(theta) * np.cos(a),
                         -np.cos(theta)])
        return np.asarray(rows, dtype=float)

    def matrix4(self) -> np.ndarray:
        """4x4 transform [u,v,w,err] = matrix4 @ beam."""
        r = self.radial_matrix()
        g = r[:, :3]
        pinv = np.linalg.pinv(g)                 # 3x4 least-squares rows
        null = self.null_vector()
        return np.vstack([pinv, null[np.newaxis, :]])

    def null_vector(self) -> np.ndarray:
        """Unit vector n with g.T-like null: radials projected sum zero."""
        n = np.array([1.0, -1.0, 1.0, -1.0])
        # Normalise so that err = n . beam carries the documented scale.
        theta = np.deg2rad(self.beam_angle_deg)
        return n / (4.0 * np.cos(theta))

    def solve4(self, beam: np.ndarray) -> dict:
        """4-beam solution. Returns u,v,w and error velocity."""
        m = self.matrix4()
        u, v, w, err = m @ np.asarray(beam, float)
        # least-squares residual cross-check
        g = self.radial_matrix()
        pred = g @ np.array([u, v, w])
        residual = float(np.linalg.norm(np.asarray(beam, float) - pred))
        return {"u": float(u), "v": float(v), "w": float(w),
                "error_velocity": float(err), "ls_residual": residual,
                "mode": "4-beam"}

    def solve3_candidates(self, beam: Sequence[Optional[float]],
                          valid: Sequence[bool]) -> list[dict]:
        """All legal exact solutions using any 3 of the surviving beams."""
        g = self.radial_matrix()
        idxs = [i for i, ok in enumerate(valid) if ok and beam[i] is not None]
        if len(idxs) < 3:
            return []
        cands = []
        for combo in itertools.combinations(idxs, 3):
            sub = g[list(combo), :]
            if abs(np.linalg.det(sub)) < 1e-9:
                continue
            vals = np.array([beam[i] for i in combo], float)
            u, v, w = np.linalg.solve(sub, vals)
            missing = tuple(i for i in range(self.nbeams) if i not in combo)
            cands.append({
                "mode": "3-beam",
                "beams": list(combo),
                "missing_beams": list(missing),
                "u": float(u), "v": float(v), "w": float(w),
                "error_velocity": None,
                "candidate_id": "b" + "".join(str(i + 1) for i in combo),
                "ambiguous": False,
            })
        # when exactly 3 beams survive there is a single candidate; with a
        # fourth marginal beam we may emit several -> genuinely multi-valued
        return cands

    def fingerprint_inputs(self) -> dict:
        return {
            "geometry_id": self.id,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
            "convention": self.convention,
            "coord_frame": self.coord_frame,
        }


DEFAULT_JANUS = BeamGeometry(
    beam_angle_deg=20.0,
    azimuths_deg=(0.0, 90.0, 180.0, 270.0),
)


def fingerprint(prefix: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return prefix + "_" + hashlib.sha256(blob.encode()).hexdigest()[:16]
