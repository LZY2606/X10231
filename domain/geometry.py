"""Beam geometry: transducer configuration as a versioned, fingerprinted object."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    """Janus convex beam geometry.

    beam_order: instrument frame axes per beam (RDI convention):
        b1 -> (+Y, +Z), b2 -> (-Y, +Z), b3 -> (-X, +Z), b4 -> (+X, +Z)
    beam_sign: +1 means positive radial velocity is motion away from transducer.
    """

    beam_angle_deg: float = 20.0
    beam_order: str = "b1:+Y+Z,b2:-Y+Z,b3:-X+Z,b4:+X+Z"
    beam_sign: float = 1.0
    n_beams: int = 4
    name: str = "rdi-janus-convex-20deg"

    def beam_unit_vectors(self) -> np.ndarray:
        """Return (n_beams, 3) unit vectors in instrument XYZ."""
        sa = math.sin(math.radians(self.beam_angle_deg))
        ca = math.cos(math.radians(self.beam_angle_deg))
        axes = {
            "+Y+Z": np.array([0.0, sa, ca]),
            "-Y+Z": np.array([0.0, -sa, ca]),
            "-X+Z": np.array([-sa, 0.0, ca]),
            "+X+Z": np.array([sa, 0.0, ca]),
        }
        vecs = []
        for part in self.beam_order.split(","):
            _, key = part.split(":")
            vecs.append(axes[key])
        return np.array(vecs[: self.n_beams])

    def forward_matrix(self) -> np.ndarray:
        """G such that radial = G @ v_instr (rows are beam unit vectors)."""
        return self.beam_sign * self.beam_unit_vectors()

    def inverse_matrix(self) -> np.ndarray:
        """Least-squares inverse mapping radial -> instrument XYZ."""
        return np.linalg.pinv(self.forward_matrix())

    def null_vector(self) -> np.ndarray:
        """Unit vector spanning the null space of G (error-velocity direction)."""
        g = self.forward_matrix()
        _, _, vt = np.linalg.svd(g)
        return vt[-1]

    def fingerprint(self) -> str:
        payload = {
            "beam_angle_deg": self.beam_angle_deg,
            "beam_order": self.beam_order,
            "beam_sign": self.beam_sign,
            "n_beams": self.n_beams,
            "name": self.name,
        }
        return "geom:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "beam_angle_deg": self.beam_angle_deg,
            "beam_order": self.beam_order,
            "beam_sign": self.beam_sign,
            "n_beams": self.n_beams,
            "name": self.name,
            "fingerprint": self.fingerprint(),
        }

    @staticmethod
    def from_dict(d: Dict) -> "BeamGeometry":
        return BeamGeometry(
            beam_angle_deg=d["beam_angle_deg"],
            beam_order=d["beam_order"],
            beam_sign=d.get("beam_sign", 1.0),
            n_beams=d.get("n_beams", 4),
            name=d.get("name", "custom"),
        )


DEFAULT_GEOMETRY = BeamGeometry()
