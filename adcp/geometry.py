"""Transducer geometry: beam directions, radial projection, least squares.

Frame convention (instrument frame): x forward, y starboard, z upward.
A downward-looking convex Janus head beams point downward (z negative) and
outward. ``radial_sign`` = +1 means positive beam velocity is away from the
transducer along the beam direction; beam radial velocity is therefore

    b_i = s * d_i . v        (s = radial_sign)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransducerConfig:
    config_id: str
    n_beams: int
    beam_angle_deg: float          # angle from vertical axis
    azimuths_deg: tuple[float, ...]  # azimuth of each beam, 0 = +x forward
    convex: bool = True
    radial_sign: float = 1.0
    orientation: str = "down"

    def direction_matrix(self) -> np.ndarray:
        """Return (n_beams, 3) matrix of unit beam vectors (instrument frame)."""
        theta = math.radians(self.beam_angle_deg)
        # angle from the +z axis. Down-looking convex beams: z component -cos.
        z = -math.cos(theta) if (self.convex and self.orientation == "down") else math.cos(theta)
        horiz = math.sin(theta)
        rows = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            rows.append([horiz * math.cos(a), horiz * math.sin(a), z])
        mat = np.asarray(rows, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / norms


def rdi_janus(config_id: str, beam_angle_deg: float = 20.0,
              radial_sign: float = 1.0) -> TransducerConfig:
    """Standard RD-Instruments 4-beam head.

    Azimuths: 45 (fwd-starboard), 135 (aft-starboard), 225 (aft-port),
    315 (fwd-port).
    """
    return TransducerConfig(
        config_id=config_id,
        n_beams=4,
        beam_angle_deg=beam_angle_deg,
        azimuths_deg=(45.0, 135.0, 225.0, 315.0),
        convex=True,
        radial_sign=radial_sign,
        orientation="down",
    )


def solve_radial(cfg: TransducerConfig, radial: np.ndarray,
                 beams: tuple[int, ...]) -> dict:
    """Recover instrument-frame velocity from a set of valid beams.

    3 beams -> exact solve, ``error_velocity`` is None (not estimable).
    4 beams -> least squares plus error velocity projected along the null
    direction of the geometry matrix; the 4-beam value is always kept
    distinct from every 3-beam candidate.
    """
    D = cfg.direction_matrix()
    s = cfg.radial_sign
    idx = list(beams)
    A = s * D[idx, :]
    b = np.asarray(radial, dtype=float)
    if len(idx) == 3:
        v, *_ = np.linalg.lstsq(A, b, rcond=None)
        return {"velocity": v.tolist(), "error_velocity": None, "kind": "3beam"}
    v, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = b - A @ v
    # Null direction of A^T for the symmetric Janus head: (1,-1,1,-1)/2.
    null_dir = np.zeros(len(idx))
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    null_dir[:] = signs[idx]
    null_dir /= np.linalg.norm(null_dir)
    err = float(np.dot(residual, null_dir)) / s
    return {"velocity": v.tolist(), "error_velocity": err, "kind": "4beam"}


def project(cfg: TransducerConfig, v_instrument: np.ndarray) -> list[float]:
    """Forward model: instrument velocity -> beam radial velocities."""
    D = cfg.direction_matrix()
    return (cfg.radial_sign * (D @ np.asarray(v_instrument, dtype=float))).tolist()
