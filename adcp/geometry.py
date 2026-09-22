"""Transducer geometry for a generic Janus ADCP.

Convention: instrument frame x=forward, y=starboard, z=up. A downward-facing
Janus head has beams pointing down, so beam k unit vector is

    u_k = (sin(theta)*cos(phi_k), sin(theta)*sin(phi_k), -cos(theta))

with phi_k the azimuth of beam k. Radial velocities are positive *toward* the
head, hence measured beam velocity v_b = -u_k . v_instrument.

Geometry is a versioned input (beam angle, azimuths, facing); it participates
in the derived fingerprint and is never hard coded into derived results.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

N_BEAMS = 4
DEFAULT_BEAM_ANGLE_DEG = 20.0
# Diagonal Janus: beams over front-right, front-left, aft-left, aft-right.
DEFAULT_AZIMUTHS_DEG = (45.0, 135.0, 225.0, 315.0)


def beam_unit_vectors(beam_angle_deg: float = DEFAULT_BEAM_ANGLE_DEG,
                      azimuths_deg: Sequence[float] = DEFAULT_AZIMUTHS_DEG,
                      facing: str = "down") -> np.ndarray:
    theta = math.radians(beam_angle_deg)
    st, ct = math.sin(theta), math.cos(theta)
    zsign = -1.0 if facing == "down" else 1.0
    rows = []
    for az in azimuths_deg:
        a = math.radians(az)
        rows.append([st * math.cos(a), st * math.sin(a), zsign * ct])
    return np.asarray(rows, dtype=float)  # shape (4,3)


def beam_matrix(geom: dict) -> np.ndarray:
    """Matrix B with v_beam = B @ v_instrument, so B = -unit_vectors."""
    return -beam_unit_vectors(
        geom["beam_angle_deg"], geom.get("azimuths_deg", DEFAULT_AZIMUTHS_DEG),
        geom.get("facing", "down"))


def _beam_space_null(b: np.ndarray) -> np.ndarray:
    """Unit vector in R^4 null(B^T): the redundancy/error direction."""
    u, _, _ = np.linalg.svd(b, full_matrices=True)
    n = u[:, 3]  # null(B^T) lives in the 4th column of U
    n = n / np.linalg.norm(n)
    if n[0] < 0:  # deterministic sign
        n = -n
    return n


def solve_four(beam_vel: Sequence[float], geom: dict) -> dict:
    """Least-squares 4-beam solution plus beam-space error velocity."""
    b = beam_matrix(geom)
    vb = np.asarray(beam_vel, dtype=float)
    vinst, *_ = np.linalg.lstsq(b, vb, rcond=None)
    n = _beam_space_null(b)
    error_velocity = float(vb @ n)
    residual = (vb - b @ vinst).tolist()
    return {
        "velocity_instrument": vinst.tolist(),
        "error_velocity": error_velocity,
        "residual": residual,
        "beams_used": [0, 1, 2, 3],
        "kind": "four_beam",
    }


def solve_three(beam_vel: Sequence[float], used: Sequence[int],
                geom: dict) -> dict:
    """Exact solution from three beams; error velocity is unobservable."""
    b = beam_matrix(geom)
    sel = list(used)
    vb = np.asarray(beam_vel, dtype=float)[sel]
    sub = b[sel]
    if abs(np.linalg.det(sub)) < 1e-9:
        return {"kind": "three_beam", "beams_used": sel,
                "velocity_instrument": None,
                "error_velocity": None,
                "diagnostic": "singular beam triplet"}
    vinst = np.linalg.solve(sub, vb)
    return {
        "velocity_instrument": vinst.tolist(),
        "error_velocity": None,
        "residual": None,
        "beams_used": sel,
        "kind": "three_beam",
        "missing_beam": [k for k in range(N_BEAMS) if k not in sel][0],
    }


def all_triplets() -> List[tuple]:
    return [tuple(k for k in range(N_BEAMS) if k != missing)
            for missing in range(N_BEAMS)]
