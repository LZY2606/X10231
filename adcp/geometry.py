"""Beam geometry and the beam -> instrument velocity inversion.

Conventions (documented here on purpose; they are part of the derivation
fingerprint):

* Instrument frame: x forward, y starboard, z down (right-handed).
* Beam radial velocity b_i is positive *away* from the transducer.
* Each beam points down-and-out at beam angle theta from vertical, at an
  azimuth phi_i measured clockwise from +x (forward) toward +y (starboard).
  The default geometry is the classic four-beam Janus arrangement:
  phi = (45, 135, 225, 315) degrees.
* v_along_beam = u . e_beam, with
  e_beam = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GEOMETRY_VERSION = "geom-janus4-v1"


@dataclass(frozen=True)
class Geometry:
    angle_deg: float
    azimuths_deg: tuple[float, ...]
    version: str = GEOMETRY_VERSION

    def beam_matrix(self) -> np.ndarray:
        """Return A with one row of unit-vector components per beam."""
        theta = math.radians(self.angle_deg)
        rows = []
        for phi_deg in self.azimuths_deg:
            phi = math.radians(phi_deg)
            rows.append(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
        return np.asarray(rows, dtype=float)


DEFAULT_GEOMETRY = Geometry(angle_deg=20.0, azimuths_deg=(45.0, 135.0, 225.0, 315.0))


def invert(b: np.ndarray, geom: Geometry) -> dict:
    """Least-squares invert beam velocities ``b`` (m/s, positive away).

    Returns instrument-frame velocity (vx forward, vy starboard, vz down),
    the normalised LS residual ("error velocity"), fit residual norm, and the
    projection matrix used.  ``b`` may contain NaN for failed beams; valid
    beams are selected automatically.
    """
    b = np.asarray(b, dtype=float)
    valid = ~np.isnan(b)
    a = geom.beam_matrix()[valid]
    bb = b[valid]
    velocity, *_ = np.linalg.lstsq(a, bb, rcond=None)
    fit = a @ velocity
    residual = bb - fit
    # Unit vector in the beam-valid subspace orthogonal to range(A): the
    # projection of residual onto it is the error-velocity analogue.
    proj = a @ np.linalg.pinv(a)
    null_vec = residual - proj @ residual
    norm = float(np.linalg.norm(null_vec))
    error_velocity = float(norm)  # magnitude; signedness would need a chosen beam basis
    return {
        "vx": float(velocity[0]),
        "vy": float(velocity[1]),
        "vz": float(velocity[2]),
        "error_velocity": error_velocity,
        "residual_norm": float(np.linalg.norm(residual)),
        "valid_beams": [int(i) for i, ok in enumerate(valid) if ok],
    }


def classic_error_velocity(b: np.ndarray, geom: Geometry) -> float:
    """Classic Janus error velocity from the two opposing-beam vz estimates.

    b = b1,b2,b3,b4 ; beam pairs (1,3) and (2,4) are opposing.
    vz_pair = (b_i + b_j)/(2 cos theta); e = (vz13 - vz24)/2.
    Requires all four beams finite.
    """
    b = np.asarray(b, dtype=float)
    if np.isnan(b).any():
        return float("nan")
    c = math.cos(math.radians(geom.angle_deg))
    vz13 = (b[0] + b[2]) / (2.0 * c)
    vz24 = (b[1] + b[3]) / (2.0 * c)
    return float(0.5 * (vz13 - vz24))
