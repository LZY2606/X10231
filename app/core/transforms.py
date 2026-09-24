"""ADCP coordinate transforms (beam -> instrument -> ship -> earth).

Conventions (fixed so geometry is reproducible):
  * axes: x forward, y starboard, z DOWN (marine ADCP convention).
  * beam radial velocity is positive for water moving TOWARD the transducer.
    ``beam_signs`` (default -1) converts radial velocities to water velocity.
  * 4 convex beams at angle ``theta`` from vertical, azimuths measured
    clockwise (x-y plane) starting at ``first_azimuth``.

The transform *order* (heading/pitch/roll composition) and the beam geometry
are both versioned inputs: see ``geometry_fingerprint`` /
``order_fingerprint``. Every derived fingerprint includes them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

SUPPORTED_ORDERS = ("HPR_RDI", "RPH_EXTRINSIC", "HP_EXTRINSIC")


@dataclass(frozen=True)
class TransducerGeometry:
    beam_count: int = 4
    theta_deg: float = 20.0          # beam angle from vertical
    first_azimuth_deg: float = 45.0  # beam 1 azimuth, clockwise from +x
    azimuth_step_deg: float = 90.0
    beam_signs: tuple = (-1.0, -1.0, -1.0, -1.0)

    def unit_vectors(self) -> np.ndarray:
        """Return (beam_count, 3) unit vectors along each beam, z down."""
        theta = math.radians(self.theta_deg)
        vectors = []
        for i in range(self.beam_count):
            az = math.radians(self.first_azimuth_deg + i * self.azimuth_step_deg)
            vectors.append(
                [math.sin(theta) * math.cos(az),
                 math.sin(theta) * math.sin(az),
                 math.cos(theta)]
            )
        return np.asarray(vectors, dtype=float)

    def beam_matrix(self) -> np.ndarray:
        """Matrix A mapping water velocity to measured radial velocity.

        measured = A @ v  (per beam: u_i = sign * dot(beam_unit, v)).
        """
        units = self.unit_vectors()
        signs = np.asarray(self.beam_signs[: self.beam_count], dtype=float)
        return units * signs[:, None]

    def fingerprint(self) -> str:
        units = np.round(self.unit_vectors(), 9)
        return (
            f"geom b={self.beam_count} theta={self.theta_deg}"
            f" az0={self.first_azimuth_deg} step={self.azimuth_step_deg}"
            f" signs={self.beam_signs} vec={units.tolist()}"
        )


# ---------------------------------------------------------------------------
# Rotation matrices (right-handed x-forward / y-starboard / z-DOWN)
# ---------------------------------------------------------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(b: float) -> np.ndarray:
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(g: float) -> np.ndarray:
    c, s = math.cos(g), math.sin(g)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def order_fingerprint(order: str) -> str:
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"unsupported transform order {order!r}; supported {SUPPORTED_ORDERS}")
    return f"order={order}"


def earth_from_ship(hpr: Sequence[float], order: str = "HPR_RDI") -> np.ndarray:
    """Rotation taking a ship-frame vector into earth frame.

    Angles are radians. ``order`` is an explicit versioned rule parameter.
    """
    heading, pitch, roll = (float(v) for v in hpr)
    if order == "HPR_RDI":
        # Classical RDI marine sequence: Rz(H) Ry(P) Rx(R), z-down.
        return rot_z(heading) @ rot_y(pitch) @ rot_x(roll)
    if order == "RPH_EXTRINSIC":
        return rot_x(roll) @ rot_y(pitch) @ rot_z(heading)
    if order == "HP_EXTRINSIC":
        return rot_x(0.0) @ rot_y(pitch) @ rot_z(heading)
    raise ValueError(f"unsupported transform order {order!r}")


def ship_from_instrument(mount: Sequence[float]) -> np.ndarray:
    """Mount/installation angles [dx, dy, dz] degrees -> rotation matrix."""
    dx, dy, dz = (math.radians(float(v)) for v in mount)
    return rot_z(dz) @ rot_y(dy) @ rot_x(dx)


def beam_to_instrument(radial: np.ndarray, geom: TransducerGeometry) -> dict:
    """Convert 3-4 radial beam velocities to instrument-frame candidates.

    Returns:
      {"candidates": [ {method, missing_beam, velocity:[u,v,w], residual, ...} ],
       "quality": {...}}

    Four good beams give a least-squares solution PLUS every legal 3-beam
    solution; three good beams give the exact 3-beam solution. Candidates are
    never collapsed into one value when more than one legal solution exists.
    """
    radial = np.asarray(radial, dtype=float)
    good = [i for i, v in enumerate(radial) if not math.isnan(v)]
    a_full = geom.beam_matrix()
    candidates: list[dict] = []

    if len(good) >= 4:
        velocity, *_ = np.linalg.lstsq(a_full, radial, rcond=None)
        predicted = a_full @ velocity
        candidates.append({
            "method": "four_beam_lsq",
            "missing_beam": None,
            "velocity": [float(x) for x in velocity],
            "residual": float(np.linalg.norm(predicted - radial)),
        })
        # also retain the 3-beam alternatives implied by the same ping
        for miss in range(geom.beam_count):
            idx = [i for i in range(geom.beam_count) if i != miss]
            sub_a = a_full[idx]
            sub_u = radial[idx]
            velocity = np.linalg.solve(sub_a, sub_u)
            candidates.append({
                "method": "three_beam_exact",
                "missing_beam": miss,
                "velocity": [float(x) for x in velocity],
                "residual": 0.0,
            })
    elif len(good) == 3:
        missing = [i for i in range(geom.beam_count) if i not in good][0]
        idx = good
        velocity = np.linalg.solve(a_full[idx], radial[idx])
        candidates.append({
            "method": "three_beam_exact",
            "missing_beam": missing,
            "velocity": [float(x) for x in velocity],
            "residual": 0.0,
        })

    # error velocity: spread of independent w estimates across candidates
    w_values = [c["velocity"][2] for c in candidates]
    quality = {
        "good_beams": good,
        "n_good": len(good),
        "w_spread": float(max(w_values) - min(w_values)) if w_values else None,
        "ambiguous": sum(c["method"] == "three_beam_exact" for c in candidates) > 1,
    }
    return {"candidates": candidates, "quality": quality}
