"""Transducer beam geometry and beam-velocity inversion.

Geometry is an explicit, versioned input: every coordinate derivation
fingerprint includes the geometry id and its full parameter set, so two
transducer configurations can never silently share one result.

Convention (``trdi-janus4-v1``)
-------------------------------
* Instrument frame: x forward, y starboard, z downward.
* Each beam points *away* from the transducer with a downward component.
* Raw radial velocity is positive when water moves **toward** the
  transducer (classic RDI sign convention), so ``r_b = -v . u_b``.
* Beams are axis-aligned Janus pairs at azimuths 0 / 90 / 180 / 270 deg,
  beam 0..3 corresponding to the classic beam 1 / 2 / 3 / 4 labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fingerprint import fingerprint

#: sin/cos names used in documentation and test expectations.
DEFAULT_GEOMETRY_ID = "trdi-janus4-v1"


@dataclass(frozen=True)
class BeamGeometry:
    geometry_id: str
    beam_angle_deg: float
    azimuths_deg: tuple[float, ...]
    sign_convention: str = "positive_toward_transducer"

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def content(self) -> dict:
        return {
            "geometry_id": self.geometry_id,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
            "sign_convention": self.sign_convention,
            "frame": "instrument: x forward, y starboard, z down",
        }

    def fingerprint(self) -> str:
        return fingerprint("geom", self.content())

    def beam_matrix(self) -> np.ndarray:
        """Rows map instrument velocity to raw radial velocities ``r = B v``."""
        theta = math.radians(self.beam_angle_deg)
        radial_sign = -1.0 if self.sign_convention == "positive_toward_transducer" else 1.0
        rows = []
        for azimuth in self.azimuths_deg:
            phi = math.radians(azimuth)
            away = np.array(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
            rows.append(radial_sign * away)
        return np.vstack(rows)

    def radial_from_instrument(self, velocity_xyz: np.ndarray) -> np.ndarray:
        """Forward model used by fixtures and round-trip tests."""
        return self.beam_matrix() @ np.asarray(velocity_xyz, dtype=float)


def default_geometry() -> BeamGeometry:
    return BeamGeometry(
        geometry_id=DEFAULT_GEOMETRY_ID,
        beam_angle_deg=20.0,
        azimuths_deg=(0.0, 90.0, 180.0, 270.0),
    )


def _null_vector(matrix: np.ndarray) -> np.ndarray:
    """Unit null vector of B.T (radial-space redundancy direction)."""
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    null = vh[-1]
    # Deterministic orientation.
    first_nonzero = next((i for i, value in enumerate(null) if abs(value) > 1e-12), 0)
    if null[first_nonzero] < 0:
        null = -null
    return null


def solve_four_beam(geometry: BeamGeometry, radial: np.ndarray) -> dict:
    """Least-squares instrument velocity plus signed error velocity.

    For an exactly redundant 4x3 geometry the least-squares velocity equals
    the classic mean of the two Janus pair estimates; error velocity is the
    difference between the two redundant vertical estimates, recovered along
    the null space of the beam matrix.
    """
    beam = geometry.beam_matrix()
    radial = np.asarray(radial, dtype=float)
    velocity, *_ = np.linalg.lstsq(beam, radial, rcond=None)
    residual = radial - beam @ velocity
    null = _null_vector(beam)
    error_velocity = float(np.dot(residual, null))
    return {
        "family": "four-beam",
        "velocity_xyz": [float(v) for v in velocity],
        "error_velocity": error_velocity,
        "used_beams": list(range(geometry.n_beams)),
    }


def solve_three_beam_candidates(
    geometry: BeamGeometry, radial: np.ndarray, available: list[int]
) -> list[dict]:
    """All legal exactly-determined solutions over 3-beam subsets.

    Every candidate is kept (see the multi-solution semantics); none is
    averaged or collapsed into the four-beam value.
    """
    from itertools import combinations

    candidates = []
    for triplet in combinations(sorted(available), 3):
        beam = geometry.beam_matrix()[list(triplet), :]
        if abs(np.linalg.det(beam)) < 1e-9:
            continue
        rhs = np.asarray([radial[b] for b in triplet], dtype=float)
        velocity = np.linalg.solve(beam, rhs)
        candidates.append(
            {
                "family": "three-beam",
                "velocity_xyz": [float(v) for v in velocity],
                "error_velocity": None,
                "used_beams": list(triplet),
            }
        )
    return candidates
