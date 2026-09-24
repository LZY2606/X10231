"""Beam geometry for the convex Janus transducer configuration.

Convention used throughout this package (documented, and versioned into the
derived fingerprint as ``beam-geom@janus-v1``):

* instrument frame: x = bow/forward, y = starboard, z = down, right handed.
* beam radial velocity is positive for water moving *away* from the
  transducer along the beam unit vector (the RDI-style outward-positive
  convention).
* beams 1/3 are a fore-aft pair, beams 2/4 a starboard-port pair; a Janus
  beam at angle ``theta`` from vertical points outward by 90-theta degrees.
* b_k = unit vector of beam k; measured component  r_k = b_k . v_instrument.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GEOMETRY_RULE = "beam-geom@janus-v1"

# Beam azimuths in the instrument frame (radians, measured clockwise from
# x/forward toward y/starboard): beam1 fore, beam2 starboard, beam3 aft,
# beam4 port.  Beam 3 points aft (-x) and beam 4 port (-y).
_BEAM_AZIMUTH = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)


@dataclass(frozen=True)
class BeamGeometry:
    """Transducer geometry with all angles in radians."""

    beam_angle_rad: float  # angle of each beam from vertical (z-down)
    n_beams: int = 4
    azimuths_rad: tuple[float, ...] = _BEAM_AZIMUTH

    @property
    def rule(self) -> str:
        return GEOMETRY_RULE

    def directions(self) -> np.ndarray:
        """Return shape (n_beams, 3): b_k rows in instrument frame.

        Fore/aft beams tilt toward +-x; starboard/port toward +-y.
        """
        theta = self.beam_angle_rad
        tilt = math.pi / 2.0 - theta  # outward tilt from horizontal
        horiz = math.cos(tilt)  # = sin(theta)
        vert = math.sin(tilt)  # = cos(theta), downward component
        rows = []
        for az in self.azimuths_rad[: self.n_beams]:
            rows.append(
                [
                    horiz * math.cos(az),
                    horiz * math.sin(az),
                    vert,
                ]
            )
        return np.asarray(rows, dtype=float)

    def B(self) -> np.ndarray:
        """Full beam matrix (n_beams x 3)."""
        return self.directions()


def janus_matrix(beam_angle_deg: float) -> np.ndarray:
    return BeamGeometry(math.radians(beam_angle_deg)).B()


def selected_matrix(geom: BeamGeometry, beams: tuple[int, ...]) -> np.ndarray:
    """Rows of the beam matrix for a sorted tuple of 0-based beam indices."""
    full = geom.B()
    return np.vstack([full[k] for k in sorted(beams)])


def solve_radial(B_sel: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Invert ``r = B_sel v`` by least squares.

    Returns ``(v_instrument, error_velocity)``.  ``error_velocity`` is the
    signed residual norm ``||r - B v||`` (None when the system is exactly
    determined, i.e. for a 3-beam solution, which cannot self-diagnose).
    """
    r = np.asarray(r, dtype=float)
    v, *_ = np.linalg.lstsq(B_sel, r, rcond=None)
    err: float | None = None
    if B_sel.shape[0] > B_sel.shape[1]:
        resid = r - B_sel @ v
        err = float(np.linalg.norm(resid))
    return v, err


def threebeam_inverse(B3: np.ndarray) -> np.ndarray:
    """Exact inverse for a non-singular 3x3 beam matrix."""
    if B3.shape != (3, 3):
        raise ValueError("three-beam inverse requires a 3x3 matrix")
    return np.linalg.inv(B3)
