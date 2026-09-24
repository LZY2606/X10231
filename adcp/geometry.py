"""Beam geometry and beam-radial -> instrument-frame velocity solutions.

Convention (documented and versioned in the geometry fact):
  * instrument axes: x points from beam 3 toward beam 4, y from beam 1 toward
    beam 2, z up;
  * 4-beam Janus convex configuration, beam i has horizontal azimuth phi_i
    measured from +y toward +x, and beam angle theta from horizontal;
  * beam look unit vector n_i = (cos(theta) sin(phi_i),
    cos(theta) cos(phi_i), cos(theta vertical sign); beam radial velocity is
    b_i = v . n_i (positive along the outward look direction).

Three-beam ambiguity is never collapsed: when one beam fails the system
records BOTH legal candidates (minimal-norm 3x3 least squares, and the
opposite-pair solution under the explicit w=0 assumption) and never lets a
three-beam value be mistaken for the four-beam solution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_BEAM_AZIMUTH_DEG = (45.0, 135.0, 225.0, 315.0)
DEFAULT_BEAM_ANGLE_DEG = 20.0


def look_matrix(beam_angle_deg: float, azimuths_deg=DEFAULT_BEAM_AZIMUTH_DEG):
    """Rows are beam look unit vectors n_i in instrument coordinates."""
    theta = math.radians(beam_angle_deg)
    rows = []
    for phi_deg in azimuths_deg:
        phi = math.radians(phi_deg)
        rows.append((
            math.cos(theta) * math.sin(phi),
            math.cos(theta) * math.cos(phi),
            math.sin(theta),
        ))
    return np.asarray(rows, dtype=float)


@dataclass(frozen=True)
class BeamSolution:
    family: str            # "four_beam", "three_beam_lsq", "three_beam_wzero"
    velocity: np.ndarray   # instrument-frame (x, y, z), m/s
    error_velocity: float | None  # m/s, only defined for the four-beam family
    used_beams: tuple[int, ...]   # 0-based beam indices
    assumption: str | None
    residual: float


def _opposite_pair_index(g: np.ndarray) -> tuple[int, int]:
    """Default Janus indexing: beams (0,2) and (1,3) face each other."""
    return 0, 2


def _null_error_velocity(g: np.ndarray, b: np.ndarray, v: np.ndarray) -> float:
    """RDI-style error velocity: difference of the two redundant w estimates.

    Opposite beams have equal vertical direction cosine and opposite
    horizontal ones, so w_pair = (b_i + b_j) / (2 * n_z).
    """
    i, k = _opposite_pair_index(g)
    j, m = 1, 3
    w_a = float((b[i] + b[k]) * 0.5 / g[i, 2])
    w_b = float((b[j] + b[m]) * 0.5 / g[j, 2])
    return w_a - w_b


def solve_four_beam(g: np.ndarray, b: np.ndarray) -> BeamSolution:
    """Full 4x3 over-determined solution; all four beams required."""
    v, *_ = np.linalg.lstsq(g, b, rcond=None)
    residual = float(np.linalg.norm(g @ v - b))
    err = _null_error_velocity(g, b, v)
    return BeamSolution("four_beam", v, err, (0, 1, 2, 3), None, residual)


def _intact_opposite_pair(g: np.ndarray, good: list[int]) -> tuple[int, int] | None:
    """An opposite pair faces each other: equal vertical direction cosine and
    opposite horizontal look components (z components are NOT negated in the
    upward-slanting Janus convex geometry)."""
    for i in good:
        for j in good:
            if j <= i:
                continue
            horiz_i, horiz_j = g[i].copy(), g[j].copy()
            horiz_i[2] = horiz_j[2] = 0.0
            if (np.isclose(g[i, 2], g[j, 2], atol=1e-9)
                    and np.allclose(horiz_i, -horiz_j, atol=1e-9)):
                return i, j
    return None


def solve_three_beam(g: np.ndarray, b: np.ndarray, good: list[int]) -> list[BeamSolution]:
    """Return every legal 3-beam candidate; empty if fewer than 3 good beams."""
    good = sorted(good)
    if len(good) < 3:
        return []
    g3 = g[good]
    b3 = b[good]

    # Candidate A: minimal-norm least squares on the three available rows.
    va, *_ = np.linalg.lstsq(g3, b3, rcond=None)
    res_a = float(np.linalg.norm(g3 @ va - b3))
    cand_a = BeamSolution(
        "three_beam_lsq", va, None, tuple(good),
        "minimal-norm fit to 3 radial measurements", res_a,
    )

    # Candidate B: opposite-pair horizontal component + w = 0 assumption.
    pair = _intact_opposite_pair(g, good)
    if pair is None:
        return [cand_a]
    i, j = pair
    v = np.zeros(3)
    # Opposite pair: b_i - b_j = 2 n_horiz . v_h fixes one horizontal axis;
    # b_i + b_j = 2 n_z w. Under the explicit w=0 assumption a third beam
    # then fixes the orthogonal horizontal component.
    n = g[i]
    horiz = n.copy()
    horiz[2] = 0.0
    axis_component = (b[i] - b[j]) / (2.0 * np.linalg.norm(horiz))
    axis_unit = horiz / np.linalg.norm(horiz)
    v += axis_unit * axis_component  # one horizontal component fixed
    v[2] = 0.0
    # Any third beam fixes the orthogonal horizontal component.
    third = next(k for k in good if k not in pair)
    rest = b[third] - float(np.dot(g[third], v))
    perp = g[third].copy()
    perp -= axis_unit * float(np.dot(g[third], axis_unit))
    perp[2] = 0.0
    # With w fixed at zero, only the horizontal projection drives the solve;
    # subtract the vertical contribution the third beam would otherwise leak.
    rest -= g[third, 2] * v[2]
    if np.linalg.norm(perp) > 1e-12:
        perp_unit = perp / np.linalg.norm(perp)
        v += perp_unit * (rest / float(np.dot(g[third], perp_unit)))
    res_b = float(np.linalg.norm(g3 @ v - b3))
    cand_b = BeamSolution(
        "three_beam_wzero", v, None, tuple(good),
        "opposite-pair horizontal component with explicit w=0 assumption",
        res_b,
    )
    return [cand_a, cand_b]


def solve(g: np.ndarray, b: np.ndarray, bad_beams: list[int]):
    """Dispatch to the 4-beam or 3-beam family. Never mixes the families."""
    good = [i for i in range(len(g)) if i not in bad_beams]
    if len(good) == len(g):
        return [solve_four_beam(g, b)]
    return solve_three_beam(g, b, good)
