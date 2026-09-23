"""Deterministic ADCP beam geometry and attitude transformations.

The conventions intentionally live in this pure module so every derived value
can explain which beam geometry, attitude, magnetic correction, and rotation
order produced it.  Angles in the public fixture records remain in their
original unit.  This layer accepts normalized degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TransformOrder = Literal["heading_pitch_roll", "pitch_roll_heading"]


@dataclass(frozen=True)
class BeamGeometry:
    """Four-beam Janus geometry with explicit units and axes."""

    beam_angle_deg: float
    convex: bool = True

    @property
    def fingerprint(self) -> str:
        sign = "convex" if self.convex else "concave"
        return f"beam4x/{self.beam_angle_deg:.6f}deg/{sign}"

    def matrix(self) -> np.ndarray:
        theta = np.deg2rad(self.beam_angle_deg)
        s = np.sin(theta)
        # Beam radial coordinates: x, x, y, y.  Opposite convex beams have
        # opposite horizontal signs.  b_i is water velocity projected on the
        # beam axis; a positive radial velocity is away from the transducer.
        if self.convex:
            return np.array(
                [
                    [-s, 0.0, np.cos(theta)],
                    [s, 0.0, np.cos(theta)],
                    [0.0, -s, np.cos(theta)],
                    [0.0, s, np.cos(theta)],
                ],
                dtype=float,
            )
        return np.array(
            [
                [s, 0.0, np.cos(theta)],
                [-s, 0.0, np.cos(theta)],
                [0.0, s, np.cos(theta)],
                [0.0, -s, np.cos(theta)],
            ],
            dtype=float,
        )


def wrap_360(degrees: float) -> float:
    return float(degrees % 360.0)


def wrapped_delta(to_deg: float, from_deg: float) -> float:
    """Smallest signed angular change, correctly crossing 0/360."""

    return float(((to_deg - from_deg + 180.0) % 360.0) - 180.0)


def interpolate_attitude(
    before: dict[str, float],
    after: dict[str, float],
    fraction: float,
) -> dict[str, float]:
    """Linearly interpolate heading with circular unwrapping.

    Caller is responsible for deciding whether a temporal gap is bridgeable.
    This function never decides to cross a missing attitude sample.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("attitude interpolation fraction must be in [0, 1]")
    return {
        "heading_deg": wrap_360(
            before["heading_deg"]
            + fraction * wrapped_delta(after["heading_deg"], before["heading_deg"])
        ),
        "pitch_deg": before["pitch_deg"]
        + fraction * (after["pitch_deg"] - before["pitch_deg"]),
        "roll_deg": before["roll_deg"]
        + fraction * (after["roll_deg"] - before["roll_deg"]),
    }


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def attitude_matrix(
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: TransformOrder = "heading_pitch_roll",
) -> np.ndarray:
    """Instrument to ship/earth rotation for column vectors.

    The canonical order is heading <- pitch <- roll (Rz * Ry * Rx).  Keeping
    the alternate product makes rotation-order mistakes visible in fingerprints
    and tests instead of silently changing velocities.
    """

    r_h = _rotation_z(heading_deg)
    r_p = _rotation_y(pitch_deg)
    r_r = _rotation_x(roll_deg)
    if order == "heading_pitch_roll":
        return r_h @ r_p @ r_r
    if order == "pitch_roll_heading":
        return r_r @ r_p @ r_h
    raise ValueError(f"unsupported transform order: {order}")


def corrected_heading_deg(
    heading_deg: float,
    declination_deg: float,
) -> float:
    """Convert magnetic heading to true heading.

    Declination is positive east.  The signed source value is retained in
    storage; only the derivation uses this normalized convention.
    """

    return wrap_360(heading_deg + declination_deg)


def solve_beams(
    beam_velocity_m_s: list[float],
    geometry: BeamGeometry,
    bad_beams: list[bool] | None = None,
) -> dict[str, object]:
    """Solve a four- or three-beam velocity equation.

    A known bad beam is excluded.  Three-beam ambiguity is handled by the
    caller generating every admissible missing beam; this function never folds
    those distinct candidates into one value.
    """

    values = np.asarray(beam_velocity_m_s, dtype=float)
    if values.shape != (4,):
        raise ValueError("exactly four beam velocities are required")
    flags = list(bad_beams or [False] * 4)
    if len(flags) != 4:
        raise ValueError("bad_beams must contain four flags")
    valid = [index for index, bad in enumerate(flags) if not bad]
    if len(valid) == 4:
        matrix = geometry.matrix()
        velocity, residuals, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
        predicted = matrix @ velocity
        residual_rms = float(np.sqrt(np.mean((values - predicted) ** 2)))
        return {
            "kind": "four_beam",
            "velocity_m_s": [float(x) for x in velocity],
            "beams_used": [1, 2, 3, 4],
            "missing_beam": None,
            "residual_rms_m_s": residual_rms,
            "matrix_rank": int(rank),
        }
    if len(valid) == 3:
        missing = next(i for i, bad in enumerate(flags) if bad) + 1
        matrix = geometry.matrix()[valid, :]
        velocity, residuals, rank, _ = np.linalg.lstsq(
            matrix, values[valid], rcond=None
        )
        predicted = matrix @ velocity
        residual_rms = float(np.sqrt(np.mean((values[valid] - predicted) ** 2)))
        return {
            "kind": "three_beam",
            "velocity_m_s": [float(x) for x in velocity],
            "beams_used": [index + 1 for index in valid],
            "missing_beam": missing,
            "residual_rms_m_s": residual_rms,
            "matrix_rank": int(rank),
        }
    return {
        "kind": "unresolved",
        "velocity_m_s": None,
        "beams_used": [index + 1 for index in valid],
        "missing_beam": None,
        "residual_rms_m_s": None,
        "matrix_rank": 0,
        "diagnostic": "fewer than three usable beams",
    }


def transform_vector(matrix: np.ndarray, vector_m_s: list[float]) -> list[float]:
    return [float(x) for x in matrix @ np.asarray(vector_m_s, dtype=float)]


def valid_cell_depth(depth_m: float, blanking_m: float, cell_size_m: float) -> bool:
    return depth_m >= blanking_m and depth_m > 0.0 and cell_size_m > 0.0
