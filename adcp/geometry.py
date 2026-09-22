from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def as_vector3(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("expected a length-3 vector")
    return arr


def rot_x(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ordered_rotation(order: str, angles: Dict[str, float]) -> np.ndarray:
    """Return rotations applied in the explicit named order.

    ``heading_pitch_roll`` means a vector is transformed by roll, then pitch,
    then heading when matrices act on column vectors.
    """
    factories = {"heading": rot_z, "pitch": rot_y, "roll": rot_x}
    names = order.split("_")
    if len(names) != 3 or set(names) != {"heading", "pitch", "roll"}:
        raise ValueError("rotation order must contain heading, pitch and roll")
    result = np.eye(3)
    for name in names:
        result = factories[name](angles[name]) @ result
    return result


def beam_unit_vector(angle_deg: float, azimuth_deg: float, sign: float = 1.0) -> np.ndarray:
    angle = math.radians(angle_deg)
    azimuth = math.radians(azimuth_deg)
    horizontal = math.sin(angle)
    return np.array(
        [
            sign * horizontal * math.cos(azimuth),
            sign * horizontal * math.sin(azimuth),
            sign * math.cos(angle),
        ]
    )


def geometry_matrix(beams: Sequence[Dict[str, object]]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for beam in beams:
        rows.append(
            beam_unit_vector(
                float(beam["angle_deg"]),
                float(beam["azimuth_deg"]),
                float(beam.get("sign", 1.0)),
            )
        )
    return np.vstack(rows)


def solve_radials(matrix: np.ndarray, radials: np.ndarray) -> Tuple[np.ndarray, float]:
    if matrix.shape[0] != radials.shape[0] or matrix.shape[1] != 3:
        raise ValueError("radial matrix/vector shape mismatch")
    solution, residuals, rank, _ = np.linalg.lstsq(matrix, radials, rcond=None)
    residual = float(np.linalg.norm(matrix @ solution - radials))
    if rank < 3:
        raise np.linalg.LinAlgError("beam geometry is rank deficient")
    return solution, residual


def wrap_360(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def interpolate_heading(left_t: float, left: float, right_t: float, right: float, t: float) -> float:
    if right_t == left_t:
        return wrap_360(left)
    fraction = (t - left_t) / (right_t - left_t)
    delta = ((right - left + 180.0) % 360.0) - 180.0
    return wrap_360(left + fraction * delta)


def interpolate_linear(left_t: float, left: float, right_t: float, right: float, t: float) -> float:
    if right_t == left_t:
        return float(left)
    return float(left + (right - left) * (t - left_t) / (right_t - left))
