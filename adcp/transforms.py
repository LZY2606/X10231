from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


BEAM_FRAME = "beam_radial"
INSTRUMENT_FRAME = "instrument_xyz"
SHIP_FRAME = "ship_xyz"
EARTH_FRAME = "enu"


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def beam_geometry(beam_angle_deg: float, beam_azimuth_deg: Sequence[float]) -> np.ndarray:
    """Return rows mapping instrument XYZ velocity to positive-away radial velocity."""
    theta = math.radians(beam_angle_deg)
    rows = []
    for azimuth_deg in beam_azimuth_deg:
        azimuth = math.radians(azimuth_deg)
        rows.append(
            [
                math.sin(theta) * math.cos(azimuth),
                math.sin(theta) * math.sin(azimuth),
                math.cos(theta),
            ]
        )
    return np.asarray(rows, dtype=float)


def solve_beams(matrix: np.ndarray, radial: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve a full-column beam subset. Residual is zero for three beams."""
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("beam geometry must be n by 3")
    if matrix.shape[0] < 3:
        raise ValueError("at least three beams are required")
    velocity, residuals, rank, _ = np.linalg.lstsq(matrix, radial, rcond=None)
    if rank < 3:
        raise ValueError("beam subset geometry is rank deficient")
    residual = float(np.linalg.norm(matrix @ velocity - radial)) if matrix.shape[0] > 3 else 0.0
    return velocity, residual


def mount_rotation(mount: dict[str, Any]) -> np.ndarray:
    order = mount.get("rotation_order", "ZYX")
    angles = [
        math.radians(float(mount.get("yaw_offset_deg", 0.0))),
        math.radians(float(mount.get("pitch_offset_deg", 0.0))),
        math.radians(float(mount.get("roll_offset_deg", 0.0))),
    ]
    rotators = {"X": _rot_x(angles[2]), "Y": _rot_y(angles[1]), "Z": _rot_z(angles[0])}
    result = np.eye(3)
    for axis in order:
        result = rotators[axis] @ result
    return result


def attitude_rotation(heading_deg: float, pitch_deg: float, roll_deg: float, declination_deg: float, order: str) -> np.ndarray:
    if order == "ZYX_magnetic_heading":
        return _rot_z(math.radians(heading_deg + declination_deg)) @ _rot_y(math.radians(pitch_deg)) @ _rot_x(math.radians(roll_deg))
    if order == "ZXY_magnetic_heading":
        return _rot_z(math.radians(heading_deg + declination_deg)) @ _rot_x(math.radians(roll_deg)) @ _rot_y(math.radians(pitch_deg))
    raise ValueError(f"unsupported coordinate transform order: {order}")


@dataclass(frozen=True)
class AttitudePoint:
    time_s: float
    heading_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None


def interpolate_attitude(samples: Sequence[AttitudePoint], time_s: float, max_gap_s: float) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda item: item.time_s)
    if not ordered:
        return {"status": "invalid", "reason": "no_attitude_samples"}
    left = None
    right = None
    for index, sample in enumerate(ordered):
        if sample.time_s <= time_s:
            left = sample
        if sample.time_s >= time_s and right is None:
            right = sample
    if left is None or right is None:
        return {"status": "invalid", "reason": "attitude_not_bracketed"}
    gap = right.time_s - left.time_s
    if gap > max_gap_s + 1e-9:
        return {"status": "invalid", "reason": "attitude_gap", "gap_s": gap, "max_gap_s": max_gap_s}
    if gap == 0:
        fraction = 0.0
    else:
        fraction = (time_s - left.time_s) / gap
    heading = shortest_angle_interpolate(left.heading_deg, right.heading_deg, fraction)
    return {
        "status": "ok",
        "time_s": time_s,
        "heading_deg": heading,
        "pitch_deg": _linear(left.pitch_deg, right.pitch_deg, fraction),
        "roll_deg": _linear(left.roll_deg, right.roll_deg, fraction),
        "left_time_s": left.time_s,
        "right_time_s": right.time_s,
        "gap_s": gap,
    }


def shortest_angle_interpolate(left: float | None, right: float | None, fraction: float) -> float:
    if left is None or right is None:
        raise ValueError("heading is absent")
    difference = (right - left + 180.0) % 360.0 - 180.0
    return (left + difference * fraction) % 360.0


def _linear(left: float | None, right: float | None, fraction: float) -> float:
    if left is None or right is None:
        raise ValueError("attitude angle is absent")
    return left + (right - left) * fraction
