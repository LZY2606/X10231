from __future__ import annotations

import math
from typing import Any

import numpy as np


BEAM_GEOMETRY_ID = "janus4-convex-20deg-clockwise-v1"
TRANSFORM_ORDER_STANDARD = "beam->instrument->mount->heading-pitch-roll->magnetic-declination-v1"
TRANSFORM_ORDER_ALTERNATE = "beam->instrument->heading-pitch-roll->mount->magnetic-declination-v2"


def janus_geometry(beam_angle_deg: float = 20.0) -> dict[str, Any]:
    theta = math.radians(beam_angle_deg)
    azimuths = [45.0, 135.0, 225.0, 315.0]
    rows = []
    for azimuth_deg in azimuths:
        az = math.radians(azimuth_deg)
        rows.append(
            [
                math.sin(theta) * math.cos(az),
                math.sin(theta) * math.sin(az),
                -math.cos(theta),
            ]
        )
    return {
        "geometry_id": BEAM_GEOMETRY_ID,
        "beam_angle_from_transducer_axis": {"value": beam_angle_deg, "unit": "degree"},
        "azimuths_in_instrument_frame": [
            {"beam": index + 1, "value": value, "unit": "degree_clockwise_from_instrument_x"}
            for index, value in enumerate(azimuths)
        ],
        "direction_cosine_matrix": rows,
        "radial_velocity_convention": "positive_along_outward_beam",
        "instrument_frame": "x_forward_y_starboard_z_up",
    }


def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _attitude_matrix(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    return rotation_z(heading) @ rotation_y(pitch) @ rotation_x(roll)


def _mount_matrix(mount_heading_deg: float, mount_pitch_deg: float, mount_roll_deg: float) -> np.ndarray:
    return (
        rotation_z(math.radians(mount_heading_deg))
        @ rotation_y(math.radians(mount_pitch_deg))
        @ rotation_x(math.radians(mount_roll_deg))
    )


def beam_to_instrument(beam_values: list[float], geometry: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(geometry["direction_cosine_matrix"], dtype=float)
    radial = np.asarray(beam_values, dtype=float)
    return np.linalg.pinv(matrix) @ radial


def three_beam_solution(
    beam_values: list[float],
    valid_beams: list[int],
    geometry: dict[str, Any],
) -> np.ndarray:
    if len(valid_beams) != 3:
        raise ValueError("exactly three valid beams are required")
    matrix = np.asarray(geometry["direction_cosine_matrix"], dtype=float)
    selected = matrix[np.asarray(valid_beams) - 1, :]
    radial = np.asarray(beam_values, dtype=float)[np.asarray(valid_beams) - 1]
    return np.linalg.solve(selected, radial)


def transform_vector(
    instrument_velocity: np.ndarray,
    *,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    mount_heading_deg: float,
    mount_pitch_deg: float,
    mount_roll_deg: float,
    declination_deg: float,
    order: str,
) -> dict[str, np.ndarray]:
    instrument = np.asarray(instrument_velocity, dtype=float)
    attitude = _attitude_matrix(heading_deg, pitch_deg, roll_deg)
    mount = _mount_matrix(mount_heading_deg, mount_pitch_deg, mount_roll_deg)
    declination = rotation_z(math.radians(declination_deg))
    if order == TRANSFORM_ORDER_STANDARD:
        ship = mount @ instrument
        earth_magnetic = attitude @ ship
        earth_true = declination @ earth_magnetic
    elif order == TRANSFORM_ORDER_ALTERNATE:
        pre_mount_instrument = attitude @ instrument
        ship = mount @ instrument
        earth_magnetic = mount @ pre_mount_instrument
        earth_true = declination @ earth_magnetic
    else:
        raise ValueError(f"unsupported transform order: {order}")
    return {
        "instrument": instrument,
        "ship": ship,
        "earth_magnetic": earth_magnetic,
        "earth_true": earth_true,
    }


def interpolate_attitude(before: dict[str, Any], after: dict[str, Any], at_time: float, max_gap_s: float) -> dict[str, Any]:
    t0 = float(before["time_s"])
    t1 = float(after["time_s"])
    if not t0 <= at_time <= t1:
        raise ValueError("attitude samples do not bracket ping time")
    gap = t1 - t0
    if gap > max_gap_s:
        raise ValueError("attitude interpolation refuses to cross a time gap")
    if gap == 0:
        weight = 0.0
    else:
        weight = (at_time - t0) / gap

    heading0 = float(before["heading_deg"])
    heading1 = float(after["heading_deg"])
    delta = (heading1 - heading0 + 180.0) % 360.0 - 180.0
    heading = (heading0 + delta * weight) % 360.0
    return {
        "time_s": at_time,
        "heading_deg": heading,
        "pitch_deg": float(before["pitch_deg"]) + (float(after["pitch_deg"]) - float(before["pitch_deg"])) * weight,
        "roll_deg": float(before["roll_deg"]) + (float(after["roll_deg"]) - float(before["roll_deg"])) * weight,
        "interpolated": True,
        "source_before": before["record_id"],
        "source_after": after["record_id"],
        "unit": {"heading": "degree_magnetic_north", "pitch": "degree", "roll": "degree"},
    }

