from __future__ import annotations

import math

import numpy as np


STANDARD_ORDER = (
    "beam_to_instrument",
    "instrument_to_ship",
    "ship_to_earth",
)


def rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def beam_direction_matrix(beam_angle_deg: float, azimuths_deg: list[float]) -> np.ndarray:
    theta = math.radians(beam_angle_deg)
    rows = []
    for azimuth_deg in azimuths_deg:
        azimuth = math.radians(azimuth_deg)
        rows.append(
            (
                math.sin(theta) * math.cos(azimuth),
                math.sin(theta) * math.sin(azimuth),
                -math.cos(theta),
            )
        )
    return np.asarray(rows, dtype=float)


def solve_beams(
    beam_values: dict[int, float],
    geometry_matrix: np.ndarray,
    beam_index_by_number: dict[int, int],
) -> np.ndarray:
    beam_numbers = sorted(beam_values)
    observations = np.asarray([beam_values[number] for number in beam_numbers], dtype=float)
    rows = np.asarray([geometry_matrix[beam_index_by_number[number]] for number in beam_numbers])
    if len(beam_numbers) == 4:
        vector, *_ = np.linalg.lstsq(rows, observations, rcond=None)
        return vector
    return np.linalg.solve(rows, observations)


def transform_stages(
    instrument_vector: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    mount_z_deg: float,
    declination_deg: float,
    order: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray | None]:
    matrices = {
        "beam_to_instrument": np.eye(3),
        "instrument_to_ship": rotation_z(-mount_z_deg) @ rotation_y(pitch_deg) @ rotation_x(roll_deg),
        "ship_to_earth": rotation_z(-(heading_deg + declination_deg)),
    }
    vectors: dict[str, np.ndarray | None] = {"beam": None, "instrument": instrument_vector}
    current = instrument_vector
    for stage in order:
        if stage == "beam_to_instrument":
            current = instrument_vector
        else:
            current = matrices[stage] @ current
        if stage == "beam_to_instrument":
            vectors["instrument"] = current
        elif stage == "instrument_to_ship":
            vectors["ship"] = current
        elif stage == "ship_to_earth":
            vectors["earth"] = current
    if "ship" not in vectors:
        vectors["ship"] = None
    if "earth" not in vectors:
        vectors["earth"] = None
    return vectors
