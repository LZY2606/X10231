"""Beam geometry and the beam -> instrument -> ship -> earth rotation chain."""
from __future__ import annotations

import numpy as np

from .units import normalize_quantity

CANONICAL_ORDER = "beam/instrument/ship/earth@v1"


def beam_unit_vectors(beam_angle_deg: float, azimuths_deg: list[float]) -> np.ndarray:
    """Return unit vectors in forward/starboard/up instrument coordinates."""
    theta = np.deg2rad(beam_angle_deg)
    vectors = []
    for azimuth in azimuths_deg:
        az = np.deg2rad(azimuth)
        vectors.append(
            [
                np.sin(theta) * np.cos(az),
                np.sin(theta) * np.sin(az),
                -np.cos(theta),
            ]
        )
    return np.asarray(vectors, dtype=float).T


def solve_beams(
    beam_velocities_m_s: list[float | None],
    valid_beams: list[bool],
    geometry: dict,
    residual_threshold_m_s: float = 0.06,
) -> dict:
    """Solve instrument velocity.

    Four beams produce a least-squares solution.  A hard failure leaves one
    three-beam solution.  A low-quality but internally inconsistent fourth beam
    exposes both the four-beam and every legal three-beam candidate instead of
    forcing a single answer.
    """
    angle, angle_unit = normalize_quantity(geometry["beam_angle_raw"], geometry.get("beam_angle_unit"), "angle")
    if angle is None:
        return {"status": "fatal", "diagnostics": [f"beam geometry: {angle_unit}"], "candidates": []}
    matrix = beam_unit_vectors(angle, geometry["azimuths_deg"])
    valid_idx = [i for i, valid in enumerate(valid_beams) if valid]
    velocities = np.asarray(beam_velocities_m_s, dtype=float)
    candidates: list[dict] = []
    diagnostics: list[str] = []

    if len(valid_idx) < 3:
        return {
            "status": "insufficient_beams",
            "diagnostics": [f"only {len(valid_idx)} valid beams; at least three required"],
            "candidates": [],
        }

    if len(valid_idx) == 3:
        submatrix = matrix[:, valid_idx]
        vector = velocities[valid_idx]
        solved = np.linalg.solve(submatrix, vector)
        return {
            "status": "three_beam",
            "diagnostics": [f"three-beam solution using beams {[i + 1 for i in valid_idx]}"],
            "candidates": [
                {
                    "method": "three_beam_exact",
                    "omitted_beams": sorted(set(range(len(velocities))) - set(valid_idx)),
                    "used_beams": valid_idx,
                    "instrument_velocity_m_s": np.round(solved, 9).tolist(),
                    "residual_m_s": 0.0,
                }
            ],
        }

    four_x, residuals, *_ = np.linalg.lstsq(matrix.T, velocities, rcond=None)
    residual_vector = velocities - matrix.T @ four_x
    residual_rms = float(np.sqrt(np.mean(residual_vector**2)))
    candidates.append(
        {
            "method": "four_beam_lsq",
            "omitted_beams": [],
            "used_beams": valid_idx,
            "instrument_velocity_m_s": np.round(four_x, 9).tolist(),
            "residual_m_s": residual_rms,
        }
    )
    if residual_rms > residual_threshold_m_s:
        diagnostics.append(
            f"redundant-beam residual {residual_rms:.4f} m/s exceeds {residual_threshold_m_s:.4f}"
        )
        for omitted in range(4):
            used = [i for i in range(4) if i != omitted]
            solved = np.linalg.solve(matrix[:, used], velocities[used])
            reconstructed = matrix[:, used].T @ solved - velocities[used]
            candidates.append(
                {
                    "method": "three_beam_exact",
                    "omitted_beams": [omitted],
                    "used_beams": used,
                    "instrument_velocity_m_s": np.round(solved, 9).tolist(),
                    "residual_m_s": float(np.sqrt(np.mean(reconstructed**2))),
                }
            )
        status = "ambiguous"
    else:
        status = "ok"
    return {"status": status, "diagnostics": diagnostics, "candidates": candidates}


def instrument_to_ship(vector_m_s: list[float], installation_alignment_deg: float) -> list[float]:
    angle = np.deg2rad(installation_alignment_deg)
    c, s = np.cos(angle), np.sin(angle)
    fx, sy, up = vector_m_s
    return [c * fx - s * sy, s * fx + c * sy, up]


def _heading_matrix(true_heading_deg: float) -> np.ndarray:
    h = np.deg2rad(true_heading_deg)
    return np.asarray(
        [
            [np.sin(h), np.cos(h), 0.0],
            [np.cos(h), -np.sin(h), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _pitch_matrix(pitch_deg: float) -> np.ndarray:
    p = np.deg2rad(pitch_deg)
    c, s = np.cos(p), np.sin(p)
    return np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _roll_matrix(roll_deg: float) -> np.ndarray:
    r = np.deg2rad(roll_deg)
    c, s = np.cos(r), np.sin(r)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ship_to_earth(
    ship_velocity_m_s: list[float],
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    magnetic_declination_deg: float,
) -> list[float]:
    true_heading = (heading_deg + magnetic_declination_deg) % 360.0
    rotation = _heading_matrix(true_heading) @ _pitch_matrix(pitch_deg) @ _roll_matrix(roll_deg)
    return np.round(rotation @ np.asarray(ship_velocity_m_s), 9).tolist()


def rotate_alternative_order(vector_m_s: list[float], attitude_deg: list[float], declination_deg: float) -> list[float]:
    """Deliberately different order used only by the rejection test."""
    heading, pitch, roll = attitude_deg
    true_heading = (heading + declination_deg) % 360.0
    rotation = _roll_matrix(roll) @ _pitch_matrix(pitch) @ _heading_matrix(true_heading)
    return np.round(rotation @ np.asarray(vector_m_s), 9).tolist()
