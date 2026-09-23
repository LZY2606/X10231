"""Instrument -> ship -> earth coordinate rotation chain.

Frames (right-handed):

* instrument (x forward / y starboard / z up, aligned with beams);
* ship (same axes after removing transducer mount misalignment + pitch/roll);
* earth (x east / y north / z up after true-heading rotation).

Transform order is part of the derived fingerprint. The documented physical
order is tilt first (ship), then heading (earth). Applying them swapped is a
recognizable alternative chain kept for the "coordinate transform order"
counterexample; the two agree only when tilt is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRANSFORM_RULE_VERSION = "transforms-2.0.0"

ORDER_TILT_THEN_HEADING = "tilt_then_heading"
ORDER_HEADING_THEN_TILT = "heading_then_tilt"


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def tilt_matrix(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Instrument -> ship.

    Pitch and roll here follow the documented ADCP convention: pitch is a
    rotation about the starboard axis and roll about the forward axis.
    """
    return rot_x(roll_rad) @ rot_y(pitch_rad)


def heading_matrix(true_heading_rad: float) -> np.ndarray:
    """Ship -> earth: rotate the ship frame by its true heading.

    With heading 0 the ship points north, so forward (+x) maps to +y north
    and starboard (+y) maps to +x east.
    """
    return rot_z(math.pi / 2.0 - true_heading_rad)


def apply_install_offsets(pitch_deg: float, roll_deg: float, heading_deg: float,
                          install: dict) -> tuple[float, float, float]:
    """Fold transducer mounting-angle offsets into attitude (degrees).

    Offsets are plain additive corrections in each axis; heading is wrapped.
    """
    p = pitch_deg + float(install.get("pitch_offset_deg", 0.0))
    r = roll_deg + float(install.get("roll_offset_deg", 0.0))
    h = heading_deg + float(install.get("heading_offset_deg", 0.0))
    return p, r, h % 360.0


@dataclass(frozen=True)
class ChainResult:
    instrument: tuple[float, float, float]
    ship: tuple[float, float, float]
    earth: tuple[float, float, float]
    matrices: dict[str, list[list[float]]]

    def to_dict(self) -> dict:
        return {
            "instrument": list(self.instrument),
            "ship": list(self.ship),
            "earth": list(self.earth),
            "matrices": {k: np.asarray(v).tolist() for k, v in self.matrices.items()},
        }


def run_chain(
    instrument_velocity: tuple[float, float, float],
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    order: str = ORDER_TILT_THEN_HEADING,
) -> ChainResult:
    """Rotate one velocity vector through instrument -> ship -> earth.

    True heading = magnetic heading + magnetic declination.
    """
    h_true = math.radians((heading_deg + declination_deg) % 360.0)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    t = tilt_matrix(pitch, roll)
    h = heading_matrix(h_true)
    vi = np.asarray(instrument_velocity, dtype=float)
    if order == ORDER_TILT_THEN_HEADING:
        vs = t @ vi
        ve = h @ vs
        r_tilt, r_head = t, h
    elif order == ORDER_HEADING_THEN_TILT:
        # Deliberately swapped chain for the transform-order counterexample.
        ve_partial = h @ vi
        ve = t @ ve_partial
        vs = t @ vi  # ship stage physically unchanged
        r_tilt, r_head = t, h
    else:
        raise ValueError(f"unknown_transform_order:{order}")
    return ChainResult(
        tuple(float(x) for x in vi),
        tuple(float(x) for x in vs),
        tuple(float(x) for x in ve),
        {"tilt": r_tilt.tolist(), "heading": r_head.tolist()},
    )


def horizontal_earth_to_instrument(
    earth_uvw: tuple[float, float, float],
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
) -> tuple[float, float, float]:
    """Inverse of :func:`run_chain` (tilt-then-heading order), fixture helper."""
    h_true = math.radians((heading_deg + declination_deg) % 360.0)
    t = tilt_matrix(math.radians(pitch_deg), math.radians(roll_deg))
    h = heading_matrix(h_true)
    return tuple(float(x) for x in np.linalg.solve(t, np.linalg.solve(h, np.asarray(earth_uvw))))
