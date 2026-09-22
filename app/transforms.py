"""Instrument -> ship -> earth coordinate transforms.

Angles in degrees. Right-handed column-vector convention:
    v_earth = R @ v_ship,  x=starboard(E), y=forward(N), z=up
Rotation matrices follow the aerospace sequence used by RDI:
    tilt      T = Rx(roll) @ Ry(pitch)
    standard  R = Rz(heading) @ Rx(roll) @ Ry(pitch)
The alternative sequence is retained as a named, fingerprinted variant so the
case archive can prove the transform order enters the derived fingerprint.
Magnetic declination is signed east-positive: heading_true = mag + decl.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

TRANSFORM_ORDERS = ("RDI_HRP", "ALT_PRH")


def _r(deg: float) -> float:
    return math.radians(deg)


def Rx(a: float) -> np.ndarray:
    c, s = math.cos(_r(a)), math.sin(_r(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(a: float) -> np.ndarray:
    c, s = math.cos(_r(a)), math.sin(_r(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(a: float) -> np.ndarray:
    c, s = math.cos(_r(a)), math.sin(_r(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def tilt_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    return Rx(roll_deg) @ Ry(pitch_deg)


def earth_matrix(heading_deg: float, pitch_deg: float, roll_deg: float,
                 declination_deg: float = 0.0, order: str = "RDI_HRP") -> np.ndarray:
    if order not in TRANSFORM_ORDERS:
        raise ValueError("unknown transform order %r" % order)
    h = (heading_deg + declination_deg) % 360.0
    if order == "RDI_HRP":
        return Rz(h) @ Rx(roll_deg) @ Ry(pitch_deg)
    # deliberately reversed multiply order: changes the result, must never be
    # silently conflated with the standard order.
    return Ry(pitch_deg) @ Rx(roll_deg) @ Rz(h)


def ship_matrix(mount_heading_deg: float, mount_pitch_deg: float,
                mount_roll_deg: float) -> np.ndarray:
    """Static instrument-on-ship alignment (heading alignment + tilt offsets)."""
    return (Rz(mount_heading_deg) @ Rx(mount_roll_deg) @ Ry(mount_pitch_deg))


@dataclass
class TransformStep:
    name: str
    matrix: np.ndarray
    basis: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "matrix": np.round(self.matrix, 9).tolist(),
            "basis": self.basis,
        }


def chain(
    v_instrument: np.ndarray,
    mount_heading_deg: float,
    mount_pitch_deg: float,
    mount_roll_deg: float,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    order: str = "RDI_HRP",
) -> Tuple[np.ndarray, np.ndarray, TransformStep, TransformStep]:
    """Apply the instrument->ship->earth chain; return vectors and steps."""
    Ms = ship_matrix(mount_heading_deg, mount_pitch_deg, mount_roll_deg)
    Me = earth_matrix(heading_deg, pitch_deg, roll_deg, declination_deg, order)
    v_ship = Ms @ np.asarray(v_instrument, dtype=float)
    v_earth = Me @ v_ship
    step_ship = (
        TransformStep("instrument_to_ship", Ms, {
            "mount_heading_deg": mount_heading_deg,
            "mount_pitch_deg": mount_pitch_deg,
            "mount_roll_deg": mount_roll_deg,
        }),
    )[0]
    step_earth = (
        TransformStep("ship_to_earth", Me, {
            "heading_deg": heading_deg, "pitch_deg": pitch_deg,
            "roll_deg": roll_deg, "declination_deg": declination_deg,
            "transform_order": order,
        }),
    )[0]
    return v_ship, v_earth, step_ship, step_earth
