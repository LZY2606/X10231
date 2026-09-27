"""Coordinate-frame rotations.

Frames, in derivation order:

    beam -> instrument -> ship -> earth

* instrument: x forward, y starboard, z down (transducer head frame)
* ship: vessel frame, related to instrument by fixed mount angles
  (heading/pitch/roll misalignment of the transducer).
* earth: east/north/down; heading rotates about the downward z axis.

Attitude convention: heading measured clockwise from magnetic north;
``declination`` is added to obtain true heading.

Two genuine rotation *orders* are provided; the order is part of every
derivation fingerprint and audits may keep both as competing candidates:

* ``HPR``  : R = R_h(heading) @ R_p(pitch) @ R_r(roll)
* ``PROH`` : R = R_h(heading) @ R_r(roll)    @ R_p(pitch)

They agree for small tilts and diverge for large pitch/roll.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import deg_to_rad, normalize_degrees


def rx(angle_deg: float) -> np.ndarray:
    a = deg_to_rad(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def ry(angle_deg: float) -> np.ndarray:
    a = deg_to_rad(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def rz(angle_deg: float) -> np.ndarray:
    """Rotation about z (down); positive angle clockwise viewed from above."""
    a = deg_to_rad(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class MountAngles:
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def matrix_inst_to_ship(self) -> np.ndarray:
        return rz(self.heading_deg) @ ry(self.pitch_deg) @ rx(self.roll_deg)


@dataclass(frozen=True)
class Attitude:
    heading_deg: float
    pitch_deg: float
    roll_deg: float

    def true_heading(self, declination_deg: float) -> float:
        return normalize_degrees(self.heading_deg + declination_deg)

    def matrix_ship_to_earth(self, declination_deg: float, order: str = "HPR") -> np.ndarray:
        heading = self.true_heading(declination_deg)
        if order == "HPR":
            return rz(heading) @ ry(self.pitch_deg) @ rx(self.roll_deg)
        if order == "PROH":
            return rz(heading) @ rx(self.roll_deg) @ ry(self.pitch_deg)
        raise ValueError(f"unknown rotation order {order!r}")


def instrument_to_ship(v_inst: np.ndarray, mount: MountAngles) -> np.ndarray:
    return mount.matrix_inst_to_ship() @ np.asarray(v_inst, dtype=float)


def ship_to_earth(v_ship: np.ndarray, att: Attitude, declination_deg: float,
                  order: str = "HPR") -> np.ndarray:
    return att.matrix_ship_to_earth(declination_deg, order) @ np.asarray(v_ship, dtype=float)


def instrument_to_earth(v_inst: np.ndarray, mount: MountAngles, att: Attitude,
                        declination_deg: float, order: str = "HPR") -> np.ndarray:
    v_ship = instrument_to_ship(v_inst, mount)
    return ship_to_earth(v_ship, att, declination_deg, order)
