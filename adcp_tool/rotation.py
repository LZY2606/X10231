"""Rotation stages: instrument -> ship -> earth.

The transform ORDER is part of the derivation fingerprint. Two supported
orders are provided; changing the order changes every downstream id rather
than silently overwriting numbers.

Stages (x forward, y starboard, z up; right-hand positive rotations):

    v_ship = R_pitch R_roll v_inst                       (tilt only)
    v_ship = R_mount v_ship                              (install angles)
    psi     = heading + heading_alignment                (mounting revision)
    v_earth_mag = Rz(psi) v_ship
    v_earth     = Rz(declination) v_earth_mag            (declination version)

Heading alignment / installation angles and magnetic declination are separate
versioned inputs, so a mount revision and a declination revision are distinct
dependency changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# documented order variants
TILT_ORDERS = {
    "roll-pitch": ("roll", "pitch"),
    "pitch-roll": ("pitch", "roll"),
}


def tilt_matrix(roll_rad: float, pitch_rad: float,
                order: str = "roll-pitch") -> np.ndarray:
    stages = TILT_ORDERS[order]
    m = np.eye(3)
    for name in reversed(stages):  # rightmost applied first to the vector
        m = m @ (_rx(roll_rad) if name == "roll" else _ry(pitch_rad))
    return m


@dataclass(frozen=True)
class MountAngles:
    """Transducer installation angles relative to the vessel frame."""

    x_deg: float = 0.0          # installation angle around x
    y_deg: float = 0.0          # installation angle around y
    heading_alignment_deg: float = 0.0
    version: str = "mount-v1"

    def matrix(self, order: str = "roll-pitch") -> np.ndarray:
        # fixed install rotations in the same tilt-order convention
        return tilt_matrix(math.radians(self.x_deg),
                           math.radians(self.y_deg), order)


@dataclass(frozen=True)
class TransformConfig:
    tilt_order: str = "roll-pitch"
    apply_declination: bool = True

    def id(self) -> str:
        return f"xform:{self.tilt_order}:decl={int(self.apply_declination)}"


def instrument_to_ship(v_inst, roll_deg: float, pitch_deg: float,
                       mount: MountAngles, cfg: TransformConfig):
    vi = np.asarray(v_inst, float)
    r_tilt = tilt_matrix(math.radians(roll_deg), math.radians(pitch_deg),
                         cfg.tilt_order)
    v_ship = r_tilt @ vi
    v_mount = mount.matrix(cfg.tilt_order) @ v_ship
    return {"v_ship": v_ship.tolist(), "v_ship_pre_mount": v_ship.tolist(),
            "R_tilt": r_tilt.tolist(), "R_mount": mount.matrix(cfg.tilt_order).tolist()}


def ship_to_earth(v_ship, heading_deg: float, mount: MountAngles,
                  declination_deg: float, cfg: TransformConfig):
    psi = math.radians(heading_deg + mount.heading_alignment_deg)
    r_head = _rz(psi)
    v_mag = r_head @ np.asarray(v_ship, float)
    decl = math.radians(declination_deg) if cfg.apply_declination else 0.0
    r_decl = _rz(decl)
    v_earth = r_decl @ v_mag
    return {
        "v_earth_mag": v_mag.tolist(),
        "v_earth": v_earth.tolist(),
        "heading_used_deg": heading_deg + mount.heading_alignment_deg,
        "declination_deg": declination_deg if cfg.apply_declination else 0.0,
        "R_head": r_head.tolist(),
        "R_decl": r_decl.tolist(),
    }
