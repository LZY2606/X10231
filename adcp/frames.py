"""Instrument -> ship -> earth rotations.

Two transform orders are implemented on purpose: the order of mount angles
versus attitude rotations is itself a derived-fingerprint ingredient and a
counterexample case.

Order "mount_first" (default): v_ship = Ry(pm) Rx(rm) v_instrument then
attitude heading/pitch/roll. Order "attitude_first" applies heading/pitch/roll
before the mount rotation; the two differ and are never silently merged.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def mount_matrix(mount: dict) -> np.ndarray:
    """Instrument to ship: pitch mount then roll mount about corrected frame."""
    pm = math.radians(float(mount.get("pitch_deg", 0.0)))
    rm = math.radians(float(mount.get("roll_deg", 0.0)))
    ym = math.radians(float(mount.get("heading_deg", 0.0)))
    return _rot_z(ym) @ _rot_y(pm) @ _rot_x(rm)


def attitude_matrix(heading_deg: float, pitch_deg: float, roll_deg: float,
                    declination_deg: float = 0.0) -> np.ndarray:
    """Ship to earth (NED-like: x east, y north handled in heading sign).

    Heading is measured clockwise from north; magnetic heading is corrected to
    true by *adding* the declination (east positive).
    """
    true_heading = heading_deg + declination_deg
    h = math.radians(true_heading)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    return _rot_z(-h) @ _rot_y(p) @ _rot_x(r)


def to_ship(v_instrument: Sequence[float], mount: dict,
            attitude: dict, order: str = "mount_first") -> Dict[str, object]:
    vi = np.asarray(v_instrument, dtype=float)
    m = mount_matrix(mount)
    a = attitude_matrix(attitude["heading_deg"], attitude["pitch_deg"],
                        attitude["roll_deg"],
                        attitude.get("declination_deg", 0.0))
    if order == "mount_first":
        vs = m @ vi
        ve = a @ vs
    elif order == "attitude_first":
        vs = vi  # attitude applied before mount is an explicit alternative
        ve = m @ (a @ vi)
    else:
        raise ValueError(f"unknown transform order: {order}")
    return {"velocity_ship": vs.tolist(), "velocity_earth": ve.tolist(),
            "order": order,
            "true_heading_deg": (attitude["heading_deg"]
                                 + attitude.get("declination_deg", 0.0)) % 360.0}
