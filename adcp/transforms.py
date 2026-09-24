"""Coordinate transforms: beam -> instrument -> ship -> earth.

Frames
------
instrument : x forward, y starboard, z up (z-up variant of the RDI head frame)
ship       : x forward, y starboard, z up after pitch/roll compensation
earth      : x east, y north, z up (ENU)

Transform ordering matters and is part of the derived fingerprint. The
canonical rule ``RDI_PR`` applies roll then pitch matrices to the
instrument vector (pitch matrix outermost); the alternative ``ALT_RP``
swaps them and is retained as a separate, non-merged derivation.
"""
from __future__ import annotations

import math

import numpy as np

RDI_PR = "RDI_PR"   # ship = R_pitch * R_roll * instrument
ALT_RP = "ALT_RP"   # ship = R_roll  * R_pitch * instrument (alternative)


def _rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def instrument_to_ship_matrix(pitch_deg: float, roll_deg: float,
                              order: str = RDI_PR) -> np.ndarray:
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    Rp, Rr = _roty(p), _rotx(r)
    if order == RDI_PR:
        return Rp @ Rr
    if order == ALT_RP:
        return Rr @ Rp
    raise ValueError(f"unknown transform order: {order!r}")


def ship_to_earth_matrix(heading_true_deg: float) -> np.ndarray:
    """Rotate a z-up x-forward/y-starboard vector to east/north/up.

    Heading is measured clockwise from north. The x-axis therefore maps to
    (sin H, cos H) and the y-axis to (cos H, -sin H).
    """
    h = math.radians(heading_true_deg)
    c, s = math.cos(h), math.sin(h)
    return np.array([[s, c, 0.0], [c, -s, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def correct_attitude(heading_mag_deg: float, pitch_deg: float, roll_deg: float,
                     declination_deg: float,
                     install: dict | None = None) -> dict:
    """Apply magnetic declination and transducer mount angles.

    Heading: measured clockwise from magnetic north; add east-positive
    declination, then subtract the heading installation misalignment
    (angle of the head from the ship centreline, clockwise positive).
    """
    install = install or {}
    h = heading_mag_deg + declination_deg - float(install.get("heading_offset_deg", 0.0))
    p = pitch_deg - float(install.get("pitch_offset_deg", 0.0))
    r = roll_deg - float(install.get("roll_offset_deg", 0.0))
    return {"heading_true": h % 360.0, "pitch": p, "roll": r}


def transform_chain(v_instrument: list[float], attitude: dict,
                    declination_deg: float, install: dict | None = None,
                    order: str = RDI_PR) -> dict:
    """Run one instrument vector through the full chain, returning each stage."""
    corr = correct_attitude(attitude["heading"], attitude["pitch"], attitude["roll"],
                            declination_deg, install)
    M_is = instrument_to_ship_matrix(corr["pitch"], corr["roll"], order)
    v_ship = M_is @ np.asarray(v_instrument, dtype=float)
    M_se = ship_to_earth_matrix(corr["heading_true"])
    v_earth = M_se @ v_ship
    return {
        "attitude_corrected": corr,
        "v_ship": v_ship.tolist(),
        "v_earth": v_earth.tolist(),
        "order": order,
    }
