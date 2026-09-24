"""Instrument -> ship -> earth coordinate transformations.

Frames
------
* instrument: x forward, y starboard, z down (right handed).
* ship:       x forward, y starboard, z down.
* earth:      ENU, x east, y north, z up (right handed).

Angles
------
* heading h: clockwise from true north to the ship's forward axis [0, 2pi).
* pitch p: bow-up positive, rotation about ship y.
* roll r: starboard-down positive, rotation about ship x.
* mount: heading/pitch/roll offsets of the instrument relative to the ship.
* magnetic declination delta: east-positive; true heading =
  magnetic heading + delta.

Both rotation order and the declination/mount versions are versioned and
enter the derived fingerprint; ``TRANSFORM_RULE`` is the accepted order and
``LEGACY_TRANSFORM_RULE`` is a deliberately wrong "heading before mount"
order used by the counter-example archive.
"""
from __future__ import annotations

import math

import numpy as np

TRANSFORM_RULE = "transform-order@mount-then-attitude-v1"
LEGACY_TRANSFORM_RULE = "transform-order@heading-first-v0"


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def instrument_to_ship(
    mount_heading: float, mount_pitch: float, mount_roll: float
) -> np.ndarray:
    """R_is: rotate a vector expressed in instrument frame into ship frame.

    Offset order is fixed and documented: Rz(mount_heading) Ry(mount_pitch)
    Rx(mount_roll).  All angles in radians.
    """
    return rot_z(mount_heading) @ rot_y(mount_pitch) @ rot_x(mount_roll)


def ship_to_earth(heading: float, pitch: float, roll: float) -> np.ndarray:
    """R_se for a z-down ship frame into ENU earth frame.

    Level hull sanity check: heading=0 (north) maps ship forward -> north,
    starboard -> east, down -> down(=-up).
    """
    c_h, s_h = math.cos(heading), math.sin(heading)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_r, s_r = math.cos(roll), math.sin(roll)
    r11 = s_h * c_p
    r12 = -s_h * s_p * s_r + c_h * c_r
    r13 = -s_h * s_p * c_r - c_h * s_r
    r21 = c_h * c_p
    r22 = -c_h * s_p * s_r - s_h * c_r
    r23 = -c_h * s_p * c_r + s_h * s_r
    r31 = s_p
    r32 = c_p * s_r
    r33 = c_p * c_r
    return np.array(
        [
            [r11, r12, r13],
            [r21, r22, r23],
            [-r31, -r32, -r33],
        ]
    )


def earth_to_ship(heading: float, pitch: float, roll: float) -> np.ndarray:
    return ship_to_earth(heading, pitch, roll).T


def true_heading(magnetic_heading: float, declination: float) -> float:
    """Apply east-positive declination and normalize to [0, 2pi)."""
    return float((magnetic_heading + declination) % (2.0 * math.pi))


def transform_pipeline(
    v_instrument: np.ndarray,
    *,
    magnetic_heading: float,
    pitch: float,
    roll: float,
    declination: float,
    mount_heading: float,
    mount_pitch: float,
    mount_roll: float,
    rule: str = TRANSFORM_RULE,
) -> dict[str, np.ndarray]:
    """Run one full beam->...->earth transform and keep every intermediate.

    The step vectors are exactly what the review page renders.
    """
    h_true = true_heading(magnetic_heading, declination)
    v_inst = np.asarray(v_instrument, dtype=float)
    if rule == TRANSFORM_RULE:
        r_is = instrument_to_ship(mount_heading, mount_pitch, mount_roll)
        v_ship = r_is @ v_inst
        r_se = ship_to_earth(h_true, pitch, roll)
        v_earth = r_se @ v_ship
    elif rule == LEGACY_TRANSFORM_RULE:
        # Deliberately wrong archival order: heading is applied *before* the
        # mount offset is removed.  Kept only to prove order is fingerprinted.
        r_se = ship_to_earth(h_true, pitch, roll)
        v_ship = r_se @ v_inst
        r_is = instrument_to_ship(mount_heading, mount_pitch, mount_roll)
        v_earth = r_is @ v_ship
    else:
        raise ValueError(f"unknown transform rule {rule!r}")
    return {
        "instrument": v_inst,
        "ship": v_ship,
        "earth": v_earth,
        "R_is": r_is,
        "R_se": r_se,
        "true_heading": h_true,
        "rule": rule,
    }
