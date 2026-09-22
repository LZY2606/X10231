"""Attitude and coordinate-frame rotations.

Frames (right-handed):
  instrument : x across face 1->2, y across face 3->4, z up
  ship       : x starboard, y forward, z up
  earth      : x east, y north, z up

Recipes are versioned.  The transform *order* is part of the derivation
fingerprint, so a change of recipe forces recomputation.
"""
from __future__ import annotations

import numpy as np


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def instrument_to_ship(alpha_deg: float, beta_deg: float,
                       recipe: str = "ALPHA_BETA") -> np.ndarray:
    """Static mounting rotation: heading offset alpha, tilt beta."""
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    if recipe == "ALPHA_BETA":
        return roty(beta) @ rotz(alpha)
    if recipe == "BETA_ALPHA":
        return rotz(alpha) @ roty(beta)
    raise ValueError("unknown_mount_recipe:%s" % recipe)


def ship_to_earth(heading_deg: float, pitch_deg: float, roll_deg: float,
                  recipe: str = "HPR") -> np.ndarray:
    """Ship-to-earth matrix mapping ship-frame velocities into ENU.

    Marine conventions: heading clockwise from north, positive pitch is
    bow-up, positive roll is starboard-down.  The columns of the matrix are
    the ship basis vectors (starboard, forward, up) resolved in east, north,
    up, obtained by applying roll (about forward) then pitch (about
    starboard) to a level, heading-rotated ship.
    """
    h = np.deg2rad(heading_deg)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)
    sp, cp, sr, cr = np.sin(p), np.cos(p), np.sin(r), np.cos(r)
    if recipe == "HPR":
        return np.array([
            [cp * np.cos(h) + sp * sr * np.sin(h),
             cr * np.sin(h),
             sp * np.cos(h) - cp * sr * np.sin(h)],
            [-cp * np.sin(h) + sp * sr * np.cos(h),
             cr * np.cos(h),
             -sp * np.sin(h) - cp * sr * np.cos(h)],
            [-sp * cr, sr, cp * cr],
        ])
    if recipe == "HRP":
        # Roll is applied to the level ship before pitch instead of after.
        star0 = np.array([np.cos(h), -np.sin(h), 0.0])
        fwd0 = np.array([np.sin(h), np.cos(h), 0.0])
        up0 = np.array([0.0, 0.0, 1.0])
        star = cr * star0 - sr * up0
        up = sr * star0 + cr * up0
        fwd = cp * fwd0 + sp * up
        up2 = -sp * fwd0 + cp * up
        return np.column_stack([star, fwd, up2])
    raise ValueError("unknown_rotation_recipe:%s" % recipe)


def true_heading(magnetic_heading_deg: float, declination_deg: float,
                 convention: str = "east_positive") -> float:
    """True heading.  East declination is positive (standard convention)."""
    if convention == "east_positive":
        return magnetic_heading_deg + declination_deg
    if convention == "west_positive":
        return magnetic_heading_deg - declination_deg
    raise ValueError("unknown_declination_convention:%s" % convention)


def unwrap_angles(degrees) -> np.ndarray:
    """Unwrap an angle series in degrees (handles the 360 crossover)."""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(degrees, dtype=float))))
