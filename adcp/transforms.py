"""Attitude handling and frame rotation matrices.

Frames:

* beam -> instrument: handled by :mod:`adcp.geometry`
* instrument -> ship: mounting-angle rotation (heading offset; pitch/roll zeroed
  misalignment is captured by the install-angle version)
* ship -> earth: heading/pitch/roll tilt rotation plus magnetic declination.

Heading wrap at 360 degrees is handled explicitly; interpolation never bridges
a time gap.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

DEG = math.pi / 180.0


def wrap_360(deg: float) -> float:
    """Wrap any angle into [0, 360)."""
    return deg % 360.0


def angle_signed_delta(a: float, b: float) -> float:
    """Shortest signed delta ``b - a`` for angles in degrees, wrap aware."""
    return (b - a + 180.0) % 360.0 - 180.0


def crosses_heading_wrap(h0: float, h1: float) -> bool:
    """True when consecutive headings cross the 0/360 seam.

    Raw deltas with magnitude above 180 imply the physical motion went the
    short way around through zero, e.g. 359 -> 1 is a +2 degree change, not
    -358. This flags the seam rather than a spin.
    """
    return abs(h1 - h0) > 180.0


def unwrap_headings(headings: list[float]) -> list[float]:
    """Unwrap a heading series through the 0/360 seam for interpolation."""
    if not headings:
        return []
    out = [float(headings[0])]
    for h in headings[1:]:
        out.append(out[-1] + angle_signed_delta(out[-1] % 360.0, float(h)))
    return out


def interpolate_attitude(
    t: float,
    t0: float,
    t1: float,
    v0: float,
    v1: float,
    *,
    is_heading: bool = False,
) -> float:
    """Linear interpolation; heading-aware via unwrapped deltas.

    Callers must guarantee ``t0 < t < t1`` and that the interval has no time
    gap. This function never bridges a gap itself.
    """
    if t1 == t0:
        return v0
    frac = (t - t0) / (t1 - t0)
    if is_heading:
        return wrap_360(v0 + angle_signed_delta(v0, v1) * frac)
    return v0 + (v1 - v0) * frac


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(deg * DEG), math.sin(deg * DEG)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(deg * DEG), math.sin(deg * DEG)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(deg * DEG), math.sin(deg * DEG)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)


def tilt_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Ship -> earth without azimuth: pitch then roll (RDI convention)."""
    return rot_x(roll_deg) @ rot_y(pitch_deg)


def heading_matrix(true_heading_deg: float) -> np.ndarray:
    return rot_z(true_heading_deg)


def ship_to_earth(
    vel_ship: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    *,
    order: str = "hpr_standard",
) -> np.ndarray:
    """Rotate ship-frame velocity to true-north ENU earth frame.

    ``order`` selects the transform order and enters the derivation
    fingerprint:

    * ``hpr_standard``: heading applied after tilt to the tilted vector
      (R = Rz(H+d) @ Rx(roll) @ Ry(pitch)).
    * ``legacy_hp_r``: legacy ordering pitch -> heading -> roll
      (R = Rx(roll) @ Rz(H+d) @ Ry(pitch)); retained so reviewers can see why
      order changes results and pick the adopted rule.
    """
    v = np.asarray(vel_ship, dtype=float)
    h = wrap_360(heading_deg + declination_deg)
    if order == "hpr_standard":
        r = rot_z(h) @ rot_x(roll_deg) @ rot_y(pitch_deg)
    elif order == "legacy_hp_r":
        r = rot_x(roll_deg) @ rot_z(h) @ rot_y(pitch_deg)
    else:
        raise ValueError(f"unknown transform order: {order}")
    return r @ v


def instrument_to_ship(vel_instrument: np.ndarray, mount_angle_deg: float) -> np.ndarray:
    """Remove mounting alignment: rotate by the configured install angle.

    Only a heading-plane alignment offset is modeled here; pitch/roll
    misalignment lives in the install-angle version record.
    """
    return rot_z(wrap_360(mount_angle_deg)) @ np.asarray(vel_instrument, dtype=float)
