"""Coordinate-frame rotations.

Frames (all right handed):
  instrument: X forward, Y starboard, Z up (relative to the ADCP head)
  ship:       X forward, Y starboard, Z up (relative to the vessel)
  earth:      X east,  Y north,   Z up

Angles are held internally in radians.

ship = R_mount instrument, where R_mount rotates about heading-alignment,
pitch-mount and roll-mount angles of the transducer installation.

earth = R_hpr(heading_true, pitch, roll) ship using the classic ADCP order:
  R_hpr = Rz(heading) @ Ry(-pitch) @ Rx(roll).

A second, explicitly labelled order "rph" (Rx(roll) @ Ry(-pitch) @
Rz(heading), i.e. applying heading first as an intrinsic rotation) is
computed as a separate candidate so that transform-order revisions remain
visible in the derived fingerprint instead of overwriting the primary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

TRANSFORM_ORDERS = ("hpr", "rph")


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def heading_true_deg(heading_mag_deg: float, declination_deg: float) -> float:
    """True heading = magnetic heading + east-positive declination."""
    return (heading_mag_deg + declination_deg) % 360.0


def mount_matrix(alignment_deg: float, pitch_mount_deg: float,
                 roll_mount_deg: float) -> np.ndarray:
    """R_mount: instrument -> ship."""
    a = math.radians(alignment_deg)
    p = math.radians(pitch_mount_deg)
    r = math.radians(roll_mount_deg)
    return rotz(a) @ roty(-p) @ rotx(r)


def ship_to_earth(heading_true_deg: float, pitch_deg: float, roll_deg: float,
                  order: str = "hpr") -> np.ndarray:
    if order not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order {order!r}")
    h = math.radians(heading_true_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    heading_r = rotz(h)
    pitch_r = roty(-p)
    roll_r = rotx(r)
    if order == "hpr":
        return heading_r @ pitch_r @ roll_r
    return roll_r @ pitch_r @ heading_r


@dataclass(frozen=True)
class Transform:
    vector: np.ndarray
    matrix: np.ndarray
    frame: str
    order: str


def apply(matrix: np.ndarray, vec: Sequence[float]) -> np.ndarray:
    return matrix @ np.asarray(vec, dtype=float)
