"""ADCP coordinate transforms.

Frames (all right-handed):
  beam        : radial velocity along each beam, + away from transducer
  instrument  : x ~ beam3 direction, y ~ beam1 direction, z down
  ship        : x bow, y starboard, z down
  earth       : x north, y east, z down  (up = -down exposed in payloads)

Beam layout (Janus-4, convex): beam1 +y, beam2 -y, beam3 +x, beam4 -x,
each tilted `beam_angle_deg` away from vertical (z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GEOMETRY_VERSION = "janus4-convex-v1"


@dataclass(frozen=True)
class BeamGeometry:
    n_beams: int = 4
    beam_angle_deg: float = 20.0
    convex: bool = True
    pattern: str = "janus"
    version: str = GEOMETRY_VERSION


def beam_unit_vectors(geom: BeamGeometry) -> np.ndarray:
    a = math.radians(geom.beam_angle_deg)
    s, c = math.sin(a), math.cos(a)
    return np.array([
        [0.0, s, c],
        [0.0, -s, c],
        [s, 0.0, c],
        [-s, 0.0, c],
    ])


def beam_to_instrument(radial, geom: BeamGeometry):
    """4-beam least-squares inverse. Returns (v_instr[3], error_velocity)."""
    radial = np.asarray(radial, dtype=float)
    a = beam_unit_vectors(geom)
    v, *_ = np.linalg.lstsq(a, radial, rcond=None)
    err = float((radial[0] + radial[1] - radial[2] - radial[3])
                / (4.0 * math.cos(math.radians(geom.beam_angle_deg))))
    return v, err


def three_beam_solution(radial, exclude: int, geom: BeamGeometry):
    """Exact inverse using the three beams other than `exclude`."""
    radial = np.asarray(radial, dtype=float)
    a = beam_unit_vectors(geom)
    idx = [i for i in range(geom.n_beams) if i != exclude]
    return np.linalg.solve(a[idx], radial[idx])


def three_beam_candidates(radial, geom: BeamGeometry, bad_beams=()):
    """All legal 3-beam candidates.

    Never collapses to a single value: each candidate keeps its own method tag
    so a 3-beam solution is never mixed with the 4-beam solution.
    """
    cands = []
    for ex in range(geom.n_beams):
        if ex in bad_beams:
            continue
        v = three_beam_solution(radial, ex, geom)
        cands.append({
            "id": f"3beam:ex={ex}",
            "method": "3beam",
            "excluded_beam": ex,
            "value": [float(x) for x in v],
            "unit": "m/s",
            "basis": "instrument",
        })
    return cands


def rot_x(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_yaw(deg):
    """Yaw about z-down, clockwise from north (compass convention)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


TRANSFORM_ORDERS = ("HPR", "PRH")


def attitude_matrix(heading, pitch, roll, order="HPR"):
    if order == "HPR":
        return rot_yaw(heading) @ rot_y(pitch) @ rot_x(roll)
    if order == "PRH":
        return rot_y(pitch) @ rot_x(roll) @ rot_yaw(heading)
    raise ValueError(f"unknown transform order {order!r}")


def instrument_to_ship(v_instr, mounting_angle_deg):
    return rot_yaw(mounting_angle_deg) @ np.asarray(v_instr, dtype=float)


def ship_to_earth(v_ship, heading, pitch, roll, declination_deg, order="HPR"):
    m = attitude_matrix(heading, pitch, roll, order)
    v_mag = m @ np.asarray(v_ship, dtype=float)
    return rot_yaw(declination_deg) @ v_mag
