"""Instrument -> ship -> earth coordinate transforms.

Frames: instrument (X starboard, Y forward, Z up), ship (X starboard,
Y forward, Z up), earth (X east, Y north, Z up).  Magnetic declination is
applied when rotating into earth coordinates.  The transform *order*, beam
geometry id and declination version are explicit parameters and enter the
derived fingerprint, so e.g. changing rotation order creates distinct
results rather than overwriting history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .units import wrap360, wrap180

ORDER_CANONICAL = "HPR_canonical"      # R = Rh . Rp . Rr (heading/pitch/roll)
ORDER_TILT_HEADING = "tilp_then_heading"  # deliberate alternative ordering


def rot_pitch(p_deg: float) -> np.ndarray:
    # pitch positive = bow up
    p = math.radians(p_deg)
    c, s = math.cos(p), math.sin(p)
    return np.array([[1, 0, 0],
                     [0, c, s],
                     [0, -s, c]], float)


def rot_roll(r_deg: float) -> np.ndarray:
    # roll positive = starboard down
    r = math.radians(r_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], float)


def rot_heading(h_deg: float) -> np.ndarray:
    # earth = (east, north, up); heading measured clockwise from north
    h = math.radians(h_deg)
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], float)


def rotation_matrix(order: str, heading_deg: float, pitch_deg: float,
                    roll_deg: float) -> np.ndarray:
    Rp, Rr, Rh = rot_pitch(pitch_deg), rot_roll(roll_deg), rot_heading(heading_deg)
    if order == ORDER_CANONICAL:
        return Rh @ Rp @ Rr
    if order == ORDER_TILT_HEADING:
        # alternative (tilt alignment applied after the heading matrix)
        return Rp @ Rh @ Rr
    raise ValueError(f"unknown transform order: {order}")


@dataclass
class MountingAngles:
    heading_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    version: str = "mount-v1"


@dataclass
class TransformStep:
    name: str
    vector: tuple
    frame: str
    matrix: Optional[list] = None
    params: dict = field(default_factory=dict)


@dataclass
class TransformChain:
    steps: List[TransformStep]
    ship_xyz_mps: tuple
    earth_enu_mps: tuple
    declination_version: str
    mount_version: str
    order: str
    diagnostics: List[dict] = field(default_factory=list)


def transform_chain(instrument_xyz, attitude_heading_deg, pitch_deg, roll_deg,
                    declination_deg, mounting: MountingAngles,
                    order: str = ORDER_CANONICAL,
                    declination_version: str = "decl-v1") -> TransformChain:
    v_inst = np.array(instrument_xyz, float)
    steps: List[TransformStep] = []
    steps.append(TransformStep("instrument_raw", tuple(float(x) for x in v_inst),
                               "instrument"))

    # mounting-angle alignment is part of the instrument->ship step
    hp = wrap180(pitch_deg + mounting.pitch_offset_deg)
    hr = wrap180(roll_deg + mounting.roll_offset_deg)
    R_is = rotation_matrix(ORDER_CANONICAL, 0.0, hp, hr)
    v_ship = R_is @ v_inst
    steps.append(TransformStep(
        "instrument_to_ship", tuple(float(x) for x in v_ship), "ship",
        matrix=R_is.tolist(),
        params={"pitch_used_deg": hp, "roll_used_deg": hr,
                "mount_version": mounting.version}))

    heading_mag = wrap360(attitude_heading_deg + mounting.heading_offset_deg)
    heading_true = wrap360(heading_mag + declination_deg)
    R_se = rotation_matrix(order, heading_true, 0.0, 0.0)
    v_earth = R_se @ v_ship
    steps.append(TransformStep(
        "ship_to_earth", tuple(float(x) for x in v_earth), "earth",
        matrix=R_se.tolist(),
        params={"heading_magnetic_deg": heading_mag,
                "declination_deg": declination_deg,
                "heading_true_deg": heading_true,
                "order": order}))

    return TransformChain(
        steps=steps,
        ship_xyz_mps=tuple(float(x) for x in v_ship),
        earth_enu_mps=tuple(float(x) for x in v_earth),
        declination_version=declination_version, mount_version=mounting.version, order=order)
