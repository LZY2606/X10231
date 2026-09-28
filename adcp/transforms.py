"""Instrument -> ship -> earth(ENU) rotations, versioned transform order.

Frames (explicit, tested on canonical cases):
- instrument/ship: X forward, Y starboard, Z *down* (right handed marine frame)
- earth: ENU (east, north, up)

Angles:
- heading: degrees clockwise from *magnetic* north in the attitude record
- declination: east-positive degrees; true heading = heading + declination
- pitch: bow-up positive; roll: starboard-down positive

Two transform-order definitions are shipped deliberately. The order, the
beam geometry and the declination/mount versions all enter the derived
fingerprint, so revising any of them invalidates the right downstream nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def tilt_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Body FSD -> level FSD: roll(starboard-down) after pitch(bow-up)."""
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    return _rx(r) @ _ry(-p)


def mount_matrix(mount_deg: float) -> np.ndarray:
    """Instrument -> ship: transducer azimuth m clockwise from bow (top view)."""
    m = math.radians(mount_deg)
    c, s = math.cos(m), math.sin(m)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def heading_to_enu_matrix(true_heading_deg: float) -> np.ndarray:
    """Level FSD -> ENU: forward->north, starboard->east, down->(-up)."""
    h = math.radians(true_heading_deg)
    s, c = math.sin(h), math.cos(h)
    return np.array([[s, c, 0], [c, -s, 0], [0, 0, -1]], dtype=float)


ORDERS = {
    # mount applied in body frame, then tilt, then heading (RDI HPR style)
    "ord-v1-rdi-body": ["mount", "tilt", "heading"],
    # alternative legal reading: tilt first, mount azimuth at level
    "ord-v2-level-mount": ["tilt", "mount", "heading"],
}


@dataclass
class TransformResult:
    instrument: list
    ship: list
    earth: list | None
    true_heading: float | None
    steps: list                 # [{name, matrix, in_frame, out_frame, vector}]
    order_id: str


def instrument_to_ship(v_instrument, mount_deg: float, order_id: str):
    v = np.asarray(v_instrument, dtype=float)
    M = mount_matrix(mount_deg)
    Rt = None
    steps = [{
        "name": "mount", "matrix": [list(row) for row in M],
        "in_frame": "instrument", "out_frame": "ship",
        "vector": [float(x) for x in M @ v],
    }]
    v_ship = M @ v
    return v_ship, steps


def transform(v_instrument, heading_deg, pitch_deg, roll_deg,
              declination_deg, mount_deg, order_id: str) -> TransformResult:
    """Full chain. Returns each level vector and every matrix step."""
    v = np.asarray(v_instrument, dtype=float)
    order = ORDERS[order_id]
    Mm = mount_matrix(mount_deg)
    Rt = tilt_matrix(pitch_deg, roll_deg)
    true_h = heading_deg + declination_deg
    H = heading_to_enu_matrix(true_h)

    mats = {"mount": (Mm, "ship"), "tilt": (Rt, "level-ship"),
            "heading": (H, "earthENU")}
    vec = {"mount": v, "tilt": v, "heading": v}
    cur = v
    cur_frame = "instrument"
    steps = []
    for name in order:
        mat, out_frame = mats[name]
        cur = mat @ cur
        steps.append({"name": name,
                      "matrix": [list(row) for row in mat],
                      "in_frame": cur_frame, "out_frame": out_frame,
                      "vector": [float(x) for x in cur]})
        cur_frame = out_frame
    v_earth = cur
    # ship vector (before heading) is tilt-complete, mount-included
    v_ship = None
    cur2 = v
    for name in order:
        cur2 = mats[name][0] @ cur2
        if name == "tilt":
            v_ship = cur2
    if v_ship is None:  # pragma: no cover
        v_ship = v_earth
    return TransformResult(
        instrument=[float(x) for x in np.asarray(v_instrument, dtype=float)],
        ship=[float(x) for x in v_ship],
        earth=[float(x) for x in v_earth],
        true_heading=true_h % 360.0,
        steps=steps, order_id=order_id)
