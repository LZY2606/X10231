"""Instrument -> ship -> earth rotation matrices.

Angles are supplied in radians.  Both coordinate frames are right-handed:
x forward, y starboard, z down (FRD).  Earth output is ENU, i.e. the
z-down water column velocity becomes a negative-up component.

The transform order is itself a versioned rule so it enters every derived
fingerprint:

* ``hpr-v1``: intrinsic heading -> pitch -> roll,
  ``R = Rz(h) @ Ry(p) @ Rx(r)``  (hydrology convention used here);
* ``rph-v1``: intrinsic roll -> pitch -> heading,
  ``R = Rx(r) @ Ry(p) @ Rz(h)``  (reverse order, used to prove order matters).
"""
from __future__ import annotations

import numpy as np

ORDERS = ("hpr-v1", "rph-v1")
ROTATION_RULE = "frd-rotation-v1"

# FRD(x forward,y starboard,z down) -> ENU(east,north,up)
FRD_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def frd_rotation(heading_rad, pitch_rad, roll_rad, order="hpr-v1"):
    """Attitude rotation in the FRD frame for the named transform order."""
    h, p, r = heading_rad, pitch_rad, roll_rad
    if order == "hpr-v1":
        return rz(h) @ ry(p) @ rx(r)
    if order == "rph-v1":
        return rx(r) @ ry(p) @ rz(h)
    raise ValueError("unknown transform order: %r" % order)


def instrument_to_ship(vec, ih_deg, ip_deg, ir_deg, order="hpr-v1"):
    """Rotate an instrument-frame vector into the ship frame.

    ``ih/ip/ir`` are the transducer mounting alignment in degrees.
    """
    r = frd_rotation(np.deg2rad(ih_deg), np.deg2rad(ip_deg), np.deg2rad(ir_deg), order)
    return r @ np.asarray(vec, dtype=float)


def ship_to_earth(vec, heading_rad, pitch_rad, roll_rad, order="hpr-v1"):
    """Rotate a ship-frame vector into ENU earth coordinates."""
    r = frd_rotation(heading_rad, pitch_rad, roll_rad, order)
    return FRD_TO_ENU @ r @ np.asarray(vec, dtype=float)


def transform_chain(
    instrument_vec,
    heading_rad,
    pitch_rad,
    roll_rad,
    mount_align,
    order="hpr-v1",
):
    """Return per-stage vectors for the UI: beam->instrument caller-supplied."""
    ship = instrument_to_ship(
        instrument_vec,
        mount_align.get("heading_deg", 0.0),
        mount_align.get("pitch_deg", 0.0),
        mount_align.get("roll_deg", 0.0),
        order,
    )
    earth = ship_to_earth(ship, heading_rad, pitch_rad, roll_rad, order)
    return {
        "instrument": [float(x) for x in instrument_vec],
        "ship": [float(x) for x in ship],
        "earth_ENU": [float(x) for x in earth],
        "order": order,
    }
