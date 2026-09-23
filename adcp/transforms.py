"""Coordinate transforms: beam -> instrument -> ship -> earth.

Conventions (documented, versioned via GEOMETRY_VERSION):
  * Instrument frame: X forward, Y starboard, Z up.
  * 4-beam Janus, convex, beam angle theta from vertical; beams point down.
  * Beam unit directions in the instrument frame:
      b1=(+s,0,-c) b2=(-s,0,-c) b3=(0,+s,-c) b4=(0,-s,-c), s=sin(t), c=cos(t).
  * Radial beam velocity is positive away from the transducer.
  * instrument->ship: rotation about vertical by the mounting angle.
  * ship->earth: rotation steps applied in the configured order; each step
    left-multiplies the accumulated rotation. Steps:
      "tilt"        = Ry(pitch) @ Rx(roll)
      "heading"     = Rz(heading)
      "declination" = Rz(magnetic_declination)
    The canonical order is ["tilt", "heading", "declination"].

The transform order, beam geometry version and magnetic declination version
all feed the derived fingerprint (see adcp.fingerprint).
"""

from __future__ import annotations

import numpy as np

CANONICAL_ORDER = ["tilt", "heading", "declination"]


def beam_directions(theta_deg: float) -> np.ndarray:
    s = np.sin(np.radians(theta_deg))
    c = np.cos(np.radians(theta_deg))
    return np.array(
        [[s, 0.0, -c], [-s, 0.0, -c], [0.0, s, -c], [0.0, -s, -c]]
    )


def beam_to_instrument_four(beam_vel, theta_deg: float):
    """Four-beam solution -> (instrument velocity (u,v,w), error velocity)."""
    s = np.sin(np.radians(theta_deg))
    c = np.cos(np.radians(theta_deg))
    b1, b2, b3, b4 = beam_vel
    u = (b1 - b2) / (2.0 * s)
    v = (b3 - b4) / (2.0 * s)
    w = -(b1 + b2 + b3 + b4) / (4.0 * c)
    err = (b1 + b2 - b3 - b4) / (4.0 * c)
    return np.array([u, v, w]), err


def beam_to_instrument_three(beam_vel_valid, valid_idx, theta_deg: float,
                             err_assumed: float = 0.0) -> np.ndarray:
    """Three-beam solution for one dropped beam.

    Solves the exact 3x3 system after removing the assumed error-velocity
    contribution. Never merged with a four-beam value by callers; it is one
    legal candidate among possibly several.
    """
    A = beam_directions(theta_deg)[list(valid_idx)]
    sign = np.array([1.0, 1.0, -1.0, -1.0])
    c = np.cos(np.radians(theta_deg))
    b = np.array(beam_vel_valid, dtype=float) - sign[list(valid_idx)] * c * err_assumed
    return np.linalg.solve(A, b)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def instrument_to_ship(v_inst, mounting_deg: float) -> np.ndarray:
    return rot_z(np.radians(mounting_deg)) @ np.asarray(v_inst, dtype=float)


def _step_matrix(step: str, heading_deg, pitch_deg, roll_deg, decl_deg):
    if step == "tilt":
        return rot_y(np.radians(pitch_deg)) @ rot_x(np.radians(roll_deg))
    if step == "heading":
        return rot_z(np.radians(heading_deg))
    if step == "declination":
        return rot_z(np.radians(decl_deg))
    raise ValueError(f"unknown transform step {step!r}")


def ship_to_earth(v_ship, heading_deg, pitch_deg, roll_deg, decl_deg,
                  order=CANONICAL_ORDER) -> np.ndarray:
    """Rotate a ship-frame vector to earth frame using the configured order."""
    R = np.eye(3)
    for step in order:
        R = _step_matrix(step, heading_deg, pitch_deg, roll_deg, decl_deg) @ R
    return R @ np.asarray(v_ship, dtype=float)
