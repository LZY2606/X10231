"""ADCP coordinate transforms.

Conventions (documented, part of the derived fingerprint):
- Instrument frame: x forward (bow), y port, z up.
- Beam geometry "janus-convex-4": beam azimuths B1=0deg, B2=180deg, B3=90deg,
  B4=270deg, each tilted `beam_angle_deg` from vertical toward its azimuth.
    b1 =  u*sin(t) + w*cos(t)
    b2 = -u*sin(t) + w*cos(t)
    b3 =  v*sin(t) + w*cos(t)
    b4 = -v*sin(t) + w*cos(t)
- Instrument -> ship: rotation about z by the mounting (alignment) angle.
- Ship -> earth: rotations about x (roll), y (pitch), z (heading) applied in
  the order named by `order` (e.g. "RPH" = roll, then pitch, then heading).
  The order string is part of the derived fingerprint; different orders give
  different results and are never silently equated.
- Magnetic heading is corrected to true heading by adding the declination of a
  named declination version (also part of the fingerprint).
"""
from __future__ import annotations

import math
import numpy as np

BEAM_GEOMETRY_ID = "janus-convex-4"
BEAM_AZIMUTHS_DEG = (0.0, 180.0, 90.0, 270.0)
VALID_ORDERS = ("RPH", "RHP", "PRH", "PHR", "HRP", "HPR")


def beam_to_instrument(beam_vels, beam_angle_deg):
    """Four-beam solution. beam_vels: sequence of 4 radial velocities (m/s).

    Returns dict(u, v, w, err). This is the *four-beam* solution; it is never
    merged with a three-beam solution into a single value.
    """
    b1, b2, b3, b4 = [float(v) for v in beam_vels]
    s = math.sin(math.radians(beam_angle_deg))
    c = math.cos(math.radians(beam_angle_deg))
    return {
        "u": (b1 - b2) / (2.0 * s),
        "v": (b3 - b4) / (2.0 * s),
        "w": (b1 + b2 + b3 + b4) / (4.0 * c),
        "err": (b1 + b2 - b3 - b4) / (4.0 * c),
        "solution": "four_beam",
    }


def three_beam_candidates(beam_vels, bad_beam, beam_angle_deg, prior_err=None):
    """Three-beam solutions when one beam (index 0..3) is bad.

    Returns a list of legitimate candidate dicts. The list is never collapsed
    to one value by the math layer; selection is a recorded human/policy act.
    Candidates:
      - "assume_zero_error": solve the 3x3 system with err = 0.
      - "carry_prior_error": solve with err taken from the previous good ping
        (only present when prior_err is not None).
    A candidate is marked indeterminate when the remaining geometry is
    ill-conditioned.
    """
    s = math.sin(math.radians(beam_angle_deg))
    c = math.cos(math.radians(beam_angle_deg))
    rows = []
    for az in BEAM_AZIMUTHS_DEG:
        a = math.radians(az)
        rows.append([math.cos(a) * s, math.sin(a) * s, c])
    rows = np.array(rows)  # beam = M @ [u, v, w]
    keep = [i for i in range(4) if i != bad_beam]
    b = np.array([float(beam_vels[i]) for i in keep])

    candidates = []
    m3 = rows[keep]
    cond = float(np.linalg.cond(m3))
    sol, *_ = np.linalg.lstsq(m3, b, rcond=None)
    candidates.append({
        "method": "assume_zero_error",
        "u": float(sol[0]), "v": float(sol[1]), "w": float(sol[2]),
        "err_assumed": 0.0,
        "condition": cond,
        "determinate": bool(cond < 1e6),
        "solution": "three_beam",
    })
    if prior_err is not None:
        # Replace the dropped beam's equation by the error-velocity constraint
        # b1 + b2 - b3 - b4 = 4*cos(t)*err, using the three good beams plus
        # this constraint on the *known* beams only is not closed, so instead
        # solve the 4-beam system with the bad beam synthesised from the
        # constraint: b_bad = (sum_known_signed - 4c*err) with sign pattern.
        signs = [1.0, 1.0, -1.0, -1.0]
        known = sum(signs[i] * float(beam_vels[i]) for i in keep)
        b_bad = signs[bad_beam] * (4.0 * c * float(prior_err) - known)
        full = [float(beam_vels[i]) if i != bad_beam else b_bad for i in range(4)]
        r = beam_to_instrument(full, beam_angle_deg)
        candidates.append({
            "method": "carry_prior_error",
            "u": r["u"], "v": r["v"], "w": r["w"],
            "err_assumed": float(prior_err),
            "condition": cond,
            "determinate": bool(cond < 1e6),
            "solution": "three_beam",
        })
    return candidates


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _vec3(vec):
    if isinstance(vec, dict):
        return np.array([vec["u"], vec["v"], vec["w"]], dtype=float)
    return np.array(vec, dtype=float)


def instrument_to_ship(vec, mounting_angle_deg):
    """Rotate instrument vector by the mounting (alignment) angle about z."""
    m = rot_z(float(mounting_angle_deg))
    out = m @ _vec3(vec)
    return {"u": float(out[0]), "v": float(out[1]), "w": float(out[2])}


def ship_to_earth(vec, heading_deg, pitch_deg, roll_deg, order="RPH"):
    """Rotate ship vector to earth frame applying rotations in `order`.

    order is a permutation of "R" (roll, about x), "P" (pitch, about y),
    "H" (heading, about z). Rotations compose left-to-right:
    order="RPH" -> v_earth = Rz(H) @ Ry(P) @ Rx(R) @ v_ship.
    """
    if order not in VALID_ORDERS:
        raise ValueError(f"unknown transform order {order!r}")
    mats = {"R": rot_x(roll_deg), "P": rot_y(pitch_deg), "H": rot_z(heading_deg)}
    m = np.eye(3)
    for step in order:  # first listed applied first (rightmost)
        m = mats[step] @ m
    out = m @ _vec3(vec)
    return {"u": float(out[0]), "v": float(out[1]), "w": float(out[2]),
            "order": order}


def magnetic_to_true(heading_mag_deg, declination_deg):
    return (float(heading_mag_deg) + float(declination_deg)) % 360.0
