"""Beam geometry and 3-D coordinate transforms.

Conventions (all versioned, they enter every derived fingerprint):

* 4-beam convex Janus head, beams numbered 1..4.
* Beam unit direction vectors (pointing *away* from the transducer, downward
  for a down-looking ADCP), expressed in the instrument frame::

        b1 = ( s*sin45,  s*sin45, c)
        b2 = (-s*sin45, -s*sin45, c)
        b3 = (-s*sin45,  s*sin45, c)
        b4 = ( s*sin45, -s*sin45, c)

  with s = sin(beam_angle), c = cos(beam_angle).
* Beam velocities are positive for scatterer motion *away* from the
  transducer along the beam, so ``v_beam = B @ v_instrument`` where row ``i``
  of ``B`` is ``b_i``.
* Instrument frame: x forward, y starboard, z down.
* Ship frame: x forward (bow), y starboard, z down.
* Earth frame: east, north, up (ENU).

Two rotation *orders* are implemented deliberately. The shipped rule version
is ``hpr-v1`` (heading applied last, which yields the RDI-standard result);
``hpr-v2`` (heading applied first) is retained as a named alternative so the
case pack can demonstrate that transform order changes the fingerprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TRANSFORM_ORDER_VERSIONS = ("hpr-v1", "hpr-v2")


@dataclass(frozen=True)
class BeamGeometry:
    """Transducer geometry configuration (an independently recorded source)."""

    geometry_id: str
    beam_angle_deg: float = 20.0
    n_beams: int = 4
    convention: str = "convex-janus"
    beam_sign_convention: str = "positive-away"
    # Instrument-frame unit direction vectors, n_beams x 3.

    def directions(self) -> np.ndarray:
        s = math.sin(math.radians(self.beam_angle_deg))
        c = math.cos(math.radians(self.beam_angle_deg))
        q = s / math.sqrt(2.0)
        return np.array(
            [
                [q, q, c],
                [-q, -q, c],
                [-q, q, c],
                [q, -q, c],
            ],
            dtype=float,
        )

    def beam_matrix(self) -> np.ndarray:
        return self.directions()

    def to_dict(self) -> dict:
        return {
            "geometry_id": self.geometry_id,
            "beam_angle_deg": self.beam_angle_deg,
            "n_beams": self.n_beams,
            "convention": self.convention,
            "beam_sign_convention": self.beam_sign_convention,
        }


def solve_instrument(
    beam_velocity: np.ndarray,
    geometry: BeamGeometry,
    valid: np.ndarray,
) -> dict:
    """Solve beam velocities into instrument-frame velocity.

    Returns a dict with keys:

    * ``v``           - least-squares instrument velocity (3,) or None
    * ``rank``        - rank of the active beam matrix rows
    * ``error_velocity`` - RDI-style 4-beam error velocity, or None
    * ``error_direction`` - the q unit 4-vector (1,1,-1,-1)/2, or None
    * ``residual``    - beam-fit residual vector on active beams
    * ``n_beams``     - number of beams used

    With exactly 3 active beams the system is square-exact (rank 3); the
    missing-beam component is not guessed. With 4 beams the solution is
    overdetermined; the component of beam velocity orthogonal to the 3-D
    instrument space is the error velocity.
    """

    bm = geometry.beam_matrix()
    active = np.asarray(valid, dtype=bool)
    vb = np.asarray(beam_velocity, dtype=float)
    a = bm[active]
    y = vb[active]
    rank = int(np.linalg.matrix_rank(a, tol=1e-10))
    result = {
        "v": None,
        "rank": rank,
        "error_velocity": None,
        "error_direction": None,
        "residual": None,
        "n_beams": int(active.sum()),
    }
    if rank < 3:
        return result
    v, *_ = np.linalg.lstsq(a, y, rcond=None)
    result["v"] = v
    result["residual"] = (y - a @ v)
    if active.sum() == 4 and geometry.n_beams == 4:
        q = np.array([0.5, 0.5, -0.5, -0.5])
        full_fit = bm @ v
        result["error_direction"] = q
        # For the symmetric Janus geometry q is exactly orthogonal to the
        # column space of B, so q . vb is the full error velocity and
        # q . (vb - B v_hat) is the same number (checked in tests).
        result["error_velocity"] = float(q @ vb)
    return result


def error_velocity_explicit(
    beam_velocity: np.ndarray,
    geometry: BeamGeometry,
    v_instrument: np.ndarray,
) -> float:
    """4-beam error velocity computed explicitly (used by the tests).

    e = q . (v_beam - B v_hat) = q . v_beam  because q . B == 0.
    """

    q = np.array([0.5, 0.5, -0.5, -0.5])
    vb = np.asarray(beam_velocity, dtype=float)
    bm = geometry.beam_matrix()
    return float(q @ (vb - bm @ np.asarray(v_instrument, dtype=float)))


# ---------------------------------------------------------------------------
# Instrument -> ship (mount / alignment correction about the vertical)
# ---------------------------------------------------------------------------


def rz(angle_deg: float) -> np.ndarray:
    """Active rotation about the z (down) axis, right-handed positive."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def instrument_to_ship(v_instrument: np.ndarray, mount_angle_deg: float) -> np.ndarray:
    """Rotate instrument frame into ship frame.

    ``mount_angle_deg`` is the recorded transducer alignment: the angle
    (clockwise looking down) from ship's bow axis to the instrument x axis.
    ``v_ship = Rz(mount) v_instrument``.
    """

    return rz(mount_angle_deg) @ np.asarray(v_instrument, dtype=float)


# ---------------------------------------------------------------------------
# Ship -> earth via heading / pitch / roll
# ---------------------------------------------------------------------------


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _heading_enu(heading_true_deg: float) -> np.ndarray:
    """Rotate ship-frame (forward,starboard,down) to earth-frame E,N,U.

    Heading 0 -> ship forward points north; 90 -> east.
    A vector (f, st, dn) maps to::

        e =  f sin h + st cos h
        n =  f cos h - st sin h
        u = -dn
    """

    h = math.radians(heading_true_deg)
    c, s = math.cos(h), math.sin(h)
    return np.array(
        [
            [s, c, 0.0],
            [c, -s, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def ship_to_earth(
    v_ship: np.ndarray,
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "hpr-v1",
) -> np.ndarray:
    """Rotate ship-frame vector into earth ENU.

    ``hpr-v1`` (shipped): first undo roll, then pitch, then apply the heading
    to level coordinates - the order used by RDI ADCP processing::

        v_earth = H(h) @ Ry(p) @ Rx(r) @ v_ship

    ``hpr-v2`` (retained alternative): heading applied *before* tilt, which is
    a different coordinate transform and must produce a different fingerprint.
    """

    vs = np.asarray(v_ship, dtype=float)
    h = heading_true_deg
    if order == "hpr-v1":
        return _heading_enu(h) @ _ry(pitch_deg) @ _rx(roll_deg) @ vs
    if order == "hpr-v2":
        # Deliberately non-standard ordering for the order-sensitivity case.
        return _ry(pitch_deg) @ _rx(roll_deg) @ _heading_enu(h) @ vs
    raise ValueError(f"unknown transform order: {order}")


def transform_chain(
    beam_velocity: np.ndarray,
    valid: np.ndarray,
    geometry: BeamGeometry,
    mount_angle_deg: float,
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "hpr-v1",
) -> dict:
    """Full beam -> instrument -> ship -> earth chain for one (ping, cell)."""

    solved = solve_instrument(beam_velocity, geometry, valid)
    out: dict = {
        "beam": np.asarray(beam_velocity, dtype=float),
        "valid": np.asarray(valid, dtype=bool),
        "instrument": None,
        "ship": None,
        "earth": None,
        "error_velocity": solved["error_velocity"],
        "rank": solved["rank"],
        "n_beams": solved["n_beams"],
        "residual": solved["residual"],
        "order": order,
    }
    if solved["v"] is None:
        return out
    vi = solved["v"]
    vs = instrument_to_ship(vi, mount_angle_deg)
    ve = ship_to_earth(vs, heading_true_deg, pitch_deg, roll_deg, order)
    out["instrument"] = vi
    out["ship"] = vs
    out["earth"] = ve
    return out


def bottom_track_range_depth(
    ranges: np.ndarray,
    valid: np.ndarray,
    geometry: BeamGeometry,
) -> dict:
    """Vertical bottom depth from slant ranges.

    Slant range along a beam converts to vertical depth by cos(beam_angle);
    only valid beams participate, and the median of their vertical ranges is
    returned. Fewer than 2 valid beams => no bottom depth.
    """

    mask = np.asarray(valid, dtype=bool)
    rng = np.asarray(ranges, dtype=float)
    if mask.sum() < 2:
        return {"depth_m": None, "n_beams": int(mask.sum()), "per_beam_m": []}
    vertical = rng[mask] * math.cos(math.radians(geometry.beam_angle_deg))
    return {
        "depth_m": float(np.median(vertical)),
        "n_beams": int(mask.sum()),
        "per_beam_m": [float(x) for x in vertical],
    }
