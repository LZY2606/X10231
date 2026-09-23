"""Coordinate conversion rules: beam -> instrument -> ship -> earth.

All functions are pure.  Angles are passed in degrees, velocities in m/s.
Quantities are represented as plain dicts carrying ``unit`` and ``frame`` so
that every value keeps its unit / coordinate reference (see pipeline).

Conventions used by the bundled "flowbeam bench" case (documented here rather
than hidden in the UI):

* Four Janus beams, convex downward-looking head, beam angle ``theta`` from
  vertical; beams 1/3 form the fore-aft pair, beams 2/4 the athwart pair.
* Beam velocities are positive away from the transducer and instrument axes
  are x=fore, y=starboard, z=up.
* Ship axes match instrument axes up to the mounting angle (rotation about z,
  clockwise/positive when instrument x is turned toward starboard of bow).
* Earth axes are east, north, up.  Heading is degrees clockwise from north;
  the magnetic declination is added to the magnetic heading with east
  positive (``true = magnetic + declination``).
* ``transform_order`` describes the heading/pitch/roll application order;
  ``"HPR"`` is R = H @ P @ R (heading applied last), ``"PRH"`` is the
  reversed order.  The chosen order is part of every derived fingerprint.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading_rot(heading_true_deg: float) -> np.ndarray:
    """Ship->earth heading rotation.

    Heading h measured clockwise from north: at h=0 ship x (bow) -> north,
    ship y (starboard) -> east.  Earth coords are (east, north, up).
    """
    a = math.radians(heading_true_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[s, c, 0.0], [c, -s, 0.0], [0.0, 0.0, 1.0]])


def ship_to_earth_matrix(
    heading_mag_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    transform_order: str,
) -> np.ndarray:
    """Rotation matrix from ship frame to earth frame."""
    h_true = heading_mag_deg + declination_deg
    mh = heading_rot(h_true)
    mp = _rot_y(pitch_deg)
    mr = _rot_x(roll_deg)
    if transform_order == "HPR":
        return mh @ mp @ mr
    if transform_order == "PRH":
        return mr @ mp @ mh
    raise ValueError("unsupported transform_order: %r" % (transform_order,))


def mounting_rot(mounting_angle_deg: float) -> np.ndarray:
    """Instrument->ship rotation about z (mounting angle)."""
    return _rot_z(-mounting_angle_deg)


def beam_to_instrument(
    beam_velocities: Sequence[Optional[float]],
    beam_angle_deg: float,
) -> Dict[str, object]:
    """Convert up to four beam velocities to instrument-frame velocity.

    Returns a status report rather than a single magic vector:

    * ``status="four_beam"`` -> one solution plus error velocity.
    * ``status="three_beam"`` -> ``candidates`` lists every legal solution
      implied by a documented missing-beam assumption.  Candidates are never
      collapsed into one value and are never mixed with the four-beam
      solution of neighbouring pings.
    * ``status="impossible"`` -> fewer than three usable beams.

    A beam entry of ``None`` is treated as invalid/failed.
    """
    if len(beam_velocities) != 4:
        raise ValueError("exactly four beam velocities expected")
    s = math.sin(math.radians(beam_angle_deg))
    c = math.cos(math.radians(beam_angle_deg))
    good = [i for i, v in enumerate(beam_velocities) if v is not None]
    if len(good) == 4:
        b1, b2, b3, b4 = [float(v) for v in beam_velocities]
        u = (b1 - b3) / (2.0 * s)
        v = (b2 - b4) / (2.0 * s)
        w = -(b1 + b2 + b3 + b4) / (4.0 * c)
        w_13 = -(b1 + b3) / (2.0 * c)
        w_24 = -(b2 + b4) / (2.0 * c)
        err = w_13 - w_24
        return {
            "status": "four_beam",
            "solution": {"candidate": "four_beam", "vec": [u, v, w]},
            "candidates": [{"candidate": "four_beam", "vec": [u, v, w]}],
            "error_velocity": err,
            "failed_beams": [],
        }
    if len(good) == 3:
        failed = [i + 1 for i in range(4) if i not in good]
        bv = [float(v) if v is not None else None for v in beam_velocities]

        # Beam model: b1 = u*s - w*c, b3 = -u*s - w*c,
        #             b2 = v*s - w*c, b4 = -v*s - w*c.
        def u_from_beams(w):
            if bv[0] is not None:
                return (bv[0] + w * c) / s
            return -(bv[2] + w * c) / s

        def v_from_beams(w):
            if bv[1] is not None:
                return (bv[1] + w * c) / s
            return -(bv[3] + w * c) / s

        def pair_w(pair):
            i, j = pair
            return -0.5 * (bv[i - 1] + bv[j - 1]) / c

        candidates: List[Dict[str, object]] = []
        # Candidate A: take w from the intact opposing pair (fore-aft or
        # athwart), then solve the two horizontal components.
        if failed[0] in (1, 3):
            w_a = pair_w((2, 4))
            u_a = u_from_beams(w_a)
            v_a = (bv[1] - bv[3]) / (2.0 * s)
            candidates.append({
                "candidate": "intact_pair_w",
                "assumption": "vertical from intact beams 2+4, "
                              "error velocity unobservable",
                "vec": [u_a, v_a, w_a],
            })
        else:
            w_a = pair_w((1, 3))
            u_a = (bv[0] - bv[2]) / (2.0 * s)
            v_a = v_from_beams(w_a)
            candidates.append({
                "candidate": "intact_pair_w",
                "assumption": "vertical from intact beams 1+3, "
                              "error velocity unobservable",
                "vec": [u_a, v_a, w_a],
            })

        # Candidate B: three-beam planar solution, w = -mean(beams)/c,
        # horizontal component from the surviving member of the broken pair.
        vals = [bv[i - 1] for i in good]
        w_b = -sum(vals) / (3.0 * c)
        u_b = u_from_beams(w_b)
        v_b = v_from_beams(w_b)
        candidates.append({
            "candidate": "three_beam_mean_w",
            "assumption": "vertical from mean of the three surviving beams, "
                          "error velocity unobservable",
            "vec": [u_b, v_b, w_b],
        })
        return {
            "status": "three_beam",
            "solution": None,
            "candidates": candidates,
            "error_velocity": None,
            "failed_beams": failed,
        }
    return {
        "status": "impossible",
        "solution": None,
        "candidates": [],
        "error_velocity": None,
        "failed_beams": [i + 1 for i in range(4) if i not in good],
    }


def instrument_to_ship(vec: Sequence[float], mounting_angle_deg: float):
    return (mounting_rot(mounting_angle_deg) @ np.asarray(vec, float)).tolist()


def ship_to_earth(
    vec: Sequence[float],
    heading_mag_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    transform_order: str,
):
    mat = ship_to_earth_matrix(
        heading_mag_deg, pitch_deg, roll_deg, declination_deg, transform_order
    )
    return (mat @ np.asarray(vec, float)).tolist()


def bottom_track_ship_to_earth(
    bt_ship: Sequence[float],
    heading_mag_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    transform_order: str,
):
    """Bottom-track ship velocity over ground, rotated into earth frame."""
    return ship_to_earth(
        bt_ship, heading_mag_deg, pitch_deg, roll_deg,
        declination_deg, transform_order,
    )


def earth_water_velocity(water_earth, ship_earth):
    """Absolute water velocity = measured (relative) + ship motion."""
    return [float(a) + float(b) for a, b in zip(water_earth, ship_earth)]
