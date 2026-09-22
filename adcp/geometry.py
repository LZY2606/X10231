"""Beam geometry and coordinate transforms.

Frames (documented, versioned, part of every derived fingerprint):
  ISF instrument/ship frame, right handed: X forward(bow), Y starboard, Z down.
  Earth frame: NED, X north, Y east, Z down. The UI renders ENU by axis swap.
Beam velocity sign convention: positive means flow AWAY from the transducer
along the beam. Janus layout, azimuth clockwise from bow viewed from above:
  beam1 forward, beam2 aft, beam3 port, beam4 starboard.
All conventions are fields of Geometry so changing geometry changes fingerprints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

BEAM_COUNT = 4
BEAM_NAMES = ("b1", "b2", "b3", "b4")
# Adopted chain order and the deliberately retained alternative.
ORDER_HPR = "rz(H)-ry(P)-rx(R)"
ORDER_XYZ = "rx(R)-ry(P)-rz(H)"


@dataclass(frozen=True)
class Geometry:
    beam_angle_deg: float = 20.0          # angle from vertical
    azimuths_deg: Tuple[float, float, float, float] = (0.0, 180.0, 270.0, 90.0)
    sign_convention: str = "positive-away"
    frame_instrument: str = "ISF: x-forward y-starboard z-down"
    frame_ship: str = "SHIP: x-forward y-starboard z-down"
    frame_earth: str = "NED: x-north y-east z-down"
    geometry_version: str = "geo-1"

    def directions(self) -> np.ndarray:
        theta = math.radians(self.beam_angle_deg)
        s, c = math.sin(theta), math.cos(theta)
        rows = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            rows.append([s * math.cos(a), s * math.sin(a), c])
        return np.asarray(rows, dtype=float)

    def to_json(self) -> Dict[str, object]:
        return {
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
            "sign_convention": self.sign_convention,
            "frame_instrument": self.frame_instrument,
            "frame_ship": self.frame_ship,
            "frame_earth": self.frame_earth,
            "geometry_version": self.geometry_version,
        }

    @staticmethod
    def from_json(data: Dict[str, object]) -> "Geometry":
        return Geometry(
            beam_angle_deg=float(data["beam_angle_deg"]),
            azimuths_deg=tuple(data["azimuths_deg"]),
            sign_convention=data["sign_convention"],
            frame_instrument=data["frame_instrument"],
            frame_ship=data["frame_ship"],
            frame_earth=data["frame_earth"],
            geometry_version=data["geometry_version"],
        )


def solve_beams(beam_si: Sequence[Optional[float]], geom: Geometry,
                error_tolerance: float = 0.05
                ) -> Dict[str, object]:
    """Solve one depth cell. Never collapses ambiguity into a single answer.

    Always returns the four-beam solution when all beams are usable, plus every
    legal three-beam candidate. Three- and four-beam solutions stay distinct
    values. Error velocity above tolerance flags inconsistency but keeps both.
    """
    values = [None if v is None else float(v) for v in beam_si]
    usable = [i for i, v in enumerate(values) if v is not None]
    dirs = geom.directions()
    out = {"four": None, "triples": [], "usable": usable, "status": "insufficient"}

    def ls(idx: Sequence[int]) -> np.ndarray:
        sub = dirs[list(idx), :]
        rhs = np.array([values[i] for i in idx], dtype=float)
        vec, *_ = np.linalg.lstsq(sub, rhs, rcond=None)
        return vec

    if len(usable) == 4:
        theta = math.radians(geom.beam_angle_deg)
        s, c = math.sin(theta), math.cos(theta)
        b1, b2, b3, b4 = values
        vel = np.array([
            (b1 - b2) / (2 * s),
            (b4 - b3) / (2 * s),
            (b1 + b2 + b3 + b4) / (4 * c),
        ])
        w12 = (b1 + b2) / (2 * c)
        w34 = (b3 + b4) / (2 * c)
        error_velocity = w12 - w34
        consistent = abs(error_velocity) <= error_tolerance
        out["four"] = {
            "kind": "four-beam",
            "beams": (0, 1, 2, 3),
            "velocity": [float(x) for x in vel],
            "frame": "ISF",
            "unit": "m/s",
            "error_velocity": float(error_velocity),
            "status": "ok" if consistent else "inconsistent",
        }
        for missing in range(BEAM_COUNT):
            idx = [i for i in range(BEAM_COUNT) if i != missing]
            vec = ls(idx)
            out["triples"].append({
                "kind": "three-beam",
                "beams": tuple(idx),
                "missing": missing,
                "velocity": [float(x) for x in vec],
                "frame": "ISF",
                "unit": "m/s",
                "error_velocity": None,
                "status": "candidate",
            })
        out["status"] = "ok" if consistent else "ambiguous"
    elif len(usable) == 3:
        vec = ls(usable)
        out["triples"].append({
            "kind": "three-beam",
            "beams": tuple(usable),
            "missing": next(i for i in range(BEAM_COUNT) if i not in usable),
            "velocity": [float(x) for x in vec],
            "frame": "ISF",
            "unit": "m/s",
            "error_velocity": None,
            "status": "candidate",
        })
        out["status"] = "three-beam"
    return out


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def mount_rotation(heading_mount_rad: float, pitch_mount_rad: float,
                   roll_mount_rad: float) -> np.ndarray:
    return _rx(roll_mount_rad) @ _ry(pitch_mount_rad) @ _rz(heading_mount_rad)


def attitude_rotation(heading_true_rad: float, pitch_rad: float, roll_rad: float,
                      order: str = ORDER_HPR) -> np.ndarray:
    if order == ORDER_HPR:
        return _rz(heading_true_rad) @ _ry(pitch_rad) @ _rx(roll_rad)
    if order == ORDER_XYZ:
        return _rx(roll_rad) @ _ry(pitch_rad) @ _rz(heading_true_rad)
    raise ValueError("unknown transform order: %s" % order)
