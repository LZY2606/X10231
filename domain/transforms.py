"""Coordinate transforms: beam -> instrument -> ship -> earth.

Core semantics enforced here:
- three-beam solutions are never merged with four-beam solutions into one value;
- attitude interpolation never crosses a time gap;
- transform order, beam geometry and magnetic-declination version all enter
  the derived fingerprints;
- ambiguity is preserved as a candidate set, never silently collapsed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import BeamGeometry

GAP_SECONDS = 60.0  # attitude samples farther apart than this form a gap


def _fp(kind: str, payload: Dict) -> str:
    return kind + ":" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# attitude: unwrap, gap-aware interpolation
# ---------------------------------------------------------------------------

def unwrap_headings(headings_deg: Sequence[float]) -> np.ndarray:
    """Unwrap headings so 359.9 -> 0.1 becomes a +0.2 deg step, not -359.8."""
    arr = np.asarray(headings_deg, dtype=float)
    return np.degrees(np.unwrap(np.radians(arr)))


def interpolate_attitude(
    times: Sequence[float],
    series: Dict[str, Sequence[float]],
    t_query: float,
    gap_seconds: float = GAP_SECONDS,
) -> Tuple[Optional[Dict[str, float]], str]:
    """Linear interpolation of heading/pitch/roll at t_query.

    Heading is interpolated on the unwrapped circle. If t_query falls inside a
    gap (no samples on both sides within gap_seconds of each other), returns
    (None, reason) -- interpolation never crosses a gap.
    """
    t = np.asarray(times, dtype=float)
    order = np.argsort(t)
    t = t[order]
    idx = np.searchsorted(t, t_query)
    if idx < len(t) and t[idx] == t_query:
        i = int(idx)
        return {k: float(np.asarray(v)[order][i]) for k, v in series.items()}, "exact"
    if idx == 0 or idx >= len(t):
        return None, "outside-attitude-range"
    t0, t1 = t[idx - 1], t[idx]
    if t1 - t0 > gap_seconds:
        return None, "attitude-gap"
    w = (t_query - t0) / (t1 - t0)
    out: Dict[str, float] = {}
    for key, vals in series.items():
        v = np.asarray(vals, dtype=float)[order]
        if key == "heading_deg":
            vu = unwrap_headings(v)
            h = vu[idx - 1] + w * (vu[idx] - vu[idx - 1])
            out[key] = float(h % 360.0)
        else:
            out[key] = float(v[idx - 1] + w * (v[idx] - v[idx - 1]))
    return out, "interpolated"


# ---------------------------------------------------------------------------
# stage 1: beam -> instrument (with three-beam candidates)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    kind: str  # "four_beam" | "three_beam"
    dropped_beam: Optional[int]
    xyz: Optional[np.ndarray]
    err: Optional[float]
    status: str  # "ok" | "ambiguous" | "failed"
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "dropped_beam": self.dropped_beam,
            "xyz": None if self.xyz is None else [float(x) for x in self.xyz],
            "err": self.err,
            "status": self.status,
            "note": self.note,
        }


def beam_to_instrument(
    radial: Sequence[Optional[float]],
    geometry: BeamGeometry,
) -> List[Candidate]:
    """Invert radial beam velocities to instrument XYZ.

    radial: per-beam radial velocity in m/s (None = beam flagged bad).
    Returns every legal candidate; three-beam results stay separate from the
    four-beam result. Never merges them into a single value.
    """
    g = geometry.forward_matrix()
    ginv = geometry.inverse_matrix()
    null = geometry.null_vector()
    r = np.array(
        [np.nan if v is None else float(v) for v in radial], dtype=float
    )
    valid = [i for i in range(len(r)) if not math.isnan(r[i])]
    candidates: List[Candidate] = []

    if len(valid) == geometry.n_beams:
        xyz = ginv @ r
        err = float(null @ r)
        candidates.append(Candidate("four_beam", None, xyz, err, "ok"))
    elif len(valid) == geometry.n_beams - 1:
        dropped = [i for i in range(len(r)) if i not in valid][0]
        for drop in range(len(r)):
            keep = [i for i in range(len(r)) if i != drop]
            if not all(k in valid for k in keep):
                continue
            g3 = g[keep, :]
            xyz3, residuals, rank, _ = np.linalg.lstsq(g3, r[keep], rcond=None)
            if rank < 3:
                continue
            resid = float(np.linalg.norm(g3 @ xyz3 - r[keep]))
            note = "assumes error velocity = 0"
            if drop != dropped:
                note = "counterfactual: beam %d was actually good" % (drop + 1)
            candidates.append(
                Candidate("three_beam", drop, xyz3, None, "ambiguous", note + "; residual=%.4g" % resid)
            )
        if not candidates:
            candidates.append(Candidate("three_beam", dropped, None, None, "failed", "singular 3-beam subset"))
    else:
        candidates.append(
            Candidate("insufficient", None, None, None, "failed",
                      "only %d valid beams" % len(valid))
        )
    return candidates


# ---------------------------------------------------------------------------
# stage 2: instrument -> ship (mounting angle, versioned)
# ---------------------------------------------------------------------------

def instrument_to_ship(
    xyz: np.ndarray, mounting_angle_deg: float
) -> np.ndarray:
    """Rotate instrument XYZ into ship frame (starboard, forward, up).

    mounting_angle_deg: heading offset of instrument X axis relative to ship
    forward, positive clockwise (starboard). Applied as Rz(-mounting).
    """
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    a = math.radians(mounting_angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # instrument X->ship fwd, Y->ship stbd after derotation by mounting angle
    fwd = ca * y - sa * x
    stbd = sa * y + ca * x
    return np.array([stbd, fwd, z])


# ---------------------------------------------------------------------------
# stage 3: ship -> earth (order + declination version enter the fingerprint)
# ---------------------------------------------------------------------------

def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ship_to_earth(
    ship_vec: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    declination_deg: float,
    order: str = "HPR",
) -> np.ndarray:
    """Rotate ship vector (stbd, fwd, up) to earth (east, north, up).

    heading_deg: magnetic heading, degrees clockwise from magnetic north.
    declination_deg: magnetic declination, positive east; added to heading.
    order: rotation composition, "HPR" (heading-pitch-roll, RDI default) or
    "RPY" (roll-pitch-yaw). The order is part of the derived fingerprint.
    """
    heading_true = (heading_deg + declination_deg) % 360.0
    # heading matrix: clockwise rotation about up axis == Rz(-(90-h)) in ENU
    h = math.radians(90.0 - heading_true)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    m_frame = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    v_enu = m_frame @ np.asarray(ship_vec, dtype=float)
    if order == "HPR":
        rot = _rz(h) @ _ry(-p) @ _rx(r)
    elif order == "RPY":
        rot = _rz(h) @ (_rx(r) @ _ry(-p))
    else:
        raise ValueError("unknown transform order: %s" % order)
    return rot @ v_enu


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------

def fingerprint_beam_to_instr(geometry: BeamGeometry, radial_raw: List) -> str:
    return _fp("beam2instr", {"geom": geometry.fingerprint(), "radial_raw": radial_raw})


def fingerprint_instr_to_ship(mounting_angle_deg: float, upstream: str) -> str:
    return _fp("instr2ship", {"mounting_angle_deg": mounting_angle_deg, "up": upstream})


def fingerprint_ship_to_earth(
    order: str,
    declination_version: str,
    declination_deg: float,
    attitude_fp: str,
    upstream: str,
) -> str:
    return _fp(
        "ship2earth",
        {
            "order": order,
            "declination_version": declination_version,
            "declination_deg": declination_deg,
            "attitude_fp": attitude_fp,
            "up": upstream,
        },
    )


def fingerprint_attitude(times: Sequence[float], series: Dict[str, Sequence[float]]) -> str:
    return _fp(
        "attitude",
        {
            "times": [float(x) for x in times],
            "heading_deg": [float(x) for x in series["heading_deg"]],
            "pitch_deg": [float(x) for x in series["pitch_deg"]],
            "roll_deg": [float(x) for x in series["roll_deg"]],
        },
    )
