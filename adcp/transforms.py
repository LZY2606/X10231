"""ADCP coordinate transforms.

Stages: beam -> instrument -> ship -> earth.

Rules enforced here (mirrored as machine-readable events by the pipeline):
* three-beam and four-beam solutions are never merged into one value;
* attitude is never interpolated across a time gap;
* heading interpolation unwraps across the 360-degree seam;
* transform order, beam geometry and magnetic-declination version feed the
  derived fingerprint (see adcp.fingerprint);
* ambiguous states (e.g. degenerate three-beam geometry) keep every legal
  candidate instead of being collapsed to a single answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Beam geometry
# ---------------------------------------------------------------------------

# Convex Janus configuration: beams 1..4, azimuths 45/135/225/315 deg is one
# common layout; we use the RDI-style convention where beam k points along
# unit vector (sin(a)*cos(phi_k), sin(a)*sin(phi_k), -cos(a)) in instrument
# coords (z up, so beams point down).  Radial velocity along the beam is
# positive away from the transducer.
BEAM_AZIMUTHS_DEG = {1: 270.0, 2: 90.0, 3: 0.0, 4: 180.0}


def beam_unit_vectors(beam_angle_deg: float) -> Dict[int, np.ndarray]:
    a = np.radians(beam_angle_deg)
    out = {}
    for k, phi_deg in BEAM_AZIMUTHS_DEG.items():
        phi = np.radians(phi_deg)
        out[k] = np.array([
            np.sin(a) * np.sin(phi),
            np.sin(a) * np.cos(phi),
            -np.cos(a),
        ])
    return out


def beam_matrix(beam_angle_deg: float, beams: Sequence[int]) -> np.ndarray:
    """Rows map instrument velocity v to radial beam velocity: b = M @ v."""
    uv = beam_unit_vectors(beam_angle_deg)
    return np.stack([uv[k] for k in beams])


@dataclass
class BeamSolution:
    kind: str                       # 'beam4' or 'beam3'
    beams: Tuple[int, ...]
    velocity: Optional[np.ndarray]  # instrument-frame velocity, None if none
    error_velocity: Optional[float] = None
    status: str = "ok"              # ok | ambiguous | impossible
    candidates: List[np.ndarray] = field(default_factory=list)
    diagnostic: Optional[str] = None


def solve_four_beam(radials: Dict[int, float], beam_angle_deg: float) -> BeamSolution:
    beams = (1, 2, 3, 4)
    M = beam_matrix(beam_angle_deg, beams)
    b = np.array([radials[k] for k in beams])
    vel, *_ = np.linalg.lstsq(M, b, rcond=None)
    resid = b - M @ vel
    err = float(np.linalg.norm(resid) / 2.0)
    return BeamSolution("beam4", beams, vel, error_velocity=err)


def solve_three_beam(radials: Dict[int, float], beams: Sequence[int],
                     beam_angle_deg: float, rank_tol: float = 1e-9) -> BeamSolution:
    """Three-beam solution.  Never merged with the four-beam value.

    If the beam geometry is (near-)degenerate the system has a null space:
    every legal candidate along it is returned instead of one picked value.
    """
    beams = tuple(beams)
    M = beam_matrix(beam_angle_deg, list(beams))
    b = np.array([radials[k] for k in beams])
    u, s, vt = np.linalg.svd(M)
    rank = int((s > rank_tol).sum())
    if rank >= 3:
        vel = np.linalg.solve(M, b)
        return BeamSolution("beam3", beams, vel)
    # degenerate: particular solution + null-space family of candidates
    particular = np.linalg.lstsq(M, b, rcond=None)[0]
    null = vt[rank:]                      # rows spanning the null space
    candidates: List[np.ndarray] = []
    if null.shape[0] > 0:
        # legal candidates: particular plus/minus unit steps along each null
        # direction (bounded so the list stays finite and inspectable)
        for direction in null:
            for scale in (-1.0, 0.0, 1.0):
                candidates.append(particular + scale * direction)
    return BeamSolution(
        "beam3", beams, None, status="ambiguous",
        candidates=candidates,
        diagnostic=(f"three-beam geometry rank {rank}<3: solution not unique; "
                    f"{len(candidates)} legal candidates retained"),
    )


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


#: recognised transform orders for ship->earth attitude application
TRANSFORM_ORDERS = ("heading-pitch-roll", "heading-roll-pitch")


def instrument_to_ship(vel_instr: np.ndarray, mounting_angle_deg: float) -> np.ndarray:
    """Rotate instrument velocity into the ship frame by the transducer
    mounting (alignment) angle about the vertical axis."""
    return rot_z(mounting_angle_deg) @ vel_instr


def ship_to_earth(vel_ship: np.ndarray, heading_deg: float, pitch_deg: float,
                  roll_deg: float, declination_deg: float,
                  order: str = "heading-pitch-roll") -> np.ndarray:
    """Rotate ship-frame velocity to earth frame (true north).

    ``order`` selects the rotation sequence and is part of the derived
    fingerprint; magnetic declination converts magnetic heading to true.
    """
    if order not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order: {order}")
    true_heading = (heading_deg + declination_deg) % 360.0
    R_h = rot_z(-true_heading)
    R_p = rot_y(pitch_deg)
    R_r = rot_x(roll_deg)
    if order == "heading-pitch-roll":
        R = R_h @ R_p @ R_r
    else:
        R = R_h @ R_r @ R_p
    return R @ vel_ship


# ---------------------------------------------------------------------------
# Attitude sampling: 360-degree unwrap, no interpolation across gaps
# ---------------------------------------------------------------------------

@dataclass
class AttitudeSample:
    t: float            # seconds, epoch-relative
    heading: float      # deg, magnetic, 0..360
    pitch: float        # deg
    roll: float         # deg


def unwrap_headings(headings: Sequence[float]) -> np.ndarray:
    """Unwrap a heading series so interpolation never jumps across 0/360."""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(headings, dtype=float))))


def interpolate_attitude(samples: Sequence[AttitudeSample], t: float,
                         max_gap_s: float = 30.0) -> Tuple[Optional[AttitudeSample], Optional[str]]:
    """Interpolate attitude at time ``t``.

    Returns (sample, diagnostic).  No interpolation across a time gap: if the
    bracketing samples are more than ``max_gap_s`` apart the result is None
    and the caller must keep the ping unconverted with the diagnostic.
    """
    if not samples:
        return None, "no-attitude-samples"
    samples = sorted(samples, key=lambda s: s.t)
    if t <= samples[0].t:
        s = samples[0]
        return AttitudeSample(t, s.heading, s.pitch, s.roll), None
    if t >= samples[-1].t:
        s = samples[-1]
        return AttitudeSample(t, s.heading, s.pitch, s.roll), None
    lo = None
    hi = None
    for a, b in zip(samples[:-1], samples[1:]):
        if a.t <= t <= b.t:
            lo, hi = a, b
            break
    if lo is None or hi is None:
        return None, "no-bracketing-samples"
    if hi.t - lo.t > max_gap_s:
        return None, (f"attitude-gap:{hi.t - lo.t:.1f}s>{max_gap_s:.1f}s: "
                      "interpolation-across-gap-refused")
    span = hi.t - lo.t
    w = 0.0 if span == 0 else (t - lo.t) / span
    headings = unwrap_headings([lo.heading, hi.heading])
    heading = float(headings[0] + w * (headings[1] - headings[0])) % 360.0
    return AttitudeSample(t, heading,
                          lo.pitch + w * (hi.pitch - lo.pitch),
                          lo.roll + w * (hi.roll - lo.roll)), None
