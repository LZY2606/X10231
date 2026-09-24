"""Beam velocity -> instrument frame.

4-beam solutions (one unknown error velocity) and 3-beam solutions are kept
strictly separate: they never collapse into a single value.  Three-beam
situations are mathematically ambiguous (3 radial equations for 3 velocity
components but every estimator carries an assumption); we retain *all* legal
candidates and never silently pick one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .geometry import BeamGeometry, JANUS_CONVEX


@dataclass
class BeamCandidate:
    """One legal instrument-frame solution for one ping x cell."""
    estimator: str                 # "beam4_nullspace" | "beam3_direct" | "beam3_wzero"
    n_beams: int
    instrument_xyz_mps: tuple      # (vx, vy, vz) m/s, X=starboard,Y=forward,Z=up
    error_velocity_mps: Optional[float] = None
    assumption: Optional[str] = None
    used_beams: tuple = ()
    diagnostics: List[dict] = field(default_factory=list)


def _is_janus(geo: BeamGeometry) -> bool:
    if geo.n_beams != 4 or abs(geo.beam_angle_deg - 20.0) > 1e-9:
        return False
    want = (45.0, 315.0, 135.0, 225.0)
    return all(abs(a - b) < 1e-9 for a, b in zip(geo.azimuths_deg, want))


def _rdy_pair_values(r: Sequence[float], theta: float):
    """Pair-based values for the standard Janus convex layout.

    r = [b1,b2,b3,b4] all present.
      vx = (b1 - b2) / (2 sin t);  vy = (b1 + b2 - b3 - b4) / (4 sin t)
      wz pairs: p13 = (b1+b3)/(-2cos t); p24 = (b2+b4)/(-2cos t)
    RDI-style error velocity reported here is (p13 - p24)/2.
    """
    s, c = math.sin(theta), math.cos(theta)
    b1, b2, b3, b4 = r
    vx = (b1 - b2) / (2 * s)
    vy = (b1 + b2 - b3 - b4) / (4 * s)
    w13 = (b1 + b3) / (-2 * c)
    w24 = (b2 + b4) / (-2 * c)
    vz = (w13 + w24) / 2.0
    e_pair = (w13 - w24) / 2.0
    return vx, vy, vz, e_pair


def solve_cell(radials_mps, beam_good, geometry: BeamGeometry = JANUS_CONVEX,
               corr_ok=None) -> List[BeamCandidate]:
    """All legal candidates for a single cell.

    radials_mps / beam_good are per-beam sequences length n_beams; entries may
    be None when the raw value did not normalise.  corr_ok marks beams that
    passed correlation/amplitude quality gating.
    """
    dirs = geometry.directions()
    n = geometry.n_beams
    good = [False] * n
    for i in range(n):
        if (beam_good is not None and i < len(beam_good) and not beam_good[i]):
            continue
        if corr_ok is not None and i < len(corr_ok) and not corr_ok[i]:
            continue
        r = radials_mps[i] if i < len(radials_mps) else None
        good[i] = r is not None and np.isfinite(r)
    used = tuple(i + 1 for i, g in enumerate(good) if g)
    n_good = len(used)
    if n_good < 3:
        return []

    theta = math.radians(geometry.beam_angle_deg)
    if n_good == 4 and _is_janus(geometry):
        r = [float(radials_mps[i]) for i in range(4)]
        vx, vy, vz, e_pair = _rdy_pair_values(r, theta)
        cand = BeamCandidate(
            estimator="beam4_nullspace", n_beams=4,
            instrument_xyz_mps=(vx, vy, vz), error_velocity_mps=e_pair,
            used_beams=used,
            diagnostics=[{"code": "FOUR_BEAM", "error_convention":
                          "rdi_pair_half_difference"}],
        )
        return [cand]

    if n_good == 4:
        A = dirs
        sol, *_ = np.linalg.lstsq(A, np.array(radials_mps, float), rcond=None)
        resid = float(np.linalg.norm(dirs @ sol - np.array(radials_mps, float)))
        return [BeamCandidate(
            estimator="beam4_lstsq", n_beams=4,
            instrument_xyz_mps=tuple(float(x) for x in sol),
            error_velocity_mps=resid, used_beams=used,
            diagnostics=[{"code": "NON_JANUS_LEAST_SQUARES", "residual": resid}])]

    idx = [i for i, g in enumerate(good) if g]
    A3 = dirs[idx]
    r3 = np.array([radials_mps[i] for i in idx], float)

    # Candidate 1: direct inversion of the three radial equations.
    det = float(np.linalg.det(A3))
    out: List[BeamCandidate] = []
    if abs(det) > 1e-9:
        v3 = np.linalg.solve(A3, r3)
        out.append(BeamCandidate(
            estimator="beam3_direct", n_beams=3,
            instrument_xyz_mps=tuple(float(x) for x in v3),
            assumption="exact three radial equations; vertical velocity "
                       "implicitly fixed by the surviving geometry",
            used_beams=used,
            diagnostics=[{"code": "THREE_BEAM", "missing_beams":
                          tuple(i + 1 for i in range(n) if not good[i]),
                          "determinant": det}]))
    else:
        out.append(BeamCandidate(
            estimator="beam3_direct", n_beams=3, instrument_xyz_mps=(None,)*3,
            assumption="exact three radial equations", used_beams=used,
            diagnostics=[{"code": "SINGULAR_TRIO", "determinant": det}]))

    # Candidate 2: explicitly assume zero vertical velocity (another legal,
    # physically different estimator).  Solve 2 horizontal unknowns by LS.
    Ah = A3[:, :2]
    vh, *_ = np.linalg.lstsq(Ah, r3, rcond=None)
    out.append(BeamCandidate(
        estimator="beam3_wzero", n_beams=3,
        instrument_xyz_mps=(float(vh[0]), float(vh[1]), 0.0),
        assumption="vertical velocity assumed zero (no vertical information "
                   "independent of the chosen beams)",
        used_beams=used,
        diagnostics=[{"code": "THREE_BEAM_WZERO", "missing_beams":
                      tuple(i + 1 for i in range(n) if not good[i])}]))
    return out
