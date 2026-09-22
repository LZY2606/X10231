"""Beam geometry and the beam <-> instrument velocity transform.

Documented convention used everywhere in this codebase:

* Instrument frame: ``x`` starboard, ``y`` forward, ``z`` upward
  (right-handed).
* Beam azimuth is measured clockwise from instrument ``+y`` (forward), in
  degrees.
* Beam direction vectors point *away* from the transducer, into the water.
  For a downward-looking head their ``z`` component is negative.
* Radial beam velocity ``r_i = u . b_i`` is positive *away* from the
  transducer.  The transducer configuration carries ``radial_sign``
  (``"away"`` or ``"toward"``); raw measurements are flipped on import when
  the instrument reports positive toward the transducer.

The geometry is fully data-driven (azimuth list + beam angle), so the beam
geometry version becomes part of every derived fingerprint.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GEOMETRY_SCHEMA_VERSION = "beam-geometry/1"

# RDI-style convex Janus default: 4 beams, 20 degrees from vertical.
DEFAULT_AZIMUTHS_DEG: Tuple[float, float, float, float] = (45.0, 315.0, 225.0, 135.0)
DEFAULT_BEAM_ANGLE_DEG = 20.0


@dataclass(frozen=True)
class BeamGeometry:
    azimuths_deg: Tuple[float, ...]
    beam_angle_deg: float
    radial_sign: str = "away"  # or "toward"
    look_direction: str = "down"  # "down" or "up"
    convention_version: str = GEOMETRY_SCHEMA_VERSION

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def directions(self) -> np.ndarray:
        """Return ``(n_beams, 3)`` unit vectors pointing into the water."""
        theta = math.radians(self.beam_angle_deg)
        s, c = math.sin(theta), math.cos(theta)
        z_sign = -1.0 if self.look_direction == "down" else 1.0
        rows = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            rows.append((s * math.sin(a), s * math.cos(a), z_sign * c))
        return np.asarray(rows, dtype=float)

    def beam_matrix(self) -> np.ndarray:
        """Matrix ``B`` with ``radials = B @ u_instrument``."""
        return self.directions()

    def null_unit(self) -> np.ndarray:
        """Unit vector spanning the null space of ``B.T`` (error-velocity axis)."""
        b = self.beam_matrix()
        # SVD: last right? for B.T null space -> null of B.T is vector n in
        # beam-space; compute via null space of (B.T).
        _, _, vt = np.linalg.svd(b.T)
        n = vt[-1]
        # Deterministic sign: first nonzero component positive.
        for comp in n:
            if abs(comp) > 1e-12:
                if comp < 0:
                    n = -n
                break
        return n / np.linalg.norm(n)

    def to_fingerprint(self) -> Dict[str, object]:
        return {
            "schema": self.convention_version,
            "azimuths_deg": [round(a, 9) for a in self.azimuths_deg],
            "beam_angle_deg": round(self.beam_angle_deg, 9),
            "radial_sign": self.radial_sign,
            "look_direction": self.look_direction,
        }


def canonical_radials(raw_radials: Sequence[float], geom: BeamGeometry) -> np.ndarray:
    """Flip instrument sign convention into the canonical 'away' convention."""
    r = np.asarray(raw_radials, dtype=float)
    if geom.radial_sign == "toward":
        r = -r
    return r


def solve_four_beam(radials: Sequence[float], geom: BeamGeometry) -> Dict[str, object]:
    """Four-beam Janus solution.

    Returns instrument-frame velocity plus error velocity (the beam-space
    component that no horizontal/vertical velocity can explain).
    """
    r = np.asarray(radials, dtype=float)
    b = geom.beam_matrix()
    btbi = np.linalg.inv(b.T @ b)
    u = btbi @ (b.T @ r)
    n = geom.null_unit()
    error_velocity = float(n @ r)
    residual = float(np.linalg.norm(r - b @ u - n * (n @ r)))
    return {
        "u_instrument": (float(u[0]), float(u[1]), float(u[2])),
        "error_velocity_mps": error_velocity,
        "residual_mps": residual,
        "solution_kind": "four_beam",
    }


def solve_three_beam(
    radials: Sequence[float],
    geom: BeamGeometry,
    dropped: int,
) -> Dict[str, object]:
    """Single 3-beam candidate: solve exactly after dropping beam ``dropped``.

    A 3-beam solution implicitly assumes the projection of the true flow onto
    the missing-beam null direction is zero; that assumption is recorded.
    """
    r = np.asarray(radials, dtype=float)
    keep = [i for i in range(geom.n_beams) if i != dropped]
    if len(keep) != 3:
        raise ValueError("three-beam solve requires exactly 3 retained beams")
    b3 = geom.beam_matrix()[keep, :]
    r3 = r[keep]
    u = np.linalg.solve(b3, r3)
    # Reconstruct all four radials to expose the implied value on the dropped
    # beam; its disagreement with measurement is what distinguishes candidates.
    implied = geom.beam_matrix() @ u
    return {
        "u_instrument": (float(u[0]), float(u[1]), float(u[2])),
        "dropped_beam": int(dropped),
        "retained_beams": [int(i) for i in keep],
        "implied_dropped_radial_mps": float(implied[dropped]),
        "measured_dropped_radial_mps": float(r[dropped]),
        "dropped_radial_gap_mps": float(r[dropped] - implied[dropped]),
        "solution_kind": "three_beam",
        "assumes_error_velocity_zero": True,
    }


def three_beam_candidates(
    radials: Sequence[float],
    geom: BeamGeometry,
    bad_beams: Sequence[int],
    ambiguous: bool = False,
) -> List[Dict[str, object]]:
    """Return every *legal* 3-beam candidate.

    * Normal failure (``bad_beams`` marks exactly one beam, ``ambiguous`` is
      False): one candidate, dropping that beam.
    * Ambiguous quality (suspect beam(s) or ``ambiguous=True``): all subsets of
      three beams that contain no hard-bad beam are retained as distinct
      candidates.  Multiplicity is never collapsed to a single value.
    """
    bad = set(int(i) for i in bad_beams)
    candidates: List[Dict[str, object]] = []
    if not ambiguous and len(bad) == 1:
        (dropped,) = bad
        return [solve_three_beam(radials, geom, dropped)]
    if len(bad) > 2:
        return []
    for dropped in range(geom.n_beams):
        if dropped in bad:
            continue
        candidates.append(solve_three_beam(radials, geom, dropped))
    return candidates


def instrument_to_radials(
    u_instrument: Sequence[float],
    geom: BeamGeometry,
    error_velocity: float = 0.0,
) -> Tuple[float, ...]:
    """Forward model used by the synthetic fixture generator."""
    b = geom.beam_matrix()
    r = b @ np.asarray(u_instrument, dtype=float)
    if error_velocity:
        r = r + geom.null_unit() * error_velocity
    return tuple(float(x) for x in r)
