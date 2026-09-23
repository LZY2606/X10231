"""Transducer beam geometry and beam<->instrument velocity transforms.

Convention (documented so the transform is never guessed):

* instrument frame: X = starboard/right, Y = forward, Z = up
* beam ``i`` aims downward with beam angle ``theta`` from vertical and
  azimuth ``phi_i`` measured clockwise from +Y (forward)
* radial sign ``s_i`` selects +1 (convex / radial-positive-toward-transducer
  convention) or -1 (concave); the matrix is built explicitly from it

    A_i = s_i * (sin(theta) sin(phi_i), sin(theta) cos(phi_i), -cos(theta))

so beam velocities satisfy ``b = A v_inst``.  The four-beam null-space
projection is the RDI-style "error velocity"; the geometry itself determines
the null vector, and three- and four-beam solutions always keep separate
identities.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class BeamGeometry:
    n_beams: int
    beam_angle_deg: float
    azimuths_deg: Tuple[float, ...]
    signs: Tuple[int, ...]
    convention: str = "X-right_Y-forward_Z-up; azimuth clockwise from +Y; b=A v"
    manufacturer: str = "generic-janus"

    @classmethod
    def rdi_janus(cls, beam_angle_deg: float = 20.0) -> "BeamGeometry":
        az = (45.0, 135.0, 225.0, 315.0)
        return cls(4, beam_angle_deg, az, (1, 1, 1, 1), manufacturer="rdi-janus")

    def matrix(self) -> np.ndarray:
        if len(self.azimuths_deg) != self.n_beams:
            raise GeometryError("azimuth count != n_beams")
        if len(self.signs) != self.n_beams:
            raise GeometryError("sign count != n_beams")
        th = math.radians(self.beam_angle_deg)
        rows = []
        for phi_deg, s in zip(self.azimuths_deg, self.signs):
            phi = math.radians(phi_deg)
            rows.append(
                s
                * np.array(
                    [
                        math.sin(th) * math.sin(phi),
                        math.sin(th) * math.cos(phi),
                        -math.cos(th),
                    ]
                )
            )
        return np.vstack(rows)

    def null_vector(self) -> Optional[np.ndarray]:
        """Unit null vector of A (error-velocity direction), or None."""
        a = self.matrix()
        if self.n_beams <= 3:
            return None
        _, _, vt = np.linalg.svd(a, full_matrices=True)
        n = vt[-1]
        first = next((c for c in n if abs(c) > 1e-12), 0.0)
        if first < 0:
            n = -n
        return n / np.linalg.norm(n)

    def fingerprint(self) -> Dict[str, object]:
        return {
            "n_beams": self.n_beams,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
            "signs": list(self.signs),
            "convention": self.convention,
            "manufacturer": self.manufacturer,
        }

    def geom_id(self) -> str:
        blob = json.dumps(self.fingerprint(), sort_keys=True, separators=(",", ":"))
        return "geom_" + hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class InstSolution:
    method: str                 # "beam4" | "beam3" | ...
    velocity: Tuple[float, float, float]   # m/s, instrument frame
    error_velocity: Optional[float]        # m/s, null-space projection
    used_beams: Tuple[int, ...]
    residual: Optional[float]               # m/s, |b - A v|
    null_vector: Optional[Tuple[float, ...]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "velocity": list(self.velocity),
            "error_velocity": self.error_velocity,
            "used_beams": list(self.used_beams),
            "residual": self.residual,
            "null_vector": list(self.null_vector) if self.null_vector else None,
        }


def solve_four(geom: BeamGeometry, beam_vel: Sequence[float]) -> InstSolution:
    """Exact four-beam (or N-beam square) instrument solution + error vel."""
    if geom.n_beams != 4:
        raise GeometryError("solve_four needs a 4-beam geometry")
    a = geom.matrix()
    b = np.asarray(beam_vel, dtype=float)
    v = np.linalg.solve(a, b)
    n = geom.null_vector()
    err = float(n @ b) if n is not None else None
    res = float(np.linalg.norm(b - a @ v))
    return InstSolution(
        method="beam4",
        velocity=tuple(float(x) for x in v),
        error_velocity=err,
        used_beams=tuple(range(4)),
        residual=res,
        null_vector=tuple(float(x) for x in n) if n is not None else None,
    )


def solve_three(
    geom: BeamGeometry, beam_vel: Sequence[float], omit: int
) -> InstSolution:
    """Three-beam instrument solution obtained by omitting beam ``omit``.

    A single missing beam leaves an exactly-determined 3x3 problem: it is one
    legal solution, *not* the same value as the four-beam solution.
    """
    if not (0 <= omit < geom.n_beams):
        raise GeometryError("omit index out of range")
    a = geom.matrix()
    b = np.asarray(beam_vel, dtype=float)
    keep = [i for i in range(geom.n_beams) if i != omit]
    if len(keep) != 3:
        raise GeometryError("three-beam solve needs exactly 3 remaining beams")
    v = np.linalg.solve(a[keep, :], b[keep])
    full = a @ v
    n = geom.null_vector()
    err = float(n @ b) if n is not None else None
    res = float(np.linalg.norm(b[keep] - full[keep]))
    return InstSolution(
        method=f"beam3_omit{omit + 1}",
        velocity=tuple(float(x) for x in v),
        error_velocity=err,
        used_beams=tuple(keep),
        residual=res,
        null_vector=tuple(float(x) for x in n) if n is not None else None,
    )


def three_beam_candidates(
    geom: BeamGeometry, beam_vel: Sequence[float], hard_bad: Sequence[bool]
) -> List[InstSolution]:
    """Every legal three-beam candidate consistent with the hard-bad mask.

    With exactly one hard-bad beam the candidate is forced. With no hard-bad
    beam but a *questionable* one, every omit choice is legal and all
    candidates are returned: ambiguity is never collapsed to a single answer.
    """
    if sum(1 for x in hard_bad if x) > 1:
        return []
    forced = next((i for i, x in enumerate(hard_bad) if x), None)
    omits = [forced] if forced is not None else list(range(geom.n_beams))
    out: List[InstSolution] = []
    for k in omits:
        try:
            out.append(solve_three(geom, beam_vel, k))
        except np.linalg.LinAlgError:
            continue
    return out
