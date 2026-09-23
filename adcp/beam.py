"""Beam geometry and beam -> instrument velocity transformation.

Geometry convention
-------------------
Instrument frame is right-handed: x forward, y starboard, z up.
Each beam is described by an azimuth ``phi`` measured clockwise from the
instrument x axis and a tilt ``theta`` away from the vertical. Radial velocity
is positive *away* from the transducer.

The derived matrices are computed directly from this geometry, so beam
geometry versions enter the derivation fingerprint rather than being implicit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    """Janus beam configuration."""

    beam_angle_deg: float
    azimuths_deg: Tuple[float, ...]
    geometry_version: str

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def forward_matrix(self) -> np.ndarray:
        """B such that radial b = B @ instrument velocity v (n_beams x 3)."""
        theta = math.radians(self.beam_angle_deg)
        rows: List[List[float]] = []
        for phi_deg in self.azimuths_deg:
            phi = math.radians(phi_deg)
            rows.append(
                [
                    -math.sin(theta) * math.cos(phi),
                    -math.sin(theta) * math.sin(phi),
                    -math.cos(theta),
                ]
            )
        return np.asarray(rows, dtype=float)

    def null_axis(self) -> np.ndarray:
        """Unit vector spanning the null space of B.T (the error axis)."""
        bmat = self.forward_matrix()
        _, _, vh = np.linalg.svd(bmat.T)
        axis = vh[-1]
        # Anchor the sign on the first non-zero component for stable labels.
        first = next(component for component in axis if abs(component) > 1e-12)
        if first < 0:
            axis = -axis
        return axis / np.linalg.norm(axis)


def rdi_janus(beam_angle_deg: float = 20.0) -> BeamGeometry:
    """RDI-style four-beam Janus geometry.

    Beam 1 forward/port, beam 2 forward/starboard, beam 3 aft/starboard,
    beam 4 aft/port (azimuths clockwise from instrument x).
    """
    return BeamGeometry(
        beam_angle_deg=beam_angle_deg,
        azimuths_deg=(315.0, 45.0, 135.0, 225.0),
        geometry_version=f"rdi-janus-theta{beam_angle_deg:g}",
    )


@dataclass
class BeamSolution:
    """One legal beam-instrument inversion candidate."""

    kind: str  # "four_beam" | "three_beam"
    used_beams: Tuple[int, ...]
    velocity: Tuple[float, float, float]  # m/s instrument xyz
    residual: Optional[float]  # four-beam error velocity, m/s; None for 3-beam
    error_unit: Optional[Tuple[float, ...]]

    def candidate_id(self) -> str:
        if self.kind == "four_beam":
            return "four-beam"
        missing = sorted(set(range(4)) - set(self.used_beams))
        return "three-beam-missing:" + (",".join(str(m + 1) for m in missing) or "none")


def _solve_subset(bmat: np.ndarray, beams: Sequence[int], radial: Sequence[float]) -> Tuple[float, float, float]:
    sub = bmat[list(beams), :]
    rhs = np.asarray([radial[i] for i in beams], dtype=float)
    velocity, *_ = np.linalg.lstsq(sub, rhs, rcond=None)
    return tuple(float(component) for component in velocity)  # type: ignore[return-value]


def solve_beam_velocities(
    radial: Sequence[Optional[float]],
    good: Sequence[bool],
    suspect: Sequence[bool],
    geometry: BeamGeometry,
) -> Dict[str, object]:
    """Enumerate every *legal* inversion candidate without collapsing them.

    - a hard-bad beam is never used;
    - with 3 good beams the three-beam least squares solution is unique;
    - when a beam is suspect (failed quality thresholds but not hard-bad) both
      the four-beam and the drop-suspect three-beam solutions are kept;
    - fewer than three usable beams is underdetermined.

    Three-beam and four-beam answers are never averaged into a single value.
    """
    bmat = geometry.forward_matrix()
    n = geometry.n_beams
    good_idx = [i for i in range(n) if good[i] and radial[i] is not None]
    suspect_idx = [i for i in range(n) if suspect[i] and radial[i] is not None]
    usable = sorted(set(good_idx) | set(suspect_idx))

    candidates: List[BeamSolution] = []
    if len(usable) < 3:
        return {
            "status": "underdetermined",
            "reason": f"only {len(usable)} usable beams",
            "candidates": [],
            "error_unit": None,
        }

    four = len(usable) == 4
    if four:
        velocity = _solve_subset(bmat, usable, radial)
        error_axis = geometry.null_axis()
        measured = np.asarray(radial, dtype=float)
        predicted = bmat @ np.asarray(velocity)
        error_velocity = float(np.dot(error_axis, measured - predicted))
        candidates.append(
            BeamSolution(
                kind="four_beam",
                used_beams=tuple(usable),
                velocity=velocity,
                residual=error_velocity,
                error_unit=tuple(float(c) for c in error_axis),
            )
        )
        for beam in suspect_idx:
            subset = [i for i in usable if i != beam]
            candidates.append(
                BeamSolution(
                    kind="three_beam",
                    used_beams=tuple(subset),
                    velocity=_solve_subset(bmat, subset, radial),
                    residual=None,
                    error_unit=None,
                )
            )
        status = "ambiguous_multi_solution" if suspect_idx else "four_beam_ok"
        return {
            "status": status,
            "reason": "suspect beam keeps both 4-beam and 3-beam legal candidates"
            if suspect_idx
            else "",
            "candidates": candidates,
            "error_unit": candidates[0].error_unit,
        }

    if len(good_idx) == 3 and not suspect_idx:
        velocity = _solve_subset(bmat, good_idx, radial)
        candidates.append(
            BeamSolution(
                kind="three_beam",
                used_beams=tuple(good_idx),
                velocity=velocity,
                residual=None,
                error_unit=None,
            )
        )
        return {
            "status": "three_beam_determined",
            "reason": "unique inversion over the three good beams",
            "candidates": candidates,
            "error_unit": None,
        }

    # Three usable beams but the quality picture is mixed.
    subset = usable[:3] if len(usable) == 3 else usable
    if len(subset) == 3:
        velocity = _solve_subset(bmat, subset, radial)
        candidates.append(
            BeamSolution(
                kind="three_beam",
                used_beams=tuple(subset),
                velocity=velocity,
                residual=None,
                error_unit=None,
            )
        )
        return {
            "status": "three_beam_determined",
            "reason": "unique inversion over three usable beams",
            "candidates": candidates,
            "error_unit": None,
        }

    return {"status": "underdetermined", "reason": "no usable three-beam subset", "candidates": [], "error_unit": None}
