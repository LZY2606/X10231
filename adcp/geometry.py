"""Janus ADCP beam geometry and beam/instrument velocity conversion."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import convert_angle, convert_velocity


@dataclass(frozen=True)
class Geometry:
    beam_angle_rad: float
    directions: Tuple[Tuple[float, float], ...]
    beam_order: Tuple[str, ...]
    version: str

    @property
    def matrix(self) -> np.ndarray:
        ca = math.cos(self.beam_angle_rad)
        sa = math.sin(self.beam_angle_rad)
        rows = []
        for azimuth_deg, sign in self.directions:
            az = math.radians(azimuth_deg)
            # RDI convention: beams 2/4 carry a sign that makes a uniform
            # downward vertical velocity produce equal positive beam values.
            rows.append(np.array([sign * ca * math.cos(az), sign * ca * math.sin(az), sa]))
        return np.vstack(rows)


def rdi_geometry(beam_angle_unit: str = "deg", beam_angle: float = 20.0,
                 version: str = "rdi-convex-janus-v1") -> Geometry:
    angle_rad, _ = convert_angle(beam_angle, beam_angle_unit)
    return Geometry(
        beam_angle_rad=angle_rad,
        directions=((0.0, 1.0), (90.0, -1.0), (180.0, 1.0), (270.0, -1.0)),
        beam_order=("b1", "b2", "b3", "b4"),
        version=version,
    )


def _beam_values(measurements: Sequence[dict], indices: Sequence[int]) -> Optional[np.ndarray]:
    vals = []
    try:
        for idx in indices:
            rec = measurements[idx]
            value, _ = convert_velocity(rec["raw_value"], rec["raw_unit"])
            if not np.isfinite(value):
                return None
            vals.append(value)
    except (KeyError, TypeError, ValueError):
        return None
    return np.asarray(vals, dtype=float)


def instrument_candidates(geometry: Geometry, flags_by_beam: Dict[str, str],
                          measurements: Sequence[dict]) -> Dict[str, dict]:
    """Return every legal three- and four-beam candidate.

    A hard bad beam excludes candidates using that beam. A non-bad quality
    marker (for example ``suspect_unknown_failure``) does not invalidate a
    four-beam product but makes each omitting three-beam set a legal
    alternative. Values are never merged into one answer.
    """
    mat = geometry.matrix
    hard = {i for i, b in enumerate(geometry.beam_order) if flags_by_beam.get(b) == "bad"}
    unknown = {
        i for i, b in enumerate(geometry.beam_order)
        if flags_by_beam.get(b, "").startswith("suspect")
    }
    candidates: Dict[str, dict] = {}

    if not hard:
        vals = _beam_values(measurements, range(4))
        if vals is not None:
            try:
                vec = np.linalg.solve(mat, vals)
            except np.linalg.LinAlgError:
                vec = None
            if vec is not None and np.all(np.isfinite(vec)):
                candidates["C4-all"] = {
                    "candidate_id": "C4-all", "mode": "four-beam",
                    "beams": ["b1", "b2", "b3", "b4"], "vector_mps": vec.tolist(),
                    "residual_mps": (vals - mat @ vec).tolist(),
                    "status": "alternative" if unknown else "unique",
                }

    valid_three = []
    for missing in range(4):
        beam = geometry.beam_order[missing]
        if missing in hard:
            continue
        indices = [i for i in range(4) if i != missing]
        vals = _beam_values(measurements, indices)
        if vals is None:
            continue
        sub = mat[np.asarray(indices), :]
        # Project null-space ambiguity onto horizontal: solve using x,y,z,
        # record the omitted-beam direction and zero-null-projection value.
        try:
            vec, *_ = np.linalg.lstsq(sub, vals, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        cid = f"C3-omit-{beam}"
        valid_three.append(cid)
        candidates[cid] = {
            "candidate_id": cid,
            "mode": "three-beam",
            "beams": [geometry.beam_order[i] for i in indices],
            "vector_mps": vec.tolist(),
            "residual_mps": (vals - sub @ vec).tolist(),
            "null_space_beam": beam,
            "ambiguity": "underdetermined-rank-3; null-space component unobserved",
            "status": "ambiguous" if len(valid_three) > 1 or not hard else "unique",
        }

    # A hard single failure leaves exactly one legal 3-beam candidate.
    # More than one non-hard suspect marker means no beam can be uniquely
    # chosen for omission: all legal candidates remain visible.
    if len(hard) == 1 and len(valid_three) == 1:
        candidates[valid_three[0]]["status"] = "unique"
    if len(hard) > 1:
        return {}
    if unknown and not hard:
        for cid, item in list(candidates.items()):
            if item["mode"] == "three-beam":
                item["status"] = "ambiguous"
    return candidates
