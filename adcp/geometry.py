"""Transducer / beam geometry.

A 4-beam Janus ADCP maps the instrument-frame velocity ``[u, v, w]`` plus an
error-velocity channel ``e`` to four along-beam measurements through a geometry
matrix ``G``::

    b = G @ [u, v, w, e]

The default layout is the conventional RDI convex configuration for a
down-looking head, but azimuths, beam angle, orientation and the per-beam
positive convention are all configuration so the geometry becomes part of the
derived fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import cos, radians, sin
from typing import Iterable, Optional

import numpy as np

# RDI convex down-looking layout, positive beam velocity = flow *away* from
# transducer. Beam 1/3 oppose along X, beam 2/4 oppose along Y; error signs
# alternate.
RDI_AZIMUTHS_DEG = (180.0, 270.0, 0.0, 90.0)
RDI_ERROR_SIGNS = (1.0, -1.0, 1.0, -1.0)


@dataclass(frozen=True)
class BeamGeometry:
    transducer_id: str
    beam_angle_deg: float
    orientation: str  # "down" or "up"
    azimuths_deg: tuple[float, ...]
    beam_signs: tuple[float, ...]  # per-beam flip for positive convention
    error_signs: tuple[float, ...]

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def matrix(self) -> np.ndarray:
        """Return the ``(n_beams, 4)`` geometry matrix (columns u,v,w,e)."""
        theta = radians(self.beam_angle_deg)
        vz = (-cos(theta)) if self.orientation == "down" else cos(theta)
        rows = []
        for az, bsign, esign in zip(self.azimuths_deg, self.beam_signs, self.error_signs):
            phi = radians(az)
            # azimuth measured clockwise from +X in the instrument plane
            hx = bsign * sin(theta) * sin(phi)
            hy = -bsign * sin(theta) * cos(phi)
            rows.append([hx, hy, bsign * vz, esign])
        return np.asarray(rows, dtype=float)

    def submatrix(self, beams: Iterable[int]) -> np.ndarray:
        g = self.matrix()
        idx = list(beams)
        return g[np.ix_(idx, [0, 1, 2])]

    def fingerprint(self) -> str:
        payload = {
            "transducer_id": self.transducer_id,
            "beam_angle_deg": round(self.beam_angle_deg, 9),
            "orientation": self.orientation,
            "azimuths_deg": [round(a, 9) for a in self.azimuths_deg],
            "beam_signs": list(self.beam_signs),
            "error_signs": list(self.error_signs),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "geo_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "transducer_id": self.transducer_id,
            "beam_angle_deg": self.beam_angle_deg,
            "orientation": self.orientation,
            "azimuths_deg": list(self.azimuths_deg),
            "beam_signs": list(self.beam_signs),
            "error_signs": list(self.error_signs),
            "fingerprint": self.fingerprint(),
        }


def rdi_geometry(
    beam_angle_deg: float = 20.0,
    orientation: str = "down",
    *,
    transducer_id: str = "rdi-default",
    azimuths_deg: tuple[float, ...] = RDI_AZIMUTHS_DEG,
    beam_signs: Optional[tuple[float, ...]] = None,
) -> BeamGeometry:
    if beam_signs is None:
        beam_signs = (1.0, 1.0, 1.0, 1.0)
    return BeamGeometry(
        transducer_id=transducer_id,
        beam_angle_deg=beam_angle_deg,
        orientation=orientation,
        azimuths_deg=tuple(azimuths_deg),
        beam_signs=tuple(beam_signs),
        error_signs=RDI_ERROR_SIGNS,
    )


def solve_four(geometry: BeamGeometry, beam_vel: np.ndarray):
    """Solve ``b = G @ [u,v,w,e]`` for four good beams.

    Returns ``(u, v, w, err)`` in m/s. The error channel measures Janus
    redundancy inconsistency; the four-beam value is never relabeled as a
    three-beam value.
    """
    g = geometry.matrix()
    x = np.linalg.solve(g, np.asarray(beam_vel, dtype=float))
    return float(x[0]), float(x[1]), float(x[2]), float(x[3])


def solve_three(geometry: BeamGeometry, good_beams: Iterable[int], beam_vel: np.ndarray):
    """Exact three-beam inversion for ``[u, v, w]`` with error pinned to zero.

    The result is explicitly a *three-beam* estimate: it depends on which beam
    is missing and is never merged with the four-beam solution.
    """
    idx = list(good_beams)
    if len(idx) != 3:
        raise ValueError("solve_three requires exactly three beam indices")
    g3 = geometry.submatrix(idx)
    b3 = np.asarray(beam_vel, dtype=float)[idx]
    x = np.linalg.solve(g3, b3)
    return float(x[0]), float(x[1]), float(x[2])
