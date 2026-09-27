"""Transducer / beam geometry and beam-velocity <-> instrument-frame math.

Convention (self-consistent and documented, convex Janus ADCP):

* instrument frame is right-handed: x forward, y starboard, z downward.
* beam k points downward and outward along horizontal azimuth ``azimuth[k]``
  (degrees, clockwise from x) with depression ``theta`` from horizontal.
* beam velocity ``b[k]`` is the water motion projected on the beam axis,
  positive *away* from the transducer (the RDI sign convention).

The beam map rows are therefore the outward unit beam axes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import deg_to_rad


@dataclass(frozen=True)
class TransducerGeometry:
    name: str
    azimuths_deg: tuple[float, ...]  # clockwise from instrument x
    beam_angle_deg: float            # depression from horizontal
    convex: bool = True               # beams slant outward

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def beam_map(self) -> np.ndarray:
        """Matrix A so that b = A @ v_inst (beam velocities from v xyz)."""
        theta = deg_to_rad(self.beam_angle_deg)
        rows = []
        for az in self.azimuths_deg:
            a = deg_to_rad(az)
            horiz = math.cos(theta)
            vert = math.sin(theta)
            if not self.convex:
                horiz = -horiz
            rows.append([horiz * math.cos(a), horiz * math.sin(a), vert])
        return np.asarray(rows, dtype=float)

    def vertical_factor(self) -> float:
        """4-beam Janus vertical projection sin(theta)."""
        return math.sin(deg_to_rad(self.beam_angle_deg))


def solve_velocity(geo: TransducerGeometry, beams: np.ndarray,
                   valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Least-squares instrument velocity from valid beams.

    Returns (v_xyz, residual_per_beam_or_none, error_velocity_or_none).
    """
    a = geo.beam_map()
    mask = np.asarray(valid, dtype=bool)
    sol, *_ = np.linalg.lstsq(a[mask], np.asarray(beams, dtype=float)[mask], rcond=None)
    residual = np.asarray(beams, dtype=float) - a @ sol
    error = error_velocity(geo, beams, valid) if mask.sum() >= 4 else None
    return sol, residual, error


def error_velocity(geo: TransducerGeometry, beams: np.ndarray,
                   valid: np.ndarray) -> float | None:
    """RDI Janus error velocity: ((b2+b4) - (b1+b3)) / (4 sin theta).

    Opposite beam pairs are matched by azimuth differing by 180 degrees.
    Returns None when a paired beam is missing or this is not a 4-Janus head.
    """
    if geo.n_beams != 4:
        return None
    a = geo.beam_map()
    pairs = _opposite_pairs(geo)
    if pairs is None or not np.asarray(valid, dtype=bool).all():
        return None
    (i, j), (k, l) = pairs
    return float(((beams[j] + beams[l]) - (beams[i] + beams[k]))
                 / (4.0 * geo.vertical_factor()))


def _opposite_pairs(geo: TransducerGeometry):
    az = list(geo.azimuths_deg)
    used: set[int] = set()
    pairs = []
    for i in range(len(az)):
        if i in used:
            continue
        for j in range(i + 1, len(az)):
            if j in used:
                continue
            if abs(((az[j] - az[i] + 180.0) % 360.0) - 180.0) < 1e-9:
                pairs.append((i, j))
                used.update((i, j))
                break
    if len(pairs) == 2:
        return pairs
    return None


def rdi_4_beam(name: str = "RDI-Janus-20deg") -> TransducerGeometry:
    """Standard convex RDI 4-beam head: 45/135/225/315, 20 degree depression."""
    return TransducerGeometry(name, (45.0, 135.0, 225.0, 315.0), 20.0, True)
