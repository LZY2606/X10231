"""Beam geometry and beam -> instrument velocity solutions.

Convex 4-beam Janus geometry ``adcp-janus-convex-v1``:

* beam 0 azimuth 315 deg  (forward / port)
* beam 1 azimuth  45 deg  (forward / starboard)
* beam 2 azimuth 135 deg  (aft / starboard)
* beam 3 azimuth 225 deg  (aft / port)

Azimuth is measured in the instrument frame (x forward, y starboard,
z down) from the forward axis toward starboard.  Beams slant away from the
instrument by ``beam_angle`` from the vertical; positive beam velocity is
away from the instrument along the beam.  Both the sign convention and the
azimuth order are part of the geometry version that enters every fingerprint.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .fingerprint import fingerprint
from .units import FRAME_INSTRUMENT

GEOMETRY_VERSION = "adcp-janus-convex-v1"
GEOMETRY_RULE = "beam2instrument@" + GEOMETRY_VERSION

DEFAULT_BEAM_ANGLE_DEG = 20.0
DEFAULT_AZIMUTHS_DEG = (315.0, 45.0, 135.0, 225.0)


@dataclass(frozen=True)
class BeamGeometry:
    beam_angle_deg: float = DEFAULT_BEAM_ANGLE_DEG
    azimuths_deg: tuple = DEFAULT_AZIMUTHS_DEG
    version: str = GEOMETRY_VERSION

    def matrix(self) -> np.ndarray:
        """Row i is the unit direction of beam i in the instrument frame."""
        theta = np.deg2rad(self.beam_angle_deg)
        rows = []
        for az in self.azimuths_deg:
            a = np.deg2rad(az)
            rows.append(
                [
                    np.sin(theta) * np.cos(a),
                    np.sin(theta) * np.sin(a),
                    np.cos(theta),
                ]
            )
        return np.asarray(rows, dtype=float)

    def basis(self) -> dict:
        return {
            "version": self.version,
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
        }


def _null_vector(h: np.ndarray) -> np.ndarray:
    """Unit vector spanning null(H^T); sign fixed by beam 0 coefficient >= 0."""
    _, _, vt = np.linalg.svd(h)
    n = vt[-1]
    if n[0] < 0:
        n = -n
    return n


def solve_beams(
    beam_velocities,
    valid,
    geometry: BeamGeometry,
    error_velocity_threshold: float,
):
    """Solve instrument-frame velocity from valid beams.

    ``beam_velocities``: 4 values in m/s (NaN allowed).
    ``valid``: 4 booleans from the quality tier.

    Status semantics (never collapsed into one value):

    * ``four_beam``        - unique 4-beam solution + error velocity
    * ``ambiguous``        - 4 valid beams but error velocity exceeds the
                             threshold; all four legal 3-beam candidates are
                             retained
    * ``three_beam``       - exactly 3 valid beams; one candidate
    * ``blocked``          - fewer than 3 valid beams; no velocity is produced
    """
    h = geometry.matrix()
    b = np.asarray(beam_velocities, dtype=float)
    valid = [bool(v) for v in valid]
    valid_idx = [i for i, v in enumerate(valid) if v]

    result = {
        "frame": FRAME_INSTRUMENT,
        "status": "blocked",
        "vector": None,
        "error_velocity": None,
        "candidates": [],
        "valid_beams": valid_idx,
        "geometry_version": geometry.version,
        "geometry_basis": geometry.basis(),
    }

    if len(valid_idx) < 3:
        result["diagnostic"] = (
            "only %d valid beam(s): instrument velocity underdetermined"
            % len(valid_idx)
        )
        return result

    if len(valid_idx) == 3:
        ids = tuple(i + 1 for i in valid_idx)
        m = h[valid_idx, :]
        v = np.linalg.solve(m, b[valid_idx])
        cand = _candidate(ids, v, h, b, geometry)
        result.update(
            status="three_beam",
            vector=[float(x) for x in v],
            candidates=[cand],
        )
        return result

    # Four valid beams.
    v4, *_ = np.linalg.lstsq(h, b, rcond=None)
    n = _null_vector(h)
    e = float(np.dot(n, b))
    result.update(
        vector=[float(x) for x in v4],
        null_vector=[float(x) for x in n],
        error_velocity=e,
        error_velocity_threshold=error_velocity_threshold,
    )
    if abs(e) > error_velocity_threshold:
        cands = []
        for combo in itertools.combinations(range(4), 3):
            ids = tuple(i + 1 for i in combo)
            v3 = np.linalg.solve(h[list(combo), :], b[list(combo)])
            cands.append(_candidate(ids, v3, h, b, geometry))
        result.update(status="ambiguous", candidates=cands, vector=None)
        result["diagnostic"] = (
            "|error velocity| %.4f m/s > threshold %.4f m/s; "
            "4 legal 3-beam candidates retained" % (abs(e), error_velocity_threshold)
        )
        return result
    result["status"] = "four_beam"
    return result


def _candidate(ids, v, h, b, geometry: BeamGeometry) -> dict:
    basis = {
        "beams": list(ids),
        "beam_values": [float(b[i - 1]) for i in ids],
        "geometry": geometry.basis(),
    }
    return {
        "id": "beams:" + "".join(str(i) for i in ids),
        "beams": list(ids),
        "vector": [float(x) for x in v],
        "frame": FRAME_INSTRUMENT,
        "fingerprint": fingerprint(GEOMETRY_RULE + "#candidate", basis),
    }
