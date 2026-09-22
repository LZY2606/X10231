"""Beam geometry: beam velocities -> instrument-frame XYZ velocities.

Conventions (documented, not magic):
  * Instrument axes: x = starboard, y = forward, z = up.
  * Beam azimuth is measured clockwise from the y(forward) axis looking down
    along the instrument z axis; default beam azimths 45/135/225/315 deg.
  * Beam radial velocity r_i is positive away from the transducer.
  * Downward-looking convex head: r_i = d_i . (u, v, w) with
        d_i = (sinT*sin(a), sinT*cos(a), cosT)
    so four beams point with a positive z component; r and w share sign.

The 4-beam solution and every 3-beam candidate carry *different stable
identities*. Two failed beams (or fewer than three valid) is indeterminate and
is reported as such; we never collapse it to a single guessed vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_AZIMUTHS_DEG: Tuple[float, float, float, float] = (45.0, 135.0, 225.0, 315.0)
OPPOSITE_PAIRS: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 3), (1, 2))


@dataclass(frozen=True)
class TransducerConfig:
    beam_angle_deg: float = 20.0
    azimuths_deg: Tuple[float, ...] = DEFAULT_AZIMUTHS_DEG
    orientation: str = "down"  # "down" or "up"; flips the vertical sign convention
    corr_min_counts: int = 64
    amp_min_counts: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": list(self.azimuths_deg),
            "orientation": self.orientation,
            "corr_min_counts": self.corr_min_counts,
            "amp_min_counts": self.amp_min_counts,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TransducerConfig":
        return TransducerConfig(
            beam_angle_deg=float(d["beam_angle_deg"]),
            azimuths_deg=tuple(d.get("azimuths_deg", DEFAULT_AZIMUTHS_DEG)),
            orientation=d.get("orientation", "down"),
            corr_min_counts=int(d.get("corr_min_counts", 64)),
            amp_min_counts=int(d.get("amp_min_counts", 30)),
        )


def beam_matrix(cfg: TransducerConfig) -> np.ndarray:
    """Return the 4x3 matrix B with r = B @ v_xyz (v in instrument frame)."""
    theta = math.radians(cfg.beam_angle_deg)
    s, c = math.sin(theta), math.cos(theta)
    zsign = 1.0 if cfg.orientation == "down" else -1.0
    rows = []
    for az in cfg.azimuths_deg:
        a = math.radians(az)
        rows.append([s * math.sin(a), s * math.cos(a), zsign * c])
    return np.asarray(rows, dtype=float)


@dataclass
class XYZSolution:
    candidate_id: str
    method: str
    used_beams: Tuple[int, ...]
    xyz_mm_s: Tuple[float, float, float]  # (starboard, forward, up)
    error_velocity_mm_s: Optional[float] = None
    assumption: Optional[str] = None
    residuals_mm_s: Optional[Tuple[float, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "used_beams": list(self.used_beams),
            "xyz_mm_s": list(self.xyz_mm_s),
            "error_velocity_mm_s": self.error_velocity_mm_s,
            "assumption": self.assumption,
            "residuals_mm_s": None if self.residuals_mm_s is None else list(self.residuals_mm_s),
            "frame": "instrument",
            "unit": "mm/s",
        }


def _null_error_vector() -> np.ndarray:
    # B^+ maps the 4 radials to XYZ; the homogeneous null-space direction that
    # spans all-zero XYZ is (+1,-1,+1,-1) in beam order for RDI geometry.
    n = np.array([1.0, -1.0, 1.0, -1.0])
    return n / np.linalg.norm(n)


def solve_four(radials_mm_s: Sequence[float], cfg: TransducerConfig) -> XYZSolution:
    """Least-squares XYZ from all four beams plus the RDI error velocity."""
    r = np.asarray(radials_mm_s, dtype=float)
    B = beam_matrix(cfg)
    xyz, *_ = np.linalg.lstsq(B, r, rcond=None)
    resid = r - B @ xyz
    # error velocity = projection of residual onto the homogeneous beam null.
    n = _null_error_vector()
    err = float(np.dot(n, resid))
    return XYZSolution(
        candidate_id="4b",
        method="four_beam_least_squares",
        used_beams=(0, 1, 2, 3),
        xyz_mm_s=tuple(float(v) for v in xyz),
        error_velocity_mm_s=err,
        residuals_mm_s=tuple(float(v) for v in resid),
    )


def solve_three_candidates(
    radials_mm_s: Sequence[Optional[float]],
    cfg: TransducerConfig,
) -> List[XYZSolution]:
    """Return every legal 3-beam candidate when exactly one beam is invalid.

    Candidate A ("3bA:pair(i,j)"): vertical velocity taken from the mean of the
    intact *opposite* beam pair (the classical RDI three-beam reduction); the
    remaining two good beams then determine u, v.

    Candidate B ("3bB"): exact inverse of the three good rows, equivalent to a
    four-beam solution with error velocity forced to zero at the missing beam.

    The two agree only under homogeneous flow; both are retained.
    """
    r = [None if v is None else float(v) for v in radials_mm_s]
    good = [i for i, v in enumerate(r) if v is not None]
    if len(good) != 3:
        return []
    B = beam_matrix(cfg)
    bad = [i for i in range(4) if i not in good][0]
    cands: List[XYZSolution] = []

    # ---- Candidate A: opposite-pair mean for w ---------------------------
    pair = None
    for p in OPPOSITE_PAIRS:
        if bad not in p and all(i in good for i in p):
            pair = p
            break
    if pair is not None:
        w = (r[pair[0]] + r[pair[1]]) / (2.0 * B[pair[0], 2])
        rest = [i for i in good if i not in pair]
        M = B[rest][:, :2]
        rhs = np.array([r[i] - B[i, 2] * w for i in rest])
        uv, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        xyz = np.array([uv[0], uv[1], w])
        resid = np.array([r[i] - float(B[i] @ xyz) if i in good else float("nan")
                          for i in range(4)])
        cands.append(XYZSolution(
            candidate_id="3bA:pair(%d,%d)" % (pair[0] + 1, pair[1] + 1),
            method="three_beam_opposite_pair_mean",
            used_beams=tuple(sorted(good)),
            xyz_mm_s=tuple(float(v) for v in xyz),
            assumption="vertical velocity from intact opposite beam pair mean",
            residuals_mm_s=tuple(float(v) for v in resid),
        ))

    # ---- Candidate B: exact 3x3 inverse (zero-error completion) ----------
    D = B[good]
    xyz_b = np.linalg.solve(D, np.array([r[i] for i in good]))
    resid_b = np.array([r[i] - float(B[i] @ xyz_b) if i in good else float("nan")
                        for i in range(4)])
    cands.append(XYZSolution(
        candidate_id="3bB",
        method="three_beam_exact_zero_error",
        used_beams=tuple(sorted(good)),
        xyz_mm_s=tuple(float(v) for v in xyz_b),
        assumption="error velocity zero at missing beam %d" % (bad + 1),
        residuals_mm_s=tuple(float(v) for v in resid_b),
    ))
    return cands


@dataclass
class BeamQuality:
    valid: Tuple[bool, bool, bool, bool]
    reasons: Tuple[str, str, str, str]

    @property
    def good_beams(self) -> Tuple[int, ...]:
        return tuple(i for i, ok in enumerate(self.valid) if ok)

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": list(self.valid), "reasons": list(self.reasons),
                "good_beams": list(self.good_beams)}


def classify_beams(
    radials_mm_s: Sequence[Optional[float]],
    correlation: Sequence[Optional[float]],
    amplitude: Sequence[Optional[float]],
    bad_flags: Sequence[Optional[bool]],
    cfg: TransducerConfig,
) -> BeamQuality:
    """Per-beam validity from bad flag, correlation and amplitude thresholds."""
    valid, reasons = [], []
    for i in range(4):
        why = []
        if bad_flags[i] in (True, 1):
            why.append("bad_beam_flag")
        if radials_mm_s[i] is None:
            why.append("missing_velocity")
        if correlation[i] is None:
            why.append("missing_correlation")
        elif correlation[i] < cfg.corr_min_counts:
            why.append("correlation_below_%d" % cfg.corr_min_counts)
        if amplitude[i] is None:
            why.append("missing_amplitude")
        elif amplitude[i] < cfg.amp_min_counts:
            why.append("amplitude_below_%d" % cfg.amp_min_counts)
        valid.append(not why)
        reasons.append(";".join(why) if why else "ok")
    return BeamQuality(tuple(valid), tuple(reasons))  # type: ignore[arg-type]
