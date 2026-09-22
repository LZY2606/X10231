"""Transducer beam geometry and beam <-> instrument-frame projection.

The geometry is *explicit*: the transducer configuration records the outward
unit direction of every beam, so the math does not depend on a vendor-specific
sign convention.  The one configurable convention is the radial sign:

* ``radial_positive = "toward"`` (default, RDI style): a positive beam
  velocity means water moving **toward** the transducer.
* ``radial_positive = "away"``: the opposite sign.

Semantics enforced here (see case-study requirements):

* The 4-beam least-squares solution (with error velocity) and every 3-beam
  exact solution are returned as **distinct candidates with distinct method
  ids**; they are never averaged or collapsed into one number.
* Fewer than 3 independent usable beams => no solution, only a diagnostic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransducerConfig:
    """Transducer / transducer-mount configuration (versioned source)."""

    n_beams: int
    beam_angle_deg: float
    # azimuth (deg, instrument frame, 0=+x forward, 90=+y starboard) of the
    # horizontal direction each beam tilts toward, one entry per beam.
    beam_azimuths_deg: Sequence[float]
    radial_positive: str = "toward"  # or "away"
    # fixed mount of the transducer relative to the ship frame
    mount_heading_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    mount_roll_deg: float = 0.0
    config_id: str = "rdi_janus4_20_v1"

    @staticmethod
    def rdi_janus4(beam_angle_deg: float = 20.0,
                   mount_heading_deg: float = 0.0,
                   mount_pitch_deg: float = 0.0,
                   mount_roll_deg: float = 0.0,
                   config_id: str = "rdi_janus4_20_v1") -> "TransducerConfig":
        """Standard 4-beam convex Janus: 3 fwd, 4 aft, 1 port, 2 stbd."""
        return TransducerConfig(
            n_beams=4,
            beam_angle_deg=beam_angle_deg,
            beam_azimuths_deg=[270.0, 90.0, 0.0, 180.0],
            mount_heading_deg=mount_heading_deg,
            mount_pitch_deg=mount_pitch_deg,
            mount_roll_deg=mount_roll_deg,
            config_id=config_id,
        )


def outward_vectors(cfg: TransducerConfig) -> np.ndarray:
    """Return (n_beams, 3) outward beam unit vectors in the instrument frame.

    Instrument frame is right-handed: x forward, y starboard, z downward.
    """
    theta = math.radians(cfg.beam_angle_deg)
    horizontal = math.sin(theta)
    vertical = math.cos(theta)
    dirs = []
    for az in cfg.beam_azimuths_deg:
        a = math.radians(az)
        dirs.append([horizontal * math.cos(a),
                     horizontal * math.sin(a),
                     vertical])
    d = np.asarray(dirs, dtype=float)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    return d / norms


def beam_matrix(cfg: TransducerConfig) -> np.ndarray:
    """Matrix B mapping instrument velocity w to radial beam velocities b."""
    d = outward_vectors(cfg)
    sign = -1.0 if cfg.radial_positive == "toward" else 1.0
    return sign * d


# ---------------------------------------------------------------------------
# Projection results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VelocityCandidate:
    """One legal instrument-frame velocity solution.

    Distinct ``method_id`` values are stable identities: a 4-beam solution and
    a 3-beam solution are never merged into a single value.
    """

    method_id: str
    method_kind: str          # "four_beam" | "three_beam" | "none"
    velocity: Optional[np.ndarray]  # shape (3,), m/s, basis "instrument"
    error_velocity: Optional[float]
    used_beams: tuple
    dropped_beam: Optional[int]
    assumptions: tuple
    diagnostics: tuple = field(default_factory=tuple)

    @property
    def unique(self) -> bool:
        """True only when this candidate set is forced by the data."""
        return self.method_kind == "four_beam" and self.dropped_beam is None

    def to_dict(self) -> Dict:
        v = None if self.velocity is None else [
            float(x) for x in self.velocity]
        return {
            "method_id": self.method_id,
            "method_kind": self.method_kind,
            "velocity": v,
            "error_velocity": self.error_velocity,
            "used_beams": list(self.used_beams),
            "dropped_beam": self.dropped_beam,
            "assumptions": list(self.assumptions),
            "diagnostics": list(self.diagnostics),
        }


def _four_beam(B: np.ndarray, b: np.ndarray) -> VelocityCandidate:
    bt = B.T
    w, *_ = np.linalg.lstsq(B, b, rcond=None)
    residual = b - B @ w
    # left null space of B (B is n x 3): error velocity direction
    _, _, vh = np.linalg.svd(B, full_matrices=True)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    err = float(np.dot(n, residual))
    return VelocityCandidate(
        method_id="four_beam_ls",
        method_kind="four_beam",
        velocity=np.asarray(w, dtype=float),
        error_velocity=err,
        used_beams=tuple(range(B.shape[0])),
        dropped_beam=None,
        assumptions=("homogeneous_water_mass_all_beams",),
        diagnostics=() if abs(err) < 1e-9 else
                    ("nonzero_error_velocity:%.6g_m/s" % err,),
    )


def _three_beam(B: np.ndarray, b: np.ndarray, used: Sequence[int],
                dropped: int) -> VelocityCandidate:
    try:
        w = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        return VelocityCandidate(
            method_id="three_beam_drop%d" % dropped,
            method_kind="none",
            velocity=None,
            error_velocity=None,
            used_beams=tuple(used),
            dropped_beam=dropped,
            assumptions=(),
            diagnostics=("three_remaining_beams_linearly_dependent",),
        )
    return VelocityCandidate(
        method_id="three_beam_drop%d" % dropped,
        method_kind="three_beam",
        velocity=np.asarray(w, dtype=float),
        error_velocity=None,
        used_beams=tuple(used),
        dropped_beam=dropped,
        assumptions=("homogeneous_water_mass_three_beams",
                     "error_velocity_unobservable"),
        diagnostics=("error_velocity_cannot_be_computed_from_3_beams",),
    )


def project_beams(cfg: TransducerConfig,
                  beam_vel: Sequence[float],
                  hard_bad: Sequence[bool],
                  marginal: Optional[Sequence[bool]] = None
                  ) -> List[VelocityCandidate]:
    """Project radial beam velocities to instrument-frame candidates.

    Parameters
    ----------
    beam_vel:
        Radial velocities in m/s, one entry per beam.
    hard_bad:
        Beams flagged bad (failed) are never used.
    marginal:
        Usable-but-suspect beams.  They still participate in a 4-beam least
        squares candidate, **and** each one spawns a parallel 3-beam candidate
        in which it is dropped.  Both legal answers are retained; nothing is
        averaged away.
    """
    B_full = beam_matrix(cfg)
    b = np.asarray(beam_vel, dtype=float)
    n = cfg.n_beams
    hard_bad = list(hard_bad) + [False] * (n - len(hard_bad))
    if marginal is None:
        marginal = [False] * n
    else:
        marginal = list(marginal) + [False] * (n - len(marginal))

    usable = [i for i in range(n) if not hard_bad[i]]
    bad = [i for i in range(n) if hard_bad[i]]

    if len(usable) < 3:
        return [VelocityCandidate(
            method_id="no_solution",
            method_kind="none",
            velocity=None,
            error_velocity=None,
            used_beams=tuple(usable),
            dropped_beam=None,
            assumptions=(),
            diagnostics=("fewer_than_3_usable_beams:%d" % len(usable),),
        )]

    candidates: List[VelocityCandidate] = []

    # 4-beam candidate exists only when every beam is usable.
    if len(usable) == n:
        candidates.append(_four_beam(B_full, b))

    # Exact 3-beam candidates: one for each hard-bad beam (forced geometry),
    # plus one for each marginal beam (ambiguous: inclusion vs exclusion).
    drops = sorted(set(bad + [i for i in range(n) if marginal[i]]))
    for j in drops:
        used = [i for i in range(n) if i != j]
        if len(used) < 3:
            continue
        rows = np.array(used)
        candidates.append(_three_beam(B_full[rows], b[rows], used, j))

    # Exactly 3 good beams and none marginal: single deterministic 3-beam.
    if not candidates and len(usable) == 3:
        dropped = [i for i in range(n) if i not in usable][0]
        rows = np.array(usable)
        candidates.append(_three_beam(B_full[rows], b[rows], usable, dropped))

    return candidates


def beam_geometry_fingerprint_parts(cfg: TransducerConfig) -> Dict:
    """Compact geometry description fed into derived fingerprints."""
    d = outward_vectors(cfg)
    return {
        "config_id": cfg.config_id,
        "n_beams": cfg.n_beams,
        "beam_angle_deg": cfg.beam_angle_deg,
        "beam_azimuths_deg": [float(a) for a in cfg.beam_azimuths_deg],
        "radial_positive": cfg.radial_positive,
        "beam_matrix": [[float(v) for v in row] for row in beam_matrix(cfg)],
        "outward_vectors": [[float(v) for v in row] for row in d],
    }
