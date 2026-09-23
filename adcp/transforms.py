"""Coordinate transforms: beam -> instrument -> ship -> earth.

Conventions (documented, deterministic):
  * 4-beam Janus geometry, beams numbered 1..4 at azimuths 45/135/225/315 deg,
    tilted ``beam_angle_deg`` away from the vertical axis, down-looking.
  * Instrument frame: x starboard, y forward, z up (toward the transducer).
  * Ship frame: x starboard, y bow, z up, after attitude rotation.
  * Earth frame: x east, y north, z up, after magnetic-declination rotation.

Rules enforced here (they are also emitted as machine-readable events by the
service layer):
  * a three-beam solution is never mixed into the same value as a four-beam
    solution -- it is stored as separate candidates keyed by assumption;
  * attitude interpolation never crosses a time gap;
  * transform order, beam geometry and declination version all feed the
    derived fingerprint (see fingerprint.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RULE_VERSION = "adcp-rules/1.4.0"


# ---------------------------------------------------------------- geometry

@dataclass(frozen=True)
class BeamGeometry:
    n_beams: int = 4
    beam_angle_deg: float = 20.0
    frequency_khz: int = 600
    convex: bool = True
    janus_phase_deg: float = 45.0

    def beam_unit_vectors(self) -> np.ndarray:
        """Unit vector of each beam in the instrument frame (down-looking)."""
        theta = math.radians(self.beam_angle_deg)
        vecs = []
        for k in range(self.n_beams):
            phi = math.radians(self.janus_phase_deg + k * 360.0 / self.n_beams)
            vecs.append([math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi),
                         -math.cos(theta)])
        return np.array(vecs)  # shape (n_beams, 3)

    def as_dict(self) -> dict:
        return {
            "n_beams": self.n_beams,
            "beam_angle_deg": self.beam_angle_deg,
            "frequency_khz": self.frequency_khz,
            "convex": self.convex,
            "janus_phase_deg": self.janus_phase_deg,
        }


# ------------------------------------------------------- beam -> instrument

@dataclass
class VelocitySolution:
    """One legal candidate for an instrument-frame velocity."""

    vector: tuple[float, float, float]   # x, y, z in m/s, instrument frame
    n_beams: int
    assumption: str                      # e.g. "four-beam", "errvel=0"
    error_velocity: float | None = None
    residual: float | None = None


def beam_to_instrument(beam_vels: dict[int, float],
                       geometry: BeamGeometry,
                       prev_error_velocity: float | None = None) -> list[VelocitySolution]:
    """Convert beam velocities (m/s, + away from transducer) to instrument XYZ.

    With four good beams a single solution is returned.  With exactly three
    good beams the error velocity is unobservable, so *all* legal candidates
    are returned, one per documented assumption -- never collapsed into one
    value and never mixed with a four-beam solution.
    """
    beams = sorted(beam_vels)
    u = geometry.beam_unit_vectors()
    v = np.array([beam_vels[b] for b in beams], dtype=float)
    u_sel = u[[b - 1 for b in beams]]

    if len(beams) == 4:
        sol, *_ = np.linalg.lstsq(u_sel, v, rcond=None)
        residual = v - u_sel @ sol
        err = float(np.linalg.norm(residual))
        return [VelocitySolution(tuple(float(x) for x in sol), 4,
                                 "four-beam", error_velocity=err,
                                 residual=err)]

    if len(beams) == 3:
        # determined system; the unobserved error velocity makes the answer
        # assumption-dependent, so every documented assumption is a candidate
        sol, *_ = np.linalg.lstsq(u_sel, v, rcond=None)
        candidates = [VelocitySolution(tuple(float(x) for x in sol), 3,
                                       "errvel=0", error_velocity=0.0)]
        if prev_error_velocity is not None:
            # propagate the last observed four-beam error velocity: adjust the
            # beam that is geometrically most sensitive along the null space
            null = _null_vector(u_sel)
            if null is not None:
                adjusted = sol + null * prev_error_velocity
                candidates.append(VelocitySolution(
                    tuple(float(x) for x in adjusted), 3,
                    "errvel=prev", error_velocity=prev_error_velocity))
        return candidates

    return []  # fewer than 3 beams: no legal solution


def _null_vector(u_sel: np.ndarray) -> np.ndarray | None:
    """Null-space direction of a 3x3 beam matrix (near-singular geometry)."""
    try:
        _, s, vh = np.linalg.svd(u_sel)
    except np.linalg.LinAlgError:
        return None
    if s[-1] < 1e-9:
        return vh[-1]
    # full rank: use the least-constrained direction scaled by conditioning
    return vh[-1] * (s[-1] / s[0])


# ----------------------------------------------------- instrument -> ship

_ROTATIONS = {
    "H": lambda a: np.array([[math.cos(a), -math.sin(a), 0],
                             [math.sin(a), math.cos(a), 0],
                             [0, 0, 1]]),
    "P": lambda a: np.array([[math.cos(a), 0, math.sin(a)],
                             [0, 1, 0],
                             [-math.sin(a), 0, math.cos(a)]]),
    "R": lambda a: np.array([[1, 0, 0],
                             [0, math.cos(a), -math.sin(a)],
                             [0, math.sin(a), math.cos(a)]]),
}

VALID_ORDERS = ("HPR", "HRP", "PHR", "PRH", "RHP", "RPH")


def instrument_to_ship(vec: tuple[float, float, float],
                       heading_deg: float, pitch_deg: float, roll_deg: float,
                       mounting_angle_deg: float = 0.0,
                       order: str = "HPR") -> tuple[float, float, float]:
    """Rotate an instrument-frame vector into the ship frame.

    ``order`` names the intrinsic rotation sequence (H=heading, P=pitch,
    R=roll); the order is semantically significant and is part of the
    derived fingerprint.  ``mounting_angle_deg`` is the transducer
    installation offset added to heading.
    """
    if order not in VALID_ORDERS:
        raise ValueError(f"unknown rotation order: {order}")
    angles = {"H": math.radians(heading_deg + mounting_angle_deg),
              "P": math.radians(pitch_deg),
              "R": math.radians(roll_deg)}
    m = np.eye(3)
    for axis in order:  # intrinsic: left-multiply successive rotations
        m = _ROTATIONS[axis](angles[axis]) @ m
    out = m @ np.array(vec, dtype=float)
    return tuple(float(x) for x in out)


# ---------------------------------------------------------- ship -> earth

def ship_to_earth(vec: tuple[float, float, float],
                  declination_deg: float) -> tuple[float, float, float]:
    """Rotate ship-frame horizontal components into earth (E/N/Up)."""
    a = math.radians(declination_deg)
    # ship y (bow) points at magnetic heading; declination rotates to true
    rot = np.array([[math.cos(a), -math.sin(a), 0],
                    [math.sin(a), math.cos(a), 0],
                    [0, 0, 1]])
    # ship (x=starboard, y=bow) -> earth (x=east, y=north): swap then rotate
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    out = rot @ swap @ np.array(vec, dtype=float)
    return tuple(float(x) for x in out)


# ---------------------------------------------------------------- attitude

def unwrap_headings(headings_deg: list[float]) -> list[float]:
    """Unwrap a heading series so 359.5 -> 0.5 becomes continuous."""
    if not headings_deg:
        return []
    out = [headings_deg[0]]
    for h in headings_deg[1:]:
        prev = out[-1]
        while h - prev > 180.0:
            h -= 360.0
        while h - prev < -180.0:
            h += 360.0
        out.append(h)
    return out


@dataclass
class AttitudeSeries:
    """Heading/pitch/roll samples with gap-aware interpolation."""

    times: list[float]
    heading: list[float]
    pitch: list[float]
    roll: list[float]
    max_gap_s: float = 2.0

    def segments(self) -> list[tuple[int, int]]:
        """Index ranges [start, end) of runs not broken by a time gap."""
        segs, start = [], 0
        for i in range(1, len(self.times)):
            if self.times[i] - self.times[i - 1] > self.max_gap_s:
                segs.append((start, i))
                start = i
        segs.append((start, len(self.times)))
        return segs

    def interpolate(self, t: float) -> tuple[float, float, float] | None:
        """Interpolated attitude at time t, or None across a gap.

        Interpolation is confined to a single continuous segment; a query
        that would need to bridge a gap returns None (the caller records a
        diagnostic instead of silently mixing segments).
        """
        unwrapped = unwrap_headings(self.heading)
        for start, end in self.segments():
            seg_t = self.times[start:end]
            if seg_t[0] - 1e-9 <= t <= seg_t[-1] + 1e-9:
                h = float(np.interp(t, seg_t, unwrapped[start:end])) % 360.0
                p = float(np.interp(t, seg_t, self.pitch[start:end]))
                r = float(np.interp(t, seg_t, self.roll[start:end]))
                return h, p, r
        return None


# ------------------------------------------------------------- cell quality

@dataclass
class QualityThresholds:
    min_correlation: float = 64.0      # counts
    max_correlation: float = 255.0
    min_amplitude_db: float = 5.0
    max_amplitude_db: float = 160.0
    blanking_m: float = 0.5


def evaluate_cell(correlation: dict[int, float],
                  amplitude: dict[int, float],
                  bad_beams: list[int],
                  depth_m: float,
                  bottom_depth_m: float | None,
                  th: QualityThresholds = QualityThresholds()) -> list[str]:
    """Return exclusion reasons for one depth cell (empty list = valid)."""
    reasons: list[str] = []
    if depth_m < th.blanking_m:
        reasons.append("below-blanking")
    if bottom_depth_m is not None and depth_m > bottom_depth_m:
        reasons.append("beyond-bottom")
    for beam, c in sorted(correlation.items()):
        if not (th.min_correlation <= c <= th.max_correlation):
            reasons.append(f"correlation-out-of-range:beam{beam}")
    for beam, a in sorted(amplitude.items()):
        if not (th.min_amplitude_db <= a <= th.max_amplitude_db):
            reasons.append(f"amplitude-out-of-range:beam{beam}")
    for beam in bad_beams:
        reasons.append(f"bad-beam-flag:beam{beam}")
    return reasons
