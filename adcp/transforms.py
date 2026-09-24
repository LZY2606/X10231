"""Core ADCP rules: beam geometry, rotations, quality gates.

Rule identifiers are part of every derived fingerprint:

* ``janus4-lsq-v1``  four-beam least squares + error velocity
* ``janus3-subset-v1`` three-beam exact solutions, one per legal beam subset
* ``inst2ship-mount-v1`` fixed mounting alignment rotation
* ``ship2earth-hpr-zyx-v1`` heading/pitch/roll order Rz(H) Ry(P) Rx(R)
* ``ship2earth-hpr-zxy-v1`` alternative order Rz(H) Rx(R) Ry(P)
* ``attitude-interp-v1`` linear interpolation, never across a gap
* ``declination-east-v1`` true heading = magnetic heading + east declination
* ``cell-depth-v1`` depth-cell validity gate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

GEOMETRY_RULE_ID = "janus-beam-geometry-v1"
BEAM4_RULE_ID = "janus4-lsq-v1"
BEAM3_RULE_ID = "janus3-subset-v1"
INST2SHIP_RULE_ID = "inst2ship-mount-v1"
EARTH_RULE_IDS = ("ship2earth-hpr-zyx-v1", "ship2earth-hpr-zxy-v1")
EARTH_DEFAULT_RULE_ID = "ship2earth-hpr-zyx-v1"
ATTITUDE_RULE_ID = "attitude-interp-v1"
DECLINATION_RULE_ID = "declination-east-v1"
CELL_RULE_ID = "cell-depth-v1"

ERROR_PATTERN = (1.0, -1.0, 1.0, -1.0)


@dataclass(frozen=True)
class BeamGeometry:
    beam_count: int
    azimuth_deg: tuple[float, ...]      # clockwise from instrument +x (bow)
    tilt_deg: tuple[float, ...]         # from vertical
    beam_angle_rad: tuple[float, ...]

    @property
    def matrix(self) -> np.ndarray:
        rows = []
        for az, tilt in zip(self.azimuth_deg, self.beam_angle_rad):
            a = math.radians(az)
            rows.append((math.sin(tilt) * math.cos(a),
                         math.sin(tilt) * math.sin(a),
                         math.cos(tilt)))
        return np.array(rows)


def geometry(azimuth_deg: Sequence[float], tilt_deg: Sequence[float]) -> BeamGeometry:
    if len(azimuth_deg) != len(tilt_deg):
        raise ValueError("azimuth/tilt beam count mismatch")
    return BeamGeometry(len(azimuth_deg), tuple(azimuth_deg), tuple(tilt_deg),
                        tuple(math.radians(t) for t in tilt_deg))


def default_geometry() -> BeamGeometry:
    # Four-beam Janus: beams at 45/135/225/315 deg, 20 deg from vertical.
    return geometry((45.0, 135.0, 225.0, 315.0), (20.0, 20.0, 20.0, 20.0))


@dataclass
class BeamSolution:
    velocity: tuple[float, float, float]  # instrument x/y/z m/s
    error_velocity: Optional[float]       # m/s, four-beam only
    rule_id: str
    beams: tuple[int, ...]                # 1-based beam indices used
    quality: dict


def solve_four_beam(geom: BeamGeometry, radial: np.ndarray,
                    correlations: Sequence[Optional[float]],
                    amplitudes: Sequence[Optional[float]],
                    bad_flags: Sequence[bool],
                    corr_floor: float = 60.0) -> BeamSolution:
    """Four-beam least squares. Error velocity is the null-space projection."""
    d = geom.matrix
    vel, *_ = np.linalg.lstsq(d, radial, rcond=None)
    q = np.array(ERROR_PATTERN[: geom.beam_count])
    residual = radial - d @ vel
    error = float(q @ residual / (q @ q))
    quality = beam_quality(correlations, amplitudes, bad_flags, corr_floor)
    return BeamSolution(tuple(float(v) for v in vel), error,
                        BEAM4_RULE_ID, tuple(range(1, geom.beam_count + 1)),
                        quality)


def legal_three_beam_subsets(quality: dict, corr_good: float = 150.0) -> list[tuple[int, ...]]:
    """Every 3-of-n subset that is a *legal* solution under janus3-subset-v1.

    Hard requirements per included beam: not flagged bad, correlation at or
    above the floor, amplitude interpretable. Soft requirement: the subset
    contains at least two high-correlation (``corr_good``) beams.
    """
    n = quality["beam_count"]
    usable = [i for i in range(n) if quality["usable"][i]]
    subsets: list[tuple[int, ...]] = []
    for combo in _combinations(usable, 3):
        strong = sum(1 for i in combo
                     if quality["correlation"][i] is not None
                     and quality["correlation"][i] >= corr_good)
        if strong >= 2:
            subsets.append(tuple(i + 1 for i in combo))
    return sorted(subsets)


def _combinations(items, k):
    if k == 0:
        yield ()
        return
    for i, item in enumerate(items):
        for rest in _combinations(items[i + 1:], k - 1):
            yield (item,) + rest


def solve_three_beam(geom: BeamGeometry, radial: np.ndarray,
                     beams: tuple[int, ...]) -> BeamSolution:
    idx = [b - 1 for b in beams]
    d3 = geom.matrix[idx, :]
    vel = np.linalg.solve(d3, radial[idx])
    return BeamSolution(tuple(float(v) for v in vel), None,
                        BEAM3_RULE_ID, tuple(beams),
                        {"subset": list(beams)})


def beam_quality(correlations, amplitudes, bad_flags, corr_floor) -> dict:
    usable, reasons = [], []
    for i, (corr, amp, bad) in enumerate(zip(correlations, amplitudes, bad_flags)):
        why = []
        if bad:
            why.append("bad_flag")
        if corr is None:
            why.append("correlation_missing")
        elif corr < corr_floor:
            why.append("correlation_floor")
        if amp is None:
            why.append("amplitude_unconvertible")
        usable.append(not why)
        reasons.append(why)
    return {"beam_count": len(bad_flags), "usable": usable, "reasons": reasons,
            "correlation": list(correlations), "amplitude": list(amplitudes),
            "bad_flag": list(bad_flags)}


# ---------------------------------------------------------------- rotations

def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def instrument_to_ship(velocity: Sequence[float], alignment_deg: float) -> np.ndarray:
    """Fixed mounting alignment rotation about the instrument vertical axis."""
    return _rz(math.radians(alignment_deg)) @ np.asarray(velocity, dtype=float)


def ship_to_earth(velocity: Sequence[float], heading_true_deg: float,
                  pitch_deg: float, roll_deg: float,
                  rule_id: str = EARTH_DEFAULT_RULE_ID) -> np.ndarray:
    h, p, r = (math.radians(heading_true_deg),
               math.radians(pitch_deg), math.radians(roll_deg))
    if rule_id == "ship2earth-hpr-zyx-v1":
        rot = _rz(h) @ _ry(p) @ _rx(r)
    elif rule_id == "ship2earth-hpr-zxy-v1":
        rot = _rz(h) @ _rx(r) @ _ry(p)
    else:
        raise ValueError(f"unknown earth transform order: {rule_id}")
    return rot @ np.asarray(velocity, dtype=float)


def magnetic_to_true(heading_magnetic_deg: float, declination_deg: float) -> float:
    """East-positive declination: true = magnetic + declination."""
    return (heading_magnetic_deg + declination_deg) % 360.0


# --------------------------------------------------------------- attitude

@dataclass
class AttitudeSample:
    time_s: float
    heading_mag_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class AttitudeAt:
    time_s: float
    segment: Optional[int]
    heading_true_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    diagnostic: Optional[dict] = None


def attitude_segments(samples: Sequence[AttitudeSample], max_gap_s: float) -> list[list[AttitudeSample]]:
    """Split samples into contiguous segments; gaps are never bridged."""
    ordered = sorted(samples, key=lambda s: s.time_s)
    segments: list[list[AttitudeSample]] = []
    for sample in ordered:
        if not segments or sample.time_s - segments[-1][-1].time_s > max_gap_s:
            segments.append([sample])
        else:
            segments[-1].append(sample)
    return segments


def attitude_at(samples: Sequence[AttitudeSample], time_s: float,
                declination_deg: float, max_gap_s: float) -> AttitudeAt:
    segments = attitude_segments(samples, max_gap_s)
    for seg_index, seg in enumerate(segments):
        first, last = seg[0], seg[-1]
        if time_s < first.time_s or time_s > last.time_s:
            continue
        if time_s == first.time_s:
            i0 = i1 = 0
            frac = 0.0
        else:
            i0 = max(i for i, s in enumerate(seg) if s.time_s <= time_s)
            i1 = min(i0 + 1, len(seg) - 1)
            span = seg[i1].time_s - seg[i0].time_s
            frac = 0.0 if span == 0.0 else (time_s - seg[i0].time_s) / span
        a, b = seg[i0], seg[i1]
        pitch = a.pitch_deg + (b.pitch_deg - a.pitch_deg) * frac
        roll = a.roll_deg + (b.roll_deg - a.roll_deg) * frac
        # Unwrap heading along the segment, then interpolate on the branch.
        unwrapped = [a.heading_mag_deg]
        for prev, cur in zip(seg[:-1], seg[1:]):
            unwrapped.append(unwrapped[-1] + _shortest_delta(prev.heading_mag_deg,
                                                             cur.heading_mag_deg))
        h0 = unwrapped[i0] + (unwrapped[i1] - unwrapped[i0]) * frac
        heading_true = magnetic_to_true(h0, declination_deg)
        return AttitudeAt(time_s, seg_index, heading_true % 360.0, pitch, roll)
    if samples and time_s < min(s.time_s for s in samples):
        code = "attitude_extrapolation"
    elif samples and time_s > max(s.time_s for s in samples):
        code = "attitude_extrapolation"
    else:
        code = "attitude_gap"
    return AttitudeAt(time_s, None, None, None, None,
                      {"code": code,
                       "message": "no contiguous attitude sample pair covers this ping; "
                                  "interpolation never crosses a time gap",
                       "time_s": time_s, "max_gap_s": max_gap_s})


def _shortest_delta(a: float, b: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return d
