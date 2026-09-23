"""Attitude rotations, mounting offsets and the ordered transform chain.

Frames
------
* instrument: X right, Y forward, Z up (transducer axes)
* ship:       X starboard, Y forward, Z up (vessel axes)
* earth:      X east, Y north, Z up

Heading is clockwise from north (navigation convention); pitch is tilt about
the ship X axis (bow-up positive), roll about the ship Y axis (starboard-up
positive).  Magnetic heading is turned into true heading using an explicit,
versioned declination (east-positive): ``true = wrap(magnetic + decl)``.

Every chain step carries an ordered id, so "beam geometry / mount / decl /
attitude / order" all enter the derived fingerprint; changing the order
changes the fingerprint even if the inputs are byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import lerp, lerp_angle, wrap_deg

# Default order.  "order" is part of the fingerprint, so alternatives are
# first-class chains rather than silent re-runs.
DEFAULT_ORDER: Tuple[str, ...] = (
    "mount_yaw",
    "mount_roll",
    "mount_pitch",
    "roll",
    "pitch",
    "heading",
)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    # Clockwise-from-north heading uses the transposed (clockwise) matrix.
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def inst_to_ship_matrix(
    mount_heading_deg: float,
    mount_pitch_deg: float,
    mount_roll_deg: float,
    roll_deg: float,
    pitch_deg: float,
    order: Sequence[str] = DEFAULT_ORDER,
) -> np.ndarray:
    """Compound ``v_ship = M v_inst`` with an explicit application order."""
    angles = {
        "mount_yaw": wrap_deg(mount_heading_deg),
        "mount_pitch": mount_pitch_deg,
        "mount_roll": mount_roll_deg,
        "roll": roll_deg,
        "pitch": pitch_deg,
        "heading": 0.0,
    }
    builders = {
        "mount_yaw": lambda: _rz(math.radians(angles["mount_yaw"])),
        "mount_pitch": lambda: _ry(math.radians(angles["mount_pitch"])),
        "mount_roll": lambda: _rx(math.radians(angles["mount_roll"])),
        "roll": lambda: _rx(math.radians(angles["roll"])),
        "pitch": lambda: _ry(math.radians(angles["pitch"])),
        "heading": lambda: np.eye(3),
    }
    if set(order) != set(builders):
        raise ValueError(f"bad transform order: {order!r}")
    m = np.eye(3)
    for step in order:
        m = builders[step]() @ m
    return m


def heading_matrix(true_heading_deg: float) -> np.ndarray:
    return _rz(math.radians(wrap_deg(true_heading_deg)))


@dataclass(frozen=True)
class ChainStep:
    order: int
    name: str
    angle_deg: Optional[float]
    source: str                       # node id + version of the angle source
    matrix: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class TransformChain:
    order: Tuple[str, ...]
    steps: Tuple[ChainStep, ...]
    matrix: Tuple[Tuple[float, ...], ...]

    def fingerprint(self) -> Dict[str, object]:
        return {
            "order": list(self.order),
            "steps": [
                {
                    "order": s.order,
                    "name": s.name,
                    "angle_deg": s.angle_deg,
                    "source": s.source,
                }
                for s in self.steps
            ],
        }


def build_chain(
    order: Sequence[str],
    mount_heading_deg: float,
    mount_pitch_deg: float,
    mount_roll_deg: float,
    roll_deg: float,
    pitch_deg: float,
    true_heading_deg: float,
    mount_source: str,
    attitude_source: str,
) -> TransformChain:
    """Build a fully ordered, source-attributed instrument->earth chain.

    Mount and attitude rotations go first (in ``order``), then the true
    heading rotation last.  Each step remembers the versioned node that
    supplied its angle, which is what makes recompute explanations possible.
    """
    vals = {
        "mount_yaw": (wrap_deg(mount_heading_deg), mount_source,
                      lambda a: _rz(math.radians(a))),
        "mount_pitch": (mount_pitch_deg, mount_source,
                        lambda a: _ry(math.radians(a))),
        "mount_roll": (mount_roll_deg, mount_source,
                       lambda a: _rx(math.radians(a))),
        "roll": (roll_deg, attitude_source,
                 lambda a: _rx(math.radians(a))),
        "pitch": (pitch_deg, attitude_source,
                  lambda a: _ry(math.radians(a))),
        "heading": (wrap_deg(true_heading_deg), attitude_source,
                    lambda a: _rz(math.radians(a))),
    }
    if set(order) - set(vals):
        raise ValueError(f"bad transform order: {order!r}")
    steps: List[ChainStep] = []
    m = np.eye(3)
    seq = list(order)
    if "heading" not in seq:
        seq = seq + ["heading"]
    for i, name in enumerate(seq):
        ang, src, build = vals[name]
        r = build(ang)
        m = r @ m
        steps.append(ChainStep(i, name, ang, src,
                               tuple(tuple(row) for row in r.tolist())))
    return TransformChain(tuple(seq), tuple(steps),
                          tuple(tuple(row) for row in m.tolist()))


def apply_chain(chain: TransformChain, vec: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(chain.matrix) @ np.asarray(vec))


@dataclass(frozen=True)
class AttitudeInterp:
    ok: bool
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    method: str
    diagnostic: Optional[str]
    bracket: Optional[Tuple[str, int, str, int]]
    frac: Optional[float]
    gap_s: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "heading_deg": self.heading_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "method": self.method,
            "diagnostic": self.diagnostic,
            "bracket": list(self.bracket) if self.bracket else None,
            "frac": self.frac,
            "gap_s": self.gap_s,
        }


def interpolate_attitude(
    t: float,
    before: Optional[Tuple[str, int, float, float, float, float]],
    after: Optional[Tuple[str, int, float, float, float, float]],
    max_gap_s: float,
) -> AttitudeInterp:
    """Attitude at time ``t`` bracketed by two pings.

    Each bracket sample is ``(node_id, version, time_s, heading, pitch,
    roll)`` in degrees.  Heading is interpolated on the shortest path, so a
    sequence like 358 -> 4 deg interpolates *through* 360 (359, 0, 1 ...).
    When the bracket spans a declared gap larger than ``max_gap_s`` no value
    is synthesized: interpolation never crosses a gap.
    """
    if before is None or after is None:
        return AttitudeInterp(False, None, None, None, "none",
                              "ATTITUDE_NO_BRACKET", None, None, None)
    b_id, b_ver, b_t, b_h, b_p, b_r = before
    a_id, a_ver, a_t, a_h, a_p, a_r = after
    if a_t < b_t:
        b_id, a_id = a_id, b_id
        b_ver, a_ver = a_ver, b_ver
        b_t, a_t = a_t, b_t
        b_h, a_h = a_h, b_h
        b_p, a_p = a_p, b_p
        b_r, a_r = a_r, b_r
    gap = a_t - b_t
    if gap > max_gap_s:
        return AttitudeInterp(False, None, None, None, "none",
                              "ATTITUDE_GAP_EXCEEDED",
                              (b_id, b_ver, a_id, a_ver), None, gap)
    frac = 0.0 if a_t == b_t else (t - b_t) / (a_t - b_t)
    return AttitudeInterp(
        True,
        lerp_angle(b_h, a_h, frac),
        lerp(b_p, a_p, frac),
        lerp(b_r, b_r + (a_r - b_r) * frac),
        "interp_shortest_path",
        None,
        (b_id, b_ver, a_id, a_ver),
        frac,
        gap,
    )


def true_heading(magnetic_heading_deg: float, declination_deg: float) -> float:
    """Apply east-positive magnetic declination."""
    return wrap_deg(magnetic_heading_deg + declination_deg)
