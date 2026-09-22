"""Attitude handling and frame rotation matrices.

Rules enforced here (these are product semantics, see README):

* heading is interpolated on the unwrapped angle, so motion across 0/360 deg
  (350 -> 10) interpolates the short way;
* interpolation never crosses a data gap: a ping whose surrounding attitude
  samples are too far apart (or missing/bad) is blocked with a diagnostic,
  instrument-frame velocities stay valid but ship/earth do not;
* every rotation chain declares its order; order is part of the derived
  fingerprint, so reordering is a recompute, not an overwrite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Canonical order of elementary rotations. Alternatives are allowed but never
# silently treated as equivalent.
CANONICAL_ORDER = "roll-pitch-mount-heading"
KNOWN_ORDERS = (CANONICAL_ORDER, "pitch-roll-mount-heading")


def _rx(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class MountAngles:
    """Fixed transducer -> vessel offsets (degrees)."""

    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    heading_deg: float = 0.0

    def config_id(self) -> str:
        return "mount(p=%g,r=%g,h=%g)" % (
            self.pitch_deg,
            self.roll_deg,
            self.heading_deg,
        )


def ship_from_instrument(
    v_inst: Sequence[float],
    pitch_deg: float,
    roll_deg: float,
    mount: MountAngles,
    order: str = CANONICAL_ORDER,
) -> np.ndarray:
    """Tilt correction + mount offsets, instrument -> ship frame."""
    if order not in KNOWN_ORDERS:
        raise ValueError("unknown transform order: %r" % order)
    rm = _ry(mount.pitch_deg) @ _rx(mount.roll_deg) @ _rz(mount.heading_deg)
    if order == CANONICAL_ORDER:
        tilt = _ry(pitch_deg) @ _rx(roll_deg)
    else:
        tilt = _rx(roll_deg) @ _ry(pitch_deg)
    return rm @ tilt @ np.asarray(v_inst, dtype=float)


def earth_from_ship(v_ship: Sequence[float], true_heading_deg: float) -> np.ndarray:
    """Rotate ship -> earth by true heading (magnetic + declination)."""
    return _rz(true_heading_deg) @ np.asarray(v_ship, dtype=float)


def true_heading(heading_mag_deg: float, declination_deg: float) -> float:
    """Apply magnetic declination; output normalized to [0, 360)."""
    return (heading_mag_deg + declination_deg) % 360.0


@dataclass
class AttitudeSample:
    sid: str
    time_s: float
    heading_mag_deg: float
    pitch_deg: float
    roll_deg: float
    valid: bool = True


@dataclass
class AttitudeAtPing:
    ping_id: str
    time_s: float
    heading_mag_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    blocked: bool
    diagnostic: Optional[str]
    bracket: Optional[List[str]]

    def to_dict(self) -> dict:
        return {
            "ping_id": self.ping_id,
            "time_s": self.time_s,
            "heading_mag_deg": self.heading_mag_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "blocked": self.blocked,
            "diagnostic": self.diagnostic,
            "bracket": self.bracket,
        }


def _interp(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 == t0:
        return y0
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


def interpolate_attitude(
    samples: Sequence[AttitudeSample],
    ping_id: str,
    ping_time_s: float,
    max_gap_s: float,
) -> AttitudeAtPing:
    """Interpolate attitude at one ping; block instead of crossing a gap.

    The ping must lie between two *valid* consecutive samples whose spacing is
    at most ``max_gap_s``.  No extrapolation, no interpolation over a bad
    sample or over a hole.
    """
    good = sorted((s for s in samples if s.valid), key=lambda s: s.time_s)
    if not good:
        return AttitudeAtPing(ping_id, ping_time_s, None, None, None, True,
                              "no valid attitude samples", None)
    for s0, s1 in zip(good, good[1:]):
        if s0.time_s <= ping_time_s <= s1.time_s:
            if s1.time_s - s0.time_s > max_gap_s:
                return AttitudeAtPing(
                    ping_id, ping_time_s, None, None, None, True,
                    "attitude gap %.3fs exceeds max %.3fs"
                    % (s1.time_s - s0.time_s, max_gap_s),
                    [s0.sid, s1.sid],
                )
            h0 = math.radians(s0.heading_mag_deg)
            h1 = math.radians(s1.heading_mag_deg)
            dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
            heading = math.degrees(h0 + _interp(ping_time_s, s0.time_s,
                                                s1.time_s, 0.0, dh)) % 360.0
            pitch = _interp(ping_time_s, s0.time_s, s1.time_s,
                            s0.pitch_deg, s1.pitch_deg)
            roll = _interp(ping_time_s, s0.time_s, s1.time_s,
                           s0.roll_deg, s1.roll_deg)
            return AttitudeAtPing(ping_id, ping_time_s, heading, pitch, roll,
                                  False, None, [s0.sid, s1.sid])
    return AttitudeAtPing(
        ping_id, ping_time_s, None, None, None, True,
        "ping outside attitude sample bracket (no extrapolation)", None,
    )
