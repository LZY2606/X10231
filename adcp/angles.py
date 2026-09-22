"""Attitude handling and instrument -> ship -> earth rotations.

Properties guarded here (see requirements):

* magnetic heading + magnetic declination are *separate* sourced records;
  declination carries its own model/version, which enters derived fingerprints;
* heading interpolation unwraps the 360-degree branch before interpolating;
* interpolation never crosses a declared time gap;
* the coordinate transform *order* is explicit and fingerprinted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import as_rad, wrap_360, angle_diff_short

ANGLES_SCHEMA_VERSION = "angles/1"


def rot_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Declination:
    """East-positive magnetic declination with its source model/version."""

    value_deg: float
    model: str = "WMM"
    model_version: str = "unknown"
    epoch: Optional[float] = None

    def to_fingerprint(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "model_version": self.model_version,
            "value_deg": round(self.value_deg, 9),
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class Mounting:
    """Transducer mounting angles relative to the ship frame."""

    heading_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    # Transform application order for instrument -> ship.
    order: str = "HPR"  # "HPR": Rz Ry Rx; "RPH": Rx Ry Rz
    revision: str = "rev-0"

    def to_fingerprint(self) -> Dict[str, object]:
        return {
            "heading_offset_deg": round(self.heading_offset_deg, 9),
            "pitch_offset_deg": round(self.pitch_offset_deg, 9),
            "roll_offset_deg": round(self.roll_offset_deg, 9),
            "order": self.order,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class Attitude:
    """One heading/pitch/roll sample."""

    t: float
    heading_deg: float       # 0..360
    heading_datum: str       # "magnetic" | "true"
    pitch_deg: float
    roll_deg: float


def true_heading(att: Attitude, decl: Optional[Declination]) -> float:
    h = att.heading_deg
    if att.heading_datum == "magnetic":
        if decl is None:
            raise ValueError("magnetic heading requires a declination record")
        h = h + decl.value_deg
    return wrap_360(h)


def instrument_to_ship_matrix(mounting: Mounting) -> np.ndarray:
    h = as_rad(mounting.heading_offset_deg)
    p = as_rad(mounting.pitch_offset_deg)
    r = as_rad(mounting.roll_offset_deg)
    if mounting.order == "HPR":
        return rot_z(h) @ rot_y(p) @ rot_x(r)
    if mounting.order == "RPH":
        return rot_x(r) @ rot_y(p) @ rot_z(h)
    raise ValueError("unknown mounting order %r" % (mounting.order,))


def ship_to_earth_matrix(
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "HPR",
    earth_datum: str = "ENU",
) -> np.ndarray:
    """Rotate ship-frame vector into an earth frame.

    ``order`` is explicit ("HPR" or "RPH") so a coordinate-transform order
    change yields a different derived fingerprint.  Default "HPR" is the
    classic RDI ENU convention ``Rz(H) Ry(P) Rx(R)``.
    """
    h, p, r = as_rad(heading_true_deg), as_rad(pitch_deg), as_rad(roll_deg)
    if order == "HPR":
        m = rot_z(h) @ rot_y(p) @ rot_x(r)
    elif order == "RPH":
        m = rot_x(r) @ rot_y(p) @ rot_z(h)
    else:
        raise ValueError("unknown ship->earth order %r" % (order,))
    if earth_datum == "ENU":
        return m
    if earth_datum == "NED":
        # ENU -> NED: (E,N,U) to (N,E,-D-up): swap x/y and flip z.
        swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        return swap @ m
    raise ValueError("unknown earth datum %r" % (earth_datum,))


# --- Attitude interpolation with gap protection -----------------------------


@dataclass(frozen=True)
class GapResult:
    t: float
    attitude: Optional[Attitude]
    interpolated: bool
    status: str          # "sample" | "interpolated" | "gap" | "outside"
    bracket: Optional[Tuple[int, int]] = None
    fraction: Optional[float] = None


def interpolate_attitude(
    samples: Sequence[Attitude],
    t: float,
    max_gap_s: float,
    declared_gaps: Optional[Sequence[Tuple[float, float]]] = None,
) -> GapResult:
    """Linear interpolation of attitude at time ``t``.

    * Exact sample -> that sample.
    * Bracketed samples whose span <= ``max_gap_s`` and not crossing a
      declared gap -> unwrapped linear interpolation (heading crosses 360
      correctly).
    * Otherwise no attitude is produced (a gap); callers must exclude rather
      than inventing rotation.
    """
    declared_gaps = declared_gaps or []
    if not samples:
        return GapResult(t, None, False, "gap")

    times = [s.t for s in samples]
    for lo, hi in declared_gaps:
        if lo < t < hi:
            return GapResult(t, None, False, "gap")

    # Exact sample?
    for i, s in enumerate(samples):
        if abs(s.t - t) <= 1e-9:
            return GapResult(t, s, False, "sample", (i, i), 0.0)

    if t < times[0] or t > times[-1]:
        return GapResult(t, None, False, "outside")

    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        if a.t <= t <= b.t:
            span = b.t - a.t
            crosses_declared = any(
                not (hi <= a.t or lo >= b.t) for lo, hi in declared_gaps
            )
            if span > max_gap_s or crosses_declared:
                return GapResult(t, None, False, "gap", (i, i + 1), None)
            f = 0.0 if span == 0 else (t - a.t) / span
            dh = angle_diff_short(a.heading_deg, b.heading_deg)
            heading = wrap_360(a.heading_deg + f * dh)
            pitch = a.pitch_deg + f * (b.pitch_deg - a.pitch_deg)
            roll = a.roll_deg + f * (b.roll_deg - a.roll_deg)
            datum = a.heading_datum if a.heading_datum == b.heading_datum else "magnetic"
            return GapResult(
                t,
                Attitude(t, heading, datum, pitch, roll),
                True,
                "interpolated",
                (i, i + 1),
                f,
            )
    return GapResult(t, None, False, "gap")
