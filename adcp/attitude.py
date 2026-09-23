"""Attitude handling: heading unwrap, gap-aware interpolation, rotations.

Semantics held here:
- heading is unwrapped before interpolation so a 359 deg -> 1 deg step rotates
  2 degrees instead of 358;
- interpolation never crosses a time gap (the caller declares the maximum
  legal sample spacing);
- coordinate transform *order* is explicit, so order changes show up in the
  derived fingerprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def wrap360(angle_deg: float) -> float:
    return angle_deg % 360.0


def unwrap_heading(headings_deg: Sequence[float]) -> Tuple[float, ...]:
    """Unwrap a heading series that crosses 360 degrees."""
    if not headings_deg:
        return tuple()
    out = [float(headings_deg[0])]
    for value in headings_deg[1:]:
        delta = (float(value) - out[-1] + 180.0) % 360.0 - 180.0
        out.append(out[-1] + delta)
    return tuple(out)


def rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# Rotation order tags: each tag is part of the derived fingerprint.
TRANSFORM_ORDERS = ("RDI_HPR", "aerospace_YPR", "HPR_no_tilt_correction")


@dataclass
class AttitudeSample:
    time_s: float
    heading_deg: float  # magnetic, 0-360
    pitch_deg: float
    roll_deg: float


def interpolate_attitude(
    samples: Sequence[AttitudeSample],
    at_time_s: float,
    max_gap_s: float,
) -> Dict[str, object]:
    """Linear attitude interpolation that refuses to cross a time gap."""
    times = [sample.time_s for sample in samples]
    if not samples:
        return {"status": "no_attitude", "diagnostic": "attitude_series_empty"}

    # Exact match.
    for sample in samples:
        if sample.time_s == at_time_s:
            return {
                "status": "ok",
                "time_s": at_time_s,
                "heading_deg": wrap360(sample.heading_deg),
                "pitch_deg": float(sample.pitch_deg),
                "roll_deg": float(sample.roll_deg),
                "mode": "exact",
            }

    if at_time_s < times[0] or at_time_s > times[-1]:
        return {"status": "extrapolation_refused", "diagnostic": "ping_outside_attitude_span"}

    for left, right in zip(samples, samples[1:]):
        if left.time_s <= at_time_s <= right.time_s:
            gap = right.time_s - left.time_s
            if gap > max_gap_s:
                return {
                    "status": "gap_refused",
                    "diagnostic": "attitude_gap",
                    "gap_s": gap,
                    "limit_s": max_gap_s,
                    "left_time_s": left.time_s,
                    "right_time_s": right.time_s,
                }
            frac = (at_time_s - left.time_s) / gap
            unwrapped = unwrap_heading([left.heading_deg, right.heading_deg])
            heading = wrap360(unwrapped[0] + frac * (unwrapped[1] - unwrapped[0]))
            return {
                "status": "ok",
                "time_s": at_time_s,
                "heading_deg": heading,
                "pitch_deg": left.pitch_deg + frac * (right.pitch_deg - left.pitch_deg),
                "roll_deg": left.roll_deg + frac * (right.roll_deg - left.roll_deg),
                "mode": "interpolated",
            }

    return {"status": "no_bracket", "diagnostic": "attitude_not_bracketed"}


def _tilt_matrix(order: str, pitch_deg: float, roll_deg: float) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    if order == "RDI_HPR":
        # RDI convention: roll first, then pitch (tilt correction order fixed).
        return ry(pitch) @ rx(roll)
    if order == "aerospace_YPR":
        # Alternate published convention, retained to prove order matters.
        return rx(roll) @ ry(pitch)
    if order == "HPR_no_tilt_correction":
        return np.eye(3)
    raise ValueError(f"unknown transform order: {order}")


def instrument_to_ship(velocity_xyz: np.ndarray, mount_pitch_deg: float, mount_roll_deg: float) -> np.ndarray:
    """Static transducer mounting alignment (installation angles)."""
    return ry(math.radians(mount_pitch_deg)) @ rx(math.radians(mount_roll_deg)) @ np.asarray(velocity_xyz)


def ship_to_earth(
    velocity_xyz: np.ndarray,
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "RDI_HPR",
) -> np.ndarray:
    """Ship-frame vector -> earth ENU. ``order`` is fingerprinted downstream."""
    if order not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order: {order}")
    tilt = _tilt_matrix(order, pitch_deg, roll_deg)
    heading = math.radians(heading_true_deg)
    return rz(heading) @ tilt @ np.asarray(velocity_xyz)
