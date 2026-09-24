"""Attitude rotation: instrument -> ship -> earth(ENU).

The transform order is a versioned rule and enters every derived
fingerprint.  Two orders are supported:

* ``rdi-zyx-v1``: intrinsic heading -> pitch -> roll,
  ``R = Rz(h) @ Ry(p) @ Rx(r)`` -- the classic ADCP cardan order;
* ``hpr-zxy-v1``: intrinsic heading -> roll -> pitch,
  ``R = Rz(h) @ Rx(r) @ Ry(p)`` -- a deliberately different order kept so
  an order change can be shown as an isolated, evidence-carrying mutation.

Conventions: x forward, y starboard, z down.  Positive pitch = bow up,
positive roll = port side up.  Headings rotate magnetic -> true with
``true = wrap(magnetic + declination)`` (east-positive declination).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import wrap_deg_0_360

TRANSFORM_ORDERS = ("rdi-zyx-v1", "hpr-zxy-v1")


def _x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    # positive roll rotates forward toward down / down toward starboard
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    # positive pitch = bow up: forward axis gains an upward (-z) component
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def attitude_matrix(
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order_id: str = "rdi-zyx-v1",
) -> np.ndarray:
    if order_id not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order {order_id!r}")
    heading = _z(heading_true_deg)
    pitch = _y(pitch_deg)
    roll = _x(roll_deg)
    if order_id == "rdi-zyx-v1":
        return heading @ pitch @ roll
    return heading @ roll @ pitch


def instrument_to_ship_matrix(pitch_deg: float, roll_deg: float, order_id: str) -> np.ndarray:
    if order_id == "rdi-zyx-v1":
        return _y(pitch_deg) @ _x(roll_deg)
    return _x(roll_deg) @ _y(pitch_deg)


def ship_to_earth_matrix(heading_true_deg: float, order_id: str) -> np.ndarray:
    return _z(heading_true_deg)


@dataclass
class MountAngles:
    """Transducer alignment relative to the ship frame (degrees)."""

    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    heading_offset_deg: float = 0.0
    mount_id: str = "mount-v1"

    def content(self) -> dict:
        return {
            "mount_id": self.mount_id,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "heading_offset_deg": self.heading_offset_deg,
        }


def apply_mount_angles(
    velocity_xyz: np.ndarray, mount: MountAngles, order_id: str
) -> np.ndarray:
    """Instrument -> ship including fixed transducer misalignment."""
    matrix = instrument_to_ship_matrix(mount.pitch_deg, mount.roll_deg, order_id)
    aligned = matrix @ np.asarray(velocity_xyz, dtype=float)
    if mount.heading_offset_deg:
        aligned = _z(mount.heading_offset_deg) @ aligned
    return aligned


def true_heading(magnetic_heading_deg: float, declination_deg: float) -> float:
    return wrap_deg_0_360(magnetic_heading_deg + declination_deg)


@dataclass
class AttitudeSample:
    t: float
    heading_mag: float
    pitch: float
    roll: float


@dataclass
class AttitudeResult:
    t: float
    heading_mag: float
    heading_true: float
    pitch: float
    roll: float
    interpolated: bool
    diagnostics: list[str]


def interpolate_attitude(
    samples: list[AttitudeSample],
    query_t: float,
    declination_deg: float,
    max_gap_s: float,
) -> AttitudeResult:
    """Look up / linearly interpolate attitude at ``query_t``.

    Heading is unwrapped before interpolation (359 -> 1 continuity).
    Interpolation is **never** performed across a time gap larger than
    ``max_gap_s``: the query keeps no attitude and receives a diagnostic.
    """
    from .units import unwrap_deg

    if not samples:
        return AttitudeResult(query_t, math.nan, math.nan, math.nan, math.nan,
                              False, ["attitude:no-samples"])
    times = [s.t for s in samples]
    if query_t < times[0] - 1e-9 or query_t > times[-1] + 1e-9:
        return AttitudeResult(query_t, math.nan, math.nan, math.nan, math.nan,
                              False, ["attitude:outside-time-range"])

    exact = next((s for s in samples if abs(s.t - query_t) <= 1e-9), None)
    if exact is not None:
        return AttitudeResult(
            exact.t, exact.heading_mag,
            true_heading(exact.heading_mag, declination_deg),
            exact.pitch, exact.roll, False, [],
        )

    right = next(i for i, t in enumerate(times) if t > query_t)
    left = right - 1
    gap = times[right] - times[left]
    if gap > max_gap_s:
        return AttitudeResult(
            query_t, math.nan, math.nan, math.nan, math.nan, True,
            [f"attitude:gap-exceeds-max:{gap:.3f}s>{max_gap_s:.3f}s"],
        )

    fraction = (query_t - times[left]) / gap
    headings = unwrap_deg([s.heading_mag for s in samples])
    heading = headings[left] + fraction * (headings[right] - headings[left])
    pitch = samples[left].pitch + fraction * (samples[right].pitch - samples[left].pitch)
    roll = samples[left].roll + fraction * (samples[right].roll - samples[left].roll)
    heading_mag = wrap_deg_0_360(heading)
    diagnostics = []
    if abs(
        wrap_deg_0_360(samples[right].heading_mag)
        - wrap_deg_0_360(samples[left].heading_mag)
    ) > 180:
        diagnostics.append("attitude:heading-crossed-360")
    return AttitudeResult(
        query_t, heading_mag, true_heading(heading_mag, declination_deg),
        pitch, roll, True, diagnostics,
    )
