"""Attitude handling and instrument -> ship -> earth coordinate transforms.

Angles are canonical degrees.  Heading is clockwise from the datum; magnetic
heading + magnetic declination (+ ship heading alignment) gives true heading.
The transform ORDER is an explicit, versioned choice ("hpr_rdi" standard
ship-heading convention vs "rph_alt"), so order changes change the fingerprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRANSFORM_ORDERS = ("hpr_rdi", "rph_alt")


def _d2r(x: float) -> float:
    return math.radians(x)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def instrument_to_ship(vi: np.ndarray, mounting: "Mounting") -> np.ndarray:
    """Instrument frame to ship frame.

    mounting_alignment is the fixed heading offset of the transducer relative to
    ship heading; pitch/roll offsets are small installation angles.  The
    installation-angle convention itself is versioned in Mounting.
    """
    m = (rotz(_d2r(mounting.alignment_deg))
         @ roty(_d2r(mounting.pitch_offset_deg))
         @ rotx(_d2r(mounting.roll_offset_deg)))
    return m @ np.asarray(vi, dtype=float)


def ship_to_earth(vs: np.ndarray, heading_deg: float, pitch_deg: float,
                  roll_deg: float, order: str = "hpr_rdi") -> np.ndarray:
    """Ship frame to earth frame (east, north, up) in two versioned orders."""
    h = _d2r(heading_deg)
    p = _d2r(pitch_deg)
    r = _d2r(roll_deg)
    if order == "hpr_rdi":
        # Conventional heading/pitch/roll ADCP chain.
        return rotz(h) @ roty(p) @ rotx(r) @ np.asarray(vs, dtype=float)
    if order == "rph_alt":
        # Explicitly different ordering, retained as a competing rule version.
        return rotz(h) @ rotx(r) @ roty(p) @ np.asarray(vs, dtype=float)
    raise ValueError(f"unknown transform order {order!r}")


def true_heading(magnetic_heading_deg: float, declination_deg: float,
                 alignment_deg: float = 0.0) -> float:
    """Magnetic heading -> true heading.  Declination kept separate & versioned."""
    return (magnetic_heading_deg + declination_deg + alignment_deg) % 360.0


@dataclass
class Mounting:
    mounting_id: str
    alignment_deg: float
    pitch_offset_deg: float
    roll_offset_deg: float


# ---------------------------------------------------------------------------
# Attitude interpolation: wraps 360 deg but never interpolates across a gap.
# ---------------------------------------------------------------------------

@dataclass
class InterpResult:
    time: float
    heading: float | None
    pitch: float | None
    roll: float | None
    interpolated: bool
    wrapped: bool = False
    gap: bool = False
    diagnostic: str | None = None


def unwrap_deg(angles: list[float]) -> list[float]:
    out: list[float] = []
    for i, a in enumerate(angles):
        if i == 0:
            out.append(float(a))
            continue
        d = a - out[-1]
        while d > 180.0:
            a -= 360.0
            d = a - out[-1]
        while d < -180.0:
            a += 360.0
            d = a - out[-1]
        out.append(float(a))
    return out


def interp_attitude(times: list[float], heading: list[float],
                    pitch: list[float], roll: list[float], t: float,
                    max_gap_s: float = 2.0) -> InterpResult:
    """Interpolate attitude at time t.

    Heading is unwrapped first so a 359.7 -> 0.4 transition interpolates
    correctly across zero.  If the two surrounding attitude samples are more
    than ``max_gap_s`` apart (or t falls outside the span), no interpolation is
    performed and a gap diagnostic is returned: attitude interpolation never
    crosses a time gap.
    """
    if not times:
        return InterpResult(t, None, None, None, False, diagnostic="no_attitude")
    times_a = np.asarray(times, dtype=float)
    if t < times_a[0] - 1e-9 or t > times_a[-1] + 1e-9:
        return InterpResult(t, None, None, None, False, gap=True,
                            diagnostic="outside_attitude_span")
    j = int(np.searchsorted(times_a, t))
    if j < len(times) and abs(times[j] - t) < 1e-9:
        return InterpResult(t, heading[j] % 360.0, pitch[j], roll[j], False)
    lo, hi = max(0, j - 1), min(len(times) - 1, j)
    if hi <= lo:
        return InterpResult(t, None, None, None, False, gap=True,
                            diagnostic="attitude_undersampled")
    dt = times[hi] - times[lo]
    if dt > max_gap_s + 1e-9:
        return InterpResult(t, None, None, None, False, gap=True,
                            diagnostic=f"attitude_gap_{dt:g}s")
    frac = (t - times[lo]) / dt
    hu = unwrap_deg(heading)
    wrapped = abs(heading[hi] - heading[lo]) > 180.0
    h = (hu[lo] + frac * (hu[hi] - hu[lo])) % 360.0
    p = pitch[lo] + frac * (pitch[hi] - pitch[lo])
    r = roll[lo] + frac * (roll[hi] - roll[lo])
    return InterpResult(t, h, p, r, True, wrapped=wrapped)
