"""ADCP beam geometry and coordinate-frame mathematics.

Frames:
    beam       - radial velocity along one of four Janus beams
    instrument - transducer body axes (x=starboard, y=forward, z=up)
    ship       - vessel axes after mounting angle and tilt (pitch/roll)
    earth      - east/north/up, after heading (with magnetic declination)

The rules implemented here are deliberately explicit so that the derived
fingerprint can name the exact geometry/order a value came from.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# null-space pattern of the 4x3 Janus beam matrix: error velocity
NULL_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])


def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def beam_matrix(beam_angle_deg: float, azimuths_deg: Sequence[float]) -> np.ndarray:
    """D such that v_beam = D @ V_instrument (one row per beam)."""
    theta = np.radians(beam_angle_deg)
    rows = []
    for azimuth in azimuths_deg:
        phi = np.radians(azimuth)
        rows.append(
            [np.sin(theta) * np.cos(phi),
             np.sin(theta) * np.sin(phi),
             np.cos(theta)]
        )
    return np.asarray(rows, dtype=float)


def solve_four(v_beam: Sequence[float], D: np.ndarray):
    """Four-beam least-squares solution plus error velocity."""
    v_beam = np.asarray(v_beam, dtype=float)
    V, *_ = np.linalg.lstsq(D, v_beam, rcond=None)
    residual = v_beam - D @ V
    err = float(np.dot(residual, NULL_PATTERN[: len(v_beam)]) / max(len(v_beam), 1))
    return V, err


def solve_three(v_kept: Sequence[float], kept_beams: Sequence[int],
                D: np.ndarray, error_velocity: float = 0.0) -> np.ndarray:
    """Three-beam solution.

    With three beams the (u, v, w, error) system is underdetermined;
    ``error_velocity`` fixes the free null-space coordinate.  Use
    :func:`three_beam_candidates` to retain every legal hypothesis rather
    than silently picking one.

    ``v_kept`` is aligned with ``kept_beams`` (zero-based row indices).
    """
    idx = [b for b in kept_beams]
    D3 = D[idx, :]
    v3 = np.asarray(v_kept, dtype=float)
    s3 = NULL_PATTERN[idx]
    return np.linalg.solve(D3, v3 - error_velocity * s3)


def three_beam_candidates(v_beam: Sequence[float], kept_beams: Sequence[int],
                          D: np.ndarray, err_bound: float) -> List[Dict]:
    """All legal three-beam candidates over the error-velocity bound."""
    candidates = []
    for error_velocity in (-err_bound, 0.0, err_bound):
        V = solve_three(v_beam, kept_beams, D, error_velocity)
        candidates.append({
            "error_velocity_ms": round(float(error_velocity), 6),
            "u_ms": round(float(V[0]), 6),
            "v_ms": round(float(V[1]), 6),
            "w_ms": round(float(V[2]), 6),
        })
    return candidates


def unwrap_headings(headings_deg: Sequence[float]) -> np.ndarray:
    """Continuous heading series across the 0/360 seam (e.g. 359.6 -> 360.4)."""
    return np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, dtype=float))))


def interpolate_attitude(samples: Sequence[Dict], t: float, max_gap_s: float) -> Dict:
    """Interpolate heading/pitch/roll at ping time ``t``.

    Heading is unwrapped across the 360-degree seam before interpolation.
    If the bracketing samples are missing or separated by more than
    ``max_gap_s`` the result is status=gap: attitude is **never**
    interpolated across a time gap.
    """
    if len(samples) < 2:
        return {"status": "gap", "reason": "no bracketing attitude samples", "t": t}
    samples = sorted(samples, key=lambda s: s["t"])
    ts = [float(s["t"]) for s in samples]
    headings = unwrap_headings([s["heading"] for s in samples])
    index = None
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            index = i
            break
    if index is None:
        return {"status": "gap", "reason": "ping outside attitude sample range",
                "t": t, "nearest": [ts[0], ts[-1]]}
    gap = ts[index + 1] - ts[index]
    bracket = [ts[index], ts[index + 1]]
    if gap > max_gap_s:
        return {"status": "gap", "reason": "attitude samples gap exceeds limit; "
                "no interpolation across the gap", "gap_s": round(gap, 6),
                "max_gap_s": max_gap_s, "bracket": bracket}
    frac = 0.0 if gap == 0 else (t - ts[index]) / gap
    heading_cont = headings[index] + frac * (headings[index + 1] - headings[index])
    pitch = samples[index]["pitch"] + frac * (
        samples[index + 1]["pitch"] - samples[index]["pitch"])
    roll = samples[index]["roll"] + frac * (
        samples[index + 1]["roll"] - samples[index]["roll"])
    return {
        "status": "ok",
        "t": t,
        "heading_mag_deg": round(float(heading_cont % 360.0), 6),
        "heading_continuous_deg": round(float(heading_cont), 6),
        "pitch_deg": round(float(pitch), 6),
        "roll_deg": round(float(roll), 6),
        "bracket_s": bracket,
        "method": "linear-on-unwrapped",
    }


def instrument_to_ship(v_instrument, pitch_deg: float, roll_deg: float,
                       mounting_deg: float) -> np.ndarray:
    """Instrument -> ship: mounting yaw, then roll and pitch."""
    v = np.asarray(v_instrument, dtype=float)
    return rot_y(pitch_deg) @ rot_x(roll_deg) @ rot_z(mounting_deg) @ v


def ship_to_earth(v_ship, heading_true_deg: float) -> np.ndarray:
    """Ship -> earth under the standard HPR composition (heading last)."""
    return rot_z(heading_true_deg) @ np.asarray(v_ship, dtype=float)


def instrument_to_earth_alternative(v_instrument, heading_true_deg: float,
                                    pitch_deg: float, roll_deg: float,
                                    mounting_deg: float) -> np.ndarray:
    """Alternative transform order: heading applied *before* tilt.

    This is the documented counter-example: with non-zero pitch/roll it
    disagrees with the standard order, and the order string itself enters
    the derived fingerprint so the two results can never be confused.
    """
    v = np.asarray(v_instrument, dtype=float)
    return (rot_y(pitch_deg) @ rot_x(roll_deg)
            @ rot_z(heading_true_deg) @ rot_z(mounting_deg) @ v)
