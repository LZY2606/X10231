"""Staged ADCP coordinate transforms and attitude handling.

Frames
------
- ``beam_radial``: along-beam radial velocity per beam. Raw records keep the
  original representation (mm/s integers as logged by the instrument).
- ``instrument``: x starboard, y forward, z up, relative to the transducer.
- ``ship``: instrument frame rotated by pitch/roll (attitude).
- ``earth``: x east, y north, z up. yaw = heading + magnetic declination
  + mounting angle.

Rules encoded here (mirrored by machine-readable judgment events):
- a three-beam solution is never merged into a four-beam value;
- heading interpolation unwraps around 360 deg and never crosses a time gap;
- transform order, beam geometry and declination versions all feed the
  derivation fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

ATTITUDE_POLICY_VERSION = "att-v1"  # unwrap-360, no interpolation across gaps
GEOMETRY_VERSION = "geom-v1"        # Janus 20 deg, beam azimuths 0/180/90/270
JANUS_DEG = 20.0
BEAM_AZIMUTHS_DEG = (0.0, 180.0, 90.0, 270.0)
GAP_THRESHOLD_S = 2.5
BAD_SENTINEL_MM_S = -32768          # RDI-style bad value, kept as-is in raw layer


def beam_unit_vectors(janus_deg: float = JANUS_DEG) -> list[tuple[float, float, float]]:
    """Unit vector of each beam in the instrument frame (z up)."""
    s = math.sin(math.radians(janus_deg))
    c = math.cos(math.radians(janus_deg))
    out = []
    for az in BEAM_AZIMUTHS_DEG:
        a = math.radians(az)
        out.append((s * math.cos(a), s * math.sin(a), c))
    return out


def beams_to_instrument(radial_m_s, beams_used, janus_deg: float = JANUS_DEG) -> dict:
    """Invert beam-radial velocities (m/s) to instrument-frame u/v/w.

    ``beams_used`` holds 1-based beam ids. Four beams give the redundant
    solution plus an error velocity; three beams give an exactly determined
    solution with no error velocity. The two kinds are never averaged into
    a single value: callers store them under separate identities.
    """
    vecs = beam_unit_vectors(janus_deg)
    r = np.asarray(list(radial_m_s), dtype=float)
    if len(beams_used) == 4:
        s = math.sin(math.radians(janus_deg))
        c = math.cos(math.radians(janus_deg))
        r1, r2, r3, r4 = r
        return {
            "u": float((r1 - r2) / (2 * s)),
            "v": float((r3 - r4) / (2 * s)),
            "w": float((r1 + r2 + r3 + r4) / (4 * c)),
            "err": float((r1 + r2 - r3 - r4) / (4 * c)),
            "solution_kind": "four_beam",
        }
    B = np.array([vecs[b - 1] for b in beams_used], dtype=float)
    sol, *_ = np.linalg.lstsq(B, r, rcond=None)
    return {
        "u": float(sol[0]),
        "v": float(sol[1]),
        "w": float(sol[2]),
        "err": None,
        "solution_kind": "three_beam",
    }


def rot_x(deg: float) -> np.ndarray:
    """Pitch: rotation about the instrument x (starboard) axis."""
    a = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def rot_y(deg: float) -> np.ndarray:
    """Roll: rotation about the instrument y (forward) axis."""
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    )


def instrument_to_ship(vec, pitch_deg: float, roll_deg: float, order: str = "pitch_then_roll"):
    """Rotate an instrument-frame vector into the ship frame.

    The rotation order is explicit and versioned: swapping it changes the
    result and therefore changes the derivation fingerprint.
    """
    v = np.asarray(vec, dtype=float)
    if order == "pitch_then_roll":
        m = rot_y(roll_deg) @ rot_x(pitch_deg)
    elif order == "roll_then_pitch":
        m = rot_x(pitch_deg) @ rot_y(roll_deg)
    else:
        raise ValueError(f"unknown transform order: {order}")
    return m @ v


def ship_to_earth(vec, heading_deg: float):
    """Rotate a ship-frame vector into the earth frame.

    heading_deg is clockwise from north; ship x = starboard, y = forward.
    """
    h = math.radians(heading_deg)
    u, v, w = (float(x) for x in vec)
    east = u * math.cos(h) + v * math.sin(h)
    north = -u * math.sin(h) + v * math.cos(h)
    return np.array([east, north, w])


def unwrap_headings(headings_deg) -> list[float]:
    """Unwrap a heading series so 359.5 -> 0.5 becomes 359.5 -> 360.5."""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(list(headings_deg), dtype=float)))).tolist()


class AttitudeSeries:
    """Heading/pitch/roll samples with gap-aware interpolation.

    Interpolation is linear for pitch/roll and circular (on unwrapped
    headings) for heading. A bracketing interval wider than the gap
    threshold is a gap: interpolation across it is refused.
    """

    def __init__(self, samples, gap_threshold_s: float = GAP_THRESHOLD_S):
        self.samples = sorted((dict(s) for s in samples), key=lambda s: s["t"])
        unwrapped = unwrap_headings([s["heading_deg"] for s in self.samples]) if self.samples else []
        for s, h in zip(self.samples, unwrapped):
            s["heading_unwrapped"] = h
        self.gap_threshold = gap_threshold_s
        self.gaps = [
            (a["t"], b["t"])
            for a, b in zip(self.samples, self.samples[1:])
            if b["t"] - a["t"] > gap_threshold_s
        ]

    def at(self, t: float) -> dict:
        S = self.samples
        if not S:
            return {"status": "gap", "diagnostic": "no attitude samples", "sources": []}
        for s in S:
            if s["t"] == t:
                return self._value(s, s, 0.0, "ok", None)
        if t < S[0]["t"] or t > S[-1]["t"]:
            s = S[0] if t < S[0]["t"] else S[-1]
            return {
                "status": "edge",
                "diagnostic": f"t={t} outside attitude range; nearest sample at t={s['t']} used",
                "heading_deg": s["heading_unwrapped"] % 360.0,
                "pitch_deg": s["pitch_deg"],
                "roll_deg": s["roll_deg"],
                "sources": [s.get("rid")],
            }
        a, b = next((x, y) for x, y in zip(S, S[1:]) if x["t"] < t < y["t"])
        if b["t"] - a["t"] > self.gap_threshold:
            return {
                "status": "gap",
                "diagnostic": (
                    f"attitude gap {a['t']}s..{b['t']}s: "
                    "interpolation across gaps is not allowed"
                ),
                "sources": [a.get("rid"), b.get("rid")],
            }
        f = (t - a["t"]) / (b["t"] - a["t"])
        return self._value(a, b, f, "ok", None)

    @staticmethod
    def _value(a, b, f, status, diagnostic):
        heading = (a["heading_unwrapped"] + f * (b["heading_unwrapped"] - a["heading_unwrapped"])) % 360.0
        return {
            "status": status,
            "diagnostic": diagnostic,
            "heading_deg": heading,
            "pitch_deg": a["pitch_deg"] + f * (b["pitch_deg"] - a["pitch_deg"]),
            "roll_deg": a["roll_deg"] + f * (b["roll_deg"] - a["roll_deg"]),
            "sources": [r for r in (a.get("rid"), b.get("rid")) if r],
        }


def derivation_fingerprint(config: dict) -> str:
    """Stable fingerprint of everything a derivation depends on.

    Includes transform order, beam geometry version, magnetic declination
    version, mounting-angle version, QC rule version, attitude policy and
    the raw-data revision.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
