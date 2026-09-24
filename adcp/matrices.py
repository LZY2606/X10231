"""Beam geometry and attitude rotation matrices.

Conventions (documented so the choices are part of the provenance fingerprint):

* Four-beam Janus geometry. Beams 1..4 project onto the instrument frame with
  azimuths (90, 0, 270, 180) degrees: beam1 +Y, beam2 +X, beam3 -Y, beam4 -X.
* ``b`` is positive away from the transducer.
* Radial velocity measured along a beam is ``b . v`` (receding water => positive).
* ``instrument`` frame: X beam2/starboard, Y beam1/forward, Z up.
* ``earth`` frame: X east, Y north, Z up.
* Attitude rotation convention ``HPR`` is Rz(heading) @ Ry(pitch) @ Rx(roll),
  using a frame that maps the above XYZ instrument axes to ENU. Alternatives
  (e.g. RDI instrument x=beam1) are supported as distinct, fingerprinted
  conventions and never silently substituted.
"""
from __future__ import annotations

import math

import numpy as np

BEAM_AZIMUTH_DEG = (90.0, 0.0, 270.0, 180.0)


def beam_direction(azimuth_deg: float, angle_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    th = math.radians(angle_deg)
    return np.array(
        [
            math.sin(th) * math.cos(az),
            math.sin(th) * math.sin(az),
            math.cos(th),
        ]
    )


def geometry_matrix(beam_angle_deg: float = 20.0) -> np.ndarray:
    """Return the 4x3 matrix mapping instrument velocity to four radials."""
    return np.vstack(
        [beam_direction(az, beam_angle_deg) for az in BEAM_AZIMUTH_DEG]
    )


def rdi_matrix(beam_angle_deg: float = 20.0) -> np.ndarray:
    """Classical RDI 4x3 matrix (x=beam1, y=beam2, z=up)."""
    th = math.radians(beam_angle_deg)
    s, c = math.sin(th), math.cos(th)
    return np.array(
        [
            [s, 0.0, c],
            [0.0, s, c],
            [-s, 0.0, c],
            [0.0, -s, c],
        ]
    )


def _rx(r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(p: float) -> np.ndarray:
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(h: float) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


CONVENTIONS = ("HPR", "H-PR", "PRH", "RDI-HPR")


def attitude_matrix(
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    convention: str = "HPR",
) -> np.ndarray:
    """Rotation from ship frame to earth frame (applies column vectors)."""
    h = math.radians(heading_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    if convention == "HPR":
        return _rz(h) @ _ry(p) @ _rx(r)
    if convention == "H-PR":
        return _rz(h) @ _ry(-p) @ _rx(r)
    if convention == "PRH":
        return _rx(r) @ _ry(p) @ _rz(h)
    if convention == "RDI-HPR":
        return _rz(h) @ _ry(pitch_deg and -p) @ _rx(r)
    raise ValueError(f"unknown attitude convention {convention!r}")


def mounting_matrix(mount_deg: tuple[float, float, float]) -> np.ndarray:
    """Instrument -> ship rotation from mounting angles (heading,pitch,roll)."""
    mh, mp, mr = (math.radians(a) for a in mount_deg)
    return _rz(mh) @ _ry(mp) @ _rx(mr)


def invert_instrument(
    radials: np.ndarray,
    good: np.ndarray,
    beam_angle_deg: float = 20.0,
    geometry: str = "janus",
) -> dict:
    """Solve instrument-frame velocity from good beams.

    Returns the four-beam solution or *all* legal three-beam candidates.
    Candidate ``kind`` is ``four`` or ``three``; four- and three-beam results
    are never collapsed into one value.
    """
    G = (
        rdi_matrix(beam_angle_deg)
        if geometry == "rdi"
        else geometry_matrix(beam_angle_deg)
    )
    good_idx = [i for i in range(4) if bool(good[i])]
    out: dict = {"kind": None, "candidates": [], "error_velocity": None}
    if len(good_idx) == 4:
        v, *_ = np.linalg.lstsq(G, radials, rcond=None)
        resid = radials - G @ v
        # Null-space projection of the four radials: the inconsistency
        # component that no 3D velocity can explain (RDI "error velocity").
        u, s, vh = np.linalg.svd(G, full_matrices=True)
        null = vh[-1]
        scale = 1.0 / (null @ radials and 1.0)  # placeholder, replaced below
        e = float(null @ radials)
        out.update(
            kind="four",
            velocity=[float(x) for x in v],
            residual=[float(x) for x in resid],
            error_velocity=e,
            null_vector=[float(x) for x in null],
            condition=float(s[0] / s[-1]),
        )
        return out
    if len(good_idx) == 3:
        u, s, _ = np.linalg.svd(G[good_idx], full_matrices=False)
        cond = float(s[0] / s[-1])
        cands = []
        for missing in range(4):
            idx = [i for i in range(4) if i != missing]
            if not all(i in good_idx for i in idx):
                continue
            v, *_ = np.linalg.lstsq(G[idx], radials[idx], rcond=None)
            cands.append(
                {
                    "kind": "three",
                    "missing_beam": missing + 1,
                    "velocity": [float(x) for x in v],
                    "residual": [
                        float(x) for x in (radials[good_idx] - G[good_idx] @ v)
                    ],
                    "condition": cond,
                }
            )
        cands.sort(key=lambda c: c["missing_beam"])
        out.update(kind="three", candidates=cands, condition=cond)
        return out
    out["kind"] = "unresolvable"
    out["reason"] = f"only {len(good_idx)} good beams (<3)"
    return out
