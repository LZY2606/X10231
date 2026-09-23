"""Beam geometry and instrument -> ship -> earth coordinate transforms.

Frames (right handed where marked):
  instrument_xyz: x=forward, y=starboard, z=down  (right handed)
  ship_frd:       instrument frame after mount-angle rotation (right handed)
  ship_fsu:       forward, starboard, up (display of vertical component)
  earth_enu:      east, north, up (right handed)

Sign convention documented in fixtures/cases/stream_desk_v1/MANIFEST.json:
positive beam radial velocity means water moves *toward* the transducer face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TransducerGeometry:
    """Janus geometry. Beam 1/2 are the fore pair, 3/4 the aft pair.

    ``azimuths_deg`` are instrument-frame horizontal azimuths clockwise from
    the instrument x (forward) axis, at zero heading: fore pair look forward.
    """

    beam_angle_deg: float
    azimuths_deg: tuple[float, float, float, float] = (45.0, 315.0, 225.0, 135.0)

    @property
    def beam_directions(self) -> np.ndarray:
        """Return 4x3 unit vectors pointing *along* each beam, downward."""
        theta = math.radians(self.beam_angle_deg)
        rows = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            rows.append(
                [
                    math.sin(theta) * math.cos(a),
                    math.sin(theta) * math.sin(a),
                    math.cos(theta),
                ]
            )
        return np.asarray(rows, dtype=float)

    def beam_matrix(self, mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        """Rows are beam look directions for selected (valid) beams.

        Beam velocity positive *toward* the face equals minus the along-look
        water velocity, so the measured beam vector b = -G v_xyz.
        """
        dirs = self.beam_directions
        if mask is None:
            return -dirs
        selected = [row for row, keep in zip(dirs, mask) if keep]
        return -np.asarray(selected, dtype=float)


def solve_beams(
    geometry: TransducerGeometry,
    beam_velocity_ms: Sequence[Optional[float]],
    valid: Sequence[bool],
) -> dict:
    """Solve beam velocities in the instrument frame.

    Four valid beams give the over-determined least-squares solution plus an
    explicit error velocity (the beam-space component the 3-axis model cannot
    explain). Exactly three valid beams give one exact 3-beam solution.

    The two kinds are returned under different ``kind`` values and never share
    a slot. With fewer than three beams no vector is fabricated.
    """
    idx = [i for i, ok in enumerate(valid) if ok and beam_velocity_ms[i] is not None]
    n = len(idx)
    if n < 3:
        return {
            "kind": "insufficient_beams",
            "valid_beams": idx,
            "vector_ms": None,
            "error_velocity_ms": None,
            "condition": None,
            "diagnostic": f"only_{n}_valid_beams",
        }
    mask = [i in idx for i in range(4)]
    A = geometry.beam_matrix(mask)
    b = np.asarray([beam_velocity_ms[i] for i in idx], dtype=float)
    if n == 4:
        # Exact rectangular Janus solve: least squares with a documented
        # pseudo-inverse, plus error velocity along the null/redundant axis.
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        residual = b - A @ sol
        # Error velocity: component of residual along [1,-1,1,-1]-type null
        # direction of the ideal 3-axis Janus model.
        null_axis = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
        error_v = float(np.dot(null_axis, b))
        return {
            "kind": "four_beam",
            "valid_beams": idx,
            "vector_ms": [float(x) for x in sol],
            "residual_ms": [float(x) for x in residual],
            "error_velocity_ms": error_v,
            "condition": float(np.linalg.cond(A)),
            "diagnostic": None,
        }
    sol = np.linalg.solve(A, b)
    return {
        "kind": "three_beam",
        "valid_beams": idx,
        "omitted_beams": [i for i in range(4) if i not in idx],
        "vector_ms": [float(x) for x in sol],
        "error_velocity_ms": None,
        "condition": float(np.linalg.cond(A)),
        "diagnostic": None,
    }


def three_beam_candidates(
    geometry: TransducerGeometry,
    beam_velocity_ms: Sequence[Optional[float]],
    admissible: Sequence[bool],
) -> list[dict]:
    """Enumerate every admissible 3-of-4 sub-solution.

    Used when four beams are present but none can be declared the unique
    failure: all legal candidates are retained instead of collapsing to one.
    """
    candidates = []
    for drop in range(4):
        valid = [(i != drop) and admissible[i] for i in range(4)]
        if sum(valid) != 3:
            continue
        sol = solve_beams(geometry, beam_velocity_ms, valid)
        if sol["vector_ms"] is None:
            continue
        candidates.append(
            {
                "dropped_beam": drop,
                "valid_beams": sol["valid_beams"],
                "vector_ms": sol["vector_ms"],
                "condition": sol["condition"],
            }
        )
    return candidates


def mount_rotation_matrix(
    alpha_deg: float, beta_deg: float, gamma_deg: float
) -> np.ndarray:
    """Instrument -> ship (FRD) mount rotation Rz(gamma) Ry(beta) Rx(alpha).

    alpha: roll offset about instrument x, beta: pitch offset about y,
    gamma: yaw offset about z (degrees, right hand positive).
    """
    a, b, g = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return rz @ ry @ rx


def attitude_matrix_frd_to_enu(
    heading_true_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    """Build the FRD -> ENU rotation directly from physical basis vectors.

    Conventions (documented in the case manifest):
      heading 0 -> ship x (forward) points north; positive clockwise.
      positive pitch -> bow up; positive roll -> starboard down.

    Derived by rotating basis vectors so the order of application is explicit
    and testable, instead of silently composing matrices.
    """
    h = math.radians(heading_true_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    # Heading: forward/starboard in the horizontal EN plane.
    east, north, up = np.eye(3)
    f0 = math.sin(h) * east + math.cos(h) * north
    s0 = math.cos(h) * east - math.sin(h) * north
    # Pitch about the starboard axis (bow up lifts forward toward up).
    f1 = math.cos(p) * f0 - math.sin(p) * up
    d1 = math.sin(p) * f0 + math.cos(p) * up
    # Roll about the pitched forward axis (starboard down -> toward down).
    s2 = math.cos(r) * s0 + math.sin(r) * d1
    d2 = -math.sin(r) * s0 + math.cos(r) * d1
    # Rows are the ENU components of the FRD basis axes f, s, d.
    return np.vstack([f1, s2, d2])


def frd_to_fsu(vector_frd: Sequence[float]) -> list[float]:
    """Forward/starboard/down -> forward/starboard/up (vertical sign flip)."""
    x, y, z = vector_frd
    return [x, y, -z]


def apply_transform_chain(
    beam_velocity_ms: Sequence[Optional[float]],
    valid: Sequence[bool],
    geometry: TransducerGeometry,
    mount_angles_deg: tuple[float, float, float],
    heading_true_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "v1_separated",
) -> dict:
    """Run the full beam -> instrument -> ship -> earth chain.

    ``order`` selects the rule:
      v1_separated  - mount rotation applied in FRD, then attitude to ENU.
      v2_fused_yaw  - WRONG alternative that fuses mount yaw into heading and
                      skips the mount pitch/roll step. Kept as an explicit
                      branch so reviewers can see what ordering changes.
    Every step emits its own matrix for the review UI.
    """
    beam_sol = solve_beams(geometry, beam_velocity_ms, valid)
    steps: dict[str, object] = {"beam_solution": beam_sol}
    if beam_sol["vector_ms"] is None:
        steps["ship_ms"] = None
        steps["earth_ms"] = None
        return steps
    v_inst = np.asarray(beam_sol["vector_ms"], dtype=float)
    rm = mount_rotation_matrix(*mount_angles_deg)
    if order == "v1_separated":
        v_ship_frd = rm @ v_inst
        heading_for_attitude = heading_true_deg
    elif order == "v2_fused_yaw":
        # Deliberately incorrect: fold the yaw offset into heading and ignore
        # the mount pitch/roll rotation matrix entirely.
        v_ship_frd = v_inst
        heading_for_attitude = heading_true_deg + mount_angles_deg[2]
    else:
        raise ValueError(f"unknown transform order: {order}")
    ra = attitude_matrix_frd_to_enu(
        heading_for_attitude, pitch_deg, roll_deg
    )
    v_earth = ra @ v_ship_frd
    steps.update(
        {
            "instrument_ms": [float(x) for x in v_inst],
            "mount_matrix": [[float(x) for x in row] for row in rm],
            "ship_ms": [float(x) for x in v_ship_frd],
            "ship_fsu_ms": frd_to_fsu(v_ship_frd),
            "attitude_matrix": [[float(x) for x in row] for row in ra],
            "earth_ms": [float(x) for x in v_earth],
            "heading_used_true_deg": heading_for_attitude,
        }
    )
    return steps
