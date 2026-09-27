"""Core ADCP beam geometry + coordinate transforms.

Convention (documented and fingerprinted, not assumed):
- Beam 1 points forward (x+), beam 3 aft (x-), beam 2 starboard (y+),
  beam 4 port (y-). All beams slant downward at beam angle theta.
- Beam radial velocity is positive for scatterer motion *away* from the
  transducer along the beam (standard RD/Teledyne sign convention).
- Frames are right handed: x forward, y starboard(right), z up.
- Instrument velocity v_i satisfies  b_vel = -B v_i  with rows = beam unit
  vectors; hence v_i = -pinv(B) b_vel.
- instrument -> ship uses heading mount offset alpha about z (CCW +).
- ship -> earth uses the RD order roll(R) then pitch(P) then heading(H):
      R_ship2earth = Rz(Ht) @ Ry(-P) @ Rx(R)
  with true heading Ht = heading + magnetic_declination.
The exact transform order string is part of every derivation fingerprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

BEAM_AZIMUTH = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}  # deg, CCW from forward
TRANSFORM_ORDER = "RDI:Rx(roll)->Ry(-pitch)->Rz(true_heading)"
BEAM_GEOMETRY_ID = "conv-beam1fwd-beam2star-pos-away"

# Quality tiers
Q_GOOD = "good"
Q_SUSPECT = "suspect"
Q_BAD = "bad"
Q_UNKNOWN = "unknown"

CORR_GOOD = 100.0     # counts (0..255)
CORR_SUSPECT = 64.0
AMP_GOOD = 80.0       # dB counts
AMP_SUSPECT = 40.0


def beam_unit_vector(beam: int, beam_angle_deg: float) -> np.ndarray:
    az = math.radians(BEAM_AZIMUTH[beam])
    th = math.radians(beam_angle_deg)
    return np.array([math.sin(th) * math.cos(az),
                     math.sin(th) * math.sin(az),
                     -math.cos(th)])


def beam_matrix(beam_angle_deg: float, beams=(1, 2, 3, 4)) -> np.ndarray:
    return np.vstack([beam_unit_vector(b, beam_angle_deg) for b in beams])


# ---------------------------------------------------------------- quality

def beam_tier(correlation: float | None, amplitude: float | None,
              bad_flag: int | None) -> str:
    if bad_flag:
        return Q_BAD
    if correlation is None or amplitude is None:
        return Q_UNKNOWN
    if correlation < CORR_SUSPECT or amplitude < AMP_SUSPECT:
        return Q_BAD
    if correlation < CORR_GOOD or amplitude < AMP_GOOD:
        return Q_SUSPECT
    return Q_GOOD


# ---------------------------------------------------------------- rotations

def rot_x(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_y(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rot_z(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap360(a: float) -> float:
    return a % 360.0


def instrument_to_ship(vi: np.ndarray, mount_heading_deg: float) -> np.ndarray:
    return rot_z(mount_heading_deg) @ vi


def ship_to_earth_rdi(vs: np.ndarray, heading_true: float,
                      pitch_deg: float, roll_deg: float) -> np.ndarray:
    return rot_z(heading_true) @ rot_y(-pitch_deg) @ rot_x(roll_deg) @ vs


def ship_to_earth_alt_order(vs: np.ndarray, heading_true: float,
                            pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Deliberately different (wrong) order H,P,R applied as Rz@Rx@Ry.

    Kept only as a counter-example branch; never silently used.
    """
    return rot_z(heading_true) @ rot_x(roll_deg) @ rot_y(-pitch_deg) @ vs


# ---------------------------------------------------------------- beam solving

@dataclass
class BeamSolution:
    method: str                    # "4beam" | "3beam:<missing>"
    beams: tuple[int, ...]
    velocity: np.ndarray           # instrument frame (x,y,z) m/s
    error_velocity: float | None = None
    residual: float | None = None
    diagnostics: tuple[str, ...] = ()


def solve_beams(radial: dict[int, float], beam_angle_deg: float,
                tiers: dict[int, str]) -> list[BeamSolution]:
    """Enumerate every legal solution and keep them separate.

    4-beam solution exists only when all four beams are good.
    Otherwise every good/suspect 3-beam subset that yields a solution is
    returned; multiple subsets => multiple candidates (ambiguity retained).
    """
    usable = [b for b in (1, 2, 3, 4)
              if b in radial and tiers.get(b) in (Q_GOOD, Q_SUSPECT)
              and radial[b] is not None and not math.isnan(radial[b])]
    solutions: list[BeamSolution] = []
    if len(usable) == 4:
        B = beam_matrix(beam_angle_deg)
        r = np.array([radial[b] for b in (1, 2, 3, 4)], dtype=float)
        vi = -np.linalg.pinv(B) @ r
        # RD error velocity: difference of the two vertical estimates.
        th = math.radians(beam_angle_deg)
        vert_13 = -(radial[1] - radial[3]) / (2.0 * math.cos(th))
        vert_24 = -(radial[2] - radial[4]) / (2.0 * math.cos(th))
        err = vert_13 - vert_24
        solutions.append(BeamSolution("4beam", (1, 2, 3, 4), vi,
                                      error_velocity=float(err), residual=0.0))
    elif len(usable) >= 3:
        from itertools import combinations
        for subset in combinations(sorted(usable), 3):
            B = beam_matrix(beam_angle_deg, beams=subset)
            r = np.array([radial[b] for b in subset], dtype=float)
            try:
                vi, resid, rank, _ = np.linalg.lstsq(-B, r, rcond=None)
                if rank < 3:
                    continue
                missing = tuple(b for b in (1, 2, 3, 4) if b not in subset)
                diag: tuple[str, ...] = ()
                if any(tiers.get(b) == Q_SUSPECT for b in subset):
                    diag = ("suspect_beam_in_3beam_solution",)
                solutions.append(BeamSolution(
                    f"3beam:missing{missing[0]}", tuple(subset), vi,
                    error_velocity=None,
                    residual=float(resid[0]) if len(resid) else 0.0,
                    diagnostics=diag))
            except np.linalg.LinAlgError:
                continue
    return solutions
