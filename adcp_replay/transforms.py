from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .storage import canonical_json, digest, stable_id


BEAMS = (1, 2, 3, 4)
MAX_ATTITUDE_GAP_S = 2.0


def wrap_360(deg: float) -> float:
    return float((deg + 360.0) % 360.0)


def unwrap_degrees(values: Sequence[float]) -> List[float]:
    return [float(x) for x in np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=float))))]


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def as_float(record: Optional[Dict[str, Any]]) -> Tuple[Optional[float], List[Dict[str, str]]]:
    if record is None:
        return None, [{"code": "missing-source", "message": "source record is absent"}]
    if record.get("value") is not None:
        return float(record["value"]), []
    raw = str(record.get("raw_repr", "")).strip()
    if raw.upper() in {"", "NA", "N/A", "NULL", "NONE"}:
        return None, [{"code": "unconvertible-value", "message": "raw value remains at source", "raw": raw}]
    try:
        return float(raw), []
    except ValueError:
        return None, [{"code": "unconvertible-value", "message": "raw value remains at source", "raw": raw}]


def unit_float(record: Optional[Dict[str, Any]], *, si_unit: str) -> Tuple[Optional[float], List[Dict[str, str]]]:
    value, diagnostics = as_float(record)
    if value is None:
        return None, diagnostics
    unit = (record or {}).get("unit")
    if unit == si_unit:
        return value, diagnostics
    conversions = {
        ("degree", "radian"): math.pi / 180.0,
        ("deg", "rad"): math.pi / 180.0,
        ("°", "rad"): math.pi / 180.0,
        ("cm/s", "m/s"): 0.01,
        ("mm/s", "m/s"): 0.001,
        ("m", "m"): 1.0,
        ("count", "count"): 1.0,
        ("dB", "dB"): 1.0,
    }
    factor = conversions.get((unit, si_unit))
    if factor is None:
        diagnostics.append({"code": "unknown-unit", "message": "value retained in raw source", "unit": str(unit)})
        return None, diagnostics
    return value * factor, diagnostics


def parse_geometry(record: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    if not record:
        return {"beam_azimuth_deg": [45, 135, 225, 315], "beam_angle_deg": 25.0}, "geometry-default-v0"
    raw = record.get("raw_repr")
    geometry = json.loads(raw)
    return geometry, stable_id("geom", canonical_json(geometry))


def beam_matrix(geometry: Dict[str, Any]) -> np.ndarray:
    azimuths = geometry["beam_azimuth_deg"]
    theta = math.radians(float(geometry["beam_angle_deg"]))
    rows = []
    for azimuth in azimuths:
        phi = math.radians(float(azimuth))
        rows.append([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), -math.cos(theta)])
    return np.asarray(rows, dtype=float)


def solve_beams(matrix: np.ndarray, beams: Dict[int, float], omitted: Sequence[int] = ()) -> np.ndarray:
    omitted_set = set(omitted)
    indexes = [beam - 1 for beam in BEAMS if beam not in omitted_set and beam in beams]
    answer, *_ = np.linalg.lstsq(matrix[indexes], np.asarray([beams[i + 1] for i in indexes]), rcond=None)
    return answer


def rounded_vector(vector: np.ndarray) -> List[float]:
    return [round(float(x), 9) for x in vector]


@dataclass
class BeamQuality:
    beam: int
    velocity: Optional[float]
    correlation: Optional[float]
    amplitude: Optional[float]
    flag: str
    state: str
    reasons: List[str]


def classify_beam(
    beam: int,
    velocity: Optional[float],
    correlation: Optional[float],
    amplitude: Optional[float],
    flag: Optional[str],
) -> BeamQuality:
    reasons: List[str] = []
    flag = (flag or "G").strip().upper()
    if velocity is None:
        reasons.append("unconvertible-velocity")
    if correlation is None:
        reasons.append("unconvertible-correlation")
    if amplitude is None:
        reasons.append("unconvertible-amplitude")
    if correlation is not None and correlation < 70.0:
        reasons.append("correlation-below-hard-threshold")
    if amplitude is not None and amplitude < 35.0:
        reasons.append("amplitude-below-hard-threshold")
    if flag == "B":
        reasons.append("bad-beam-flag")
    hard = bool(reasons)
    marginal: List[str] = []
    if flag == "M":
        marginal.append("suspect-beam-flag")
    if correlation is not None and 70.0 <= correlation < 80.0:
        marginal.append("correlation-marginal")
    if amplitude is not None and 35.0 <= amplitude < 45.0:
        marginal.append("amplitude-marginal")
    state = "hard-invalid" if hard else "marginal" if marginal else "good"
    return BeamQuality(beam, velocity, correlation, amplitude, flag, state, reasons + marginal)


def beam_candidates(matrix: np.ndarray, qualities: Sequence[BeamQuality]) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, str]]]:
    good_velocity = {q.beam: q.velocity for q in qualities if q.state == "good" and q.velocity is not None}
    hard = [q.beam for q in qualities if q.state == "hard-invalid"]
    marginal = [q.beam for q in qualities if q.state == "marginal"]
    candidates: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, str]] = []
    if len(hard) >= 2:
        return [], "blocked", [{"code": "insufficient-beams", "message": "at least three valid beams are required", "hard": ",".join(map(str, hard))}]
    if len(hard) == 1:
        omitted = hard[0]
        usable = {q.beam: q.velocity for q in qualities if q.beam != omitted and q.velocity is not None}
        if len(usable) != 3:
            return [], "blocked", [{"code": "insufficient-beams", "message": "three-beam solve could not use exactly three beams"}]
        candidates.append(
            {
                "candidate_id": f"c3_omit_b{omitted}",
                "rank": 3,
                "omitted_beams": [omitted],
                "vector": rounded_vector(solve_beams(matrix, usable, [omitted])),
                "assumption": "three-beam exact solution; not collapsed to a four-beam least-squares value",
            }
        )
        return candidates, "selected", diagnostics
    if len(good_velocity) == 4:
        candidates.append(
            {
                "candidate_id": "c4_lsq",
                "rank": 4,
                "omitted_beams": [],
                "vector": rounded_vector(solve_beams(matrix, {q.beam: q.velocity for q in qualities})),
                "assumption": "four-beam least-squares solution",
            }
        )
    for omitted in marginal:
        usable = {q.beam: q.velocity for q in qualities if q.beam != omitted and q.velocity is not None and q.state != "hard-invalid"}
        if len(usable) == 3:
            candidates.append(
                {
                    "candidate_id": f"c3_omit_b{omitted}",
                    "rank": 3,
                    "omitted_beams": [omitted],
                    "vector": rounded_vector(solve_beams(matrix, usable, [omitted])),
                    "assumption": "legal three-beam candidate while suspect evidence is unresolved",
                }
            )
    if marginal:
        return candidates, "ambiguous", [{"code": "ambiguous-beam-selection", "message": "all legal candidates retained", "marginal": ",".join(map(str, marginal))}]
    return candidates, "selected" if candidates else "blocked", diagnostics


def mount_rotation(mount_heading: float, mount_pitch: float, mount_roll: float) -> np.ndarray:
    return rot_z(mount_heading) @ rot_y(mount_pitch) @ rot_x(mount_roll)


def earth_rotation(heading: float, pitch: float, roll: float, declination: float, order: str) -> np.ndarray:
    attitude = rot_z(wrap_360(math.degrees(heading + declination)) * math.pi / 180.0) @ rot_y(pitch) @ rot_x(roll)
    identity = np.eye(3)
    if order in ("MOUNT_THEN_HPR", "mount_then_hpr"):
        return attitude
    if order in ("HPR_BEFORE_MOUNT", "hpr_before_mount"):
        return identity
    raise ValueError(f"unsupported transform order: {order}")


def transform_vector(
    instrument: Sequence[float],
    *,
    heading: float,
    pitch: float,
    roll: float,
    declination: float,
    mount_heading: float,
    mount_pitch: float,
    mount_roll: float,
    order: str,
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    v_instrument = np.asarray(instrument, dtype=float)
    r_mount = mount_rotation(mount_heading, mount_pitch, mount_roll)
    v_ship = r_mount @ v_instrument
    if order in ("MOUNT_THEN_HPR", "mount_then_hpr"):
        r_earth = earth_rotation(heading, pitch, roll, declination, order) @ r_mount
    else:
        attitude = rot_z(wrap_360(math.degrees(heading + declination)) * math.pi / 180.0) @ rot_y(pitch) @ rot_x(roll)
        r_earth = r_mount @ attitude
    v_earth = r_earth @ v_instrument
    return rounded_vector(v_instrument), rounded_vector(v_ship), v_ship, rounded_vector(v_earth)


def transform_fingerprint(
    geometry_id: str,
    declination_record: Optional[Dict[str, Any]],
    mount_records: Dict[str, Optional[Dict[str, Any]]],
    order_record: Optional[Dict[str, Any]],
) -> str:
    return digest(
        [
            geometry_id,
            declination_record.get("logical_id") if declination_record else None,
            declination_record.get("record_id") if declination_record else None,
            {key: (record or {}).get("record_id") for key, record in mount_records.items()},
            order_record.get("logical_id") if order_record else None,
            order_record.get("record_id") if order_record else None,
            (order_record or {}).get("raw_repr"),
        ]
    )
