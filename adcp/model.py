"""Domain records (source layer) and derived-node payload helpers.

Source records keep the *original representation* (raw + unit); the derived
layer normalises via adcp.units.  Beam velocity, correlation, amplitude and
bad-beam flags are deliberately separate record streams, as are heading/
pitch/roll, magnetic declination and transducer installation angles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class BeamVelRecord:
    ping: int
    cell: int
    beam: int
    t: float
    raw: Any
    unit: str
    value: Optional[float] = None
    diagnostic: Optional[str] = None
    kind: str = "beam_vel"


@dataclass(frozen=True)
class CorrRecord:
    ping: int
    cell: int
    beam: int
    raw: Any
    unit: str
    value: Optional[float] = None
    diagnostic: Optional[str] = None
    kind: str = "correlation"


@dataclass(frozen=True)
class AmpRecord:
    ping: int
    cell: int
    beam: int
    raw: Any
    unit: str
    value: Optional[float] = None
    diagnostic: Optional[str] = None
    kind: str = "amplitude"


@dataclass(frozen=True)
class BadFlagRecord:
    ping: int
    cell: int
    beam: int
    bad: bool
    kind: str = "bad_flag"


@dataclass(frozen=True)
class AttitudeRecord:
    rec_id: str
    t: float
    heading_raw: Any
    heading_unit: str
    heading_value: Optional[float]   # radians, normalised
    pitch_raw: Any
    pitch_unit: str
    pitch_value: Optional[float]
    roll_raw: Any
    roll_unit: str
    roll_value: Optional[float]
    diagnostic: Optional[str] = None
    kind: str = "attitude"


@dataclass(frozen=True)
class DeclinationRecord:
    decl_id: str
    value_deg: float
    raw: Any
    unit: str
    source: str
    valid_from: str
    kind: str = "declination"


@dataclass(frozen=True)
class ConfigRecord:
    config_id: str
    geometry_id: str
    beam_angle_deg: float
    heading_alignment_deg: float
    pitch_mount_deg: float
    roll_mount_deg: float
    transform_order: str
    blanking_m: float
    cell_size_m: float
    n_cells: int
    corr_floor: float
    corr_marginal: float
    amp_floor: float
    amp_marginal: float
    max_attitude_gap_s: float
    bottom_guard_m: float
    kind: str = "config"


@dataclass(frozen=True)
class BottomTrackRecord:
    ping: int
    t: float
    beams_raw: tuple[Any, Any, Any, Any]
    unit: str
    values: tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
    diagnostics: tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
    bad_flags: tuple[bool, bool, bool, bool]
    correlations: tuple[float, float, float, float]
    amplitudes: tuple[float, float, float, float]
    kind: str = "bottom_track"


@dataclass(frozen=True)
class PingMeta:
    ping: int
    t: float
    bottom_m: Optional[float]
    kind: str = "ping_meta"


@dataclass
class CaseBundle:
    case_id: str
    title: str
    description: str
    config: ConfigRecord
    declination: DeclinationRecord
    ping_meta: list[PingMeta]
    attitude: list[AttitudeRecord]
    beam_vel: list[BeamVelRecord]
    correlation: list[CorrRecord]
    amplitude: list[AmpRecord]
    bad_flags: list[BadFlagRecord]
    bottom_track: list[BottomTrackRecord]

    def source_counts(self) -> dict:
        return {
            "beam_vel": len(self.beam_vel),
            "correlation": len(self.correlation),
            "amplitude": len(self.amplitude),
            "bad_flags": len(self.bad_flags),
            "attitude": len(self.attitude),
            "declination": 1,
            "config": 1,
            "bottom_track": len(self.bottom_track),
            "ping_meta": len(self.ping_meta),
        }


def record_to_dict(rec: Any) -> dict:
    return asdict(rec)


def record_from_dict(table: str, data: dict) -> Any:
    mapping = {
        "beam_vel": BeamVelRecord,
        "correlation": CorrRecord,
        "amplitude": AmpRecord,
        "bad_flags": BadFlagRecord,
        "attitude": AttitudeRecord,
        "declination": DeclinationRecord,
        "config": ConfigRecord,
        "bottom_track": BottomTrackRecord,
        "ping_meta": PingMeta,
    }
    cls = mapping[table]
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    payload = {k: v for k, v in data.items() if k in fields}
    return cls(**payload)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=_json_default)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"cannot serialise {type(obj)!r}")
