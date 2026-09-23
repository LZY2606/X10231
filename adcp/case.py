"""In-memory model of an imported ADCP case.

Every measured quantity stays a :class:`adcp.units.Raw` (value + original
unit) until the derivation layer normalizes it. The eight separately
recorded source categories are distinct fields/records here and distinct
versioned rows in storage:

beam velocities | correlation | amplitude | bad-beam flags |
heading/pitch/roll | magnetic declination | transducer install angles |
transducer (beam geometry) configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .units import Raw


def _raw(obj: Any) -> Raw:
    return Raw.from_dict(obj)


def _raw_grid(obj: Any) -> list[list[Raw]]:
    return [[_raw(x) for x in row] for row in obj]


@dataclass
class AttitudeRecord:
    time_s: float
    heading: Raw
    pitch: Raw
    roll: Raw
    flag: Raw

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "AttitudeRecord":
        return AttitudeRecord(float(obj["time_s"]), _raw(obj["heading"]),
                              _raw(obj["pitch"]), _raw(obj["roll"]),
                              _raw(obj.get("flag", {"value": 0, "unit": "flag"})))


@dataclass
class PingRecord:
    ping_index: int
    time_s: float
    velocity: list[list[Raw]]       # [cell][beam]
    correlation: list[list[Raw]]
    amplitude: list[list[Raw]]
    bad_flag: list[list[Raw]]
    cell_depths: list[Raw]          # [cell]

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "PingRecord":
        return PingRecord(
            int(obj["ping_index"]), float(obj["time_s"]),
            _raw_grid(obj["velocity"]), _raw_grid(obj["correlation"]),
            _raw_grid(obj["amplitude"]), _raw_grid(obj["bad_flag"]),
            [_raw(x) for x in obj["cell_depths"]],
        )


@dataclass
class BottomTrackRecord:
    time_s: float
    velocity: list[Raw]             # [beam]
    ranges: list[Raw]               # [beam]
    correlation: list[Raw]
    bad_flag: list[Raw]

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "BottomTrackRecord":
        return BottomTrackRecord(
            float(obj["time_s"]), [_raw(x) for x in obj["velocity"]],
            [_raw(x) for x in obj["ranges"]], [_raw(x) for x in obj["correlation"]],
            [_raw(x) for x in obj["bad_flag"]],
        )


@dataclass
class CaseBundle:
    case_id: str
    label: str
    transducer: dict[str, Any]              # beam geometry + bin geometry, raw
    install: dict[str, Any]                 # install angles (separate source)
    declination: dict[str, Any]             # magnetic declination (separate)
    settings: dict[str, float]
    attitudes: list[AttitudeRecord] = field(default_factory=list)
    pings: list[PingRecord] = field(default_factory=list)
    bottom_tracks: list[BottomTrackRecord] = field(default_factory=list)

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "CaseBundle":
        return CaseBundle(
            case_id=str(obj["case_id"]), label=str(obj.get("label", obj["case_id"])),
            transducer=dict(obj["transducer"]), install=dict(obj["install"]),
            declination=dict(obj["declination"]), settings=dict(obj.get("settings", {})),
            attitudes=[AttitudeRecord.from_dict(x) for x in obj.get("attitudes", [])],
            pings=[PingRecord.from_dict(x) for x in obj.get("pings", [])],
            bottom_tracks=[BottomTrackRecord.from_dict(x)
                           for x in obj.get("bottom_tracks", [])],
        )

    def n_cells(self) -> int:
        return len(self.pings[0].cell_depths) if self.pings else 0
