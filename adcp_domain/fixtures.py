from __future__ import annotations

import copy

from .fingerprint import stable_json
from .geometry import STANDARD_ORDER


PING_TIMES = {
    "P001": 10.0,
    "P002": 20.0,
    "P003": 30.0,
    "P004": 40.0,
    "P005": 50.0,
    "P006": 60.0,
}

# Beam radial velocities are stored exactly as imported, in mm/s. The values
# are consistent with a convex, four-beam Janus geometry.
BEAM_MM_S = {
    "P001": (200, 152, 176, 224),
    "P002": (190, 145, 168, 215),
    "P003": (185, 140, 160, 210),
    "P004": (180, 135, 155, 205),
    "P005": (175, 130, 150, 200),
    "P006": (170, 125, 145, 195),
}
BOTTOM_MM_S = {
    "P001": (45, 12, 30, 58),
    "P002": (43, 10, 28, 56),
    "P003": (41, 9, 27, 54),
    "P004": (39, 8, 25, 52),
    "P005": (37, 7, 24, 50),
    "P006": (35, 6, 22, 48),
}


def _beam_records(kind: str, values: dict[str, tuple[int, ...]]) -> list[dict]:
    records = []
    for ping_id in PING_TIMES:
        for beam_number, value in enumerate(values[ping_id], start=1):
            records.append(
                {
                    "id": f"{kind}-{ping_id}-B{beam_number}",
                    "ping_id": ping_id,
                    "beam_number": beam_number,
                    "depth_cell_id": f"CELL-{ping_id}-01",
                    "raw_value": value,
                    "raw_unit": "mm/s",
                }
            )
    return records


def base_casepack() -> dict:
    beam_velocity = _beam_records("BV", BEAM_MM_S)
    bottom_track = _beam_records("BT", BOTTOM_MM_S)
    correlations = []
    amplitudes = []
    bad_beam_flags = []
    for ping_id in PING_TIMES:
        for beam_number in range(1, 5):
            correlation = 90
            amplitude = 100
            status = "ok"
            if ping_id == "P003" and beam_number in (3, 4):
                correlation = 55 + beam_number
                amplitude = 65 + beam_number
                status = "suspect"
            if ping_id == "P004" and beam_number == 2:
                correlation = 12
                amplitude = 18
                status = "bad"
            correlations.append(
                {
                    "id": f"COR-{ping_id}-B{beam_number}",
                    "ping_id": ping_id,
                    "beam_number": beam_number,
                    "depth_cell_id": f"CELL-{ping_id}-01",
                    "raw_value": correlation,
                    "raw_unit": "count",
                }
            )
            amplitudes.append(
                {
                    "id": f"AMP-{ping_id}-B{beam_number}",
                    "ping_id": ping_id,
                    "beam_number": beam_number,
                    "depth_cell_id": f"CELL-{ping_id}-01",
                    "raw_value": amplitude,
                    "raw_unit": "count",
                }
            )
            bad_beam_flags.append(
                {
                    "id": f"FLAG-{ping_id}-B{beam_number}",
                    "ping_id": ping_id,
                    "beam_number": beam_number,
                    "depth_cell_id": f"CELL-{ping_id}-01",
                    "raw_value": status,
                    "raw_unit": "code",
                }
            )
    attitudes = [
        ("A001", 9.0, 10.0, 1.0, -1.0),
        ("A002", 11.0, 12.0, 1.1, -0.8),
        ("A003", 19.0, 359.0, 1.2, -0.5),
        ("A004", 21.0, 1.0, 1.4, -0.3),
        ("A005", 29.0, 20.0, 1.5, 0.0),
        ("A006", 31.0, 22.0, 1.6, 0.2),
        ("A007", 39.0, 30.0, 1.7, 0.4),
        ("A008", 41.0, 32.0, 1.8, 0.6),
        ("A009", 48.0, 40.0, 1.9, 0.8),
        ("A010", 62.0, 55.0, 2.2, 1.1),
    ]
    attitude_records = [
        {
            "id": attitude_id,
            "time_epoch_s": time_value,
            "heading_deg": heading,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "raw_heading_unit": "deg",
            "raw_pitch_unit": "deg",
            "raw_roll_unit": "deg",
        }
        for attitude_id, time_value, heading, pitch, roll in attitudes
    ]
    depth_cells = []
    for ping_id, time_value in PING_TIMES.items():
        depth_value = "BLOCKED" if ping_id == "P006" else 5.0
        depth_cells.append(
            {
                "id": f"CELL-{ping_id}-01",
                "ping_id": ping_id,
                "time_epoch_s": time_value,
                "bin_number": 1,
                "raw_depth": depth_value,
                "raw_depth_unit": "m",
            }
        )
    return {
        "casepack_id": "adcp-beam-transform-quality-hierarchy-v1",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "pings": [
            {"id": ping_id, "time_epoch_s": time_value, "time_unit": "s"}
            for ping_id, time_value in PING_TIMES.items()
        ],
        "beam_velocity": beam_velocity,
        "correlation": correlations,
        "amplitude": amplitudes,
        "bad_beam_flags": bad_beam_flags,
        "attitude": attitude_records,
        "depth_cells": depth_cells,
        "bottom_track": bottom_track,
        "transducer_configuration": {
            "id": "XDC-JANUS-4",
            "geometry_id": "JANUS-4-CONVEX-20DEG-v1",
            "beam_angle_deg": 20.0,
            "beam_azimuths_deg": [45.0, 135.0, 225.0, 315.0],
            "instrument_frame": "x-forward-y-starboard-z-down",
            "ship_frame": "x-forward-y-starboard-z-down",
            "earth_frame": "east-north-up",
            "transform_order": list(STANDARD_ORDER),
        },
        "installation_alignment": [
            {
                "id": "MOUNT-001",
                "raw_installation_angle_z_deg": 1.5,
                "raw_unit": "deg",
                "effective_from_time_epoch_s": 0.0,
            }
        ],
        "magnetic_declination": [
            {
                "id": "DEC-001",
                "raw_angle_deg": 7.2,
                "raw_unit": "deg",
                "datum_version": "WMM2025",
                "effective_from_time_epoch_s": 0.0,
            }
        ],
        "rules": {
            "correlation_threshold": 64,
            "amplitude_threshold": 70,
            "max_attitude_gap_s": 4.0,
        },
        "counterexamples": _counterexamples(),
        "casepack_content_fingerprint": None,
    }


def _counterexamples() -> list[dict]:
    return [
        {
            "id": "CE-SINGLE-BEAM-FAILURE",
            "label": "单 beam 失效",
            "source_kind": "bad_beam_flags",
            "record_id": "FLAG-P001-B2",
            "replacement": {"raw_value": "bad", "raw_unit": "code"},
            "expected_scope": ["P001"],
        },
        {
            "id": "CE-HEADING-WRAP-360",
            "label": "姿态跨越三百六十度",
            "source_kind": "attitude",
            "record_id": "A004",
            "replacement": {"heading_deg": 359.0},
            "expected_scope": ["P002"],
        },
        {
            "id": "CE-TIME-GAP",
            "label": "时间缺口",
            "source_kind": "attitude",
            "record_id": "A009",
            "replacement": {"time_epoch_s": 10.0},
            "expected_scope": ["P005"],
        },
        {
            "id": "CE-THREE-BEAM-AMBIGUITY",
            "label": "三波束多解",
            "source_kind": "bad_beam_flags",
            "record_id": "FLAG-P001-B3",
            "replacement": {"raw_value": "suspect", "raw_unit": "code"},
            "expected_scope": ["P001"],
        },
        {
            "id": "CE-INSTALLATION-REVISION",
            "label": "安装角修订",
            "source_kind": "installation_alignment",
            "record_id": "MOUNT-001",
            "replacement": {"raw_installation_angle_z_deg": 2.0, "raw_unit": "deg"},
            "expected_scope": ["P001", "P002", "P003", "P004"],
        },
        {
            "id": "CE-TRANSFORM-ORDER",
            "label": "坐标变换顺序",
            "source_kind": "transducer_configuration",
            "record_id": "XDC-JANUS-4",
            "replacement": {
                "transform_order": [
                    "beam_to_instrument",
                    "ship_to_earth",
                    "instrument_to_ship",
                ]
            },
            "expected_scope": ["P001", "P002", "P003", "P004"],
        },
        {
            "id": "CE-INVALID-DEPTH-CELL",
            "label": "无效深度 cell",
            "source_kind": "depth_cells",
            "record_id": "CELL-P001-01",
            "replacement": {"raw_depth": "BLOCKED", "raw_depth_unit": "m"},
            "expected_scope": ["P001"],
        },
    ]


def replay_counterexample(casepack: dict, counterexample_id: str) -> dict:
    mutated = copy.deepcopy(casepack)
    counterexample = next(item for item in mutated["counterexamples"] if item["id"] == counterexample_id)
    source_kind = counterexample["source_kind"]
    record_id = counterexample["record_id"]
    if source_kind == "transducer_configuration":
        target = mutated[source_kind]
    else:
        target = next(record for record in mutated[source_kind] if record["id"] == record_id)
    for key, value in counterexample["replacement"].items():
        target[key] = value
    mutated["casepack_content_fingerprint"] = stable_json(mutated)
    return mutated
