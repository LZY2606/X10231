from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .transforms import beam_geometry


def q(value: Any, unit: str, datum: str | None = None) -> dict[str, Any]:
    return {"value": value, "unit": unit, "datum": datum, "raw": str(value)}


def make_baseline_case() -> dict[str, Any]:
    beam_angle = 20.0
    azimuths = [45.0, 135.0, 225.0, 315.0]
    matrix = beam_geometry(beam_angle, azimuths)
    samples = [
        {"time_s": q(0.0, "s", "case_elapsed"), "heading_deg": q(358.0, "deg", "magnetic_north"), "pitch_deg": q(-1.0, "deg", "vessel_frame"), "roll_deg": q(0.5, "deg", "vessel_frame")},
        {"time_s": q(2.0, "s", "case_elapsed"), "heading_deg": q(2.0, "deg", "magnetic_north"), "pitch_deg": q(-0.8, "deg", "vessel_frame"), "roll_deg": q(0.7, "deg", "vessel_frame")},
        {"time_s": q(4.0, "s", "case_elapsed"), "heading_deg": q(5.0, "deg", "magnetic_north"), "pitch_deg": q(-0.6, "deg", "vessel_frame"), "roll_deg": q(0.4, "deg", "vessel_frame")},
        {"time_s": q(6.0, "s", "case_elapsed"), "heading_deg": q(7.0, "deg", "magnetic_north"), "pitch_deg": q(-0.4, "deg", "vessel_frame"), "roll_deg": q(0.2, "deg", "vessel_frame")},
        {"time_s": q(11.0, "s", "case_elapsed"), "heading_deg": q(10.0, "deg", "magnetic_north"), "pitch_deg": q(0.0, "deg", "vessel_frame"), "roll_deg": q(0.0, "deg", "vessel_frame")},
        {"time_s": q(13.0, "s", "case_elapsed"), "heading_deg": q(14.0, "deg", "magnetic_north"), "pitch_deg": q(0.2, "deg", "vessel_frame"), "roll_deg": q(-0.2, "deg", "vessel_frame")},
        {"time_s": q(15.0, "s", "case_elapsed"), "heading_deg": q(18.0, "deg", "magnetic_north"), "pitch_deg": q(0.3, "deg", "vessel_frame"), "roll_deg": q(-0.4, "deg", "vessel_frame")},
    ]
    pings = []
    for ping_number, time_s in enumerate([1, 3, 5, 7, 12, 14, 15], start=1):
        instrument_velocity = np.array([0.42 + 0.01 * ping_number, -0.12 + 0.01 * ping_number, -0.03 - 0.002 * ping_number])
        radial = matrix @ instrument_velocity
        flags = {2: {2}, 5: {3, 4}}.get(ping_number, set())
        beams = []
        for beam in range(1, 5):
            amplitude = 72 - beam * 3 + ping_number
            correlation = 36 if (ping_number == 5 and beam in (3, 4)) else 82 - beam
            beams.append(
                {
                    "beam": beam,
                    "radial_velocity": q(round(float(radial[beam - 1]), 6), "m/s", "positive_away_transducer"),
                    "correlation": q(correlation, "percent", "manufacturer_quality_scale"),
                    "amplitude": q(amplitude, "dB", "ADCP_backscatter_counts"),
                    "bad_beam_flag": beam in flags,
                    "flag_source": "offline_case_quality_layer",
                }
            )
        cells = [
            {"cell": 1, "depth": q(5.0 + ping_number, "m", "transducer_vertical"), "bin_size": q(1.0, "m", "vertical")},
            {"cell": 2, "depth": q(6.0 + ping_number, "m", "transducer_vertical"), "bin_size": q(1.0, "m", "vertical")},
            {"cell": 3,
             "depth": {"value": "not-a-depth", "unit": "m", "datum": "transducer_vertical", "raw": "BAD_BLANKING_OFFSET"} if ping_number == 7 else q(7.0 + ping_number, "m", "transducer_vertical"),
             "bin_size": q(1.0, "m", "vertical")},
        ]
        pings.append(
            {
                "id": f"ping-{ping_number:02d}",
                "kind": "adcp_ping",
                "time_s": q(float(time_s), "s", "case_elapsed"),
                "ensemble": ping_number,
                "beams": beams,
                "cells": cells,
                "raw_representation": {"file": "offline_adcp_replay.adcp", "record": ping_number, "encoding": "domain_json_fixture"},
            }
        )
        bottom_radial = matrix @ np.array([0.08, -0.02, 0.0])
        bottom_beams = []
        for beam in range(1, 5):
            bottom_beams.append(
                {
                    "beam": beam,
                    "radial_velocity": q(round(float(bottom_radial[beam - 1]), 6), "m/s", "positive_away_transducer"),
                    "range": q(18.5 + 0.1 * beam, "m", "acoustic_slant_range"),
                    "correlation": q(35 if ping_number == 5 and beam in (3, 4) else 85, "percent", "manufacturer_quality_scale"),
                    "amplitude": q(86 - beam, "dB", "ADCP_bottom_backscatter"),
                    "bad_beam_flag": (ping_number == 2 and beam == 2),
                }
            )
        pings.append(
            {
                "id": f"ping-{ping_number:02d}:bottom_track",
                "kind": "bottom_track",
                "ping_id": f"ping-{ping_number:02d}",
                "time_s": q(float(time_s), "s", "case_elapsed"),
                "beams": bottom_beams,
                "raw_representation": {"file": "offline_adcp_replay.bt", "record": ping_number, "encoding": "domain_json_fixture"},
            }
        )
    resources = [
        {"id": "transducer-janus-4", "kind": "transducer_config", "beam_angle_deg": q(beam_angle, "deg", "from_transducer_axis"), "beam_azimuth_deg": azimuths, "geometry_version": "janus-4-20deg-v1"},
        {"id": "declination-local", "kind": "magnetic_declination", "declination_deg": q(7.5, "deg", "magnetic_to_true_east_positive"), "model_version": "offline-magnetic-model-2026.0"},
        {"id": "mount-config", "kind": "mounting_config", "yaw_offset_deg": q(2.5, "deg", "installation_alignment"), "pitch_offset_deg": q(0.0, "deg", "installation_alignment"), "roll_offset_deg": q(0.0, "deg", "installation_alignment"), "rotation_order": "ZYX", "revision_note": "baseline installation survey"},
        {"id": "transform-policy", "kind": "transform_policy", "coordinate_transform_order": "ZYX_magnetic_heading", "max_attitude_gap_s": 2.5, "rule_version": "adcp-rules-2026.09", "fingerprint_members": ["coordinate_transform_order", "beam_geometry_version", "magnetic_declination_version"]},
        {"id": "attitude-series", "kind": "attitude_series", "samples": samples},
        *pings,
    ]
    return {
        "case_id": "adcp-beam-attitude-quality-2026",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "domain": "ADCP beam velocity, attitude, bottom track and coordinate transformations",
        "resources": resources,
        "scenario_index": [
            {"ping": "ping-01", "case": "four-beam baseline and heading crosses 360 degrees"},
            {"ping": "ping-02", "case": "one hard failed beam yields one legal three-beam solution"},
            {"ping": "ping-03", "case": "local amplitude/correlation source correction only"},
            {"ping": "ping-04", "case": "attitude interpolation refused across a five-second gap"},
            {"ping": "ping-05", "case": "two low-correlation beams yield four-beam plus two three-beam candidates"},
            {"ping": "ping-06", "case": "mounting-angle revision changes ship and earth downstream"},
            {"ping": "ping-07", "case": "invalid depth cell remains in raw profile evidence and is excluded"},
        ],
    }


def revise_payload(case: dict[str, Any], resource_id: str, mutate) -> dict[str, Any]:
    revised = copy.deepcopy(case)
    for resource in revised["resources"]:
        if resource["id"] == resource_id:
            mutate(resource)
            return revised
    raise KeyError(resource_id)
