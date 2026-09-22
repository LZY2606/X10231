from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .transforms import beam_matrix


CASE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "adcp_case_pack.json"


def record(logical_id: str, category: str, raw_repr: Any, *, value: Optional[float] = None, unit: Optional[str] = None,
           quantity: Optional[str] = None, ping_id: Optional[str] = None, ping_time_s: Optional[float] = None,
           beam: Optional[int] = None, depth_cell: Optional[int] = None, source: str = "offline-adcp-case") -> Dict[str, Any]:
    return {
        "logical_id": logical_id,
        "category": category,
        "raw_repr": str(raw_repr),
        "value": value,
        "unit": unit,
        "quantity": quantity,
        "ping_id": ping_id,
        "ping_time_s": ping_time_s,
        "beam": beam,
        "depth_cell": depth_cell,
        "source": source,
    }


GEOMETRY = {"beam_azimuth_deg": [45, 135, 225, 315], "beam_angle_deg": 25.0}


def projected_beam_velocities(velocity: List[float]) -> List[float]:
    return [round(float(x), 6) for x in beam_matrix(GEOMETRY) @ np.asarray(velocity)]


WATER_VELOCITY = [0.25, -0.12, -0.04]
VESSEL_VELOCITY = [0.03, 0.02, 0.0]
WATER_BEAMS = projected_beam_velocities(WATER_VELOCITY)
BOTTOM_BEAMS = projected_beam_velocities(VESSEL_VELOCITY)
PINGS = [(f"ping_{i+1:02d}", float(i * 3)) for i in range(6)]
DEPTHS = [1.2, 2.2, 3.2, -1.0]


def build_case_pack() -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []
    sources.append(record("geometry.janus-25deg", "transducer_geometry", json.dumps(GEOMETRY, sort_keys=True), quantity="transducer_geometry"))
    sources.append(record("declination.2026-tokyo-v1", "magnetic_declination", "7.5", value=7.5, unit="degree", quantity="magnetic_declination"))
    sources.append(record("mount.heading.initial", "mount_heading", "0", value=0.0, unit="degree", quantity="mount_heading"))
    sources.append(record("mount.pitch.initial", "mount_pitch", "0.5", value=0.5, unit="degree", quantity="mount_pitch"))
    sources.append(record("mount.roll.initial", "mount_roll", "-0.5", value=-0.5, unit="degree", quantity="mount_roll"))
    sources.append(record("transform.order.canonical", "transform_order", "MOUNT_THEN_HPR", quantity="transform_order"))

    headings = [358.0, 359.0, 0.0, 1.0, 359.0, 2.0]
    pitches = [-0.2, -0.1, 0.0, 0.1, 0.0, 0.2]
    rolls = [0.1, 0.2, -0.1, 0.0, 0.1, -0.2]
    for index, (ping_id, ping_time) in enumerate(PINGS):
        sources.extend(
            [
                record(f"attitude.{ping_id}.heading", "heading", f"{headings[index]:.2f}", value=headings[index], unit="degree",
                       quantity="heading", ping_id=ping_id, ping_time_s=ping_time),
                record(f"attitude.{ping_id}.pitch", "pitch", f"{pitches[index]:.2f}", value=pitches[index], unit="degree",
                       quantity="pitch", ping_id=ping_id, ping_time_s=ping_time),
                record(f"attitude.{ping_id}.roll", "roll", f"{rolls[index]:.2f}", value=rolls[index], unit="degree",
                       quantity="roll", ping_id=ping_id, ping_time_s=ping_time),
            ]
        )

    for cell_index, depth in enumerate(DEPTHS):
        raw_depth = f"{depth:.1f}"
        sources.append(
            record(
                f"config.depth-cell.{cell_index}",
                "depth_cell",
                raw_depth,
                value=depth,
                unit="m",
                quantity="depth_cell_depth",
                depth_cell=cell_index,
            )
        )

    for ping_id, ping_time in PINGS:
        for beam in range(1, 5):
            water_velocity = WATER_BEAMS[beam - 1]
            bottom_velocity = BOTTOM_BEAMS[beam - 1]
            sources.extend(
                [
                    record(f"water.{ping_id}.cell.0.beam.{beam}.velocity", "water_velocity", f"{water_velocity:.6f}",
                           value=water_velocity, unit="m/s", quantity="beam_velocity", ping_id=ping_id,
                           ping_time_s=ping_time, beam=beam, depth_cell=0),
                    record(f"water.{ping_id}.cell.1.beam.{beam}.velocity", "water_velocity", f"{water_velocity + 0.01:.6f}",
                           value=water_velocity + 0.01, unit="m/s", quantity="beam_velocity", ping_id=ping_id,
                           ping_time_s=ping_time, beam=beam, depth_cell=1),
                    record(f"water.{ping_id}.cell.2.beam.{beam}.velocity", "water_velocity", f"{water_velocity - 0.005:.6f}",
                           value=water_velocity - 0.005, unit="m/s", quantity="beam_velocity", ping_id=ping_id,
                           ping_time_s=ping_time, beam=beam, depth_cell=2),
                    record(f"water.{ping_id}.cell.3.beam.{beam}.velocity", "water_velocity", f"{water_velocity + 0.02:.6f}",
                           value=water_velocity + 0.02, unit="m/s", quantity="beam_velocity", ping_id=ping_id,
                           ping_time_s=ping_time, beam=beam, depth_cell=3),
                    record(f"water.{ping_id}.beam.{beam}.correlation", "water_correlation", "92", value=92, unit="count",
                           quantity="beam_correlation", ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                    record(f"water.{ping_id}.beam.{beam}.amplitude", "water_amplitude", "68", value=68, unit="dB",
                           quantity="beam_amplitude", ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                    record(f"water.{ping_id}.beam.{beam}.bad-flag", "water_bad_flag", "G", quantity="bad_beam_flag",
                           ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                    record(f"bottom.{ping_id}.beam.{beam}.velocity", "bottom_velocity", f"{bottom_velocity:.6f}",
                           value=bottom_velocity, unit="m/s", quantity="bottom_track_beam_velocity", ping_id=ping_id,
                           ping_time_s=ping_time, beam=beam),
                    record(f"bottom.{ping_id}.beam.{beam}.correlation", "bottom_correlation", "96", value=96, unit="count",
                           quantity="bottom_track_correlation", ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                    record(f"bottom.{ping_id}.beam.{beam}.amplitude", "bottom_amplitude", "72", value=72, unit="dB",
                           quantity="bottom_track_amplitude", ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                    record(f"bottom.{ping_id}.beam.{beam}.bad-flag", "bottom_bad_flag", "G", quantity="bottom_track_bad_beam_flag",
                           ping_id=ping_id, ping_time_s=ping_time, beam=beam),
                ]
            )
