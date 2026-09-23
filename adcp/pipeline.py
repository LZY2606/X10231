"""End-to-end recomputation: beam -> instrument -> ship -> earth.

This module is pure: it takes a snapshot of source records (all seven source
kinds kept separate) plus a transducer configuration and returns a dict of
stable-identified value slots. Persistence, versioning and branching live in
:mod:`adcp.store`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .attitude import (
    AttitudeSample,
    instrument_to_ship,
    interpolate_attitude,
    ship_to_earth,
)
from .beam import BeamGeometry, BeamSolution, rdi_janus, solve_beam_velocities
from .fingerprint import derived_fingerprint, make_stable_id
from .quality import QualityThresholds, classify_beams, validate_depth_cell
from .units import Quantity, parse_quantity


def _q(value: Optional[float], unit: str, basis: str = "") -> Dict[str, Any]:
    return Quantity(value, unit, basis).to_dict()


def build_geometry(config: Dict[str, Any]) -> BeamGeometry:
    if config.get("preset") == "rdi-janus":
        return rdi_janus(float(config.get("beam_angle_deg", 20.0)))
    return BeamGeometry(
        beam_angle_deg=float(config["beam_angle_deg"]),
        azimuths_deg=tuple(float(a) for a in config["azimuths_deg"]),
        geometry_version=str(config.get("geometry_version", "custom")),
    )


def _candidate_payload(
    candidate: BeamSolution,
    attitude: Optional[Dict[str, float]],
    declination_deg: Optional[float],
    mounting: Dict[str, float],
    order: str,
    geometry_version: str,
    declination_version: str,
    mounting_version: str,
) -> Dict[str, Any]:
    instrument = np.asarray(candidate.velocity)
    ship = instrument_to_ship(instrument, mounting.get("pitch_deg", 0.0), mounting.get("roll_deg", 0.0))
    earth = None
    earth_status = "not_computed"
    true_heading = None
    if attitude is not None and declination_deg is not None:
        true_heading = (attitude["heading_deg"] + declination_deg) % 360.0
        earth = ship_to_earth(ship, true_heading, attitude["pitch_deg"], attitude["roll_deg"], order)
        earth_status = "ok"
    elif attitude is None:
        earth_status = "attitude_unavailable"
    elif declination_deg is None:
        earth_status = "declination_unavailable"

    inst_fpr = derived_fingerprint(
        "beam_to_instrument",
        {
            "candidate": candidate.candidate_id(),
            "geometry": geometry_version,
            "velocity": candidate.velocity,
        },
        [],
    )
    ship_fpr = derived_fingerprint(
        "instrument_to_ship",
        {
            "mounting_version": mounting_version,
            "mounting_pitch_deg": mounting.get("pitch_deg", 0.0),
            "mounting_roll_deg": mounting.get("roll_deg", 0.0),
            "instrument_fingerprint": inst_fpr,
        },
        [inst_fpr],
    )
    earth_inputs = {
        "transform_order": order,
        "declination_version": declination_version,
        "declination_deg": declination_deg,
        "attitude": attitude,
        "true_heading_deg": true_heading,
        "ship_fingerprint": ship_fpr,
    }
    earth_fpr = derived_fingerprint("ship_to_earth", earth_inputs, [ship_fpr]) if earth is not None else None

    return {
        "candidate_id": candidate.candidate_id(),
        "kind": candidate.kind,
        "used_beams": list(candidate.used_beams),
        "instrument_velocity": {
            "xyz": [_q(float(v), "m/s", "instrument") for v in instrument],
            "fingerprint": inst_fpr,
        },
        "ship_velocity": {
            "xyz": [_q(float(v), "m/s", "ship") for v in ship],
            "fingerprint": ship_fpr,
        },
        "earth_velocity": {
            "xyz": [_q(float(v), "m/s", "earth_enu") for v in earth] if earth is not None else None,
            "status": earth_status,
            "true_heading_deg": true_heading,
            "fingerprint": earth_fpr,
        },
        "error_velocity_mps": candidate.residual,
    }


def _pick_declination(records: Sequence[Dict[str, Any]], time_s: float) -> Optional[Dict[str, Any]]:
    covering = [r for r in records if r["valid_from_s"] <= time_s < r["valid_to_s"]]
    if not covering:
        return None
    return sorted(covering, key=lambda r: r["valid_from_s"])[-1]


def recompute(snapshot: Dict[str, Any], thresholds: Optional[QualityThresholds] = None) -> Dict[str, Any]:
    """Run the full derivation over a source snapshot.

    Expected snapshot keys:
      transducer: {preset|beam_angle_deg/azimuths_deg/geometry_version,
                   min_valid_distance_m, max_attitude_gap_s, transform_order}
      cells: [{cell_index, bin_distance_m(raw)}]
      pings: [{ping_id, time_s, beams:[{velocity(raw), correlation(raw),
                                        amplitude(raw), bad_flag} x4]}]
      attitudes: [{time_s, heading_deg, pitch_deg, roll_deg}]
      declinations: [{declination_version, valid_from_s, valid_to_s,
                      declination_deg(raw)}]
      mountings: [{mounting_version, valid_from_s, valid_to_s,
                   pitch_deg(raw), roll_deg(raw)}]
      bottom_tracks: [{ping_id, range_m(raw 4), velocity...(raw 4)}]
    """
    thresholds = thresholds or QualityThresholds()
    config = snapshot["transducer"]
    geometry = build_geometry(config)
    order = config.get("transform_order", "RDI_HPR")
    max_gap_s = float(config.get("max_attitude_gap_s", 1.0))
    min_valid_distance_m = float(config.get("min_valid_distance_m", 0.0))

    attitudes = [
        AttitudeSample(
            time_s=float(a["time_s"]),
            heading_deg=float(a["heading_deg"]),
            pitch_deg=float(a["pitch_deg"]),
            roll_deg=float(a["roll_deg"]),
        )
        for a in snapshot.get("attitudes", [])
    ]

    # Raw quantities are preserved at source level; parsed values feed derive.
    cells = []
    for cell in snapshot.get("cells", []):
        distance = parse_quantity(cell["bin_distance_m"], "m")
        cells.append(
            {
                "cell_index": int(cell["cell_index"]),
                "bin_distance": distance.to_dict(),
            }
        )

    bottom_by_ping = {bt["ping_id"]: bt for bt in snapshot.get("bottom_tracks", [])}

    pings_out: List[Dict[str, Any]] = []
    for ping in snapshot["pings"]:
        ping_id = ping["ping_id"]
        time_s = float(ping["time_s"])
        beam_meas: List[Dict[str, Any]] = []
        correlations: List[Optional[float]] = []
        amplitudes: List[Optional[float]] = []
        bad_flags: List[bool] = []
        radials: List[Optional[float]] = []
        for index, beam in enumerate(ping["beams"]):
            velocity_q = parse_quantity(beam["velocity"], "m/s", "beam")
            correlation_q = parse_quantity(beam["correlation"], "counts")
            amplitude_q = parse_quantity(beam["amplitude"], "counts")
            bad = bool(beam.get("bad_flag", False))
            beam_meas.append(
                {
                    "beam_index": index,
                    "stable_id": make_stable_id(["beam", ping_id, index]),
                    "velocity": velocity_q.to_dict(),
                    "correlation": correlation_q.to_dict(),
                    "amplitude": amplitude_q.to_dict(),
                    "bad_flag": bad,
                }
            )
            radials.append(velocity_q.value)
            correlations.append(correlation_q.value)
            amplitudes.append(amplitude_q.value)
            bad_flags.append(bad)

        quality = classify_beams(correlations, amplitudes, bad_flags, thresholds)

        attitude_result = interpolate_attitude(attitudes, time_s, max_gap_s)
        attitude_payload = None
        if attitude_result["status"] == "ok":
            attitude_payload = {
                "heading_deg": float(attitude_result["heading_deg"]),
                "pitch_deg": float(attitude_result["pitch_deg"]),
                "roll_deg": float(attitude_result["roll_deg"]),
            }

        declination_record = _pick_declination(snapshot.get("declinations", []), time_s)
        declination_deg = None
        if declination_record is not None:
            parsed = parse_quantity(declination_record["declination_deg"], "deg", "magnetic")
            declination_deg = parsed.value

        mounting_record = _pick_declination(snapshot.get("mountings", []), time_s)
        mounting: Dict[str, float] = {"pitch_deg": 0.0, "roll_deg": 0.0}
        mounting_version = "none"
        if mounting_record is not None:
            pitch_q = parse_quantity(mounting_record["pitch_deg"], "deg")
            roll_q = parse_quantity(mounting_record["roll_deg"], "deg")
            if pitch_q.value is not None:
                mounting["pitch_deg"] = pitch_q.value
            if roll_q.value is not None:
                mounting["roll_deg"] = roll_q.value
            mounting_version = str(mounting_record["mounting_version"])

        bottom = bottom_by_ping.get(ping_id)
        bottom_distance = None
        if bottom is not None:
            ranges = [parse_quantity(r, "m").value for r in bottom.get("range_m", [])]
            valid_ranges = [r for r in ranges if r is not None]
            if valid_ranges:
                bottom_distance = min(valid_ranges)

        cell_results: List[Dict[str, Any]] = []
        for cell in cells:
            cell_index = cell["cell_index"]
            validity = validate_depth_cell(
                cell_index, cell["bin_distance"]["value"], bottom_distance, min_valid_distance_m
            )
            cell_result: Dict[str, Any] = {
                "cell_index": cell_index,
                "stable_id": make_stable_id(["cell", ping_id, cell_index]),
                "bin_distance": cell["bin_distance"],
                "validity": validity,
            }
            if not validity["valid"]:
                cell_result["status"] = "excluded"
                cell_result["candidates"] = []
                cell_results.append(cell_result)
                continue

            # Depth-indexed fixtures store velocities per cell per ping.
            cell_beam_data = ping.get("cell_beams", {}).get(str(cell_index))
            if cell_beam_data is not None:
                cell_radials: List[Optional[float]] = []
                cell_corr: List[Optional[float]] = []
                cell_amp: List[Optional[float]] = []
                for beam in cell_beam_data:
                    velocity_q = parse_quantity(beam["velocity"], "m/s", "beam")
                    correlation_q = parse_quantity(beam["correlation"], "counts")
                    amplitude_q = parse_quantity(beam["amplitude"], "counts")
                    cell_radials.append(velocity_q.value)
                    cell_corr.append(correlation_q.value)
                    cell_amp.append(amplitude_q.value)
                cell_quality = classify_beams(cell_corr, cell_amp, [bool(b.get("bad_flag")) for b in cell_beam_data], thresholds)
                solve = solve_beam_velocities(
                    cell_radials, cell_quality["good"], cell_quality["suspect"], geometry
                )
                cell_result["quality"] = cell_quality
            else:
                # Ping-level beam velocities apply to every in-water cell in
                # the compact fixture format.
                solve = solve_beam_velocities(radials, quality["good"], quality["suspect"], geometry)
                cell_result["quality"] = quality

            candidates = [
                _candidate_payload(
                    candidate,
                    attitude_payload,
                    declination_deg,
                    mounting,
                    order,
                    geometry.geometry_version,
                    str(declination_record["declination_version"]) if declination_record else "none",
                    mounting_version,
                )
                for candidate in solve["candidates"]
            ]
            cell_result["status"] = solve["status"]
            cell_result["solve_reason"] = solve["reason"]
            cell_result["candidates"] = candidates
            cell_result["attitude"] = attitude_result
            cell_results.append(cell_result)

        pings_out.append(
            {
                "ping_id": ping_id,
                "stable_id": make_stable_id(["ping", ping_id]),
                "time_s": time_s,
                "beams": beam_meas,
                "quality": quality,
                "attitude": attitude_result,
                "declination_version": str(declination_record["declination_version"])
                if declination_record
                else "none",
                "mounting_version": mounting_version,
                "cells": cell_results,
                "bottom_track": _bottom_track_payload(
                    bottom, geometry, attitude_payload, declination_deg, mounting, order,
                    geometry.geometry_version,
                    str(declination_record["declination_version"]) if declination_record else "none",
                    mounting_version,
                ),
            }
        )

    return {
        "geometry_version": geometry.geometry_version,
        "transform_order": order,
        "pings": pings_out,
        "cells": cells,
    }


def _bottom_track_payload(
    bottom: Optional[Dict[str, Any]],
    geometry: BeamGeometry,
    attitude: Optional[Dict[str, float]],
    declination_deg: Optional[float],
    mounting: Dict[str, float],
    order: str,
    geometry_version: str,
    declination_version: str,
    mounting_version: str,
) -> Optional[Dict[str, Any]]:
    if bottom is None:
        return None
    radials = [parse_quantity(v, "m/s", "beam").value for v in bottom.get("velocity", [])]
    if len([v for v in radials if v is not None]) < 3:
        return {"status": "bottom_track_underdetermined", "raw_velocity": bottom.get("velocity", [])}
    good = [v is not None for v in radials]
    solve = solve_beam_velocities(radials, good, [False] * len(good), geometry)
    candidates = [
        _candidate_payload(
            candidate, attitude, declination_deg, mounting, order, geometry_version,
            declination_version, mounting_version,
        )
        for candidate in solve["candidates"]
    ]
    return {"status": solve["status"], "candidates": candidates}
