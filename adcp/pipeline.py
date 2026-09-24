"""Stepwise velocity derivation: beam -> instrument -> ship -> earth."""
from __future__ import annotations

import math

from .fingerprint import fingerprint
from .geometry import BeamGeometry
from .model import PingBundle
from .quality import CandidateSet, QualityRules, evaluate_cell
from .rotation import (
    AttitudeResult,
    MountAngles,
    apply_mount_angles,
    attitude_matrix,
    instrument_to_ship_matrix,
    ship_to_earth_matrix,
)


def _round_vector(vector):
    return [round(float(v), 9) for v in vector]


def transform_candidates(
    candidate_set: CandidateSet,
    attitude: AttitudeResult,
    geometry: BeamGeometry,
    mount: MountAngles,
    declination_deg: float,
    order_id: str,
) -> dict:
    """Apply the full rotation chain to every candidate of one cell."""
    transformed = []
    attitude_diag = list(attitude.diagnostics)
    for idx, candidate in enumerate(candidate_set.candidates):
        used = candidate["used_beams"]
        candidate_id = fingerprint(
            "cand",
            {
                "family": candidate["family"],
                "used_beams": used,
                "velocity_xyz": candidate["velocity_xyz"],
            },
        )
        entry = {
            "candidate_id": candidate_id,
            "family": candidate["family"],
            "used_beams": used,
            "instrument_xyz_mps": _round_vector(candidate["velocity_xyz"]),
            "error_velocity_mps": (
                round(candidate["error_velocity"], 9)
                if candidate["error_velocity"] is not None
                else None
            ),
        }
        if any(d.startswith("attitude:gap") for d in attitude_diag) or any(
            math.isnan(v) for v in (attitude.pitch, attitude.roll, attitude.heading_true)
        ):
            entry["ship_xyz_mps"] = None
            entry["earth_enu_mps"] = None
            entry["diagnostics"] = attitude_diag + ["transform:attitude-unavailable"]
        else:
            ship = apply_mount_angles(candidate["velocity_xyz"], mount, order_id)
            heading = ship_to_earth_matrix(attitude.heading_true, order_id)
            earth = heading @ ship
            inst_to_ship = instrument_to_ship_matrix(
                attitude.pitch, attitude.roll, order_id
            )
            # Full attitude chain (without static mount), for display.
            dynamic_ship = inst_to_ship @ candidate["velocity_xyz"]
            entry["ship_xyz_mps"] = _round_vector(ship)
            entry["earth_enu_mps"] = _round_vector(earth)
            entry["dynamic_ship_xyz_mps"] = _round_vector(dynamic_ship)
            entry["rotation"] = {
                "heading_true_deg": round(attitude.heading_true, 9),
                "heading_mag_deg": round(attitude.heading_mag, 9),
                "pitch_deg": round(attitude.pitch, 9),
                "roll_deg": round(attitude.roll, 9),
                "declination_deg": declination_deg,
                "order_id": order_id,
                "mount": mount.content(),
                "geometry_fingerprint": geometry.fingerprint(),
            }
            entry["diagnostics"] = attitude_diag
            if attitude.interpolated:
                entry["diagnostics"] = entry["diagnostics"] + ["attitude:interpolated"]
        transformed.append(entry)
    return {
        "bin_index": candidate_set.bin_index,
        "cell_status": candidate_set.cell_status,
        "diagnostics": sorted(set(candidate_set.diagnostics + attitude_diag)),
        "candidates": transformed,
    }


def derive_ping(
    bundle: PingBundle,
    attitude: AttitudeResult,
    geometry: BeamGeometry,
    mount: MountAngles,
    declination_deg: float,
    rules: QualityRules,
    order_id: str,
) -> dict:
    """Derive the complete evidence record for one ping."""
    profile_cells = []
    for bin_index in range(bundle.bins):
        candidate_set = evaluate_cell(
            bin_index,
            bundle.radial[bin_index],
            bundle.raw_bad[bin_index],
            bundle.correlation[bin_index],
            bundle.amplitude[bin_index],
            bundle.bin_invalid[bin_index],
            bundle.bin_invalid_reason[bin_index],
            geometry,
            rules,
        )
        profile_cells.append(
            transform_candidates(
                candidate_set, attitude, geometry, mount, declination_deg, order_id
            )
        )

    bt_set = evaluate_cell(
        None,
        bundle.bottom_track_radial,
        bundle.bottom_track_bad,
        bundle.bottom_track_corr,
        bundle.bottom_track_amp,
        False,
        None,
        geometry,
        rules,
    )
    bottom_track = transform_candidates(
        bt_set, attitude, geometry, mount, declination_deg, order_id
    )

    n_unique = sum(1 for c in profile_cells if c["cell_status"] == "unique")
    n_ambiguous = sum(1 for c in profile_cells if c["cell_status"] == "ambiguous")
    n_excluded = sum(1 for c in profile_cells if c["cell_status"] == "excluded")
    n_none = sum(1 for c in profile_cells if c["cell_status"] == "no-solution")

    return {
        "ping_index": bundle.ping_index,
        "t": bundle.t,
        "attitude": {
            "t": attitude.t,
            "heading_mag_deg": (None if math.isnan(attitude.heading_mag)
                                else round(attitude.heading_mag, 9)),
            "heading_true_deg": (None if math.isnan(attitude.heading_true)
                                 else round(attitude.heading_true, 9)),
            "pitch_deg": (None if math.isnan(attitude.pitch)
                          else round(attitude.pitch, 9)),
            "roll_deg": (None if math.isnan(attitude.roll)
                         else round(attitude.roll, 9)),
            "interpolated": attitude.interpolated,
            "diagnostics": attitude.diagnostics,
        },
        "inputs": {
            "geometry_id": geometry.geometry_id,
            "geometry_fingerprint": geometry.fingerprint(),
            "mount": mount.content(),
            "declination_deg": declination_deg,
            "quality_rules": rules.content(),
            "transform_order": order_id,
        },
        "profile_cells": profile_cells,
        "bottom_track": bottom_track,
        "summary": {
            "unique_cells": n_unique,
            "ambiguous_cells": n_ambiguous,
            "excluded_cells": n_excluded,
            "no_solution_cells": n_none,
        },
        "unconvertible": bundle.unconvertible,
    }
