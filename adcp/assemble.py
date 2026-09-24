"""Assemble separated raw records into parameters and per-ping bundles.

Raw representations are preserved untouched on the records themselves.
Normalization happens here, at the derivation boundary: a value that cannot
be converted stays at its original location (record + per-cell diagnostic)
instead of being dropped.
"""
from __future__ import annotations

import math

from .geometry import BeamGeometry, default_geometry
from .model import PingBundle, RawRecord
from .quality import QualityRules
from .rotation import MountAngles, TRANSFORM_ORDERS
from .units import UnitError, normalize

PROFILE_KINDS = ("beam_velocity", "correlation", "amplitude", "bad_beam_flag")


def _num(record: dict, diagnostic_target: list[dict], label: str):
    try:
        quantity = normalize(record["raw_value"], record.get("unit") or "count",
                             record.get("datum"))
        return quantity.value
    except UnitError as exc:
        diagnostic_target.append(
            {"uid": record["uid"], "label": label, "reason": str(exc),
             "raw_value": record["raw_value"], "unit": record.get("unit")}
        )
        return None


def _records_by_key(records: list[dict]) -> dict[tuple, dict]:
    return {
        (r["ping_index"], r["bin_index"], r["beam_index"]): r for r in records
    }


def load_parameters(records: list[dict]):
    geometry = default_geometry()
    mount = MountAngles()
    declination = 0.0
    rules = QualityRules(rules_id="quality-default-v1")
    order_id = "rdi-zyx-v1"
    profile_config = {"n_bins": 4, "blanking_m": 1.0, "cell_size_m": 1.0}

    for record in records:
        value = record["raw_value"]
        if record["kind"] == "geometry":
            geometry = BeamGeometry(
                geometry_id=value["geometry_id"],
                beam_angle_deg=float(value["beam_angle_deg"]),
                azimuths_deg=tuple(float(a) for a in value["azimuths_deg"]),
                sign_convention=value.get("sign_convention",
                                          "positive_toward_transducer"),
            )
        elif record["kind"] == "mount_angles":
            mount = MountAngles(
                pitch_deg=float(value["pitch_deg"]),
                roll_deg=float(value["roll_deg"]),
                heading_offset_deg=float(value.get("heading_offset_deg", 0.0)),
                mount_id=value.get("mount_id", "mount-v1"),
            )
        elif record["kind"] == "declination":
            declination = float(normalize(value, record.get("unit") or "deg").value)
        elif record["kind"] == "quality_rules":
            rules = QualityRules(
                rules_id=value["rules_id"],
                correlation_min=float(value["correlation_min"]),
                amplitude_min=float(value["amplitude_min"]),
                correlation_suspect_band=float(value.get("correlation_suspect_band", 15.0)),
                amplitude_suspect_band=float(value.get("amplitude_suspect_band", 8.0)),
                max_attitude_gap_s=float(value.get("max_attitude_gap_s", 2.0)),
            )
        elif record["kind"] == "transform_order":
            order_id = str(value)
            if order_id not in TRANSFORM_ORDERS:
                raise ValueError(f"unsupported transform order {order_id!r}")
        elif record["kind"] == "profile_config":
            profile_config = {
                "n_bins": int(value["n_bins"]),
                "blanking_m": float(value["blanking_m"]),
                "cell_size_m": float(value["cell_size_m"]),
            }
    return geometry, mount, declination, rules, order_id, profile_config


def assemble_pings(records: list[dict]) -> list[PingBundle]:
    geometry, mount, declination, rules, order_id, profile_config = load_parameters(records)
    n_beams = geometry.n_beams
    n_bins = profile_config["n_bins"]

    ping_indices = sorted({r["ping_index"] for r in records if r["ping_index"] is not None})
    radial_records = _records_by_key([r for r in records if r["kind"] == "beam_velocity"])
    corr_records = _records_by_key([r for r in records if r["kind"] == "correlation"])
    amp_records = _records_by_key([r for r in records if r["kind"] == "amplitude"])
    bad_records = _records_by_key([r for r in records if r["kind"] == "bad_beam_flag"])
    invalid_records = {
        (r["ping_index"], r["bin_index"]): r
        for r in records if r["kind"] == "bin_invalid"
    }
    bt_radial = {(r["ping_index"], r["beam_index"]): r
                 for r in records if r["kind"] == "bottom_track"}
    ping_times = {r["ping_index"]: r["t"]
                  for r in records if r["kind"] == "ping_meta" and r["t"] is not None}

    bundles = []
    for ping in ping_indices:
        radial = [[None] * n_beams for _ in range(n_bins)]
        correlation = [[None] * n_beams for _ in range(n_bins)]
        amplitude = [[None] * n_beams for _ in range(n_bins)]
        raw_bad = [[False] * n_beams for _ in range(n_bins)]
        bin_invalid = [False] * n_bins
        bin_invalid_reason: list[str | None] = [None] * n_bins
        unconvertible: list[dict] = []

        for b in range(n_bins):
            for beam in range(n_beams):
                key = (ping, b, beam)
                if key in radial_records:
                    radial[b][beam] = _num(radial_records[key], unconvertible,
                                           f"beam_velocity bin={b} beam={beam}")
                if key in corr_records:
                    correlation[b][beam] = _num(corr_records[key], unconvertible,
                                                f"correlation bin={b} beam={beam}")
                if key in amp_records:
                    amplitude[b][beam] = _num(amp_records[key], unconvertible,
                                              f"amplitude bin={b} beam={beam}")
                if key in bad_records:
                    raw_value = bad_records[key]["raw_value"]
                    raw_bad[b][beam] = bool(raw_value)
            inv = invalid_records.get((ping, b))
            if inv is not None and bool(inv["raw_value"].get("invalid", False)):
                bin_invalid[b] = True
                bin_invalid_reason[b] = inv["raw_value"].get("reason")

        bt_v = [None] * n_beams
        bt_c = [None] * n_beams
        bt_a = [None] * n_beams
        bt_bad = [False] * n_beams
        for beam in range(n_beams):
            if (ping, None, beam) in radial_records:
                rec = radial_records[(ping, None, beam)]
                bt_v[beam] = _num(rec, unconvertible, f"bottom_track beam={beam}")
            if (ping, beam) in bt_radial:
                rec = bt_radial[(ping, beam)]
                bt_v[beam] = _num(rec, unconvertible, f"bottom_track beam={beam}")
            if (ping, None, beam) in corr_records:
                bt_c[beam] = _num(corr_records[(ping, None, beam)], unconvertible,
                                  f"bt correlation beam={beam}")
            if (ping, None, beam) in amp_records:
                bt_a[beam] = _num(amp_records[(ping, None, beam)], unconvertible,
                                  f"bt amplitude beam={beam}")
            if (ping, None, beam) in bad_records:
                bt_bad[beam] = bool(bad_records[(ping, None, beam)]["raw_value"])

        bundles.append(PingBundle(
            ping_index=ping,
            t=ping_times.get(ping, float(ping)),
            bins=n_bins,
            radial=radial,
            correlation=correlation,
            amplitude=amplitude,
            raw_bad=raw_bad,
            bin_invalid=bin_invalid,
            bin_invalid_reason=bin_invalid_reason,
            bottom_track_radial=bt_v,
            bottom_track_corr=bt_c,
            bottom_track_amp=bt_a,
            bottom_track_bad=bt_bad,
            unconvertible=unconvertible,
        ))
    return bundles


def cell_depths(profile_config: dict) -> list[float]:
    return [
        round(profile_config["blanking_m"] + profile_config["cell_size_m"] * (b + 0.5), 6)
        for b in range(profile_config["n_bins"])
    ]
