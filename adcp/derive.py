from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any

from .attitude import AttitudeSample, interpolate_attitude, parse_time
from .geometry import CANONICAL_ORDER, instrument_to_ship, ship_to_earth, solve_beams
from .units import normalize_quantity


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def current_records(records: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for record in records:
        key = tuple(record["logical_key"])
        if key not in best or (record["seq"], record["source_id"]) > (best[key]["seq"], best[key]["source_id"]):
            best[key] = record
    return list(best.values())


def record_value(record: dict) -> Any:
    return copy.deepcopy(record["raw"])


def _f(record: dict | None, key: str, kind: str, default_unit: str | None = None) -> tuple[float | None, str | None, str | None]:
    if not record:
        return None, None, f"missing {key}"
    raw = record["raw"]
    unit = raw.get("unit", default_unit)
    value, diag = normalize_quantity(raw.get("value"), unit, kind)
    return value, unit, diag


def index_records(records: list[dict]) -> dict[tuple, dict]:
    return {tuple(record["logical_key"]): record for record in current_records(records)}


def build_geometry(config: dict | None, declination: dict | None) -> tuple[dict, list[str], str | None]:
    diagnostics: list[str] = []
    if not config:
        return {}, ["missing transducer configuration"], None
    raw = config["raw"]
    beam_angle, beam_unit, angle_diag = _f(config, "beam_angle", "angle", "deg")
    if angle_diag:
        diagnostics.append(angle_diag)
    azimuths = raw.get("azimuths_deg")
    if not isinstance(azimuths, list) or len(azimuths) != 4:
        diagnostics.append("beam geometry requires four azimuths_deg")
    install, _, install_diag = _f(config, "installation_alignment", "angle", "deg")
    if install_diag:
        diagnostics.append(install_diag)
    order = raw.get("transform_order") or CANONICAL_ORDER
    dec_value = 0.0
    if declination:
        dec_value, _, dec_diag = _f(declination, "declination", "angle", "deg")
        if dec_diag:
            diagnostics.append(dec_diag)
    geometry = {
        "beam_angle_raw": raw.get("beam_angle", {}).get("value") if isinstance(raw.get("beam_angle"), dict) else raw.get("beam_angle"),
        "beam_angle_unit": beam_unit,
        "beam_angle_deg": beam_angle,
        "azimuths_deg": azimuths,
        "installation_alignment_deg": install,
        "transform_order": order,
        "declination_deg": dec_value,
    }
    fingerprint_inputs = {
        "config_source_id": config["source_id"],
        "beam_angle": [geometry["beam_angle_raw"], beam_unit],
        "azimuths": azimuths,
        "installation": [raw.get("installation_alignment", {}).get("value") if isinstance(raw.get("installation_alignment"), dict) else raw.get("installation_alignment"), raw.get("installation_alignment", {}).get("unit", "deg") if isinstance(raw.get("installation_alignment"), dict) else "deg"],
        "transform_order": order,
        "declination_source_id": declination["source_id"] if declination else None,
        "declination": [declination["raw"].get("declination", {}).get("value"), declination["raw"].get("declination", {}).get("unit", "deg")] if declination else [0.0, "deg"],
    }
    return geometry, diagnostics, stable_hash(fingerprint_inputs)


def _raw_quantity(record: dict | None, field: str) -> dict[str, Any]:
    if not record:
        return {"value": None, "unit": None, "diagnostic": f"missing {field}"}
    return copy.deepcopy(record["raw"].get(field, record["raw"]))


def _quality(subject: str, ping: str, idx: dict[tuple, dict]) -> tuple[dict, list[dict], list[dict]]:
    beam_rows = []
    dependencies = []
    diagnostics = []
    for beam in range(1, 5):
        velocity = idx.get(("beam_velocity", subject, ping, beam))
        correlation = idx.get(("correlation", subject, ping, beam))
        amplitude = idx.get(("amplitude", subject, ping, beam))
        bad = idx.get(("bad_beam", subject, ping, beam))
        for record in (velocity, correlation, amplitude, bad):
            if record:
                dependencies.append(record["source_id"])
        vel_m_s, _, vel_diag = _f(velocity, "beam velocity", "velocity", "m/s")
        corr = correlation["raw"].get("value") if correlation else None
        amp = amplitude["raw"].get("value") if amplitude else None
        flagged = bool(bad and bad["raw"].get("value"))
        reasons = []
        if flagged:
            reasons.append("source bad-beam flag")
        if isinstance(corr, (int, float)) and corr < 60:
            reasons.append("correlation below 60 counts")
        if isinstance(amp, (int, float)) and amp < 70:
            reasons.append("amplitude below 70 counts")
        if vel_diag:
            reasons.append(vel_diag)
        beam_rows.append(
            {
                "beam": beam,
                "velocity_source_id": velocity["source_id"] if velocity else None,
                "correlation_source_id": correlation["source_id"] if correlation else None,
                "amplitude_source_id": amplitude["source_id"] if amplitude else None,
                "bad_beam_source_id": bad["source_id"] if bad else None,
                "velocity_raw": _raw_quantity(velocity, "velocity"),
                "velocity_m_s": vel_m_s,
                "correlation_raw": corr,
                "correlation_unit": correlation["raw"].get("unit", "counts") if correlation else "counts",
                "amplitude_raw": amp,
                "amplitude_unit": amplitude["raw"].get("unit", "counts") if amplitude else "counts",
                "bad_flag": flagged,
                "valid_for_solve": not reasons,
                "reasons": reasons,
            }
        )
        diagnostics.extend([f"beam {beam}: {reason}" for reason in reasons])
    identity = stable_hash(["quality", subject, ping, dependencies])
    node = {
        "node_id": f"quality:{identity}",
        "node_type": "beam_quality",
        "subject": subject,
        "ping": ping,
        "status": "ok" if all(row["valid_for_solve"] for row in beam_rows) else "degraded",
        "beams": beam_rows,
        "diagnostics": diagnostics,
        "fingerprint": identity,
        "source_ids": dependencies,
    }
    return node, [node], dependencies


def _depth_cell(subject: str, ping: str, idx: dict[tuple, dict]) -> dict:
    cell = idx.get(("depth_cell", subject, ping))
    depth_m = None
    diagnostics = []
    raw = None
    source_id = None
    if cell:
        source_id = cell["source_id"]
        raw = copy.deepcopy(cell["raw"])
        depth_m, _, diag = _f(cell, "depth", "depth", "m")
        if diag:
            diagnostics.append(diag)
    else:
        diagnostics.append("missing depth cell")
    identity = stable_hash(["depth", subject, ping, source_id, raw])
    return {
        "node_id": f"depth:{identity}",
        "node_type": "depth_cell",
        "subject": subject,
        "ping": ping,
        "status": "excluded" if diagnostics else "ok",
        "depth_m": depth_m,
        "raw": raw,
        "diagnostics": diagnostics,
        "fingerprint": identity,
        "source_ids": [source_id] if source_id else [],
    }


def _attitude_at(subject: str, ping_record: dict, idx: dict[tuple, dict]) -> dict:
    ping_time = ping_record["raw"].get("time")
    attitude_records = [record for record in idx.values() if record["type"] == "attitude" and tuple(record["logical_key"])[1] == subject]
    samples = []
    source_ids = []
    for record in attitude_records:
        source_ids.append(record["source_id"])
        raw = record["raw"]
        heading, _, hdiag = _f(record, "heading", "angle", "deg")
        pitch, _, pdiag = _f(record, "pitch", "angle", "deg")
        roll, _, rdiag = _f(record, "roll", "angle", "deg")
        if heading is not None and pitch is not None and roll is not None:
            samples.append(AttitudeSample(record["source_id"], parse_time(raw["time"]), heading, pitch, roll))
    interpolated, diagnostics = interpolate_attitude(parse_time(ping_time), samples) if ping_time else (None, ["missing ping time"])
    identity = stable_hash(["attitude", subject, ping_record["source_id"], ping_time, source_ids])
    return {
        "node_id": f"attitude:{identity}",
        "node_type": "attitude",
        "subject": subject,
        "ping": ping_record["logical_key"][1],
        "status": "ok" if interpolated else "excluded",
        "attitude": interpolated,
        "diagnostics": diagnostics,
        "fingerprint": identity,
        "source_ids": source_ids + [ping_record["source_id"]],
    }


def _candidate_nodes(subject: str, ping: str, quality: dict, geometry: dict, geom_fp: str | None, attitude: dict, depth: dict) -> tuple[list[dict], list[dict], list[str]]:
    nodes: list[dict] = []
    dependencies = list(quality["source_ids"]) + attitude["source_ids"]
    if geom_fp:
        dependencies.append(geom_fp)
    if depth["source_ids"]:
        dependencies.extend(depth["source_ids"])

    if depth["status"] == "excluded":
        identity = stable_hash(["excluded-cell", subject, ping, depth["fingerprint"]])
        node = {
            "node_id": f"exclusion:{identity}",
            "node_type": "excluded_cell",
            "subject": subject,
            "ping": ping,
            "status": "excluded",
            "diagnostics": depth["diagnostics"],
            "fingerprint": identity,
            "source_ids": depth["source_ids"],
        }
        return [node], [node], depth["source_ids"]

    velocities = [row["velocity_m_s"] for row in quality["beams"]]
    valid = [row["valid_for_solve"] for row in quality["beams"]]
    solved = solve_beams(velocities, valid, geometry)
    if not solved["candidates"]:
        identity = stable_hash(["insufficient", subject, ping, quality["fingerprint"], geom_fp])
        node = {
            "node_id": f"insufficient:{identity}",
            "node_type": "insufficient_beams",
            "subject": subject,
            "ping": ping,
            "status": "insufficient_beams",
            "diagnostics": solved["diagnostics"],
            "fingerprint": identity,
            "source_ids": dependencies,
        }
        return [node], [node], dependencies

    candidates = []
    for index, candidate in enumerate(solved["candidates"]):
        method = candidate["method"]
        omitted = candidate["omitted_beams"]
        beam_dep = [quality["beams"][i]["velocity_source_id"] for i in candidate["used_beams"] if quality["beams"][i]["velocity_source_id"]]
        instrument_fp = stable_hash(["instrument", subject, ping, method, omitted, candidate["instrument_velocity_m_s"], beam_dep, geom_fp])
        instrument_vector = candidate["instrument_velocity_m_s"]
        installation = geometry.get("installation_alignment_deg")
        if installation is None:
            ship_vector = None
            earth_vector = None
            install_diag = ["installation alignment unavailable"]
        else:
            ship_vector = instrument_to_ship(instrument_vector, installation)
            install_diag = []
            earth_vector = None
        if attitude["status"] == "excluded":
            earth_vector = None
        elif ship_vector is not None:
            earth_vector = ship_to_earth(
                ship_vector,
                attitude["attitude"]["heading_deg"],
                attitude["attitude"]["pitch_deg"],
                attitude["attitude"]["roll_deg"],
                geometry["declination_deg"],
            )
        ship_fp = stable_hash(["ship", instrument_fp, ship_vector, geom_fp])
        earth_fp = stable_hash(["earth", ship_fp, earth_vector, attitude["fingerprint"], geom_fp])
        candidate_id = stable_hash(["candidate", subject, ping, method, omitted, index, beam_dep])
        candidate_obj = {
            "candidate_id": f"candidate:{candidate_id}",
            "node_type": "velocity_candidate",
            "subject": subject,
            "ping": ping,
            "method": method,
            "omitted_beams": omitted,
            "used_beams": candidate["used_beams"],
            "residual_m_s": candidate["residual_m_s"],
            "instrument": {
                "frame": "instrument FSU",
                "velocity_m_s": instrument_vector,
                "unit": "m/s",
                "fingerprint": instrument_fp,
            },
            "ship": {
                "frame": "ship forward/starboard/up",
                "velocity_m_s": ship_vector,
                "unit": "m/s",
                "fingerprint": ship_fp,
                "diagnostics": install_diag,
            },
            "earth": {
                "frame": "earth east/north/up",
                "velocity_m_s": earth_vector,
                "unit": "m/s",
                "fingerprint": earth_fp,
                "attitude_mode": attitude["attitude"]["mode"] if attitude["attitude"] else None,
                "diagnostics": attitude["diagnostics"] + install_diag,
            },
            "fingerprint": stable_hash([instrument_fp, ship_fp, earth_fp, geom_fp, attitude["fingerprint"]]),
            "source_ids": dependencies,
        }
        nodes.append(candidate_obj)
        candidates.append(candidate_obj)
    identity = stable_hash(["candidate-set", subject, ping, [candidate["candidate_id"] for candidate in candidates], solved["status"]])
    set_node = {
        "node_id": f"candidate-set:{identity}",
        "node_type": "candidate_set",
        "subject": subject,
        "ping": ping,
        "status": solved["status"],
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "methods": [candidate["method"] for candidate in candidates],
        "diagnostics": solved["diagnostics"],
        "fingerprint": identity,
        "source_ids": dependencies,
    }
    nodes.insert(0, set_node)
    return nodes, candidates, dependencies
