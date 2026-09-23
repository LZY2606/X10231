from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .storage import Store, canonical_json
from .transforms import (
    BEAM_FRAME,
    EARTH_FRAME,
    INSTRUMENT_FRAME,
    SHIP_FRAME,
    attitude_rotation,
    beam_geometry,
    interpolate_attitude,
    mount_rotation,
    solve_beams,
)

RULE_VERSION = "adcp-rules-2026.09"
CORRELATION_AMBIGUOUS_BELOW = 40.0


@dataclass(frozen=True)
class Resource:
    id: str
    revision_id: str
    kind: str
    payload: dict[str, Any]


class Snapshot:
    def __init__(self, resources: dict[str, Resource]):
        self.resources = resources

    def payload(self, resource_id: str) -> dict[str, Any]:
        return self.resources[resource_id].payload

    def revision(self, resource_id: str) -> str:
        return self.resources[resource_id].revision_id


def snapshot_from_store(store: Store, version_id: str) -> Snapshot:
    resources = {}
    for resource_id, revision_id in store.version_resource_map(version_id).items():
        row = store.get_resource_revision(revision_id)
        resources[resource_id] = Resource(row["id"], row["revision_id"], row["kind"], row["payload"])
    return Snapshot(resources)


def derive_version(
    store: Store,
    version_id: str,
    label: str,
    parent_run_id: str | None = None,
    decisions: dict[str, str] | None = None,
    emit_events: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    snapshot = snapshot_from_store(store, version_id)
    decisions = decisions or {}
    resources = snapshot.resources
    transducer = next(item for item in resources.values() if item.kind == "transducer_config")
    declination = next(item for item in resources.values() if item.kind == "magnetic_declination")
    mount = next(item for item in resources.values() if item.kind == "mounting_config")
    transform = next(item for item in resources.values() if item.kind == "transform_policy")
    attitude = next(item for item in resources.values() if item.kind == "attitude_series")
    pings = [item for item in resources.values() if item.kind == "adcp_ping"]
    fingerprint_parts = {
        "rule_version": RULE_VERSION,
        "resources": {key: value.revision_id for key, value in sorted(resources.items())},
        "decisions": dict(sorted(decisions.items())),
    }
    run_fingerprint = Store.digest(fingerprint_parts)
    events: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    with store.transaction() as conn:
        run_id = store.insert_derivation_run(conn, version_id, parent_run_id, RULE_VERSION, run_fingerprint)
        geometry_payload = transducer.payload
        matrix = beam_geometry(float(geometry_payload["beam_angle_deg"]["value"]), geometry_payload["beam_azimuth_deg"])
        attitude_points = [
            type("Point", (), {
                "time_s": float(sample["time_s"]["value"]),
                "heading_deg": _optional_number(sample.get("heading_deg")),
                "pitch_deg": _optional_number(sample.get("pitch_deg")),
                "roll_deg": _optional_number(sample.get("roll_deg")),
            })()
            for sample in attitude.payload["samples"]
        ]
        declination_deg = float(declination.payload["declination_deg"]["value"])
        transform_order = transform.payload["coordinate_transform_order"]
        mount_matrix = mount_rotation(mount.payload)
        for ping in sorted(pings, key=lambda item: float(item.payload["time_s"]["value"])):
            pending_events.extend(_derive_ping(conn, store, run_id, snapshot, ping, matrix, mount_matrix, attitude_points, declination_deg, transform_order, decisions))
        store.complete_run(conn, run_id)
        store.set_current(version_id, run_id, conn)
    if emit_events:
        for pending in pending_events:
            saved = store.add_event(
                pending["event_type"],
                pending["payload"],
                actor="derivation",
                refs=pending["refs"],
                version_id=version_id,
                run_id=run_id,
            )
            events.append(saved)
        summary = store.add_event(
            "derivation.completed",
            {"label": label, "rule_version": RULE_VERSION, "fingerprint": run_fingerprint, "event_count": len(events)},
            refs=[("derivation_run", run_id), ("version", version_id)],
            version_id=version_id,
            run_id=run_id,
        )
        events.append(summary)
    return run_id, events


def _derive_ping(conn, store: Store, run_id: str, snapshot: Snapshot, ping: Resource, matrix: np.ndarray, mount_matrix: np.ndarray, attitude_points, declination_deg: float, transform_order: str, decisions: dict[str, str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ping_id = ping.id
    payload = ping.payload
    cells = payload.get("cells", [])
    beam_statuses = []
    for beam in payload["beams"]:
        reason = None
        if beam.get("bad_beam_flag"):
            reason = "hard_bad_beam_flag"
        correlation = _optional_number(beam.get("correlation"))
        if reason is None and correlation is not None and correlation < CORRELATION_AMBIGUOUS_BELOW:
            reason = "correlation_ambiguous"
        amplitude = _optional_number(beam.get("amplitude"))
        if reason == "correlation_ambiguous" and amplitude is not None and amplitude < 30.0:
            reason = "correlation_and_amplitude_ambiguous"
        beam_statuses.append({"beam": int(beam["beam"]), "hard_bad": bool(beam.get("bad_beam_flag")), "correlation_ok": reason != "hard_bad_beam_flag" and correlation is not None and correlation >= CORRELATION_AMBIGUOUS_BELOW, "reason": reason})
    hard_good = [item["beam"] - 1 for item in beam_statuses if not item["hard_bad"] and item["reason"] is None]
    hard_bad = [item["beam"] - 1 for item in beam_statuses if item["hard_bad"]]
    marginal = [item["beam"] - 1 for item in beam_statuses if item["reason"] and not item["hard_bad"]]
    quality_deps = [(ping_id, snapshot.revision(ping_id))]
    quality_fingerprint = _fingerprint(store, ["quality", ping_id, snapshot.revision(ping_id), RULE_VERSION])
    quality_node = {
        "node_id": stable_node_id(run_id, ping_id, "beam_quality"),
        "run_id": run_id,
        "subject_id": ping_id,
        "stage": "beam_quality",
        "fingerprint": quality_fingerprint,
        "status": "ok" if not hard_bad and not marginal else "ambiguous" if not hard_bad else "degraded",
        "accepted": True,
        "value_hash": _fingerprint(store, beam_statuses),
        "diagnostic": {"beams": beam_statuses},
        "deps": quality_deps,
        "results": [
            {
                "result_id": stable_result_id(stable_node_id(run_id, ping_id, "beam_quality"), f"beam-{item['beam']}"),
                "node_id": stable_node_id(run_id, ping_id, "beam_quality"),
                "candidate_key": f"beam-{item['beam']}",
                "kind": "beam_quality",
                "value": item,
                "unit": "counts_and_correlation",
                "frame": BEAM_FRAME,
                "status": "excluded" if item["hard_bad"] else "ambiguous" if item["reason"] else "accepted",
                "adopted": not item["hard_bad"] and item["reason"] is None,
                "diagnostic": None if item["reason"] is None else {"reason": item["reason"]},
            }
            for item in beam_statuses
        ],
    }
    store.insert_node(conn, quality_node)
    if hard_bad:
        events.append(_pending_event(store, conn, "beam.failed", ping_id, quality_node, {"bad_beams_1based": [index + 1 for index in hard_bad]}))
    if marginal:
        events.append(_pending_event(store, conn, "beam.ambiguous", ping_id, quality_node, {"marginal_beams_1based": [index + 1 for index in marginal]}))

    valid_cells = []
    excluded_cells = []
    for cell in cells:
        depth = _optional_number(cell.get("depth"))
        depth_unit = (cell.get("depth") or {}).get("unit")
        if depth is None or depth <= 0 or depth_unit != "m":
            excluded_cells.append({"cell": cell.get("cell"), "reason": "invalid_depth_cell", "raw": cell.get("depth")})
        else:
            valid_cells.append(cell)
    profile_node = {
        "node_id": stable_node_id(run_id, ping_id, "profile_cells"),
        "run_id": run_id,
        "subject_id": ping_id,
        "stage": "profile_cells",
        "fingerprint": _fingerprint(store, ["profile", ping_id, cells, snapshot.revision(ping_id), RULE_VERSION]),
        "status": "ok" if not excluded_cells else "partial",
        "accepted": True,
        "value_hash": _fingerprint(store, [valid_cells]),
        "diagnostic": {"excluded_cells": excluded_cells},
        "deps": [(ping_id, snapshot.revision(ping_id))],
        "results": [
            {
                "result_id": stable_result_id(stable_node_id(run_id, ping_id, "profile_cells"), f"cell-{cell.get('cell')}"),
                "node_id": stable_node_id(run_id, ping_id, "profile_cells"),
                "candidate_key": f"cell-{cell.get('cell')}",
                "kind": "depth_cell",
                "value": cell,
                "unit": "m",
                "frame": "depth_bins",
                "status": "excluded" if any(item["cell"] == cell.get("cell") for item in excluded_cells) else "accepted",
                "adopted": not any(item["cell"] == cell.get("cell") for item in excluded_cells),
                "diagnostic": next((item for item in excluded_cells if item["cell"] == cell.get("cell")), None),
            }
            for cell in cells
        ],
    }
    store.insert_node(conn, profile_node)
    for excluded in excluded_cells:
        events.append(_pending_event(store, conn, "cell.excluded", ping_id, profile_node, excluded))

    attitude = interpolate_attitude(attitude_points, float(payload["time_s"]["value"]), float(snapshot.payload("transform_policy")["max_attitude_gap_s"]))
    attitude_node = {
        "node_id": stable_node_id(run_id, ping_id, "attitude"),
        "run_id": run_id,
        "subject_id": ping_id,
        "stage": "attitude",
        "fingerprint": _fingerprint(store, ["attitude", ping_id, payload["time_s"], snapshot.revision(_resource_id(snapshot, "attitude_series")), attitude, RULE_VERSION]),
        "status": "ok" if attitude["status"] == "ok" else "excluded",
        "accepted": attitude["status"] == "ok",
        "value_hash": _fingerprint(store, attitude),
        "diagnostic": None if attitude["status"] == "ok" else attitude,
        "deps": [(_resource_id(snapshot, "attitude_series"), snapshot.revision(_resource_id(snapshot, "attitude_series")))],
        "results": [
            {
                "result_id": stable_result_id(stable_node_id(run_id, ping_id, "attitude"), "interpolated-attitude"),
                "node_id": stable_node_id(run_id, ping_id, "attitude"),
                "candidate_key": "interpolated-attitude",
                "kind": "attitude",
                "value": attitude,
                "unit": "deg,s",
                "frame": "heading_pitch_roll",
                "status": "accepted" if attitude["status"] == "ok" else "excluded",
                "adopted": attitude["status"] == "ok",
                "diagnostic": None if attitude["status"] == "ok" else attitude,
            }
        ],
    }
    store.insert_node(conn, attitude_node)
    if attitude["status"] != "ok":
        events.append(_pending_event(store, conn, "attitude.unavailable", ping_id, attitude_node, attitude))

    instrument_nodes_by_cell: dict[int, str] = {}
    ship_nodes_by_cell: dict[int, str] = {}
    earth_nodes_by_cell: dict[int, str] = {}
    for cell in valid_cells:
        instrument_node = _instrument_cell(conn, store, run_id, snapshot, ping, cell, matrix, hard_good, marginal, hard_bad, decisions)
        instrument_nodes_by_cell[int(cell["cell"])] = instrument_node["node_id"]
        if instrument_node["status"] in {"unique", "adopted_multi"}:
            ship_node = _ship_cell(conn, store, run_id, snapshot, ping, cell, instrument_node, mount_matrix)
            ship_nodes_by_cell[int(cell["cell"])] = ship_node["node_id"]
            if attitude["status"] == "ok":
                earth_node = _earth_cell(conn, store, run_id, snapshot, ping, cell, ship_node, attitude, declination_deg, transform_order)
                earth_nodes_by_cell[int(cell["cell"])] = earth_node["node_id"]
        if instrument_node["status"] == "ambiguous":
            events.append(_pending_event(store, conn, "beam.three_beam_ambiguity", ping_id, instrument_node, {"candidate_keys": [result["candidate_key"] for result in instrument_node["results"]], "selected_candidate_key": instrument_node.get("selected_candidate_key")}))
        if instrument_node["status"] == "invalid":
            events.append(_pending_event(store, conn, "velocity.indeterminate", ping_id, instrument_node, instrument_node.get("diagnostic") or {}))

    events.extend(_bottom_track(conn, store, run_id, snapshot, ping, matrix, hard_good, marginal, hard_bad, attitude, mount_matrix, declination_deg, transform_order))
    return events


def stable_node_id(run_id: str, subject_id: str, stage: str) -> str:
    return Store.digest(["node", run_id, subject_id, stage])


def stable_result_id(node_id: str, candidate_key: str) -> str:
    return Store.digest(["result", node_id, candidate_key])


def _fingerprint(store: Store, value: Any) -> str:
    return store.digest([canonical_json(value)])


def _resource_id(snapshot: Snapshot, kind: str) -> str:
    return next(item.id for item in snapshot.resources.values() if item.kind == kind)


def _optional_number(quantity: dict[str, Any] | None) -> float | None:
    if not quantity or quantity.get("value") is None:
        return None
    try:
        number = float(quantity["value"])
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pending_event(store: Store, conn, event_type: str, subject_id: str, node: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "payload": {**payload, "subject_id": subject_id, "stage": node["stage"], "fingerprint": node["fingerprint"]},
        "refs": [("derivation_node", node["node_id"])]
        + [("derivation_result", result["result_id"]) for result in node.get("results", []) if result.get("adopted") or event_type.startswith("beam.three")],
    }


def _instrument_cell(conn, store: Store, run_id: str, snapshot: Snapshot, ping: Resource, cell: dict[str, Any], matrix: np.ndarray, hard_good: list[int], marginal: list[int], hard_bad: list[int], decisions: dict[str, str]) -> dict[str, Any]:
    node_id = stable_node_id(run_id, ping.id, f"instrument:cell-{cell['cell']}")
    radial = []
    raw_ok = []
    for index, beam in enumerate(ping.payload["beams"]):
        quantity = beam.get("radial_velocity")
        value = _optional_number(quantity)
        unit_ok = quantity is not None and quantity.get("unit") == "m/s" and quantity.get("datum") == "positive_away_transducer"
        if value is None or not unit_ok or index in hard_bad:
            raw_ok.append(False)
            radial.append(np.nan)
        else:
            raw_ok.append(True)
            radial.append(value)
    candidates: list[tuple[str, list[int], bool]] = []
    eligible = [index for index, ok in enumerate(raw_ok) if ok and index not in hard_bad]
    if len(eligible) == 4:
        candidates.append(("four-beam", [0, 1, 2, 3], False))
        for excluded in marginal:
            subset = [index for index in eligible if index != excluded]
            candidates.append((f"three-beam-drop-{excluded + 1}", subset, True))
    elif len(eligible) == 3 and not marginal:
        dropped = next(index for index in range(4) if index not in eligible)
        candidates.append((f"three-beam-drop-{dropped + 1}", sorted(eligible), True))
    results = []
    selected = decisions.get(node_id)
    diagnostics = []
    for candidate_key, subset, three_beam in candidates:
        subset_matrix = matrix[subset, :]
        subset_radial = np.asarray([radial[index] for index in subset], dtype=float)
        try:
            velocity, residual = solve_beams(subset_matrix, subset_radial)
            diagnostic = {"beams_1based": [index + 1 for index in subset], "mode": "three-beam" if three_beam else "four-beam"}
            status = "candidate" if len(candidates) > 1 else "accepted"
            adopted = len(candidates) == 1 or selected == candidate_key
            value = {
                "velocity_m_s": [float(item) for item in velocity],
                "residual_m_s": float(residual),
                "beams_1based": [index + 1 for index in subset],
                "mode": "three-beam" if three_beam else "four-beam",
            }
            results.append(_result(node_id, candidate_key, "water_velocity", value, "m/s", INSTRUMENT_FRAME, status, adopted, diagnostic))
        except (ValueError, np.linalg.LinAlgError) as exc:
            diagnostics.append({"candidate_key": candidate_key, "reason": str(exc)})
    if not results:
        status = "invalid"
        adopted = False
        diagnostic = {"reason": "fewer_than_three_valid_beams", "hard_bad_beams_1based": [index + 1 for index in hard_bad]}
        selected = None
    elif len(results) == 1:
        status = "unique"
        adopted = True
        diagnostic = results[0]["diagnostic"]
        selected = results[0]["candidate_key"]
    else:
        selected_result = next((item for item in results if item["candidate_key"] == selected), None)
        if selected_result is None:
            status = "ambiguous"
            adopted = False
            diagnostic = {"reason": "multiple_legal_candidates", "candidate_keys": [item["candidate_key"] for item in results], "modes_distinct": True}
            selected = None
            for result in results:
                result["adopted"] = False
                result["status"] = "candidate"
        else:
            status = "adopted_multi"
            adopted = True
            diagnostic = {"reason": "human_selected_candidate", "candidate_keys": [item["candidate_key"] for item in results], "selected_candidate_key": selected, "unselected_retained": True, "modes_distinct": True}
            for result in results:
                result["adopted"] = result["candidate_key"] == selected
                result["status"] = "accepted" if result["adopted"] else "candidate"
    transducer_id = _resource_id(snapshot, "transducer_config")
    node = {
        "node_id": node_id,
        "run_id": run_id,
        "subject_id": ping.id,
        "stage": f"instrument:cell-{cell['cell']}",
        "fingerprint": _fingerprint(store, ["instrument", run_id, ping.id, cell, snapshot.revision(ping.id), snapshot.revision(transducer_id), RULE_VERSION, [bool(item) for item in raw_ok]]),
        "status": status,
        "accepted": adopted,
        "value_hash": _fingerprint(store, [result["value"] for result in results]),
        "diagnostic": diagnostic if not diagnostics else {"candidate_errors": diagnostics},
        "deps": [(stable_node_id(run_id, ping.id, "beam_quality"), snapshot.revision(ping.id)), (stable_node_id(run_id, ping.id, "profile_cells"), snapshot.revision(ping.id)), (transducer_id, snapshot.revision(transducer_id))],
        "results": results,
        "selected_candidate_key": selected,
    }
    store.insert_node(conn, node)
    return node


def _result(node_id: str, candidate_key: str, kind: str, value: Any, unit: str, frame: str, status: str, adopted: bool, diagnostic: Any = None) -> dict[str, Any]:
    return {"result_id": stable_result_id(node_id, candidate_key), "node_id": node_id, "candidate_key": candidate_key, "kind": kind, "value": value, "unit": unit, "frame": frame, "status": status, "adopted": adopted, "diagnostic": diagnostic}


def _adopted_result(node: dict[str, Any]) -> dict[str, Any] | None:
    return next((result for result in node["results"] if result["adopted"]), None)


def _ship_cell(conn, store: Store, run_id: str, snapshot: Snapshot, ping: Resource, cell: dict[str, Any], instrument_node: dict[str, Any], mount_matrix: np.ndarray) -> dict[str, Any]:
    instrument_result = _adopted_result(instrument_node)
    velocity = np.asarray(instrument_result["value"]["velocity_m_s"])
    ship_velocity = mount_matrix @ velocity
    mount_id = _resource_id(snapshot, "mounting_config")
    value = {
        "instrument_velocity_m_s": [float(item) for item in velocity],
        "ship_velocity_m_s": [float(item) for item in ship_velocity],
        "mount_matrix": mount_matrix.tolist(),
        "selected_candidate_key": instrument_result["candidate_key"],
        "instrument_mode": instrument_result["value"]["mode"],
    }
    node_id = stable_node_id(run_id, ping.id, f"ship:cell-{cell['cell']}")
    node = {
        "node_id": node_id,
        "run_id": run_id,
        "subject_id": ping.id,
        "stage": f"ship:cell-{cell['cell']}",
        "fingerprint": _fingerprint(store, ["ship", run_id, ping.id, cell, instrument_node["fingerprint"], snapshot.revision(mount_id), RULE_VERSION]),
        "status": "unique",
        "accepted": True,
        "value_hash": _fingerprint(store, value),
        "diagnostic": {"mount_revision": snapshot.revision(mount_id), "instrument_mode_distinct_from_ship": True},
        "deps": [(instrument_node["node_id"], instrument_node["fingerprint"]), (mount_id, snapshot.revision(mount_id))],
        "results": [_result(node_id, instrument_result["candidate_key"], "water_velocity", value, "m/s", SHIP_FRAME, "accepted", True)],
    }
    store.insert_node(conn, node)
    return node


def _earth_cell(conn, store: Store, run_id: str, snapshot: Snapshot, ping: Resource, cell: dict[str, Any], ship_node: dict[str, Any], attitude: dict[str, Any], declination_deg: float, transform_order: str) -> dict[str, Any]:
    ship_result = _adopted_result(ship_node)
    ship_velocity = np.asarray(ship_result["value"]["ship_velocity_m_s"])
    rotation = attitude_rotation(attitude["heading_deg"], attitude["pitch_deg"], attitude["roll_deg"], declination_deg, transform_order)
    earth_velocity = rotation @ ship_velocity
    declination_id = _resource_id(snapshot, "magnetic_declination")
    policy_id = _resource_id(snapshot, "transform_policy")
    attitude_id = _resource_id(snapshot, "attitude_series")
    value = {
        "ship_velocity_m_s": [float(item) for item in ship_velocity],
        "earth_velocity_m_s": [float(item) for item in earth_velocity],
        "attitude_rotation_matrix": rotation.tolist(),
        "heading_magnetic_deg": attitude["heading_deg"],
        "declination_deg": declination_deg,
        "heading_true_deg": (attitude["heading_deg"] + declination_deg) % 360.0,
        "transform_order": transform_order,
        "selected_candidate_key": ship_result["candidate_key"],
    }
    node_id = stable_node_id(run_id, ping.id, f"earth:cell-{cell['cell']}")
    node = {
        "node_id": node_id,
        "run_id": run_id,
        "subject_id": ping.id,
        "stage": f"earth:cell-{cell['cell']}",
        "fingerprint": _fingerprint(store, ["earth", run_id, ping.id, cell, ship_node["fingerprint"], attitude, snapshot.revision(declination_id), snapshot.revision(policy_id), snapshot.revision(attitude_id), transform_order, RULE_VERSION]),
        "status": "unique",
        "accepted": True,
        "value_hash": _fingerprint(store, value),
        "diagnostic": {"declination_revision": snapshot.revision(declination_id), "transform_order": transform_order, "attitude_bracket_s": [attitude.get("left_time_s"), attitude.get("right_time_s")]},
        "deps": [(ship_node["node_id"], ship_node["fingerprint"]), (attitude_id, snapshot.revision(attitude_id)), (declination_id, snapshot.revision(declination_id)), (policy_id, snapshot.revision(policy_id))],
        "results": [_result(node_id, ship_result["candidate_key"], "water_velocity", value, "m/s", EARTH_FRAME, "accepted", True)],
    }
    store.insert_node(conn, node)
    return node


def _bottom_track(conn, store: Store, run_id: str, snapshot: Snapshot, ping: Resource, matrix: np.ndarray, hard_good: list[int], marginal: list[int], hard_bad: list[int], attitude: dict[str, Any], mount_matrix: np.ndarray, declination_deg: float, transform_order: str) -> list[dict[str, Any]]:
    bottom_id = f"{ping.id}:bottom_track"
    if bottom_id not in snapshot.resources:
        return []
    bottom = snapshot.resources[bottom_id]
    events: list[dict[str, Any]] = []
    raw_ok = []
    radial = []
    for index, beam in enumerate(bottom.payload["beams"]):
        flag = bool(beam.get("bad_beam_flag"))
        value = _optional_number(beam.get("radial_velocity"))
        raw_ok.append(not flag and value is not None and beam.get("radial_velocity", {}).get("unit") == "m/s")
        radial.append(value if value is not None else np.nan)
    valid_ranges = []
    excluded_ranges = []
    for index, beam in enumerate(bottom.payload["beams"]):
        range_value = _optional_number(beam.get("range"))
        item = {"beam_1based": index + 1, "range": beam.get("range"), "accepted": range_value is not None and range_value > 0 and beam.get("range", {}).get("unit") == "m"}
        if item["accepted"]:
            valid_ranges.append(item)
        else:
            excluded_ranges.append({**item, "reason": "invalid_bottom_range"})
    node_id = stable_node_id(run_id, bottom_id, "bottom_instrument")
    candidates = []
    eligible = [index for index, ok in enumerate(raw_ok) if ok and index not in hard_bad]
    if len(eligible) == 4:
        candidates.append(("four-beam-bottom", eligible, False))
        for excluded in marginal:
            subset = [index for index in eligible if index != excluded]
            candidates.append((f"three-beam-bottom-drop-{excluded + 1}", subset, True))
    elif len(eligible) == 3 and not marginal:
        dropped = next(index for index in range(4) if index not in eligible)
        candidates.append((f"three-beam-bottom-drop-{dropped + 1}", sorted(eligible), True))
    results = []
    for candidate_key, subset, three_beam in candidates:
        try:
            velocity, residual = solve_beams(matrix[subset, :], np.asarray([radial[index] for index in subset], dtype=float))
            value = {"bottom_track_velocity_m_s": [float(item) for item in velocity], "residual_m_s": float(residual), "beams_1based": [index + 1 for index in subset], "mode": "three-beam" if three_beam else "four-beam"}
            adopted = len(candidates) == 1
            results.append(_result(node_id, candidate_key, "bottom_track_velocity", value, "m/s", INSTRUMENT_FRAME, "accepted" if adopted else "candidate", adopted))
        except (ValueError, np.linalg.LinAlgError):
            pass
    status = "unique" if len(results) == 1 else "ambiguous" if len(results) > 1 else "invalid"
    node = {
        "node_id": node_id,
        "run_id": run_id,
        "subject_id": bottom_id,
        "stage": "bottom_instrument",
        "fingerprint": _fingerprint(store, ["bottom", run_id, bottom_id, snapshot.revision(bottom_id), snapshot.revision(_resource_id(snapshot, "transducer_config")), RULE_VERSION]),
        "status": status,
        "accepted": any(result["adopted"] for result in results),
        "value_hash": _fingerprint(store, [result["value"] for result in results]),
        "diagnostic": {"ranges": valid_ranges, "excluded_ranges": excluded_ranges, "candidate_count": len(results)},
        "deps": [(bottom_id, snapshot.revision(bottom_id)), (_resource_id(snapshot, "transducer_config"), snapshot.revision(_resource_id(snapshot, "transducer_config")))],
        "results": results,
    }
    store.insert_node(conn, node)
    if len(results) > 1:
        events.append(_pending_event(store, conn, "bottom.ambiguous", bottom_id, node, {"candidate_keys": [item["candidate_key"] for item in results]}))
    if excluded_ranges:
        events.append(_pending_event(store, conn, "bottom.range_excluded", bottom_id, node, {"excluded": excluded_ranges}))
    return events
