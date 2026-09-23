from __future__ import annotations

import copy
from typing import Any

from .case_builder import make_baseline_case
from .derive import derive_version
from .storage import ConflictError, Store


def import_case(store: Store, case: dict[str, Any] | None = None, label: str = "baseline separated ADCP sources") -> tuple[str, str, dict[str, Any]]:
    case = case or make_baseline_case()
    resources: dict[str, str] = {}
    with store.transaction() as conn:
        for resource in case["resources"]:
            payload = {key: value for key, value in resource.items() if key not in {"id", "kind"}}
            revision_id = store.add_resource_revision(conn, resource["id"], resource["kind"], payload)
            resources[resource["id"]] = revision_id
        version_id = store.create_version(resources, label, complete=True)
    run_id, events = derive_version(store, version_id, label, emit_events=True)
    store.add_event(
        "case.imported",
        {"case_id": case.get("case_id"), "title": case.get("title"), "resource_count": len(case["resources"]), "separate_sources": True},
        refs=[("version", version_id), ("derivation_run", run_id)],
        version_id=version_id,
        run_id=run_id,
    )
    return version_id, run_id, case


def revise_resource(store: Store, resource_id: str, payload: dict[str, Any], label: str, actor: str = "reviewer", event_type: str = "source.revised", event_payload: dict[str, Any] | None = None) -> tuple[str, str]:
    current_version = store.current_version_id()
    if not current_version:
        raise ValueError("no current version")
    current_map = store.version_resource_map(current_version)
    if resource_id not in current_map:
        raise KeyError(resource_id)
    parent_revision_id = current_map[resource_id]
    with store.transaction() as conn:
        revision_id = store.add_resource_revision(conn, resource_id, store.get_resource_revision(parent_revision_id)["kind"], payload, parent_revision_id)
        resources = dict(current_map)
        resources[resource_id] = revision_id
        version_id = store.create_version(resources, label, current_version, True)
    run_id, events = derive_version(store, version_id, label, parent_run_id=store.current_run_id())
    store.add_event(
        event_type,
        {"label": label, "resource_id": resource_id, "parent_revision_id": parent_revision_id, "revision_id": revision_id, **(event_payload or {})},
        actor=actor,
        refs=[("resource", revision_id), ("version", version_id), ("derivation_run", run_id)],
        version_id=version_id,
        run_id=run_id,
    )
    return version_id, run_id


def payload_for(store: Store, version_id: str, resource_id: str) -> dict[str, Any]:
    revision_id = store.version_resource_map(version_id)[resource_id]
    return copy.deepcopy(store.get_resource_revision(revision_id)["payload"])


def run_mount_revision(store: Store) -> tuple[str, str]:
    payload = payload_for(store, store.current_version_id(), "mount-config")
    payload["yaw_offset_deg"] = {"value": 4.5, "unit": "deg", "datum": "installation_alignment", "raw": "4.5 deg installation survey revision"}
    payload["revision_note"] = "installation survey correction: yaw 2.5 -> 4.5 degrees"
    return revise_resource(store, "mount-config", payload, "installation angle revision", event_type="mount.revised", event_payload={"yaw_offset_before_deg": 2.5, "yaw_offset_after_deg": 4.5})


def run_local_source_correction(store: Store) -> tuple[str, str]:
    version_id = store.current_version_id()
    payload = payload_for(store, version_id, "ping-03")
    for beam in payload["beams"]:
        if beam["beam"] == 1:
            beam["correlation"]["value"] = 91
            beam["correlation"]["raw"] = "91 reprocessed correlation from local beam record"
            beam["amplitude"]["value"] = 79
            beam["amplitude"]["raw"] = "79 reprocessed amplitude from local beam record"
    return revise_resource(store, "ping-03", payload, "local ping-03 quality correction", event_type="source.local_correction", event_payload={"scope": "ping-03 beam-1 correlation and amplitude", "dependency_propagation": "quality only"})


def run_transform_order_branch(store: Store) -> tuple[str, str]:
    payload = payload_for(store, store.current_version_id(), "transform-policy")
    payload["coordinate_transform_order"] = "ZXY_magnetic_heading"
    payload["revision_note"] = "rejected branch: altered earth-coordinate rotation order"
    version_id, run_id = revise_resource(store, "transform-policy", payload, "alternate coordinate transform order", actor="branch-reviewer", event_type="branch.transform_order", event_payload={"before": "ZYX_magnetic_heading", "after": "ZXY_magnetic_heading", "adopted_as_current": True})
    return version_id, run_id


def record_human_decision(store: Store, node_id: str, candidate_key: str, action: str, actor: str, base_version_id: str | None = None, note: str = "") -> dict[str, Any]:
    current_version_id = store.current_version_id()
    base_version_id = base_version_id or current_version_id
    if base_version_id != current_version_id:
        raise ConflictError({"error": "version_conflict", "base_version_id": base_version_id, "current_version_id": current_version_id, "resolution": "rebase decision after comparing branches"})
    run_id = store.current_run_id()
    payload = {"node_id": node_id, "candidate_key": candidate_key, "action": action, "note": note, "base_version_id": base_version_id}
    return store.add_event(
        "decision.recorded",
        payload,
        actor=actor,
        refs=[("derivation_node", node_id), ("version", base_version_id), ("derivation_run", run_id)],
        version_id=current_version_id,
        run_id=run_id,
    )


def resolve_conflict(store: Store, original_event_id: str, resolution: str, actor: str = "lead-hydrologist") -> dict[str, Any]:
    original = store.get_event(original_event_id)
    return store.add_event(
        "decision.conflict_resolved",
        {"original_event_id": original_event_id, "resolution": resolution, "original_payload": original["payload"]},
        actor=actor,
        refs=[("event", original_event_id), ("version", store.current_version_id()), ("derivation_run", store.current_run_id())],
        version_id=store.current_version_id(),
        run_id=store.current_run_id(),
    )


def run_full_demo(store: Store) -> dict[str, Any]:
    baseline_version, baseline_run, case = import_case(store)
    mount_version, mount_run = run_mount_revision(store)
    local_version, local_run = run_local_source_correction(store)
    branch_version, branch_run = run_transform_order_branch(store)
    node_id = None
    for node in store.list_nodes(branch_run):
        if node["subject_id"] == "ping-05" and node["stage"].startswith("instrument:") and len(node["results"]) > 1:
            node_id = node["node_id"]
            break
    first = record_human_decision(store, node_id, "three-beam-drop-3", "accept", "reviewer-a", baseline_version, "branch A chooses beam drop 3")
    try:
        record_human_decision(store, node_id, "three-beam-drop-4", "accept", "reviewer-b", baseline_version, "concurrent stale branch chooses beam drop 4")
    except ConflictError as exc:
        conflict_resolution = resolve_conflict(store, first["event_id"], "rebase reviewer-b onto current branch; retain both candidates and require explicit lead decision")
        conflict = exc.args[0]
    else:
        conflict = None
        conflict_resolution = None
    accepted_event = record_human_decision(store, node_id, "three-beam-drop-3", "accept", "lead-hydrologist", branch_version, "post-rebase selection; four-beam candidate remains stored")
    return {
        "case_id": case["case_id"],
        "baseline_version": baseline_version,
        "baseline_run": baseline_run,
        "mount_version": mount_version,
        "mount_run": mount_run,
        "local_version": local_version,
        "local_run": local_run,
        "branch_version": branch_version,
        "branch_run": branch_run,
        "ambiguous_node": node_id,
        "stale_conflict": conflict,
        "conflict_resolution_event": conflict_resolution and conflict_resolution["event_id"],
        "accepted_event": accepted_event["event_id"],
    }
