"""Turn source snapshots into derivation nodes, events and new versions.

Each value displayed on the review page gets a stable slot id plus a
fingerprinted value id, so export/delete/import preserves identity, order and
provenance.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fingerprint import fingerprint, make_stable_id
from .graph import DependencyGraph
from .pipeline import recompute
from .store import VersionStore


def _event(event_id: str, event_type: str, scope: str, affected: Sequence[str],
           status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "scope": scope,
        "affected": list(affected),
        "status": status,
        "payload": payload,
    }


def _source_nodes(sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    transducer = sources.get("transducer") or {}
    nodes.append({
        "stable_id": "config/transducer",
        "node_kind": "source_transducer",
        "parents": [],
        "rule": "transducer_config",
        "value": transducer,
        "value_fingerprint": fingerprint(["transducer", transducer]),
    })
    for index, attitude in enumerate(sources.get("attitudes", [])):
        stable_id = make_stable_id(["attitude", index])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": "source_attitude",
            "parents": [],
            "rule": "attitude_record",
            "value": attitude,
            "value_fingerprint": fingerprint(["attitude", attitude]),
        })
    for declination in sources.get("declinations", []):
        stable_id = make_stable_id(["declination", declination["declination_version"]])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": "source_declination",
            "parents": [],
            "rule": "declination_record",
            "value": declination,
            "value_fingerprint": fingerprint(["declination", declination]),
        })
    for mounting in sources.get("mountings", []):
        stable_id = make_stable_id(["mounting", mounting["mounting_version"]])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": "source_mounting",
            "parents": [],
            "rule": "mounting_record",
            "value": mounting,
            "value_fingerprint": fingerprint(["mounting", mounting]),
        })
    for cell in sources.get("cells", []):
        stable_id = make_stable_id(["celldef", cell["cell_index"]])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": "source_cell",
            "parents": [],
            "rule": "cell_definition",
            "value": cell,
            "value_fingerprint": fingerprint(["cell", cell]),
        })
    for ping in sources.get("pings", []):
        ping_id = ping["ping_id"]
        beams = ping.get("beams", [])
        for beam in beams:
            index = beam.get("beam_index")
            for field_name, rule, kind in (
                ("velocity", "beam_velocity_raw", "source_beam_velocity"),
                ("correlation", "correlation_raw", "source_correlation"),
                ("amplitude", "amplitude_raw", "source_amplitude"),
                ("bad_flag", "bad_beam_flag", "source_bad_flag"),
            ):
                stable_id = make_stable_id(["src", ping_id, index, field_name])
                nodes.append({
                    "stable_id": stable_id,
                    "node_kind": kind,
                    "parents": [],
                    "rule": rule,
                    "value": beam[field_name] if field_name in beam else None,
                    "value_fingerprint": fingerprint([field_name, beam.get(field_name)]),
                })
        for cell_index, cell_beams in (ping.get("cell_beams") or {}).items():
            for beam in cell_beams:
                index = beam.get("beam_index")
                for field_name, rule, kind in (
                    ("velocity", "beam_velocity_raw", "source_beam_velocity"),
                    ("correlation", "correlation_raw", "source_correlation"),
                    ("amplitude", "amplitude_raw", "source_amplitude"),
                    ("bad_flag", "bad_beam_flag", "source_bad_flag"),
                ):
                    stable_id = make_stable_id(
                        ["src", ping_id, f"cell{cell_index}", index, field_name]
                    )
                    nodes.append({
                        "stable_id": stable_id,
                        "node_kind": kind,
                        "parents": [],
                        "rule": rule,
                        "value": beam.get(field_name),
                        "value_fingerprint": fingerprint([field_name, beam.get(field_name)]),
                    })
    for bottom in sources.get("bottom_tracks", []):
        stable_id = make_stable_id(["bottomtrack", bottom["ping_id"]])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": "source_bottom_track",
            "parents": [],
            "rule": "bottom_track_raw",
            "value": bottom,
            "value_fingerprint": fingerprint(["bottomtrack", bottom]),
        })
    return nodes


def _derived_nodes(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], DependencyGraph, Dict[str, str]]:
    nodes: List[Dict[str, Any]] = []
    graph = DependencyGraph()
    value_ids: Dict[str, str] = {}

    def add(stable_id: str, kind: str, parents: Sequence[str], rule: str, value: Any) -> None:
        value_fpr = fingerprint([rule, parents, value])
        nodes.append({
            "stable_id": stable_id,
            "node_kind": kind,
            "parents": list(parents),
            "rule": rule,
            "value": value,
            "value_fingerprint": value_fpr,
        })
        graph.add_node(stable_id, parents, rule)
        value_ids[stable_id] = value_fpr

    for ping in result["pings"]:
        ping_id = ping["ping_id"]
        attitude_status = ping["attitude"]["status"]
        attitude_id = make_stable_id(["attitude-at-ping", ping_id])
        add(attitude_id, "attitude_at_ping", ["config/transducer"],
            "attitude_interpolation", ping["attitude"])
        quality_id = make_stable_id(["quality", ping_id])
        add(quality_id, "beam_quality", [], "quality_hierarchy",
            {"good": ping["quality"]["good"], "suspect": ping["quality"]["suspect"],
             "hard_bad": ping["quality"]["hard_bad"], "reasons": ping["quality"]["reasons"]})

        if ping.get("bottom_track"):
            add(make_stable_id(["bottom-earth", ping_id]), "bottom_track",
                [make_stable_id(["bottomtrack", ping_id]), attitude_id],
                "bottom_track_transform", ping["bottom_track"])

        for cell in ping["cells"]:
            cell_index = cell["cell_index"]
            slot = make_stable_id(["vec", ping_id, cell_index])
            validity_id = make_stable_id(["validity", ping_id, cell_index])
            validity_parents = [make_stable_id(["celldef", cell_index])]
            if make_stable_id(["bottomtrack", ping_id]):
                validity_parents.append(make_stable_id(["bottomtrack", ping_id]))
            add(validity_id, "cell_validity", validity_parents,
                "depth_cell_validation", cell["validity"])

            if cell["status"] == "excluded":
                add(slot, "excluded_cell", [validity_id], "exclude_depth_cell",
                    {"status": "excluded", "reason": cell["validity"]["reason"]})
                continue

            add(make_stable_id(["solve", ping_id, cell_index]), "beam_solve",
                [quality_id], "beam_inversion",
                {"status": cell["status"], "reason": cell.get("solve_reason", "")})
            for candidate in cell["candidates"]:
                cid = candidate["candidate_id"]
                inst_id = make_stable_id(["inst", ping_id, cell_index, cid])
                add(inst_id, "instrument_velocity", [quality_id],
                    "beam_to_instrument", candidate["instrument_velocity"])
                ship_id = make_stable_id(["ship", ping_id, cell_index, cid])
                add(ship_id, "ship_velocity", [inst_id],
                    "instrument_to_ship", candidate["ship_velocity"])
                earth_id = make_stable_id(["earth", ping_id, cell_index, cid])
                add(earth_id, "earth_velocity", [ship_id, attitude_id],
                    "ship_to_earth", candidate["earth_velocity"])
                graph.add_node(slot, [earth_id, ship_id, inst_id], "candidate_set")
            value_ids.setdefault(slot, fingerprint([c["candidate_id"] for c in cell["candidates"]]))
            nodes.append({
                "stable_id": slot,
                "node_kind": "candidate_set",
                "parents": [],
                "rule": "candidate_enumeration",
                "value": {
                    "status": cell["status"],
                    "candidate_ids": [c["candidate_id"] for c in cell["candidates"]],
                },
                "value_fingerprint": value_ids[slot],
            })
    return nodes, graph, value_ids


def build_events(sources: Dict[str, Any], result: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.append(_event(
        f"evt-{label}-import", "source_import", "dataset",
        ["config/transducer"], "accepted",
        {"separated_kinds": [
            "beam_velocity", "correlation", "amplitude", "bad_beam_flag",
            "heading_pitch_roll", "declination", "installation_angles"],
         "ping_count": len(sources.get("pings", [])),
         "geometry_version": result["geometry_version"],
         "transform_order": result["transform_order"]},
    ))
    for ping in result["pings"]:
        ping_id = ping["ping_id"]
        bad_beams = [i for i, bad in enumerate(ping["quality"]["hard_bad"]) if bad]
        suspect_beams = [i for i, bad in enumerate(ping["quality"]["suspect"]) if bad]
        if bad_beams:
            events.append(_event(
                f"evt-beamfail-{ping_id}", "single_beam_failure", f"ping/{ping_id}",
                [make_stable_id(["src", ping_id, b, "velocity"]) for b in bad_beams],
                "applied", {"bad_beams": bad_beams,
                            "reasons": ping["quality"]["reasons"]}))
        attitude = ping["attitude"]
        if attitude["status"] == "gap_refused":
            events.append(_event(
                f"evt-gap-{ping_id}", "attitude_time_gap", f"ping/{ping_id}",
                [make_stable_id(["attitude-at-ping", ping_id])],
                "earth_withheld", attitude))
        elif attitude["status"] == "ok" and attitude.get("mode") == "interpolated":
            events.append(_event(
                f"evt-attinterp-{ping_id}", "attitude_interpolated", f"ping/{ping_id}",
                [make_stable_id(["attitude-at-ping", ping_id])],
                "applied", {"mode": "interpolated", "heading_unwrapped": True}))
        for cell in ping["cells"]:
            if cell["status"] == "ambiguous_multi_solution":
                events.append(_event(
                    f"evt-multi-{ping_id}-{cell['cell_index']}",
                    "three_beam_multi_solution",
                    f"ping/{ping_id}/cell/{cell['cell_index']}",
                    [make_stable_id(["vec", ping_id, cell["cell_index"]])],
                    "unresolved_all_candidates_retained",
                    {"candidate_ids": [c["candidate_id"] for c in cell["candidates"]],
                     "rule": "three_and_four_beam_not_merged"}))
            elif cell["status"] == "three_beam_determined":
                events.append(_event(
                    f"evt-three-{ping_id}-{cell['cell_index']}",
                    "three_beam_solution",
                    f"ping/{ping_id}/cell/{cell['cell_index']}",
                    [make_stable_id(["vec", ping_id, cell["cell_index"]])],
                    "determined_unique",
                    {"candidate_ids": [c["candidate_id"] for c in cell["candidates"]]}))
            if cell["status"] == "excluded":
                events.append(_event(
                    f"evt-excl-{ping_id}-{cell['cell_index']}",
                    "invalid_depth_cell",
                    f"ping/{ping_id}/cell/{cell['cell_index']}",
                    [make_stable_id(["validity", ping_id, cell["cell_index"]])],
                    "excluded_evidence_retained", cell["validity"]))
    return events


def derive_version_payload(sources: Dict[str, Any], label: str) -> Dict[str, Any]:
    result = recompute(sources)
    source_nodes = _source_nodes(sources)
    derived, graph, value_ids = _derived_nodes(result)
    events = build_events(sources, result, label)
    return {
        "result": result,
        "nodes": source_nodes + derived,
        "events": events,
        "graph": graph,
        "value_ids": value_ids,
    }
