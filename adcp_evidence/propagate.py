"""Incremental dependency-graph propagation between two source views."""
from __future__ import annotations

from typing import Any


def changed_records(old_sources: dict, new_sources: dict) -> dict[str, str]:
    """Map record_id -> change kind: added|modified|deleted|unchanged(filt)."""
    changes = {}
    for rid, rec in new_sources.items():
        if rid not in old_sources:
            changes[rid] = "added"
        elif old_sources[rid]["version"] != rec["version"]:
            changes[rid] = "modified"
    for rid in old_sources:
        if rid not in new_sources:
            changes[rid] = "deleted"
    return changes


def affected_slots(graph: dict[str, list[str]],
                   changed: dict[str, str]) -> set[str]:
    """Every slot whose dependency record changed, transitively one hop.

    Profile and other aggregate slots depend on cell slots, so recomputing all
    slots whose dependency *records* changed plus their dependents covers the
    fan-out. The engine recomputes cheaply; content addressing deduplicates.
    """
    record_changes = set(changed)
    direct = {
        slot for slot, deps in graph.items()
        if any(dep in record_changes for dep in deps)
    }
    dependents = {
        slot for slot, deps in graph.items() if any(d in direct for d in deps)
    }
    return direct | dependents


def explain_slot(
    slot: str,
    graph: dict[str, list[str]],
    changed: dict[str, str],
    old_fingerprint: str | None,
    new_fingerprint: str | None,
) -> dict:
    """Explain why a slot recomputed or stayed valid."""
    touched = [dep for dep in graph.get(slot, []) if dep in changed]
    if old_fingerprint == new_fingerprint:
        return {
            "slot": slot,
            "action": "still_valid",
            "reason": "all dependency versions, rules, geometry and "
                      "declination versions unchanged",
            "touched_dependencies": touched,
        }
    return {
        "slot": slot,
        "action": "recomputed",
        "reason": "fingerprint changed"
        if not touched else "changed dependencies: " + ",".join(sorted(touched)),
        "touched_dependencies": touched,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
    }


def propagation_plan(
    old_graph: dict, old_slots: dict, new_state: dict,
    old_sources: dict, new_sources: dict,
) -> dict:
    changed = changed_records(old_sources, new_sources)
    affected = affected_slots(old_graph, changed)
    explanations = []
    all_slots = set(old_slots) | set(new_state["slot_status"])
    for slot in sorted(all_slots):
        old_fp = (old_slots.get(slot) or {}).get("fingerprint")
        new_fp = (new_state["slot_status"].get(slot) or {}).get("fingerprint")
        if slot not in affected and old_fp is not None:
            explanations.append(explain_slot(
                slot, old_graph, changed, old_fp, new_fp))
    return {
        "changed_records": changed,
        "affected_slots": sorted(affected),
        "unaffected_explanations": explanations,
    }
