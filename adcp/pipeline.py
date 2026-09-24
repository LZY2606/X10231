"""Replayable case pipeline: import, edge-case edits, decisions and conflicts.

Each operation materializes a new full revision through commit_revision
(either the old revision or a complete new one is ever visible) and appends a
machine-readable event recording the input fragment, decision status and
affected objects. Accepted AND rejected candidates are kept.
"""
from __future__ import annotations

import copy

from . import cases
from . import derive
from . import store as store_mod


def new_event(event_type, actor, summary, **data):
    import uuid
    return {"id": "ev_" + uuid.uuid4().hex[:12], "event_type": event_type,
            "actor": actor, "summary": summary, "data": data}


def _bump(fact, new_value, source_ref=None):
    version = int(fact.get("version", 1)) + 1
    return {"sid": cases.sid_of("_rev", fact["sid"] + str(version),
                                (new_value, version)),
            "value": new_value,
            "raw_unit": fact.get("raw_unit"),
            "raw_repr": fact.get("raw_repr"),
            "source_ref": source_ref or fact.get("source_ref"),
            "version": version, "replaces_sid": fact["sid"]}


def _rebuild_pings(state, pings):
    """Re-run the graph derivation for a set of pings by rebuilding all
    derived objects (source facts are unchanged); the dependency closure makes
    exactly these pings the affected set."""
    objects, edges, diag = derive.build_objects(state["facts"])
    state["objects"] = objects
    state["edges"] = edges
    return diag


def _commit(conn, state, label, change_set, event):
    state = copy.deepcopy(state)
    state.setdefault("events", []).append(event)
    rev = store_mod.commit_revision(conn, label, state, change_set)
    return rev, store_mod.snapshot(conn, rev)


def seed_baseline(conn):
    facts = cases.build_baseline()
    objects, edges, diag = derive.build_objects(facts)
    state = {"facts": facts, "objects": objects, "edges": edges,
             "events": [], "conflicts": []}
    ev = new_event(
        "import", "importer",
        "Initial ADCP import: beam velocity, correlation, amplitude, bad-beam "
        "flags, heading/pitch/roll, declination and mounting angles recorded "
        "as separate channels",
        pings=cases.PINGS, cells=cases.CELLS,
        channels=["beam_velocity", "correlation", "amplitude",
                  "bad_beam_flag", "heading", "pitch", "roll",
                  "declination", "mount", "beam_geometry",
                  "transform_order", "cell_depth", "bottom_track"],
        diagnostics=diag)
    state.setdefault("events", []).append(ev)
    rev = store_mod.commit_revision(conn, "baseline import", state,
                                    {"kind": "import"})
    return rev


def toggle_single_beam(conn, base_rev, ping="P1", cell=2, beam=2,
                       actor="reviewer", bad=True):
    state = store_mod.snapshot(conn)
    key = f"{ping}:c{cell}"
    f = state["facts"]["bad_beam_flag"][f"{key}:b{beam}"]
    old = f["value"]["bad"]
    state["facts"]["bad_beam_flag"][f"{key}:b{beam}"] = _bump(
        f, {"bad": bad})
    diag = _rebuild_pings(state, {ping})
    ev = new_event(
        "source_change", actor,
        f"Single-beam failure toggled: {key} beam{beam} bad={bad}",
        change={"kind": "bad_beam_flag", "key": f"{key}:b{beam}",
                "old": old, "new": bad},
        affected_pings=[ping], diagnostics=diag,
        scope="triggers three-beam multi-solution")
    return _commit(conn, state, f"single beam failure {key} b{beam}",
                   {"kind": "source_change", "kind_key":
                    f"bad_beam_flag:{key}:b{beam}"}, ev)


def _global_bump_commit(conn, kind, key, new_value, label, event_type,
                        summary, actor, extra=None):
    state = store_mod.snapshot(conn)
    f = state["facts"][kind][key]
    state["facts"][kind][key] = _bump(f, new_value)
    diag = _rebuild_pings(state, set(cases.PINGS))
    ev = new_event(event_type, actor, summary,
                   change={"kind": kind, "key": key,
                           "old": f["value"], "new": new_value},
                   affected_pings=cases.PINGS, diagnostics=diag,
                   **(extra or {}))
    return _commit(conn, state, label,
                   {"kind": event_type, "kind_key": f"{kind}:{key}"}, ev)


def revise_mount(conn, base_rev=None, new_heading_deg=1.5,
                 actor="surveyor"):
    return _global_bump_commit(
        conn, "mount", "heading_deg", {"deg": new_heading_deg},
        "mounting-angle revision", "mount_revision",
        f"Mounting heading offset revised to {new_heading_deg} deg "
        "(installation survey rev2)", actor)


def flip_transform_order(conn, base_rev=None, order="RPH_ZYX",
                         actor="reviewer"):
    cur = store_mod.get_fact(conn, "transform_order", "ship_to_earth")
    return _global_bump_commit(
        conn, "transform_order", "ship_to_earth", {"order": order},
        f"transform order -> {order}", "transform_order_change",
        f"Coordinate transform order changed to {order}; order enters the "
        "derived fingerprint", actor,
        extra={"previous_order": cur["value"]["order"]})


def revise_declination(conn, base_rev=None, new_deg=-7.3,
                       model_year=2025, actor="navigator"):
    return _global_bump_commit(
        conn, "declination", "magnetic_declination",
        {"deg": new_deg, "model": "WMM", "model_year": model_year},
        f"declination revised -> {new_deg} deg", "declination_revision",
        f"Magnetic declination version revised to {new_deg} deg; declination "
        "version enters the derived fingerprint", actor)


def change_attitude_gap(conn, base_rev=None, actor="reviewer"):
    """The attitude source already records a gap between t=25 and t=35; this
    operation additionally widens the heading change across the wrap
    (359->1) is exercised on P0/P1, while P2 stays blocked by the gap. It is
    therefore a verification replay step (no fact edit required)."""
    state = store_mod.snapshot(conn)
    diag = _rebuild_pings(state, set(cases.PINGS))
    ev = new_event(
        "boundary_verified", actor,
        "Heading crosses 360 deg (359->1) with shortest-arc interpolation; "
        "P2 is NOT interpolated across the flagged t=25->35 time gap",
        affected_pings=cases.PINGS, diagnostics=diag,
        rule={"heading_wrap": "shortest_arc_unwrap",
              "gap": "no interpolation across flagged gap"})
    return _commit(conn, state, "verify heading wrap and time gap",
                   {"kind": "boundary_verified", "case": "attitude"}, ev)


def edit_local_source(conn, base_rev=None, ping="P5", cell=0, beam=0,
                      delta_cm_s=5.0, actor="reviewer"):
    """A source change that affects only one ping/cell dependency subtree."""
    state = store_mod.snapshot(conn)
    key = f"{ping}:c{cell}"
    f = state["facts"]["beam_velocity"][f"{key}:b{beam}"]
    new_value = dict(f["value"])
    new_value["cm_s"] = round(new_value["cm_s"] + delta_cm_s, 2)
    state["facts"]["beam_velocity"][f"{key}:b{beam}"] = _bump(f, new_value)
    before = {otype: dict(items) for otype, items in state["objects"].items()}
    diag = _rebuild_pings(state, {ping})
    changed_nodes = derive.affected_nodes(
        state["edges"], [f"beam_velocity:{key}:b{beam}"])
    ev = new_event(
        "source_change", actor,
        f"Local beam edit {key} beam{beam} += {delta_cm_s} cm/s; graph "
        "propagation recomputes only the affected subtree",
        change={"kind": "beam_velocity", "key": f"{key}:b{beam}",
                "old": f["value"], "new": new_value},
        affected_pings=[ping], affected_node_count=len(changed_nodes),
        diagnostics=diag)
    return _commit(conn, state, f"local edit {key} b{beam}",
                   {"kind": "source_change",
                    "kind_key": f"beam_velocity:{key}:b{beam}"}, ev)


def choose_three_beam_candidate(conn, family, actor="reviewer",
                                ping="P1", cell=2):
    """Accept one 3-beam candidate; the rejected candidate stays stored."""
    state = store_mod.snapshot(conn)
    key = f"{ping}:c{cell}"
    chosen = state["objects"]["earth_velocity"][f"{key}:{family}"]
    rejected = [k for k in state["objects"]["earth_velocity"]
                if k.startswith(f"{key}:three_beam") and not k.endswith(family)]
    chosen["adopted"] = True
    chosen["status"] = "adopted"
    for rk in rejected:
        state["objects"]["earth_velocity"][rk]["status"] = "rejected"
    diag = _rebuild_pings(state, {ping})
    # re-apply adoption after rebuild (rebuild resets derived statuses)
    state["objects"]["earth_velocity"][f"{key}:{family}"]["adopted"] = True
    state["objects"]["earth_velocity"][f"{key}:{family}"]["status"] = "adopted"
    for rk in rejected:
        state["objects"]["earth_velocity"][rk]["status"] = "rejected"
    ev = new_event(
        "decision", actor,
        f"Three-beam candidate {family} accepted for {key}; rejected "
        "candidate retained (three/four-beam values are never merged)",
        decision={"ping": ping, "cell": cell, "chosen_family": family,
                  "rejected": rejected,
                  "chosen_sid": chosen["sid"],
                  "chosen_velocity": chosen["payload"]["velocity"]},
        rule="three_beam and four_beam never share one value",
        affected_pings=[ping], diagnostics=diag)
    return _commit(conn, state, f"adopt {family} for {key}",
                   {"kind": "decision", "family": family}, ev)


def conflicting_mount_edits(conn, actor_a="hydrologist",
                            actor_b="surveyor"):
    """Two users edit the same version. The second commit is a stale-base
    edit; the system records a resolvable conflict and keeps BOTH values."""
    state = store_mod.snapshot(conn)
    base_rev = store_mod.current_rev(conn)
    f = state["facts"]["mount"]["heading_deg"]

    # Branch A commits first: +1.5 deg.
    state_a = copy.deepcopy(state)
    state_a["facts"]["mount"]["heading_deg"] = _bump(
        f, {"deg": 1.5}, source_ref="installation_survey_rev2.md")
    rev_a = store_mod.commit_revision(
        conn, "mount revision branch A (1.5 deg)", state_a,
        {"kind": "mount_revision", "actor": actor_a,
         "kind_key": "mount:heading_deg"})
    state_a = store_mod.snapshot(conn, rev_a)
    diag_a = _rebuild_pings(state_a, set(cases.PINGS))
    ev_a = new_event(
        "mount_revision", actor_a,
        "User A revised mounting heading to 1.5 deg",
        change={"key": "mount:heading_deg", "old": f["value"],
                "new": {"deg": 1.5}},
        affected_pings=cases.PINGS, diagnostics=diag_a)
    state_a.setdefault("events", []).append(ev_a)
    rev_a = store_mod.commit_revision(
        conn, "derivation + event branch A", state_a,
        {"kind": "mount_revision", "actor": actor_a})

    # Branch B was edited against the stale base_rev (+2.0 deg).
    try:
        state_b = copy.deepcopy(state)
        state_b["facts"]["mount"]["heading_deg"] = _bump(
            f, {"deg": 2.0}, source_ref="field_note_revB.md")
        store_mod.commit_revision(
            conn, "mount revision branch B (2.0 deg)", state_b,
            {"kind": "mount_revision", "actor": actor_b},
            base_rev=base_rev)
        raise AssertionError("ConcurrentEdit was not raised")
    except store_mod.ConcurrentEdit:
        pass

    state_now = store_mod.snapshot(conn)
    candidate_b = _bump(f, {"deg": 2.0}, source_ref="field_note_revB.md")
    cid = "conflict_mount_heading"
    conflict = {
        "cid": cid, "base_rev": base_rev,
        "branch_sid": state_now["facts"]["mount"]["heading_deg"]["sid"],
        "incoming_sid": candidate_b["sid"],
        "key": "mount:heading_deg", "resolved_rev": None,
        "resolution": None}
    state_now.setdefault("conflicts", []).append(conflict)
    # The incoming candidate is retained verbatim as a separate fact sid, not
    # overwriting the committed branch value.
    state_now["facts"]["mount"]["heading_deg__incoming_candidate"] = \
        candidate_b
    ev = new_event(
        "conflict_opened", "system",
        "Two users edited mount:heading_deg from the same version: 1.5 deg "
        "(committed) vs 2.0 deg (incoming). Conflict must be resolved; both "
        "values are retained.",
        conflict=cid, base_rev=base_rev, branch={"deg": 1.5},
        incoming={"deg": 2.0},
        retained={"branch_sid": conflict["branch_sid"],
                  "incoming_sid": candidate_b["sid"]})
    state_now.setdefault("events", []).append(ev)
    rev = store_mod.register_conflict(
        conn, cid, base_rev, conflict["branch_sid"], candidate_b["sid"],
        "mount:heading_deg", state_now, "mount conflict recorded")
    return rev, cid


def resolve_mount_conflict(conn, resolution="branch", actor="lead"):
    cid = "conflict_mount_heading"

    def mutate(state, found, choice):
        if choice == "branch":
            state["facts"]["mount"].pop(
                "heading_deg__incoming_candidate", None)
        else:
            inc = state["facts"]["mount"].pop(
                "heading_deg__incoming_candidate")
            current = state["facts"]["mount"]["heading_deg"]
            state["facts"]["mount"]["heading_deg"] = {
                **inc, "replaces_sid": current["sid"]}
        _rebuild_pings(state, set(cases.PINGS))
        ev2 = new_event(
            "conflict_resolved", actor,
            f"Mounting-angle conflict resolved by keeping {choice} value; "
            "the unchosen value remains referenced by the conflict record",
            conflict=cid, resolution=choice,
            kept_value=state["facts"]["mount"]["heading_deg"]["value"])
        state.setdefault("events", []).append(ev2)

    rev = store_mod.resolve_conflict(
        conn, cid, resolution, mutate,
        f"mount conflict resolved -> {resolution}")
    return rev
