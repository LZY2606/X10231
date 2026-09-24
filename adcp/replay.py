"""Deterministic replay script for the offline ADCP case pack.

The replay physically passes through: single-beam failure, 360-degree
attitude crossing, time gap, three-beam multi-solution, mounting-angle
revision, transform-order(ing), magnetic-declination revision, invalid depth
cell, plus a concurrent-edit conflict. Every step is a new revision/event and
can be navigated back to restore the dependency versions visible at that
moment.
"""
from __future__ import annotations

from . import cases
from . import pipeline
from . import store as store_mod

STEPS = [
    ("import", "Baseline import with every source channel separated"),
    ("single_beam", "P1/c2 beam2 fails -> two legal 3-beam candidates"),
    ("attitude", "Heading crosses 360 deg; P2 blocked by the time gap"),
    ("three_beam_choice", "Adopt three_beam_lsq; w0 candidate kept"),
    ("mount", "Mounting heading offset revised (1.5 deg)"),
    ("order", "Transform order switched to RPH_ZYX"),
    ("declination", "Declination revised to -7.3 deg"),
    ("local_edit", "Local beam edit limited to one dependency subtree"),
    ("conflict", "Two users edit mount angle from the same version"),
    ("resolve", "Resolve conflict by keeping branch A"),
]

BOUNDARY_CASES = {
    "single_beam_failure": {"ping": "P1", "cell": 2, "beam": 2,
                            "expect": "ambiguous, 2 candidates"},
    "heading_crosses_360": {"pings": ["P0", "P1"], "expect":
                            "shortest-arc 359->1"},
    "time_gap": {"ping": "P2", "expect":
                 "no attitude interpolation; cells excluded"},
    "three_beam_multi_solution": {"ping": "P1", "cell": 2,
                                  "families": ["three_beam_lsq",
                                               "three_beam_wzero"]},
    "two_bad_beams": {"ping": "P1", "cell": 4, "expect":
                      "excluded, too_many_bad_beams"},
    "mount_revision": {"fact": "mount:heading_deg"},
    "transform_order": {"fact": "transform_order:ship_to_earth",
                        "orders": ["RDI_HPR", "RPH_ZYX"]},
    "invalid_depth_cell": {"ping": "P3", "cell": 4,
                           "expect": "excluded, bad_depth, raw value kept"},
    "low_correlation": {"ping": "P4", "cell": 1, "beam": 3},
    "low_amplitude": {"ping": "P4", "cell": 3, "beam": 2},
    "write_interruption": {"expect": "old rev or complete new rev only"},
    "old_schema_migration": {"from": 1, "to": 2},
    "concurrent_save": {"fact": "mount:heading_deg"},
}


def run_all(conn, up_to: int | None = None, progress=None) -> list[int]:
    """Execute the scripted replay. Returns created revision numbers."""
    created = []

    def record(rev):
        created.append(rev)
        if progress:
            progress(len(created), len(STEPS), STEPS[len(created) - 1][1])
        return rev

    n = len(STEPS) if up_to is None else min(up_to, len(STEPS))
    for i in range(n):
        if store_mod.current_rev(conn) < 0 and i == 0:
            record(pipeline.seed_baseline(conn))
            continue
        kind = STEPS[i][0]
        if kind == "single_beam":
            record(pipeline.toggle_single_beam(conn, None)[0])
        elif kind == "attitude":
            record(pipeline.change_attitude_gap(conn)[0])
        elif kind == "three_beam_choice":
            record(pipeline.choose_three_beam_candidate(
                conn, "three_beam_lsq")[0])
        elif kind == "mount":
            record(pipeline.revise_mount(conn)[0])
        elif kind == "order":
            record(pipeline.flip_transform_order(conn)[0])
        elif kind == "declination":
            record(pipeline.revise_declination(conn)[0])
        elif kind == "local_edit":
            record(pipeline.edit_local_source(conn)[0])
        elif kind == "conflict":
            record(pipeline.conflicting_mount_edits(conn)[0])
        elif kind == "resolve":
            record(pipeline.resolve_mount_conflict(conn)[0])
    return created


def reset_database(db_path: str):
    import os
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except FileNotFoundError:
            pass
    conn = store_mod.connect(db_path)
    return conn


def boundary_evidence(snap) -> dict:
    """Read back, from the current state, evidence for every boundary case."""
    objs = snap["objects"]
    cq = objs.get("cell_quality", {})

    def reason(ping, cell):
        o = cq.get(f"{ping}:c{cell}")
        return None if not o else {
            "excluded": o["status"] == "excluded",
            "reason": o["payload"].get("reason"),
            "detail": o["payload"].get("detail"),
            "depth_diagnostic": o["payload"].get("depth_diagnostic")}

    three = sorted(k for k in objs.get("earth_velocity", {})
                   if k.startswith("P1:c2:three_beam"))
    four = "P1:c2:four_beam" in objs.get("earth_velocity", {})
    return {
        "single_beam_failure": reason("P1", 2),
        "three_beam_candidates": three,
        "four_beam_at_three_beam_cell": four,
        "two_bad_beams": reason("P1", 4),
        "time_gap": reason("P2", 0),
        "invalid_depth_cell": reason("P3", 4),
        "low_correlation": reason("P4", 1),
        "low_amplitude": reason("P4", 3),
        "p2_profile_attitude_status":
            objs.get("profile", {}).get("P2", {}).get("payload", {})
            .get("attitude_status"),
    }
