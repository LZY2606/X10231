"""Dependency-graph propagation.

Derived rows record the identities of the sources they were computed from.
When a new source version arrives, only the derived rows whose dependency
set intersects the changed sources (or whose fingerprint inputs changed)
are recomputed; everything else is carried into the new version unchanged
and can explain *why it is still valid*.
"""
from __future__ import annotations

import json

SOURCE_KEY_KINDS = ("beam_velocity", "correlation", "amplitude",
                    "bad_beam_flag", "heading", "pitch", "roll",
                    "declination", "mounting_angle", "bottom_track",
                    "ping_time")


def natural_key(row) -> tuple:
    return (row["kind"], row["ping"], row["cell"], row["beam"])


def changed_source_keys(old_rows, new_rows) -> set[tuple]:
    """Natural keys whose raw representation changed between versions."""
    old = {natural_key(r): r["raw"] for r in old_rows}
    new = {natural_key(r): r["raw"] for r in new_rows}
    changed = set()
    for key in set(old) | set(new):
        if old.get(key) != new.get(key):
            changed.add(key)
    return changed


def changed_source_ids(old_rows, changed_keys: set[tuple]) -> set[str]:
    return {r["id"] for r in old_rows if natural_key(r) in changed_keys}


def stale_rows(old_derived, changed_ids: set[str],
               param_stale_stages: set[str]) -> tuple[list, list]:
    """Split old derived rows into (stale, reusable)."""
    stale, reuse = [], []
    for row in old_derived:
        deps = set(json.loads(row["deps"]))
        if row["stage"] in param_stale_stages or deps & changed_ids:
            stale.append(row)
        else:
            reuse.append(row)
    return stale, reuse
