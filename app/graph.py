"""Dependency graph + incremental propagation.

A derivation is a graph of typed nodes. Each node has a content fingerprint
over (node type, upstream fingerprints, rule/geom/decl/install versions).
When a source changes we compute the affected ping set; nodes outside it keep
the SAME fingerprint and are reused (recording reused_from), nodes inside are
recomputed. This both limits work and lets the UI state *why* a result is
still valid (identical fingerprint) or required recompute (its basis changed).
"""
from __future__ import annotations

# node type -> node types it depends on
NODE_EDGES = {
    "beam_quality": [],
    "attitude": [],
    "cell_status": [],
    "water_instrument": ["beam_quality"],
    "water_ship": ["water_instrument", "attitude"],
    "water_earth": ["water_ship", "attitude"],
    "bottom_instrument": ["beam_quality"],
    "bottom_ship": ["bottom_instrument", "attitude"],
    "bottom_earth": ["bottom_ship", "attitude"],
}

# which raw source changes can affect which node types
SOURCE_TO_NODES = {
    "beam_vel": ["water_instrument", "bottom_instrument"],
    "correlation": ["beam_quality"],
    "amplitude": ["beam_quality"],
    "badbeam": ["beam_quality"],
    "attitude": ["attitude"],
    "declination": ["attitude", "water_earth", "bottom_earth"],
    "installation": ["water_ship", "bottom_ship"],
    "geom": ["water_instrument", "bottom_instrument"],
    "cells": ["cell_status"],
    "bottom_range": ["bottom_instrument"],
}

# nodes that are not ping scoped (attitude rule uses declination globally)
GLOBAL_BASIS_TABLES = {"declination", "installation", "geom"}


def downstream(node_types: set[str]) -> set[str]:
    out = set()
    stack = list(node_types)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        for child, deps in NODE_EDGES.items():
            if n in deps:
                stack.append(child)
    return out


def affected_node_types(changed_tables: set[str]) -> set[str]:
    roots: set[str] = set()
    for t in changed_tables:
        roots.update(SOURCE_TO_NODES.get(t, []))
    return downstream(roots)


def affected_pings(table: str, changed_pings: set[str] | None) -> set[str] | str:
    """Return affected ping set, or 'ALL' for a global parameter change."""
    if table in GLOBAL_BASIS_TABLES:
        return "ALL"
    return set(changed_pings or [])
