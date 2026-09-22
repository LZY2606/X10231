"""Dependency graph between source channels and derived nodes.

Scope keys are stable, human-readable coordinates such as
"ping-03.cell-2" or "config".  Nodes in one scope depend on a fixed set of
source channels; changing a source propagates only to nodes whose
dependency set contains it.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

# node kinds, in evaluation order
NODE_KINDS = [
    "cell_validity",          # per ping/cell: depth gate + raw value diagnostics
    "beam_quality",           # per ping/cell/beam: corr/amp/bad tier
    "velocity_instrument",    # per ping/cell: 4-beam and/or 3-beam candidates
    "attitude",               # per ping sample: unwrap/interpolate attitude
    "velocity_ship",          # per ping/cell
    "velocity_earth",         # per ping/cell
    "bt_instrument",          # per ping bottom track
    "bt_ship",
    "bt_earth",
    "velocity_relative",      # water minus bottom, per ping/cell
]

CONFIG_KINDS = {
    "geometry": ["geometry"],
    "mounting": ["mounting"],
    "declination": ["declination"],
    "rules": ["rules"],
}

CELL_CHANNELS = ["radial", "corr", "amp", "bad", "depth"]


def beam_dependencies(scope: str, beam: int) -> List[str]:
    return ["%s.beam.%d.%s" % (scope, beam, c) for c in CELL_CHANNELS]


def node_dependencies(kind: str, scope: str) -> List[str]:
    """Stable dependency set for a node, expressed as source_id patterns.

    Patterns use the scope prefix; the evaluator resolves them against the
    concrete channels present in a version.
    """
    if kind == "cell_validity":
        return ["%s.cell.depth" % scope]
    if kind == "beam_quality":
        return ["%s.beam.%%.corr" % scope, "%s.beam.%%.amp" % scope,
                "%s.beam.%%.bad" % scope]
    if kind == "velocity_instrument":
        return ["%s.beam.%%.radial" % scope, "%s.beam.%%.corr" % scope,
                "%s.beam.%%.amp" % scope, "%s.beam.%%.bad" % scope,
                "%s.cell.depth" % scope, "geometry", "rules"]
    if kind == "attitude":
        return ["attitude.%%.time", "attitude.%%.heading",
                "attitude.%%.pitch", "attitude.%%.roll", "rules"]
    if kind == "velocity_ship":
        return ["node:velocity_instrument:%s" % scope, "mounting"]
    if kind == "velocity_earth":
        return ["node:velocity_ship:%s" % scope,
                "node:attitude:%s" % scope, "declination"]
    if kind == "bt_instrument":
        return ["%s.bt.beam.%%.radial" % scope,
                "%s.bt.beam.%%.corr" % scope,
                "%s.bt.beam.%%.amp" % scope,
                "%s.bt.beam.%%.bad" % scope, "geometry", "rules"]
    if kind == "bt_ship":
        return ["node:bt_instrument:%s" % scope, "mounting"]
    if kind == "bt_earth":
        return ["node:bt_ship:%s" % scope,
                "node:attitude:%s" % scope, "declination"]
    if kind == "velocity_relative":
        return ["node:velocity_earth:%s" % scope,
                "node:bt_earth:%s" % scope]
    raise ValueError(kind)


def affected_scopes(changed_sources: List[str], nodes: List[dict]) -> Set[str]:
    """Given changed source ids and existing nodes, return affected scopes.

    A node is affected if any changed source matches one of its dependency
    patterns.  Nodes that do not match are reported as still valid during
    propagation (see engine.evaluate).
    """
    import fnmatch
    affected: Set[str] = set()
    for n in nodes:
        patterns = node_dependencies(n["kind"].split(":")[0]
                                     if False else n.get("kind", ""),
                                     n["scope_key"])
    # Build a flat index instead: pattern -> scopes
    return affected


def matches(pattern: str, source_id: str) -> bool:
    """Match a dependency pattern against a concrete source id.

    '%%' is the multi-segment wildcard for beam/sample indexes; ':' node
    references are handled by the caller.
    """
    import fnmatch
    return fnmatch.fnmatchcase(source_id, pattern.replace("%%", "*"))
