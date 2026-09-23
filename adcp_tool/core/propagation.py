"""Dependency-graph propagation with validity explanations.

Given the node sets of the old derivation and a freshly built candidate, the
module reports exactly which nodes changed and, for every node, why it is
still valid (identical content fingerprint) or why it had to recompute (a
direct input or a transitive dependency changed).
"""
from __future__ import annotations

from .fingerprint import fingerprint, short

PROP_RULE = "propagation-v1"


def fp_index(nodes):
    return {n["node_id"]: n["fingerprint"] for n in nodes}


def node_index(nodes):
    return {n["node_id"]: n for n in nodes}


def _direct_inputs(node):
    return list(node.get("depends_on", []))


def propagate(old_nodes, new_nodes):
    old = node_index(old_nodes)
    new = node_index(new_nodes)
    old_fp = fp_index(old_nodes)
    new_fp = fp_index(new_nodes)

    changed = {}   # node_id -> reason
    for nid, node in new.items():
        if nid not in old:
            changed[nid] = "new"
            continue
        if old_fp[nid] != new_fp[nid]:
            changed[nid] = "content-changed"
    for nid in old:
        if nid not in new:
            changed[nid] = "removed"

    # Transitive closure over the new graph edges.
    affected = set()
    queue = list(changed.keys())
    while queue:
        cur = queue.pop()
        if cur in affected and cur not in changed:
            pass
        affected.add(cur)
        for nid, node in new.items():
            if nid in affected:
                continue
            if cur in _direct_inputs(node):
                affected.add(nid)
                queue.append(nid)

    report = {"changed": sorted(changed), "recomputed": [], "unchanged": [], "reasons": {}}
    for nid in sorted(new):
        node = new[nid]
        if nid in changed and old_fp.get(nid) != new_fp[nid]:
            report["recomputed"].append(nid)
        elif nid in affected:
            report["recomputed"].append(nid)
        else:
            report["unchanged"].append(nid)

    for nid, node in new.items():
        if nid not in old:
            reason = "new node produced by this derivation"
        elif nid in changed:
            direct = [d for d in _direct_inputs(node) if d in changed]
            reason = "direct input changed: %s" % (direct or "its own basis values")
        elif nid in affected:
            via = [d for d in _direct_inputs(node) if d in affected]
            reason = "recomputed because upstream dependency changed: %s" % via
        else:
            reason = "still valid: fingerprint %s unchanged and no upstream change" % short(
                node["fingerprint"]
            )
        report["reasons"][nid] = reason

    roots = sorted(
        nid for nid, why in changed.items() if why in ("content-changed", "new", "removed")
    )
    report["roots"] = roots
    report["summary_fingerprint"] = fingerprint(
        PROP_RULE, {"roots": roots, "n_recomputed": len(report["recomputed"])}
    )
    return report
