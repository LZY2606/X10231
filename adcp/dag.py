"""Dependency graph, content fingerprints and impact propagation.

Every derived node is identified by a stable *slot* (what it is: e.g.
``earth/p03``) and has a content identity ``val_id`` (what it is worth right
now).  Slots are stable across runs and export/import; ``val_id`` changes
exactly when the value or its inputs change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

NODE_KINDS = (
    "instrument",
    "three_beam",
    "ship",
    "earth",
    "earth_bottom",
    "cell_exclusion",
)


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha12(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def val_id(basis: str, value, provenance: dict) -> str:
    """Content identity of one derived value.

    Provenance is part of the identity: same numbers produced by a different
    geometry/declination/order are a *different* value, never the same one.
    """
    return "val_" + sha12(
        stable_json({"basis": basis, "value": value, "provenance": provenance})
    )


def fingerprint(
    rulebook_version: str,
    transform_order: str,
    geom_config_id: str,
    declination_id: str,
    mount_id: str,
    quality_policy_id: str,
) -> str:
    """Derived-layer fingerprint.

    Coordinate transform order, beam geometry and the declination *version* are
    all ingredients by requirement.
    """
    return "fp_" + sha12(
        stable_json(
            {
                "rulebook": rulebook_version,
                "order": transform_order,
                "geometry": geom_config_id,
                "declination": declination_id,
                "mount": mount_id,
                "quality": quality_policy_id,
            }
        )
    )


@dataclass
class Node:
    slot: str
    kind: str
    deps: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"slot": self.slot, "kind": self.kind, "deps": list(self.deps)}


class DependencyGraph:
    """Static slot graph for one derivation run."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def add(self, slot: str, kind: str, deps: Sequence[str]) -> Node:
        if kind not in NODE_KINDS:
            raise ValueError("unknown node kind %r" % kind)
        if slot in self.nodes:
            raise ValueError("duplicate node slot %r" % slot)
        node = Node(slot=slot, kind=kind, deps=list(deps))
        self.nodes[slot] = node
        return node

    def topo(self) -> List[str]:
        order: List[str] = []
        state: Dict[str, int] = {}

        def visit(n: str, stack: List[str]) -> None:
            if state.get(n) == 1:
                raise ValueError("cycle at %s: %s" % (n, stack))
            if state.get(n) == 2:
                return
            state[n] = 1
            for d in self.nodes[n].deps:
                if d in self.nodes:
                    visit(d, stack + [n])
            state[n] = 2
            order.append(n)

        for slot in sorted(self.nodes):
            visit(slot, [])
        return order

    def downstream(self, changed: Sequence[str]) -> List[str]:
        """All transitively affected slots, returned in eval order.

        Only the affected part of the graph is re-derived; every unaffected
        node can explain "still valid" by unchanged upstream val_ids.
        """
        rev: Dict[str, List[str]] = {}
        for slot, node in self.nodes.items():
            for d in node.deps:
                rev.setdefault(d, []).append(slot)
        hit = set(changed)
        stack = list(changed)
        while stack:
            n = stack.pop()
            for child in rev.get(n, []):
                if child not in hit:
                    hit.add(child)
                    stack.append(child)
        return [s for s in self.topo() if s in hit]

    def upstream(self, slot: str) -> List[str]:
        seen: Dict[str, int] = {}

        def visit(n: str) -> None:
            for d in self.nodes[n].deps:
                if d not in seen and d in self.nodes:
                    seen[d] = 1
                    visit(d)

        visit(slot)
        return list(seen.keys())
