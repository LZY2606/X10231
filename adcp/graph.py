"""Dependency graph and incremental propagation.

Nodes are typed artifacts with stable ids and content fingerprints. Edges are
derivation dependencies. A node is valid iff its stored fingerprint matches the
fingerprint recomputed from its current dependencies; invalid nodes propagate
downstream so only the affected subgraph is recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class Node:
    id: str
    kind: str
    payload: Dict[str, Any]
    fingerprint: str
    deps: List[str] = field(default_factory=list)
    rule_version: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "payload": self.payload,
            "fingerprint": self.fingerprint, "deps": list(self.deps),
            "rule_version": self.rule_version,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Node":
        return Node(
            id=data["id"], kind=data["kind"], payload=data["payload"],
            fingerprint=data["fingerprint"], deps=list(data.get("deps", [])),
            rule_version=data.get("rule_version", ""),
        )


class DependencyGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.children: Dict[str, Set[str]] = {}
        self.compute: Dict[str, Callable[[], Dict[str, Any]]] = {}

    def add_source(self, node_id: str, kind: str, payload: Dict[str, Any],
                   fingerprint: str, rule_version: str = "") -> Node:
        node = Node(node_id, kind, payload, fingerprint, [], rule_version)
        self.nodes[node_id] = node
        self.children.setdefault(node_id, set())
        return node

    def upsert_derived(self, node_id: str, kind: str, deps: List[str],
                       payload: Dict[str, Any], fingerprint: str,
                       rule_version: str = "") -> Node:
        node = Node(node_id, kind, payload, fingerprint, list(deps), rule_version)
        self.nodes[node_id] = node
        for dep in deps:
            self.children.setdefault(dep, set()).add(node_id)
        self.children.setdefault(node_id, set())
        return node

    def descendants(self, node_ids: List[str]) -> Set[str]:
        affected: Set[str] = set()
        stack = list(node_ids)
        while stack:
            cur = stack.pop()
            for child in self.children.get(cur, ()):
                if child not in affected:
                    affected.add(child)
                    stack.append(child)
        return affected

    def validity_report(self, recompute: Callable[[Node], Optional[str]]) -> Dict[str, Any]:
        """Recompute fingerprints lazily; report why nodes are valid/stale."""
        report: Dict[str, Any] = {}
        order = list(self.nodes)
        for node_id in order:
            node = self.nodes[node_id]
            if not node.deps:
                report[node_id] = "valid"
                continue
            dep_state = [report.get(d) for d in node.deps]
            fresh = recompute(node)
            if any(s != "valid" for s in dep_state if s is not None):
                report[node_id] = "stale-dependency"
            elif fresh is None:
                report[node_id] = "cannot-recompute"
            elif fresh == node.fingerprint:
                report[node_id] = "valid"
            else:
                report[node_id] = "stale-input"
        return report

    def explain(self, node_id: str, recompute: Callable[[Node], Optional[str]]) -> Dict[str, Any]:
        node = self.nodes[node_id]
        fresh = recompute(node) if node.deps else node.fingerprint
        return {
            "id": node_id,
            "kind": node.kind,
            "deps": list(node.deps),
            "stored_fingerprint": node.fingerprint,
            "current_fingerprint": fresh,
            "state": ("source" if not node.deps else
                      "valid" if fresh == node.fingerprint else "needs-recompute"),
        }
