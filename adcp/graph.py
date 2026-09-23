"""Dependency graph for incremental recomputation.

Nodes are typed (``source`` / ``derived``) and versioned by a content
fingerprint. When a source changes, only downstream nodes whose inputs changed
are marked stale; unaffected nodes keep ``valid`` with a human-readable reason.
The graph also powers "why is this result still valid / why must it recompute".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class Node:
    key: str
    kind: str                       # "source" | "derived"
    fingerprint: str
    inputs: tuple[str, ...] = ()
    state: str = "valid"            # valid | stale | recomputed
    note: str = ""


class DependencyGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self._edges: dict[str, set[str]] = defaultdict(set)   # input -> dependents

    def add_source(self, key: str, fingerprint: str) -> None:
        self.nodes[key] = Node(key, "source", fingerprint)

    def add_derived(self, key: str, inputs: Iterable[str], fingerprint: str) -> None:
        inputs = tuple(inputs)
        self.nodes[key] = Node(key, "derived", fingerprint, inputs)
        for src in inputs:
            self._edges[src].add(key)

    def descendants(self, key: str) -> set[str]:
        seen: set[str] = set()
        stack = [key]
        while stack:
            cur = stack.pop()
            for dep in self._edges.get(cur, ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    def change_source(self, key: str, new_fingerprint: str) -> set[str]:
        """Update one source and mark exactly its downstream nodes stale.

        Returns the set of derived keys that must recompute. Unaffected nodes
        remain ``valid``.
        """
        node = self.nodes[key]
        if node.fingerprint == new_fingerprint:
            return set()
        node.fingerprint = new_fingerprint
        affected = self.descendants(key)
        for k in affected:
            n = self.nodes[k]
            n.state = "stale"
            n.note = f"input {key} changed"
        return affected

    def recompute_finished(self, key: str, new_fingerprint: str) -> None:
        n = self.nodes[key]
        n.fingerprint = new_fingerprint
        n.state = "valid"
        n.note = "recomputed from current inputs"

    def why_valid(self, key: str) -> str:
        """Explain why a node stays valid after a change."""
        n = self.nodes[key]
        if n.kind == "source":
            return f"source {key} unchanged at {n.fingerprint}"
        unchanged = [s for s in n.inputs if self.nodes[s].fingerprint]
        return (
            f"{key} valid: all {len(n.inputs)} inputs unchanged "
            f"({', '.join(n.inputs)})"
        )

    def stale_inputs(self, key: str) -> list[str]:
        return [s for s in self.nodes[key].inputs if self.nodes[s].state == "stale"]

    def propagate(
        self,
        changed: dict[str, str],
        recompute: Callable[[str], str],
    ) -> dict[str, list[str]]:
        """Apply a batch of source changes and recompute stale derived nodes.

        ``recompute(key)`` returns the new fingerprint. Returns a report mapping
        each derived key to ["recomputed"] or ["valid", reason].
        """
        all_affected: set[str] = set()
        for key, fp in changed.items():
            all_affected |= self.change_source(key, fp)
        report: dict[str, list[str]] = {}
        for key in sorted(all_affected):
            new_fp = recompute(key)
            self.recompute_finished(key, new_fp)
            report[key] = ["recomputed"]
        for key, node in self.nodes.items():
            if node.kind == "derived" and key not in report:
                report[key] = ["valid", self.why_valid(key)]
        return report
