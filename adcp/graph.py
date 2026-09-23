"""Dependency graph propagation.

The derivation is a DAG of value slots:

    beam velocity / correlation / amplitude / bad flag  (source nodes)
        -> beam quality -> instrument candidate(s)
        -> ship velocity -> earth velocity
        -> profile cell admission / exclusion

When a source node is replaced by a new version only the reachable downstream
nodes are marked ``needs_recompute``; everything else stays ``still_valid`` and
the reason for either state can be reported.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class DependencyGraph:
    # node -> direct upstream nodes
    parents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # node -> direct downstream nodes
    children: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # node -> rule tag that produced it (for explanations)
    rules: Dict[str, str] = field(defaultdict(str))

    def add_node(self, node: str, parents: Iterable[str], rule: str = "") -> None:
        self.parents[node].update(parents)
        for parent in parents:
            self.children[parent].add(node)
        if rule:
            self.rules[node] = rule

    def downstream_of(self, changed: Iterable[str]) -> Set[str]:
        """Return every node reachable from ``changed`` (excluding sources)."""
        affected: Set[str] = set()
        queue: deque = deque(changed)
        while queue:
            node = queue.popleft()
            for child in self.children.get(node, ()):
                if child not in affected:
                    affected.add(child)
                    queue.append(child)
        return affected

    def explain(self, node: str, changed: Iterable[str]) -> Dict[str, object]:
        affected = self.downstream_of(changed)
        changed_set = set(changed)
        if node in changed_set:
            return {
                "node": node,
                "state": "source_changed",
                "reason": "the source record itself was replaced",
            }
        if node in affected:
            path = self._find_path(node, changed_set)
            return {
                "node": node,
                "state": "needs_recompute",
                "reason": "a changed source reaches this node through the dependency graph",
                "changed_inputs": sorted(changed_set),
                "path": path,
                "rule": self.rules.get(node, ""),
            }
        blockers = [p for p in self.parents.get(node, ()) if p in affected]
        return {
            "node": node,
            "state": "still_valid",
            "reason": "none of its dependencies are downstream of the changed source records",
            "blocking_inputs": blockers,
            "rule": self.rules.get(node, ""),
        }

    def _find_path(self, target: str, sources: Set[str]) -> List[str]:
        """Breadth-first path from any changed source to ``target``."""
        queue: deque = deque((source, [source]) for source in sources)
        seen = set(sources)
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for child in self.children.get(node, ()):
                if child not in seen:
                    seen.add(child)
                    queue.append((child, path + [child]))
        return []
