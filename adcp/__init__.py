"""ADCP beam/attitude/coordinate-transform evidence toolkit.

Layers (raw representation is never overwritten):

raw  : beam velocity / correlation / amplitude / bad-beam flags / attitude /
       bottom track / transducer config, each stored in its own table
dag  : dependency graph that decides which derived nodes are still valid
derived: instrument -> ship -> earth velocities, per candidate assumption
event: machine-readable judgments attached to immutable versions
"""
from __future__ import annotations

RULE_VERSION = "rule-2026.09.1"
