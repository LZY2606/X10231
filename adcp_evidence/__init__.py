"""ADCP beam-velocity / attitude / coordinate-transform evidence tool.

Layers:
  units      - quantities keep their original representation, unit and datum
  geometry   - Janus beam geometry, instrument/ship/earth rotations
  quality    - beam -> ping -> cell quality hierarchy
  fingerprint- content-addressed identities for sources, rules and derivations
  store      - versioned SQLite storage with atomic commit pointers
  engine     - dependency-graph propagation and replay
"""

__version__ = "1.0.0"
RULE_VERSION = "adcp-2026.1"
