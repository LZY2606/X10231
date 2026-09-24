"""ADCP beam-velocity / attitude / coordinate-transform evidence toolkit.

Domain package layout::

    quantities   - value objects carrying units / datum plus raw preservation
    math3d       - beam geometry, Janus transform, attitude rotations
    attitudes    - heading wrap handling, gap-aware interpolation
    quality      - correlation / amplitude / bad-flag / cell hierarchy
    fingerprints - deterministic provenance hashes
    cases        - offline domain case pack (the "beam-reduction bench")
    storage      - SQLite store, atomic versions, migration, import/export
    engine       - dependency-graph propagation and replay
    events       - machine-readable event vocabulary
    api          - FastAPI application
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
