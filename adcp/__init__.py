"""ADCP beam-to-earth recomputation and evidence toolkit.

Core rule modules are deliberately free of database and web dependencies so
they can be unit tested in isolation:

- ``units``       raw representation preservation + normalisation
- ``geometry``    transducer beam geometry, beam <-> instrument projection
- ``attitude``    heading unwrap, gap-aware interpolation
- ``transforms``  instrument -> ship -> earth rotations (explicit order)
- ``quality``     per-ping/per-cell exclusion hierarchy
- ``fingerprint`` content hashing for provenance
- ``derive``      dependency-graph propagation and candidate sets
"""

__all__ = [
    "units",
    "geometry",
    "attitude",
    "transforms",
    "quality",
    "fingerprint",
    "derive",
]
