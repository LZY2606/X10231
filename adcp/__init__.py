"""ADCP beam-velocity / coordinate-transform evidence toolkit.

Stable public surface:

- :mod:`adcp.units`       quantities, units, coordinate bases
- :mod:`adcp.geometry`    Janus beam geometry + 4/3 beam solutions
- :mod:`adcp.transforms`  attitude handling, rotation matrices
- :mod:`adcp.dag`         dependency graph, fingerprints, propagation
- :mod:`adcp.store`       versioned SQLite storage (crash-safe)
- :mod:`adcp.engine`      derivation engine (beam -> instrument -> ship -> earth)
- :mod:`adcp.casepack`    offline narrative case pack
- :mod:`adcp.archive`     counterexample archive
- :mod:`adcp.viewmodel`   read models for the review UI
"""

__all__ = [
    "units",
    "geometry",
    "transforms",
    "dag",
    "store",
    "engine",
    "casepack",
    "archive",
    "viewmodel",
]
