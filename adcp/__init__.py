"""ADCP beam velocity -> instrument -> ship -> earth evidence core.

All domain quantities carry a unit or datum. Raw representations are preserved
at the import layer; this package normalizes only on the derived layer and never
silently discards values that cannot be converted.
"""

__all__ = ["units", "geometry", "transforms", "quality", "graph", "pipeline"]
