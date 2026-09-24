"""Units and normalization.

Raw records keep their original representation. Normalization happens only in
the derived layer. Values whose unit cannot be converted stay in place with a
diagnostic; they are never dropped to make plotting easier.
"""
from __future__ import annotations

VELOCITY_TO_MS = {"m/s": 1.0, "mm/s": 1e-3, "cm/s": 1e-2, "knot": 0.514444}
ANGLE_UNITS = {"deg", "degree", "degrees"}
COUNT_UNITS = {"count", "counts", "dB", "percent", "%"}


def normalize_velocity(value, unit):
    """-> (value_m_s, diagnostic_or_None). Never raises on unknown units."""
    if unit in VELOCITY_TO_MS:
        return float(value) * VELOCITY_TO_MS[unit], None
    return None, f"unconvertible velocity unit {unit!r}; raw value kept in place"


def normalize_angle(value, unit):
    if unit in ANGLE_UNITS:
        return float(value) % 360.0, None
    return None, f"unconvertible angle unit {unit!r}; raw value kept in place"
