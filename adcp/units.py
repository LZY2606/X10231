"""Unit handling.

Raw source records keep their original unit and coordinate basis. Only the
derived layer normalizes to SI. Values that cannot be converted stay in place
and carry a diagnostic; they are never silently dropped.
"""

from __future__ import annotations

# unit -> scale factor to SI
_VELOCITY = {"mm/s": 0.001, "cm/s": 0.01, "m/s": 1.0}
_ANGLE = {"deg": 1.0, "rad": 57.29577951308232}
_COUNT = {"count": 1.0}
_TIME = {"s": 1.0, "ms": 0.001}
_LENGTH = {"m": 1.0, "cm": 0.01}

_TABLES = {
    "velocity": (_VELOCITY, "m/s"),
    "angle": (_ANGLE, "deg"),
    "count": (_COUNT, "count"),
    "time": (_TIME, "s"),
    "length": (_LENGTH, "m"),
}


def normalize(value, unit: str, quantity: str):
    """Return (si_value, si_unit, diagnostic).

    diagnostic is None on success; otherwise the value is returned unchanged
    and the diagnostic explains why it was left in raw form.
    """
    table, si_unit = _TABLES[quantity]
    if unit not in table:
        return value, unit, f"cannot normalize {quantity} unit {unit!r}; kept raw"
    return value * table[unit], si_unit, None


def normalize_seq(values, unit: str, quantity: str):
    """Normalize a list; per-element diagnostics, raw values preserved on failure."""
    table, si_unit = _TABLES[quantity]
    if unit not in table:
        diag = f"cannot normalize {quantity} unit {unit!r}; kept raw"
        return list(values), unit, [diag for _ in values]
    factor = table[unit]
    return [v * factor for v in values], si_unit, [None for _ in values]
