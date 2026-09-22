"""Unit preservation and controlled normalization for ADCP quantities."""
from __future__ import annotations

from math import isfinite
from typing import Any


class UnitConversionError(ValueError):
    pass


VELOCITY_TO_M_S = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
}

DEPTH_TO_M = {
    "m": 1.0,
    "meters": 1.0,
    "ft": 0.3048,
    "feet": 0.3048,
}

ANGLE_TO_DEG = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "rad": 57.29577951308232,
    "radians": 57.29577951308232,
}


def normalize_quantity(value: Any, unit: str | None, kind: str) -> tuple[float | None, str | None]:
    if value is None or isinstance(value, bool) or value == "":
        return None, f"missing {kind}"
    if isinstance(value, str):
        return None, f"non-numeric {kind}: {value!r} retained in source"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"non-numeric {kind}: {value!r} retained in source"
    if not isfinite(number):
        return None, f"non-finite {kind} retained in source"

    table = {
        "velocity": (VELOCITY_TO_M_S, "m/s"),
        "depth": (DEPTH_TO_M, "m"),
        "angle": (ANGLE_TO_DEG, "deg"),
    }.get(kind)
    if table is None:
        return number, unit
    conversions, target = table
    factor = conversions.get(unit or "")
    if factor is None:
        return None, f"unknown {kind} unit {unit!r}; raw value retained"
    return number * factor, target


def wrap_360(angle: float) -> float:
    return angle % 360.0
