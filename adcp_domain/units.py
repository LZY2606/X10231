from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueWithUnit:
    raw: object
    unit: str
    value: float | None
    normalized: float | None
    normalized_unit: str | None
    diagnostic: str | None = None


def normalize_number(raw: object, unit: str, target_unit: str) -> ValueWithUnit:
    conversions = {
        ("mm/s", "m/s"): 0.001,
        ("m/s", "m/s"): 1.0,
        ("deg", "deg"): 1.0,
        ("count", "count"): 1.0,
    }
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return ValueWithUnit(raw, unit, None, None, None, "UNCONVERTIBLE_VALUE")
    factor = conversions.get((unit, target_unit))
    if factor is None:
        return ValueWithUnit(raw, unit, number, None, None, "UNKNOWN_UNIT_CONVERSION")
    return ValueWithUnit(raw, unit, number, number * factor, target_unit)


def as_float(raw: object) -> float | None:
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result
