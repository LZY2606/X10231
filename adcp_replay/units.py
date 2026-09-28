"""Domain quantity metadata and import-time representation handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    datum: str | None = None
    precision: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"value": self.value, "unit": self.unit}
        if self.datum:
            result["datum"] = self.datum
        if self.precision is not None:
            result["precision"] = self.precision
        return result


def q(value: Any, unit: str, datum: str | None = None) -> Quantity:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return Quantity(float("nan"), unit, datum)
    return Quantity(numeric, unit, datum)


def valid_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def quantity_dict(value: Any, unit: str, datum: str | None = None) -> dict[str, Any]:
    return {
        "raw_representation": str(value),
        "normalized": q(value, unit, datum).to_dict() if valid_number(value) else None,
        "diagnostic": None if valid_number(value) else f"value stays at source; cannot convert to {unit}",
    }
