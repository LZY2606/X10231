from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Quantity:
    value: float | None
    unit: str
    datum: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "datum": self.datum, "raw": self.raw}


def parse_quantity(payload: dict[str, Any]) -> Quantity:
    raw = payload.get("raw")
    value = payload.get("value")
    if value is None:
        return Quantity(None, str(payload.get("unit", "")), payload.get("datum"), raw)
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("value is not finite")
        return Quantity(number, str(payload.get("unit", "")), payload.get("datum"), str(raw) if raw is not None else str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot normalize quantity {raw!r}: {exc}") from exc


def require_unit(quantity: Quantity, unit: str) -> float:
    if quantity.unit != unit:
        raise ValueError(f"expected unit {unit}, got {quantity.unit}")
    if quantity.value is None:
        raise ValueError("quantity has no numeric value")
    return quantity.value
