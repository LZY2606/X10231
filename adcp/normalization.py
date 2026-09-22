from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


UNIT_CONVERSIONS = {
    ("mm/s", "m/s"): 0.001,
    ("cm/s", "m/s"): 0.01,
    ("m/s", "m/s"): 1.0,
    ("counts", "counts"): 1.0,
    ("percent", "percent"): 1.0,
    ("degree", "degree"): 1.0,
    ("degrees", "degree"): 1.0,
    ("deg", "degree"): 1.0,
    ("m", "m"): 1.0,
    ("s", "s"): 1.0,
}


def scalar(value: Any, unit: Optional[str] = None, target_unit: Optional[str] = None) -> Tuple[float, str]:
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("missing measurement")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError("non-finite measurement")
    if unit and target_unit:
        key = (unit, target_unit)
        if key not in UNIT_CONVERSIONS:
            raise ValueError("no conversion from %s to %s" % key)
        number *= UNIT_CONVERSIONS[key]
    return number, target_unit or unit or ""


def integer_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return int(value) != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "bad", "y", "yes"}:
        return True
    if text in {"0", "false", "good", "n", "no"}:
        return False
    raise ValueError("invalid boolean flag")


def require_fields(value: Dict[str, Any], fields: Tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError("missing fields: " + ",".join(missing))
