"""Unit-safe parsing used only by the derived layer.

Raw imports retain their original representation. Parsing failures do not
delete a value; callers keep the raw value and attach a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

VELOCITY_TO_MPS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001}
ANGLE_TO_RAD = {"rad": 1.0, "deg": 0.017453292519943295, "mrad": 0.001}


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    raw: Any


def convert_velocity(raw: Any, unit: str) -> Tuple[float, str]:
    factor = VELOCITY_TO_MPS.get(unit)
    if factor is None:
        raise ValueError(f"unsupported velocity unit: {unit}")
    return float(raw) * factor, "m/s"


def convert_angle(raw: Any, unit: str) -> Tuple[float, str]:
    factor = ANGLE_TO_RAD.get(unit)
    if factor is None:
        raise ValueError(f"unsupported angle unit: {unit}")
    return float(raw) * factor, "rad"


def angle_unit_metadata(unit: str) -> Dict[str, str]:
    if unit not in ANGLE_TO_RAD:
        return {"unit": unit, "basis": "unrecognized"}
    return {"unit": "rad", "source_unit": unit, "basis": "angle"}
