"""单位与坐标基准。

所有领域量都携带单位或坐标基准；导入时保留原始表示，派生层才做规范化。
无法转换的值留在原位置并附诊断，绝不为了画图而丢掉。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# unit -> (dimension, factor to canonical unit of that dimension)
UNITS = {
    "m/s": ("velocity", 1.0),
    "cm/s": ("velocity", 0.01),
    "mm/s": ("velocity", 0.001),
    "deg": ("angle", 1.0),
    "rad": ("angle", 180.0 / math.pi),
    "count": ("correlation", 1.0),
    "dB": ("echo_amplitude", 1.0),
    "s": ("time", 1.0),
    "m": ("length", 1.0),
}

FRAMES = {"beam", "instrument", "ship", "earth", "magnetic", "true", "none"}


@dataclass
class Diagnostic:
    code: str
    message: str
    raw: dict | None = None

    def as_dict(self):
        return {"code": self.code, "message": self.message, "raw": self.raw}


@dataclass
class Qty:
    value: float
    unit: str
    frame: str = "none"
    meta: dict = field(default_factory=dict)


def normalize(value, unit, target_unit):
    """规范化到目标单位。

    返回 (canonical_value, diagnostic)。无法转换时返回 (None, Diagnostic)，
    调用方必须把原值留在原位置并携带诊断。
    """
    if unit not in UNITS:
        return None, Diagnostic(
            "unsupported_unit",
            f"unit {unit!r} is not supported; raw value kept in place",
            {"value": value, "unit": unit},
        )
    dim, factor = UNITS[unit]
    if target_unit not in UNITS or UNITS[target_unit][0] != dim:
        return None, Diagnostic(
            "dimension_mismatch",
            f"cannot convert {unit!r} ({dim}) to {target_unit!r}",
            {"value": value, "unit": unit},
        )
    return value * factor / UNITS[target_unit][1], None
