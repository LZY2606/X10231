"""领域量的单位与坐标基准。

所有原始值在导入时保留原始表示（含原始单位）；只有派生层才做规范化。
无法转换的值留在原位置并附诊断，而不是被静默丢弃。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 规范单位（派生层使用 SI/通用约定）
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "time": "s",
    "depth": "m",
    "correlation": "counts",
    "amplitude": "counts",
}

# 支持的单位转换 -> 规范单位
_FACTORS = {
    ("velocity", "mm/s", "m/s"): 1e-3,
    ("velocity", "cm/s", "m/s"): 1e-2,
    ("velocity", "m/s", "m/s"): 1.0,
    ("velocity", "knot", "m/s"): 0.5144444444444445,
    ("angle", "deg", "deg"): 1.0,
    ("angle", "rad", "deg"): 180.0 / math.pi,
    ("time", "s", "s"): 1.0,
    ("time", "ms", "s"): 1e-3,
    ("depth", "m", "m"): 1.0,
    ("depth", "dm", "m"): 0.1,
    ("depth", "ft", "m"): 0.3048,
    ("correlation", "counts", "counts"): 1.0,
    ("amplitude", "counts", "counts"): 1.0,
}


class UnitError(ValueError):
    """值无法转换到规范单位时抛出；由导入层捕获并写为诊断。"""


@dataclass(frozen=True)
class Quantity:
    """带单位的标量；raw_value/raw_unit 永远保留。"""

    value: float
    unit: str
    raw_value: float
    raw_unit: str
    kind: str

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "raw_value": self.raw_value,
            "raw_unit": self.raw_unit,
            "kind": self.kind,
        }


def normalize(value, kind: str, raw_unit: str):
    """把原始表示规范化；返回 Quantity 或 (None, diagnostic)。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    target = CANONICAL[kind]
    factor = _FACTORS.get((kind, raw_unit, target))
    if factor is None:
        raise UnitError(f"unsupported {kind} unit: {raw_unit!r}")
    return Quantity(f * factor, target, f, raw_unit, kind)


def normalize_or_diag(value, kind: str, raw_unit: str):
    """成功返回 Quantity；无法转换时返回 None 并给出诊断字符串。"""
    try:
        return normalize(value, kind, raw_unit)
    except UnitError as exc:
        return None, str(exc)
