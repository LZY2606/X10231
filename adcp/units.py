"""带单位/坐标基准的领域量。

导入层保留原始表示(raw, raw_unit)，只有进入派生层才规范化为 SI；
无法转换的值停在原位置并附诊断，不为画图而丢弃。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 允许的原始单位 -> SI 因子（线性换算）
_UNIT_TO_SI = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "deg": math.pi / 180.0,
    "rad": 1.0,
    "count": 1.0,
    "pct": 1.0,  # 0-100 标度，阈值同样按百分比比较
    "m": 1.0,
    "s": 1.0,
    "": 1.0,
}


class UnitError(ValueError):
    """单位/数值无法规范化时抛出，调用方需保留原值并附诊断。"""


@dataclass(frozen=True)
class Quantity:
    """一个携带原始表示的领域量。"""

    raw: str
    raw_unit: str
    si: float | None
    diagnostic: str | None = None

    @property
    def ok(self) -> bool:
        return self.si is not None and self.diagnostic is None

    def to(self, unit: str) -> float:
        if not self.ok:
            raise UnitError(f"unconvertible quantity: {self.raw!r} {self.diagnostic or ''}".strip())
        return self.si / _UNIT_TO_SI[unit]


def parse_quantity(raw, raw_unit: str, *, expect_unit: str | None = None) -> Quantity:
    """把原始表示解析为 SI。不抛异常，失败时返回带 diagnostic 的 Quantity。"""
    if raw is None:
        return Quantity(str(raw), raw_unit, None, "missing value")
    if isinstance(raw, bool):
        return Quantity(str(raw), raw_unit, None, "boolean is not a measurement")
    text = str(raw).strip()
    unit = (raw_unit or "").strip()
    if expect_unit is not None and unit != expect_unit:
        return Quantity(text, unit, None, f"unit mismatch: expected {expect_unit}, got {unit!r}")
    if unit not in _UNIT_TO_SI:
        return Quantity(text, unit, None, f"unknown unit {unit!r}")
    try:
        value = float(text)
    except (TypeError, ValueError):
        return Quantity(text, unit, None, f"unparseable value {text!r}")
    if math.isnan(value):
        return Quantity(text, unit, None, "NaN value")
    if math.isinf(value):
        return Quantity(text, unit, None, "infinite value")
    return Quantity(text, unit, value * _UNIT_TO_SI[unit], None)


def normalize_angle_deg(value: float) -> float:
    """归一化到 [0, 360)。"""
    return value % 360.0
