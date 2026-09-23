"""单位与坐标基准。

所有领域量都携带单位或坐标基准。导入时保留原始表示（raw_repr），
派生层才做规范化；无法转换的值留在原位置并附诊断，绝不为了画图丢掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field

VELOCITY_TO_MS = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
    "kn": 0.514444,
    "knots": 0.514444,
}
ANGLE_TO_DEG = {"deg": 1.0, "rad": 57.295779513082323}
LENGTH_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}
PASSTHROUGH_UNITS = {"count", "counts", "dB", "RSSI", "s", "1"}

FRAMES = ("beam", "instrument", "ship", "earth")


@dataclass
class Quantity:
    """一个携带单位与坐标基准的领域量。"""

    value: float | None
    unit: str
    frame: str | None = None
    raw_repr: str | None = None
    diagnostic: str | None = None

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "frame": self.frame,
            "raw_repr": self.raw_repr,
            "diagnostic": self.diagnostic,
        }


def convert_value(value: float, from_unit: str, to_unit: str):
    """返回 (converted_value, diagnostic)。无法转换时返回 (None, diagnostic)。"""
    if from_unit == to_unit:
        return value, None
    for table in (VELOCITY_TO_MS, ANGLE_TO_DEG, LENGTH_TO_M):
        if from_unit in table and to_unit in table:
            return value * table[from_unit] / table[to_unit], None
    return None, f"unit_unconvertible:{from_unit}->{to_unit}"


def parse_quantity(raw: str, unit: str, frame: str | None = None) -> Quantity:
    """把导入的原始字符串解析为 Quantity；解析失败时保留原值并附诊断。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Quantity(
            value=None, unit=unit, frame=frame, raw_repr=raw,
            diagnostic=f"value_unparseable:{raw!r}",
        )
    return Quantity(value=value, unit=unit, frame=frame, raw_repr=raw)


def normalize(quantity: Quantity, to_unit: str) -> Quantity:
    """派生层规范化：转换到目标单位；无法转换时原样保留并附诊断。"""
    if quantity.value is None:
        return quantity
    converted, diagnostic = convert_value(quantity.value, quantity.unit, to_unit)
    if diagnostic is not None:
        return Quantity(
            value=None, unit=quantity.unit, frame=quantity.frame,
            raw_repr=quantity.raw_repr, diagnostic=diagnostic,
        )
    return Quantity(
        value=converted, unit=to_unit, frame=quantity.frame,
        raw_repr=quantity.raw_repr,
    )
