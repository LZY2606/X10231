"""单位与坐标基准。

语义：所有领域量都携带单位或坐标基准；导入时保留原始表示，
派生层才做规范化；无法转换的值留在原位置并附诊断，
不能为了画图而丢掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 坐标基准（参考系）
FRAMES = ("beam", "instrument", "ship", "earth", "none")

# 可互相转换的单位（同一物理量族内的线性换算）
_CONVERSIONS = {
    ("mm/s", "m/s"): 0.001,
    ("cm/s", "m/s"): 0.01,
    ("m/s", "m/s"): 1.0,
    ("rad", "deg"): 180.0 / 3.141592653589793,
    ("deg", "deg"): 1.0,
    ("m", "m"): 1.0,
}

# 已知但不可换算成物理单位的原始量（保留原值 + 诊断）
RAW_ONLY_UNITS = ("count", "dB", "vendor-count", "flag")


@dataclass
class Diagnostic:
    code: str
    message: str

    def as_dict(self):
        return {"code": self.code, "message": self.message}


@dataclass
class Quantity:
    value: float | None
    unit: str
    frame: str = "none"
    raw_repr: str = ""  # 导入时的原始表示，永不改写
    diagnostics: list = field(default_factory=list)

    def as_dict(self):
        return {
            "value": self.value,
            "unit": self.unit,
            "frame": self.frame,
            "raw_repr": self.raw_repr,
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }


def normalize(q: Quantity, target_unit: str) -> Quantity:
    """派生层规范化。无法转换时原样返回并附诊断，不丢值。"""
    if q.unit == target_unit:
        return q
    key = (q.unit, target_unit)
    if key in _CONVERSIONS and q.value is not None:
        return Quantity(
            value=q.value * _CONVERSIONS[key],
            unit=target_unit,
            frame=q.frame,
            raw_repr=q.raw_repr,
        )
    diag = Diagnostic(
        code="unit_unconvertible",
        message=f"单位 {q.unit!r} 无法转换为 {target_unit!r}；值保留在原位置，不参与派生计算",
    )
    return Quantity(
        value=q.value,
        unit=q.unit,
        frame=q.frame,
        raw_repr=q.raw_repr,
        diagnostics=list(q.diagnostics) + [diag],
    )
