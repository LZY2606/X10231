"""领域量的单位与坐标基准。

语义约定 (README "单位与原始表示" 一节对应):
- 每个领域量是一个 Quantity: 原始值 + 原始单位 + 坐标/基准 + 原始表示原文;
- 导入层只登记原始表示, 派生层 (beam/instrument/ship/earth) 才做规范化;
- 无法转换的值不抛异常丢弃, 而是返回 ``unconvertible`` 占位诊断,
  调用方必须把它留在被排除集合里, 不能为了画图而吞掉。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class UnitError(ValueError):
    """单位缺失、不认识或无法按基准转换时抛出 (仅在严格模式)。"""


# 规范单位 (派生层内部统一使用 SI/约定单位, 但只用于派生, 不改原始记录)
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "time": "s",
    "depth": "m",
    "correlation": "counts",
    "amplitude": "counts",
    "length": "m",
}

# 已知换算因子 -> 规范单位
_FACTORS = {
    ("velocity", "mm/s"): 1e-3,
    ("velocity", "cm/s"): 1e-2,
    ("velocity", "m/s"): 1.0,
    ("angle", "deg"): 1.0,
    ("angle", "rad"): 180.0 / math.pi,
    ("time", "s"): 1.0,
    ("time", "ms"): 1e-3,
    ("depth", "m"): 1.0,
    ("depth", "dm"): 0.1,
    ("length", "m"): 1.0,
    ("length", "dm"): 0.1,
    ("correlation", "counts"): 1.0,
    ("amplitude", "counts"): 1.0,
}

# 合法坐标/角度基准
FRAMES = ("beam", "instrument", "ship", "earth")
HEADING_REFERENCES = ("magnetic", "true")


@dataclass(frozen=True)
class Quantity:
    """一个带身份元数据的领域量。不可变, 便于指纹计算。"""

    value: Optional[float]
    unit: str
    kind: str
    frame: str = "none"          # beam/instrument/ship/earth/none
    reference: str = "none"      # magnetic/true/none
    raw_repr: Optional[str] = None
    diagnostic: Optional[str] = None

    def to_canonical(self) -> "Quantity":
        """派生层规范化。无法转换时返回带诊断的占位量, 不抛异常。"""
        if self.value is None:
            return self
        key = (self.kind, self.unit)
        factor = _FACTORS.get(key)
        if factor is None:
            return Quantity(
                value=None,
                unit=CANONICAL.get(self.kind, self.unit),
                kind=self.kind,
                frame=self.frame,
                reference=self.reference,
                raw_repr=self.raw_repr if self.raw_repr is not None else str(self.value),
                diagnostic="unconvertible: unknown unit %r for kind %r" % (self.unit, self.kind),
            )
        return Quantity(
            value=self.value * factor,
            unit=CANONICAL[self.kind],
            kind=self.kind,
            frame=self.frame,
            reference=self.reference,
            raw_repr=self.raw_repr,
            diagnostic=self.diagnostic,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "frame": self.frame,
            "reference": self.reference,
            "raw_repr": self.raw_repr,
            "diagnostic": self.diagnostic,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Quantity":
        return Quantity(
            value=d.get("value"),
            unit=d["unit"],
            kind=d["kind"],
            frame=d.get("frame", "none"),
            reference=d.get("reference", "none"),
            raw_repr=d.get("raw_repr"),
            diagnostic=d.get("diagnostic"),
        )


def q(value: Any, unit: str, kind: str, frame: str = "none",
      reference: str = "none") -> Quantity:
    """从导入记录构造领域量, 保留原始表示。非数值值保留为无法转换占位。"""
    raw = None if value is None else str(value)
    try:
        fvalue = float(value)
        if math.isnan(fvalue) or math.isinf(fvalue):
            raise ValueError("non-finite")
    except (TypeError, ValueError):
        return Quantity(
            value=None, unit=unit, kind=kind, frame=frame, reference=reference,
            raw_repr=raw, diagnostic="unconvertible: non-numeric value",
        )
    return Quantity(value=fvalue, unit=unit, kind=kind, frame=frame,
                    reference=reference, raw_repr=raw)


def vector(components: Dict[str, float], frame: str, unit: str = "m/s",
           reference: str = "none") -> Dict[str, Quantity]:
    return {
        axis: Quantity(value=comp, unit=unit, kind="velocity",
                       frame=frame, reference=reference)
        for axis, comp in components.items()
    }
