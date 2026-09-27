"""领域量的单位与坐标基准。

规则（对应需求“所有领域量都携带单位或坐标基准，导入时保留原始表示，
派生层才做规范化；无法转换的值留在原位置并附诊断”）：

* 导入层只产生 :class:`Quantity`，不做单位换算，``raw`` 保存原始文本表示；
* 解析失败时 ``value is None`` 且 ``ok is False``，记录 ``diagnostic``，
  调用方必须保留该记录、不得丢弃；
* 派生层通过 :meth:`Quantity.as_si` 做显式换算，换算失败返回 None 而不抛异常。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

# 到 SI 基本单位（m, m/s, rad）的换算因子
_TO_SI = {
    "mm/s": 0.001,
    "m/s": 1.0,
    "m": 1.0,
    "mm": 0.001,
    "deg": math.pi / 180.0,
    "rad": 1.0,
    "%": 1.0,
    "count": 1.0,
    "magnetic_deg": math.pi / 180.0,
    "true_deg": math.pi / 180.0,
}

# 同族单位（用于兼容性检查）
FAMILY = {
    "mm/s": "velocity",
    "m/s": "velocity",
    "m": "length",
    "mm": "length",
    "deg": "angle",
    "rad": "angle",
    "magnetic_deg": "angle",
    "true_deg": "angle",
    "%": "percent",
    "count": "count",
}

# 角度基准（坐标基准）
ANGLE_BASES = {"", "magnetic", "true"}


@dataclass(frozen=True)
class Quantity:
    """带单位/基准的标量领域量。"""

    value: Optional[float]
    unit: str
    raw: Optional[str] = None
    ok: bool = True
    diagnostic: Optional[str] = None
    basis: str = ""

    @property
    def family(self) -> str:
        return FAMILY.get(self.unit, "unknown")

    def as_si(self) -> Optional[float]:
        """换算到 SI（m/m·s⁻¹/rad）。不可换算返回 None。"""
        if not self.ok or self.value is None:
            return None
        factor = _TO_SI.get(self.unit)
        if factor is None:
            return None
        return self.value * factor

    def with_diagnostic(self, message: str) -> "Quantity":
        return replace(self, ok=False, diagnostic=message)

    def __str__(self) -> str:  # pragma: no cover - 仅便于调试
        if self.ok and self.value is not None:
            return f"{self.value:g} {self.unit}"
        return f"<invalid raw={self.raw!r} unit={self.unit} {self.diagnostic}>"


def parse_quantity(raw, unit: str, basis: str = "") -> Quantity:
    """导入层解析：保留原始表示，失败值留在原位附诊断。"""
    if basis not in ANGLE_BASES:
        raise ValueError(f"unknown angle basis: {basis!r}")
    if unit not in _TO_SI:
        return Quantity(None, unit, raw=None if raw is None else str(raw),
                        ok=False, diagnostic=f"unknown unit {unit!r}", basis=basis)
    if isinstance(raw, Quantity):
        return Quantity(raw.value, unit, raw.raw, raw.ok, raw.diagnostic, basis or raw.basis)
    text = None if raw is None else str(raw)
    if raw is None:
        return Quantity(None, unit, raw=None, ok=False,
                        diagnostic="missing value", basis=basis)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Quantity(None, unit, raw=text, ok=False,
                        diagnostic=f"unparseable value {text!r}", basis=basis)
    if math.isnan(value):
        return Quantity(None, unit, raw=text, ok=False,
                        diagnostic="NaN value", basis=basis)
    return Quantity(value, unit, raw=text, ok=True, basis=basis)


def compatible(a: Quantity, b: Quantity) -> bool:
    """两个量是否同族（velocity/length/angle/...）。"""
    return a.family != "unknown" and a.family == b.family
