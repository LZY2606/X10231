"""领域量：值 + 单位 + 坐标基准。导入保留原始表示，派生层才规范化。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional


@dataclass(frozen=True)
class Q:
    """带单位与坐标基准的领域量。raw 保存导入时的原始表示。"""

    value: float
    unit: str
    frame: Optional[str] = None
    raw: Any = None
    diagnosis: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.diagnosis is None and self.value is not None

    def with_diagnosis(self, msg: str) -> "Q":
        return replace(self, diagnosis=msg)

    def to_unit(self, target: str) -> "Q":
        factor = _CONVERSIONS.get((self.unit, target))
        if factor is None:
            if self.unit == target:
                return self
            return self.with_diagnosis(f"无法把 {self.unit} 转换为 {target}")
        return Q(self.value * factor, target, self.frame, raw=self.raw)


_CONVERSIONS = {
    ("mm/s", "m/s"): 0.001,
    ("cm/s", "m/s"): 0.01,
    ("m/s", "m/s"): 1.0,
    ("deg", "rad"): 0.017453292519943295,
    ("rad", "deg"): 57.29577951308232,
    ("deg", "deg"): 1.0,
}


# 坐标基准词汇表。任何派生向量都必须显式声明其中之一。
FRAME_BEAM = "beam"
FRAME_INSTRUMENT = "instrument"
FRAME_SHIP = "ship"
FRAME_EARTH = "earth"

VELOCITY_UNITS = {"mm/s", "cm/s", "m/s"}
ANGLE_UNITS = {"deg", "rad"}
CORRELATION_UNITS = {"counts", "percent"}
AMPLITUDE_UNITS = {"counts", "dB"}


def is_finite_number(x: Any) -> bool:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return xf == xf and xf not in (float("inf"), float("-inf"))
