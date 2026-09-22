"""领域量：始终携带单位/坐标基准；导入保留原始表示，派生层才规范化。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# 角度换算
DEG = 3.141592653589793 / 180.0

# 支持的量纲单位（派生层规范化目标）
VELOCITY_UNITS = {"mm/s", "m/s", "cm/s"}
ANGLE_UNITS = {"deg", "rad"}


class UnitError(ValueError):
    """无法在已知量纲内完成单位转换。"""


@dataclass(frozen=True)
class Quantity:
    """带单位的标量。raw_value/raw_unit 保留导入时的原始表示，不丢弃。"""

    value: float
    unit: str
    raw_value: Optional[str] = None
    raw_unit: Optional[str] = None
    valid: bool = True
    diagnostic: Optional[str] = None

    def to(self, target: str) -> "Quantity":
        if not self.valid:
            raise UnitError(f"invalid quantity cannot convert: {self.diagnostic}")
        nv = convert_velocity(self.value, self.unit, target) \
            if self.unit in VELOCITY_UNITS and target in VELOCITY_UNITS \
            else convert_angle(self.value, self.unit, target) \
            if self.unit in ANGLE_UNITS and target in ANGLE_UNITS \
            else None
        if nv is None:
            raise UnitError(f"incompatible units {self.unit!r} -> {target!r}")
        return Quantity(nv, target, self.raw_value, self.raw_unit)

    def to_float(self, target: str) -> float:
        return self.to(target).value


def convert_velocity(v: float, src: str, dst: str) -> float:
    to_ms = {"mm/s": 1e-3, "cm/s": 1e-2, "m/s": 1.0}
    if src not in to_ms or dst not in to_ms:
        raise UnitError(f"unknown velocity unit {src!r}->{dst!r}")
    return v * to_ms[src] / to_ms[dst]


def convert_angle(v: float, src: str, dst: str) -> float:
    if src == dst:
        return v
    rad = v * DEG if src == "deg" else v
    if dst == "rad":
        return rad
    if dst == "deg":
        return rad / DEG
    raise UnitError(f"unknown angle unit {src!r}->{dst!r}")


def as_deg(v: float, unit: str) -> float:
    return convert_angle(v, unit, "deg")


@dataclass(frozen=True)
class Vector:
    """三维向量，带坐标基准 (frame) 与速度单位。"""

    x: float
    y: float
    z: float
    frame: str
    unit: str = "m/s"
    extras: Dict[str, Any] = field(default_factory=dict)

    def as_array(self):
        import numpy as np

        return np.array([self.x, self.y, self.z], dtype=float)
