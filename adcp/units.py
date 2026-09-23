"""领域量：携带单位/坐标基准的数值容器与导入期转换。

规则：
- 导入时保留原始表示（raw 文本 + 原始单位）；
- 只有派生层才规范化；
- 无法转换的值留在原位置并附诊断，绝不丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class UnconvertibleError(ValueError):
    """原始值无法规范化（未知单位/无法解析）。"""


_VELOCITY_TO_MPS = {
    "m/s": 1.0,
    "cm/s": 0.01,
    "mm/s": 0.001,
}

_ANGLE_TO_DEG = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "rad": 57.29577951308232,
    "radian": 57.29577951308232,
    "radians": 57.29577951308232,
}

_LENGTH_TO_M = {"m": 1.0, "cm": 0.01, "dm": 0.1, "ft": 0.3048}


def normalize(raw: Any, unit: Optional[str], kind: str) -> float:
    """把 (raw, unit) 规范化为 SI 内部单位（m/s、deg、m）。

    raw 已经是数值时视为已规范化；否则尝试 float() 解析。
    失败抛 UnconvertibleError，调用方负责在原位置记录诊断。
    """
    table = {
        "velocity": _VELOCITY_TO_MPS,
        "angle": _ANGLE_TO_DEG,
        "length": _LENGTH_TO_M,
    }[kind]
    if unit is None:
        unit = "m/s" if kind == "velocity" else "deg" if kind == "angle" else "m"
    if unit not in table:
        raise UnconvertibleError(f"unknown {kind} unit: {unit!r}")
    if isinstance(raw, bool):
        raise UnconvertibleError("boolean is not a measurement")
    if isinstance(raw, (int, float)):
        return float(raw) * table[unit]
    try:
        return float(str(raw).strip()) * table[unit]
    except (ValueError, TypeError) as exc:
        raise UnconvertibleError(f"cannot parse {raw!r} as {kind}") from exc


@dataclass(frozen=True)
class Quantity:
    """领域量：值 + 内部单位 + 原始表示 + 原始单位 + 基准说明。"""

    value: float
    unit: str
    raw: Any
    raw_unit: Optional[str]
    datum: Optional[str] = None
    ok: bool = True
    diag: Optional[str] = None

    @staticmethod
    def from_raw(raw: Any, raw_unit: Optional[str], kind: str,
                 datum: Optional[str] = None) -> "Quantity":
        try:
            value = normalize(raw, raw_unit, kind)
        except UnconvertibleError as exc:
            return Quantity(float("nan"),
                            {"velocity": "m/s", "angle": "deg",
                             "length": "m"}[kind], raw, raw_unit, datum,
                            ok=False, diag=str(exc))
        return Quantity(value,
                        {"velocity": "m/s", "angle": "deg",
                         "length": "m"}[kind], raw, raw_unit, datum)

    def as_dict(self) -> dict:
        return {
            "value": None if self.ok is False else self.value,
            "unit": self.unit,
            "raw": self.raw,
            "raw_unit": self.raw_unit,
            "datum": self.datum,
            "ok": self.ok,
            "diag": self.diag,
        }


def wrap360(deg: float) -> float:
    """归一化到 [0, 360)。"""
    return deg % 360.0


def wrap180(deg: float) -> float:
    """归一化到 (-180, 180]。"""
    return (deg + 180.0) % 360.0 - 180.0
