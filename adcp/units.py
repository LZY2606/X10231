"""领域量与单位/坐标基准。

导入时保留原始表示（raw_value/raw_unit），派生层才做规范化；
无法转换的值在 normalize 时抛出 Unconvertible，留在原位置并附诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# 内部规范单位
CANONICAL = {
    "velocity": "m/s",
    "angle": "rad",
    "depth": "m",
    "time": "s",
    "correlation": "count",
    "amplitude": "count",
    "magnetic_field": "nT",
}

# 角度单位 -> 弧度倍率
ANGLE_TO_RAD = {
    "rad": 1.0,
    "deg": 0.017453292519943295,
    "degree": 0.017453292519943295,
    "degrees": 0.017453292519943295,
}
# 深度单位 -> 米
LENGTH_TO_M = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "ft": 0.3048}
# 流速单位 -> m/s
VEL_TO_MPS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knot": 0.5144444444}


class Unconvertible(ValueError):
    """无法把原始表示转换为规范单位；保留在原位置并附诊断。"""


@dataclass
class Quantity:
    """带单位（以及可选坐标基准）的领域量；raw_* 始终保留原始导入表示。"""

    value: float
    unit: str
    kind: str
    raw_value: Any = None
    raw_unit: Optional[str] = None
    datum: Optional[str] = None  # 坐标基准，如 MAG/TRUE、SHIP/EARTH-Z-DOWN
    diagnostic: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float(raw: Any) -> float:
    if isinstance(raw, bool):
        raise Unconvertible(f"布尔值不是有效测量: {raw!r}")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise Unconvertible(f"无法解析数值: {raw!r}") from exc


def normalize_velocity(raw: Any, raw_unit: str = "m/s", datum: Optional[str] = None) -> Quantity:
    value = _to_float(raw)
    factor = VEL_TO_MPS.get(raw_unit)
    if factor is None:
        raise Unconvertible(f"未知流速单位: {raw_unit!r}")
    return Quantity(
        value=value * factor, unit=CANONICAL["velocity"], kind="velocity",
        raw_value=raw, raw_unit=raw_unit, datum=datum,
    )


def normalize_angle(raw: Any, raw_unit: str = "deg", datum: Optional[str] = None) -> Quantity:
    value = _to_float(raw)
    factor = ANGLE_TO_RAD.get(raw_unit)
    if factor is None:
        raise Unconvertible(f"未知角度单位: {raw_unit!r}")
    return Quantity(
        value=value * factor, unit=CANONICAL["angle"], kind="angle",
        raw_value=raw, raw_unit=raw_unit, datum=datum,
    )


def normalize_depth(raw: Any, raw_unit: str = "m", datum: Optional[str] = None) -> Quantity:
    value = _to_float(raw)
    factor = LENGTH_TO_M.get(raw_unit)
    if factor is None:
        raise Unconvertible(f"未知长度单位: {raw_unit!r}")
    if value < 0:
        raise Unconvertible(f"深度/距离不能为负: {raw!r}")
    return Quantity(
        value=value * factor, unit=CANONICAL["depth"], kind="depth",
        raw_value=raw, raw_unit=raw_unit, datum=datum,
    )


def normalize_count(raw: Any, kind: str) -> Quantity:
    value = _to_float(raw)
    if value < 0:
        raise Unconvertible(f"{kind} 不能为负: {raw!r}")
    return Quantity(
        value=value, unit=CANONICAL[kind], kind=kind,
        raw_value=raw, raw_unit=CANONICAL[kind],
    )


def safe_normalize(fn, raw: Any, *args, **kwargs) -> Quantity:
    """规范化失败时不丢弃：返回停在原位并带诊断的 Quantity。"""
    try:
        return fn(raw, *args, **kwargs)
    except Unconvertible as exc:
        placeholder = 0.0
        unit = {"normalize_velocity": "m/s", "normalize_angle": "rad",
                "normalize_depth": "m"}.get(getattr(fn, "__name__", ""), "unknown")
        return Quantity(
            value=placeholder, unit=unit,
            kind="velocity" if "velocity" in fn.__name__ else
                 "angle" if "angle" in fn.__name__ else
                 "depth" if "depth" in fn.__name__ else "unknown",
            raw_value=raw, raw_unit=kwargs.get("raw_unit", args[0] if args else None),
            datum=kwargs.get("datum"), diagnostic=str(exc),
        )
