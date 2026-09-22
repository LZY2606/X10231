"""单位与坐标基准。

领域量在原始层保留原始表示（含原始单位字符串）；派生层用 :class:`Quantity`
规范化。无法转换的值停留在原位置并携带 :class:`UnitDiagnostic`，不做静默丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# 允许的换算表：from_unit -> (to_unit, factor)
_LINEAR = {
    ("mm/s", "m/s"): 0.001,
    ("cm/s", "m/s"): 0.01,
    ("m/s", "m/s"): 1.0,
    ("knot", "m/s"): 0.514444,
    ("mm", "m"): 0.001,
    ("cm", "m"): 0.01,
    ("m", "m"): 1.0,
    ("deg", "rad"): 0.017453292519943295,
    ("rad", "rad"): 1.0,
    ("degC", "degC"): 1.0,
    ("count", "count"): 1.0,
    ("dB", "dB"): 1.0,
}

DIMENSION = {
    "mm/s": "velocity", "cm/s": "velocity", "m/s": "velocity", "knot": "velocity",
    "mm": "length", "cm": "length", "m": "length",
    "deg": "angle", "rad": "angle",
    "degC": "temperature", "count": "count", "dB": "amplitude",
}


class UnconvertibleError(ValueError):
    """无法把原始单位换算到目标单位。"""


@dataclass(frozen=True)
class UnitDiagnostic:
    code: str
    message: str
    raw_unit: Optional[str] = None
    target_unit: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "raw_unit": self.raw_unit,
            "target_unit": self.target_unit,
        }


@dataclass(frozen=True)
class Quantity:
    """带单位/坐标基准的规范化标量。"""
    value: float
    unit: str
    datum: Optional[str] = None  # 例如 ENU / MAGNETIC / TRUE / ship-x / inst-x
    raw_unit: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "datum": self.datum,
            "raw_unit": self.raw_unit,
        }


def dimension_of(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    return DIMENSION.get(unit)


def convert(value: float, from_unit: Optional[str], to_unit: str) -> float:
    """线性单位换算；未知单位抛 :class:`UnconvertibleError`。"""
    if from_unit is None:
        raise UnconvertibleError("缺少单位")
    key = (from_unit, to_unit)
    if key in _LINEAR:
        return value * _LINEAR[key]
    # 同单位直通
    if from_unit == to_unit:
        return value
    # 同量纲内未登记的组合不允许
    raise UnconvertibleError(f"无法从 {from_unit!r} 换算到 {to_unit!r}")


def try_convert(
    value, from_unit: Optional[str], to_unit: str
) -> Tuple[Optional[float], Optional[UnitDiagnostic]]:
    """非抛出式换算：返回 (value|None, diagnostic|None)。

    非有限值（NaN/Inf）视为“无法转换”，保留诊断而非抛异常。
    """
    try:
        import math
        if value is None:
            return None, UnitDiagnostic("MISSING_VALUE", "值缺失", from_unit, to_unit)
        if not math.isfinite(float(value)):
            return None, UnitDiagnostic("NON_FINITE", "值不是有限数", from_unit, to_unit)
        return convert(float(value), from_unit, to_unit), None
    except UnconvertibleError as exc:
        return None, UnitDiagnostic("UNCONVERTIBLE_UNIT", str(exc), from_unit, to_unit)


def to_quantity(value, raw_unit: Optional[str], unit: str, datum: Optional[str] = None
                ) -> Tuple[Optional[Quantity], Optional[UnitDiagnostic]]:
    v, diag = try_convert(value, raw_unit, unit)
    if diag is not None:
        return None, diag
    return Quantity(v, unit, datum, raw_unit), None
