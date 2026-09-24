"""领域量：始终携带单位与坐标基准；原始表示在源层保留，仅在派生层规范化。

设计约定：
- ``Quantity`` 同时保存原始值、原始单位（或枚举）、规范化后的值与 SI 单位。
- 无法转换时 ``value`` 为 None，诊断写入 ``diagnostic``，值仍留在原位置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Any, Iterable

import numpy as np

# 以 SI（弧度、m/s、米）为规范化目标。单位键是材料包使用的稳定枚举。
_UNIT_TO_SI: dict[tuple[str, str], tuple[str, float, float]] = {
    # (量纲, 原始单位) -> (SI 单位, 乘法系数, 加法偏置)
    ("angle", "deg"): ("rad", np.pi / 180.0, 0.0),
    ("angle", "rad"): ("rad", 1.0, 0.0),
    ("angle", "gon"): ("rad", np.pi / 200.0, 0.0),
    ("speed", "mm/s"): ("m/s", 0.001, 0.0),
    ("speed", "m/s"): ("m/s", 1.0, 0.0),
    ("speed", "cm/s"): ("m/s", 0.01, 0.0),
    ("length", "m"): ("m", 1.0, 0.0),
    ("length", "cm"): ("m", 0.01, 0.0),
    ("length", "ft"): ("m", 0.3048, 0.0),
    ("time", "s"): ("s", 1.0, 0.0),
    ("time", "ms"): ("s", 0.001, 0.0),
}

VALID_FRAMES = {"beam", "instrument", "ship", "earth"}
VALID_EARTH_DATUM = ("ENU", "NED", "XYZ")


class UnitError(ValueError):
    """无法把原始表示转换为规范化表示时抛出；调用方须保留原值与诊断。"""


@dataclass(frozen=True)
class Quantity:
    """一个带单位/坐标基准的领域量；raw 字段在导入后永不改写。"""

    raw: Any
    raw_unit: str
    dimension: str
    si_unit: str | None = None
    frame: str | None = None
    datum: str | None = None
    value: float | None = None
    diagnostic: str | None = None

    @property
    def convertible(self) -> bool:
        return self.value is not None and self.diagnostic is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "raw_unit": self.raw_unit,
            "dimension": self.dimension,
            "si_unit": self.si_unit,
            "frame": self.frame,
            "datum": self.datum,
            "value": self.value,
            "diagnostic": self.diagnostic,
        }


def quantity(
    raw: Any,
    raw_unit: str,
    dimension: str,
    *,
    frame: str | None = None,
    datum: str | None = None,
) -> Quantity:
    """构造单个领域量；任何异常都转成带诊断的不可转换 Quantity。"""
    if frame is not None and frame not in VALID_FRAMES:
        return Quantity(raw, raw_unit, dimension, frame=frame, datum=datum,
                        diagnostic=f"unknown frame {frame!r}")
    if datum is not None and datum not in VALID_EARTH_DATUM:
        return Quantity(raw, raw_unit, dimension, frame=frame, datum=datum,
                        diagnostic=f"unknown earth datum {datum!r}")
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return Quantity(raw, raw_unit, dimension, frame=frame, datum=datum,
                        diagnostic=f"non-numeric raw value {raw!r}")
    if isnan(num):
        return Quantity(raw, raw_unit, dimension, frame=frame, datum=datum,
                        diagnostic="NaN raw value")
    key = (dimension, raw_unit)
    if key not in _UNIT_TO_SI:
        return Quantity(raw, raw_unit, dimension, frame=frame, datum=datum,
                        diagnostic=f"no conversion for dimension={dimension!r} unit={raw_unit!r}")
    si_unit, scale, offset = _UNIT_TO_SI[key]
    return Quantity(raw=raw, raw_unit=raw_unit, dimension=dimension, si_unit=si_unit,
                    frame=frame, datum=datum, value=num * scale + offset)


def quantity_array(
    raw: Iterable[Any],
    raw_unit: str,
    dimension: str,
    *,
    frame: str | None = None,
    datum: str | None = None,
) -> list[Quantity]:
    return [quantity(v, raw_unit, dimension, frame=frame, datum=datum) for v in raw]


def require(values: Iterable[Quantity], what: str) -> np.ndarray:
    """取出规范化数组；若存在不可转换项，抛出带定位信息的 UnitError。"""
    out: list[float] = []
    bad: list[str] = []
    for i, q in enumerate(values):
        if q.convertible:
            out.append(q.value)  # type: ignore[arg-type]
        else:
            bad.append(f"[{i}]={q.diagnostic}")
    if bad:
        raise UnitError(f"unconvertible quantity while computing {what}: {'; '.join(bad)}")
    return np.asarray(out, dtype=float)
