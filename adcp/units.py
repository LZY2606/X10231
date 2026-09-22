"""领域量：携带单位与坐标基准。

导入时保留原始表示（raw）；只有派生层才规范化到 canonical 单位。
无法转换的值停在原位置：value=None 且附带 diagnostic，绝不静默丢弃。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional

# canonical 单位：velocity -> m/s，angle -> deg
_TO_CANONICAL = {
    "velocity": {
        "m/s": 1.0,
        "m s-1": 1.0,
        "ms-1": 1.0,
        "cm/s": 0.01,
        "mm/s": 0.001,
        "knot": 0.514444,
        "knots": 0.514444,
    },
    "angle": {
        "deg": 1.0,
        "degree": 1.0,
        "degrees": 1.0,
        "rad": 180.0 / math.pi,
        "radian": 180.0 / math.pi,
    },
    "dimensionless": {"": 1.0, "count": 1.0, "%": 0.01},
    "length": {"m": 1.0, "meter": 1.0, "cm": 0.01, "ft": 0.3048, "feet": 0.3048},
}

CANONICAL_UNIT = {"velocity": "m/s", "angle": "deg", "dimensionless": "",
                   "length": "m"}


@dataclass(frozen=True)
class Quantity:
    """一个带单位/基准的领域量；不可变。"""

    value: Optional[float]
    unit: str                       # canonical 单位（规范化后）
    raw: str                        # 导入时的原始表示
    datum: Optional[str] = None     # 坐标基准，如 "magnetic"/"true"/"earth:ENU"
    kind: str = "number"            # velocity|angle|dimensionless|time|flag
    diagnostic: Optional[str] = None

    def with_value(self, **changes) -> "Quantity":
        return replace(self, **changes)


def parse_quantity(raw, declared_unit=None, datum=None, kind="number"):
    """把原始表示解析为 Quantity；永不抛异常，坏值留在原位置附诊断。"""
    raw_s = "" if raw is None else str(raw).strip()
    if kind in ("flag", "text"):
        return Quantity(value=None, unit="", raw=raw_s, datum=datum, kind=kind)
    if kind == "time":
        return _parse_time(raw_s, datum)
    unit = (declared_unit or "").strip()
    table = _TO_CANONICAL.get(kind)
    if table is None:
        return Quantity(None, unit or "", raw_s, datum, kind,
                        diagnostic=f"unknown kind '{kind}'")
    norm_unit = unit.lower()
    factor = table.get(norm_unit)
    if norm_unit not in table:
        return Quantity(None, unit, raw_s, datum, kind,
                        diagnostic=f"unconvertible unit '{unit}' for kind '{kind}'")
    try:
        num = float(raw_s)
    except (TypeError, ValueError):
        return Quantity(None, CANONICAL_UNIT[kind], raw_s, datum, kind,
                        diagnostic=f"unparseable value '{raw_s}'")
    if math.isnan(num):
        return Quantity(None, CANONICAL_UNIT[kind], raw_s, datum, kind,
                        diagnostic="NaN value")
    return Quantity(num * factor, CANONICAL_UNIT[kind], raw_s, datum, kind)


def _parse_time(raw_s, datum):
    try:
        dt = datetime.fromisoformat(raw_s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return Quantity(dt.timestamp(), "epoch_s", raw_s, datum or "utc", "time")
    except ValueError:
        return Quantity(None, "epoch_s", raw_s, datum, "time",
                        diagnostic=f"unparseable timestamp '{raw_s}'")


def convert(value, unit, kind):
    """显式把某个已解析单位的值换算到 canonical；供派生层使用。"""
    factor = _TO_CANONICAL[kind][unit.lower()]
    return value * factor


def format_q(q: Quantity, ndigits=4):
    if q.value is None:
        return f"<unconvertible:{q.raw}>"
    if q.kind == "time":
        return datetime.fromtimestamp(q.value, tz=timezone.utc).isoformat()
    return f"{round(q.value, ndigits)} {q.unit}".rstrip()
