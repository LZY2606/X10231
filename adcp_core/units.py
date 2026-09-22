"""领域量的单位与坐标基准。

导入层保留原始表示（raw_value + raw_unit），派生层才规范化为 SI。
无法转换的值不被丢弃：Q.raw() 产生一个 value=None 的量化值并带诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# 允许的单位（键是规范化单位，值是允许出现的原始别名）
_UNIT_ALIASES = {
    "m/s": ["m/s", "mps", "meter_per_second"],
    "mm/s": ["mm/s"],
    "cm/s": ["cm/s"],
    "knot": ["knot", "kn", "knots"],
    "deg": ["deg", "degree", "degrees", "°"],
    "rad": ["rad", "radian", "radians"],
    "m": ["m", "meter", "meters"],
    "bin": ["bin"],
    "s": ["s", "sec", "second", "seconds"],
    "count": ["count", "cnt", "corr_counts"],
    "dB": ["dB", "db", "decibel"],
    "percent": ["percent", "%", "pct"],
    "": ["none", "", "unitless"],
}
_TO_SI = {
    "mm/s": ("m/s", 0.001),
    "cm/s": ("m/s", 0.01),
    "knot": ("m/s", 0.5144444444444445),
    "rad": ("deg", 57.29577951308232),
}


@dataclass
class Q:
    """带单位/基准的量化值。value 为 None 表示无法转换（诊断保留在 diagnostics）。"""

    raw_value: Any
    raw_unit: str
    value: Optional[float]
    unit: str
    datum: str = ""          # 坐标基准，如 ENU、SHIP-FSD、BEAM-RDI-CONVEX，非空间量为空
    diagnostics: List[str] = field(default_factory=list)

    @staticmethod
    def of(raw_value: Any, raw_unit: str, datum: str = "") -> "Q":
        if raw_value is None:
            return Q(None, raw_unit, None, raw_unit, datum,
                     ["E_MISSING_VALUE: 原始值缺失，保留原位置"])
        try:
            v = float(raw_value)
        except (TypeError, ValueError):
            return Q(raw_value, raw_unit, None, raw_unit, datum,
                     ["E_NON_NUMERIC: 值 %r 无法转换为数值" % (raw_value,)])
        canon = _canon_unit(raw_unit)
        if canon is None:
            return Q(raw_value, raw_unit, None, raw_unit, datum,
                     ["E_UNKNOWN_UNIT: 未知单位 %r，未做规范化" % (raw_unit,)])
        if canon in _TO_SI:
            unit, factor = _TO_SI[canon]
            return Q(raw_value, raw_unit, v * factor, unit, datum)
        return Q(raw_value, raw_unit, v, canon, datum)

    @staticmethod
    def raw(raw_value: Any, raw_unit: str, diagnostic: str, datum: str = "") -> "Q":
        return Q(raw_value, raw_unit, None, raw_unit, datum, [diagnostic])

    def ok(self) -> bool:
        return self.value is not None and not any(d.startswith("E_") for d in self.diagnostics)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Q":
        return Q(**d)


def _canon_unit(unit: str) -> Optional[str]:
    u = (unit or "").strip()
    for canon, aliases in _UNIT_ALIASES.items():
        if u in aliases:
            return canon
    return None


def convert_angle(raw_value: Any, raw_unit: str) -> Q:
    """角度导入：规范化为度，保留原值。"""
    q = Q.of(raw_value, raw_unit)
    return q
