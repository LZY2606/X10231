"""领域量的单位与坐标基准。

导入层保留原始表示（raw_value/raw_unit），派生层才规范化为内部单位。
无法转换的值不丢弃，留在原位置并附诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# 内部规范单位
CANONICAL = {
    "velocity": "m/s",
    "angle": "deg",
    "time": "s",
    "length": "m",
    "dimensionless": "1",
}

# 每类量纲的线性换算因子: value_canonical = value * factor + offset
_UNIT_TABLE: Dict[tuple, tuple] = {
    # velocity
    ("m/s", "m/s"): (1.0, 0.0),
    ("mm/s", "m/s"): (1e-3, 0.0),
    ("cm/s", "m/s"): (1e-2, 0.0),
    ("knot", "m/s"): (0.5144444444, 0.0),
    # angle (offset 仅用于 kelvin 等；角度不做偏移换算)
    ("deg", "deg"): (1.0, 0.0),
    ("rad", "deg"): (57.29577951308232, 0.0),
    ("gon", "deg"): (0.9, 0.0),
    # time
    ("s", "s"): (1.0, 0.0),
    ("ms", "s"): (1e-3, 0.0),
    ("min", "s"): (60.0, 0.0),
    # length
    ("m", "m"): (1.0, 0.0),
    ("cm", "m"): (1e-2, 0.0),
    ("ft", "m"): (0.3048, 0.0),
    # dimensionless
    ("1", "1"): (1.0, 0.0),
    ("%", "1"): (0.01, 0.0),
}


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    """一个携带量纲/原始表示的值。value 是规范化后的值（无法转换时为 None）。"""

    value: Optional[float]
    canonical_unit: Optional[str]
    raw_value: Any
    raw_unit: Optional[str]
    dim: Optional[str]
    basis: Optional[str] = None  # 坐标基准，如 FRAME_BEAM / FRAME_INSTRUMENT
    diagnostics: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.canonical_unit,
            "raw_value": self.raw_value,
            "raw_unit": self.raw_unit,
            "dim": self.dim,
            "basis": self.basis,
            "diagnostics": list(self.diagnostics),
        }

    @property
    def ok(self) -> bool:
        return self.value is not None and not any(d["severity"] == "error" for d in self.diagnostics)


def normalize(raw_value: Any, raw_unit: Optional[str], dim: str, basis: Optional[str] = None) -> Quantity:
    """把原始表示规范化；单位缺失/不认识时保留原值并附诊断，绝不丢值。"""
    target = CANONICAL[dim]
    diags = []
    value: Optional[float] = None
    if raw_value is None:
        diags.append({"severity": "error", "code": "missing_value",
                      "message": "value absent; original representation retained"})
    elif raw_unit is None:
        diags.append({"severity": "error", "code": "missing_unit",
                      "message": f"unit absent for dim {dim}; kept raw {raw_value!r}"})
    else:
        key = (raw_unit, target)
        if key not in _UNIT_TABLE:
            diags.append({"severity": "error", "code": "unconvertible_unit",
                          "message": f"no conversion {raw_unit} -> {target}; raw value retained"})
        else:
            try:
                value = float(raw_value) * _UNIT_TABLE[key][0] + _UNIT_TABLE[key][1]
            except (TypeError, ValueError):
                diags.append({"severity": "error", "code": "non_numeric",
                              "message": f"raw value {raw_value!r} not numeric; retained in place"})
                value = None
    return Quantity(value=value, canonical_unit=target if value is not None else None,
                    raw_value=raw_value, raw_unit=raw_unit, dim=dim, basis=basis,
                    diagnostics=tuple(diags))
