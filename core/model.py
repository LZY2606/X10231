"""源数据模型：波束测量、correlation、amplitude、坏 beam 标志、姿态、磁偏角、
安装角、换能器配置、bottom track、深度 cell 各自独立来源、独立版本。

导入时保留原始表示（raw + raw_unit）；派生层才做规范化。无法转换的值留在
原位置并附诊断（见 core.units），不会因为要画图而被丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import TransducerConfig
from .quality import QualityRules
from .rotation import MountingAngles
from .units import Quantity, quantity

SOURCE_KINDS = (
    "beam_velocity", "correlation", "amplitude", "bad_beam",
    "attitude", "declination", "mounting", "geometry",
    "bottom_track", "depth_cell",
)


@dataclass
class PingCellRaw:
    depth_raw: Any
    depth_unit: str = "m"
    invalid: bool = False  # 显式无效深度 cell 标志（独立于质量阈值）


@dataclass
class PingRaw:
    ping: int
    time_s: float
    beam_velocity_raw: list[list[Any]]     # [cell][beam]，单位在来源头部
    velocity_unit: str
    correlation: list[list[int | None]]    # [cell][beam] counts
    amplitude: list[list[int | None]]      # [cell][beam] counts
    bad_beam: list[list[bool]]             # [cell][beam]
    cells: list[PingCellRaw]
    bottom_track_raw: list[Any] | None = None  # 4 波束底跟踪径向速度
    bottom_track_unit: str = "mm/s"

    def cell_count(self) -> int:
        return len(self.cells)


@dataclass
class ProfileSources:
    """一次计算所看到的全部来源的不可变快照（每个来源带版本 id）。"""

    pings: list[PingRaw]
    attitude: dict[str, Any]                # times/heading/pitch/roll + version
    declination: dict[str, Any]             # deg, east-positive + version
    mounting: MountingAngles
    mounting_version: str
    geometry: TransducerConfig
    geometry_version: str
    rules: QualityRules
    rules_version: str
    transform_order: str = "HPR"
    transform_version: str = "rules-v1"
    attitude_max_gap_s: float = 2.0
    source_versions: dict[str, str] = field(default_factory=dict)

    def ping(self, ping: int) -> PingRaw:
        return self.pings[ping]


def beam_quantities(raw: list[Any], unit: str) -> list[Quantity]:
    return [quantity(v, unit, "speed", frame="beam") for v in raw]


def depth_quantity(raw_cell: PingCellRaw) -> Quantity:
    return quantity(raw_cell.depth_raw, raw_cell.depth_unit, "length", frame="earth")


def validate_snapshot(data: dict[str, Any]) -> list[str]:
    """导入校验：返回诊断列表（空列表表示通过）。不删除任何原始值。"""
    diags: list[str] = []
    if "pings" not in data or not data["pings"]:
        diags.append("no pings")
        return diags
    n = None
    for p in data["pings"]:
        c = len(p["beam_velocity_raw"])
        if n is None:
            n = c
        elif c != n:
            diags.append(f"ping {p['ping']}: cell count {c} != {n}")
        for i, row in enumerate(p["beam_velocity_raw"]):
            if len(row) != 4:
                diags.append(f"ping {p['ping']} cell {i}: expected 4 beams")
    for key in ("attitude", "declination", "mounting", "geometry"):
        if key not in data:
            diags.append(f"missing source kind: {key}")
    return diags
