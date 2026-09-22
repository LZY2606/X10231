"""换能器与变换配置（全部带版本，版本进入派生指纹）。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .geometry import BeamGeometry, default_rd_geometry


@dataclass(frozen=True)
class GeometryConfig:
    version: str
    geometry: BeamGeometry

    def fingerprint_input(self) -> dict:
        return {"version": self.version, "geometry": self.geometry.fingerprint_input()}


@dataclass(frozen=True)
class InstallConfig:
    """instrument -> ship 安装角（度，顺序与 order 对应，默认 zyx）。"""
    version: str
    heading_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    order: str = "zyx"

    def angles_rad(self) -> Tuple[float, float, float]:
        return (
            math.radians(self.heading_offset_deg),
            math.radians(self.pitch_offset_deg),
            math.radians(self.roll_offset_deg),
        )

    def fingerprint_input(self) -> dict:
        return {
            "version": self.version,
            "heading_offset_deg": self.heading_offset_deg,
            "pitch_offset_deg": self.pitch_offset_deg,
            "roll_offset_deg": self.roll_offset_deg,
            "order": self.order,
        }


@dataclass(frozen=True)
class DeclinationConfig:
    """磁偏角：TRUE = MAGNETIC + declination_deg。"""
    version: str
    declination_deg: float
    source: str = "manual"

    def fingerprint_input(self) -> dict:
        return {
            "version": self.version,
            "declination_deg": self.declination_deg,
            "source": self.source,
        }


@dataclass(frozen=True)
class TransformConfig:
    """ship -> earth 的姿态旋转应用顺序（h/p/r 的 intrinsic 次序）。"""
    version: str
    ship_earth_order: str = "hpr"

    def fingerprint_input(self) -> dict:
        return {"version": self.version, "ship_earth_order": self.ship_earth_order}
