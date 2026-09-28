"""换能器波束几何（beam -> instrument 方向余弦）。

仪器坐标系：x=船首方向，y=右舷，z=向下（右手系）。
默认 convex Janus 4 波束，俯角（自垂直方向量）20°，
方位角依次 45/135/225/315 度（beam1 右前、beam2 左前、beam3 左后、beam4 右后）。
几何参数全部进入派生指纹。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

GEOMETRY_VERSION = "rdi-janus-convex-v1"
DEFAULT_AZIMUTHS = (45.0, 135.0, 225.0, 315.0)


@dataclass(frozen=True)
class TransducerGeometry:
    config_id: str
    beam_angle_deg: float = 20.0
    azimuths_deg: tuple[float, ...] = DEFAULT_AZIMUTHS
    upward: bool = False
    beam_angle_basis: str = "from_vertical"  # from_vertical / from_horizontal
    geometry_version: str = GEOMETRY_VERSION

    def __post_init__(self) -> None:
        if self.beam_angle_basis not in ("from_vertical", "from_horizontal"):
            raise ValueError(f"未知波束角基准: {self.beam_angle_basis}")
        if len(self.azimuths_deg) != 4:
            raise ValueError("当前几何版本要求 4 个波束方位角")
        if not 0.0 < self.beam_angle_deg < 90.0:
            raise ValueError("波束角必须位于 (0, 90) 度")

    @property
    def n_beams(self) -> int:
        return 4

    def beam_angle_from_vertical(self) -> float:
        if self.beam_angle_basis == "from_vertical":
            return self.beam_angle_deg
        return 90.0 - self.beam_angle_deg

    def beam_matrix(self) -> np.ndarray:
        """B[k] 为第 k 波束单位方向在仪器系下的分量（向下为 +z）。"""
        theta = math.radians(self.beam_angle_from_vertical())
        sign = -1.0 if self.upward else 1.0
        rows = []
        for az in self.azimuths_deg:
            phi = math.radians(az)
            rows.append(
                [
                    sign * math.sin(theta) * math.cos(phi),
                    sign * math.sin(theta) * math.sin(phi),
                    sign * math.cos(theta),
                ]
            )
        return np.array(rows, dtype=float)

    def subset_matrix(self, beam_indices: Sequence[int]) -> np.ndarray:
        matrix = self.beam_matrix()
        return matrix[list(beam_indices), :]

    def null_axis(self) -> np.ndarray:
        """四波束冗余零空间方向（不可观测组合），单位化。"""
        theta = math.radians(self.beam_angle_from_vertical())
        s = math.sin(theta)
        # 与默认方位角配对一致：b1,b3 抵消的水平组合
        null_4 = np.array([0.0, -s * s, 0.0, s * s])
        norm = np.linalg.norm(null_4)
        return null_4 / norm

    def fingerprint(self) -> str:
        payload = {
            "geometry_version": self.geometry_version,
            "config_id": self.config_id,
            "beam_angle_deg": round(self.beam_angle_deg, 9),
            "beam_angle_basis": self.beam_angle_basis,
            "azimuths_deg": [round(a, 9) for a in self.azimuths_deg],
            "upward": self.upward,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
