"""换能器波束几何与 beam->instrument 速度反演（4 波束 Janus）。

instrument 坐标：x 向前、y 向右舷、z 向下（右手系）。
每条波束方向在仪器系下为
    d_i = [sin θ cos az_i, sin θ sin az_i, +cos θ]
beam 测量沿波束方向（凸配置，向下向外）。
波束几何整体带版本指纹，进入所有派生结果。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    beam_angle_deg: float          # 波束与竖直轴夹角 θ
    azimuths_deg: tuple[float, ...]  # 每条波束方位角
    n_beams: int = 4

    @staticmethod
    def rdi_convex(beam_angle_deg: float = 20.0) -> "BeamGeometry":
        # RDI 惯例：beam3 前向、beam4 后向、beam1 左舷、beam2 右舷
        return BeamGeometry(beam_angle_deg, (270.0, 90.0, 0.0, 180.0))

    def directions(self) -> np.ndarray:
        theta = np.deg2rad(self.beam_angle_deg)
        az = np.deg2rad(np.asarray(self.azimuths_deg, dtype=float))
        st = np.sin(theta)
        return np.column_stack([st * np.cos(az), st * np.sin(az), np.full_like(az, np.cos(theta))])

    def matrix(self) -> np.ndarray:
        """A: (n_beams,3)，b_beam = A v_instrument。"""
        return self.directions()

    def null_unit(self) -> np.ndarray:
        """左零空间单位向量 n，满足 n @ A = 0；符号规范化以便指纹稳定。"""
        a = self.matrix()
        # n 满足 n@A=0，即 n 属于 A^T 的零空间，取全尺寸 U 的最后一列
        u, _, _ = np.linalg.svd(a, full_matrices=True)
        n = u[:, -1]
        first = next((x for x in n if abs(x) > 1e-12), 0.0)
        if first < 0:
            n = -n
        return n

    def error_velocity(self, beams: np.ndarray) -> float:
        """RDI 风格误差速度 = n·b（与坐标约定无关的失配量）。"""
        return float(self.null_unit() @ np.asarray(beams, dtype=float))

    def solve_four(self, beams: np.ndarray) -> np.ndarray:
        """四波束最小二乘解（丢弃误差模态）。"""
        a = self.matrix()
        x, *_ = np.linalg.lstsq(a, np.asarray(beams, dtype=float), rcond=None)
        return x

    def triple_indices(self) -> list[tuple[int, ...]]:
        return [tuple(i for i in range(self.n_beams) if i != m) for m in range(self.n_beams)]

    def solve_three(self, beams: np.ndarray, present: tuple[int, ...]) -> np.ndarray:
        """给定可用波束（恰 3 条）精确求逆。"""
        a = self.matrix()[list(present), :]
        b = np.asarray(beams, dtype=float)[list(present)]
        return np.linalg.solve(a, b)

    def triple_is_independent(self, present: tuple[int, ...], tol: float = 1e-9) -> bool:
        a = self.matrix()[list(present), :]
        return abs(np.linalg.det(a)) > tol

    def canonical(self) -> dict:
        return {
            "beam_angle_deg": self.beam_angle_deg,
            "azimuths_deg": [float(a) for a in self.azimuths_deg],
            "n_beams": self.n_beams,
        }

    def version_id(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return "geo-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
