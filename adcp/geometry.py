"""波束几何：Janus 四波束换能器配置。

几何版本进入派生指纹：几何修订会让所有 instrument 解失效。

约定（RDI 风格）：
  instrument 轴 X=fore(船首), Y=starboard(右舷), Z=down(向下)；右手系。
  波束顺序 0..3 = fore, starboard, aft, port（凸式/向下）。
  beam radial 正值表示沿波束向外（远离换能器）；r_i = v . d_i。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fingerprint import digest


@dataclass(frozen=True)
class BeamGeometry:
    beam_angle_deg: float = 20.0      # 波束与 Z 轴夹角
    convex: bool = True               # 凸式（向下）；False=凹式符号翻转
    orientation: str = "rdi_janus"    # fore,starboard,aft,port
    version: str = "geom:janus20:v1"

    @property
    def theta(self) -> float:
        return math.radians(self.beam_angle_deg)

    def beam_unit_vectors(self) -> np.ndarray:
        """返回 4x3 波束指向单位向量（波束向外为正）。"""
        s, c = math.sin(self.theta), math.cos(self.theta)
        sign = 1.0 if self.convex else -1.0
        # fore, starboard, aft, port —— X/Y 分量与凸式一致，z 向 sign*c
        return sign * np.array([
            [ s,  0.0, c],
            [0.0,  s, c],
            [-s,  0.0, c],
            [0.0, -s, c],
        ], dtype=float)

    def solve_matrix(self, beams) -> np.ndarray:
        """给定有效波束下标序列，返回最小二乘矩阵 A（行=有效波束）。"""
        vecs = self.beam_unit_vectors()
        return vecs[list(beams), :]

    def radial_to_xyz(self, radial: np.ndarray, beams):
        """最小二乘解 v 使 r = A v。返回 (xyz(3,), residuals)。"""
        A = self.solve_matrix(beams)
        xyz, *_ = np.linalg.lstsq(A, np.asarray(radial, float), rcond=None)
        residuals = np.asarray(radial, float) - A @ xyz
        return xyz, residuals

    @property
    def null_vector(self) -> np.ndarray:
        """四波束矩阵零空间方向（误差速度方向），归一化。"""
        return np.array([0.0, 0.0, 1.0])

    def error_velocity(self, radial: np.ndarray) -> float:
        """RDI 风格 error velocity（缩放后等价于 A 零空间投影）。

        e = (r0 - r1 + r2 - r3) / (4 cos(theta))
        无噪声、纯 3D 流时 e=0；它是四波束解独有的质量量，
        三波束解不存在单一 error velocity（故不与四波束混值）。
        """
        r = np.asarray(radial, float)
        return float((r[0] - r[1] + r[2] - r[3]) / (4.0 * math.cos(self.theta)))

    def fingerprint_material(self) -> dict:
        return {
            "geometry_version": self.version,
            "beam_angle_deg": self.beam_angle_deg,
            "convex": self.convex,
            "orientation": self.orientation,
            "vectors": np.round(self.beam_unit_vectors(), 10).tolist(),
        }

    def content_hash(self) -> str:
        return "geom:" + digest(self.fingerprint_material())[:12]
