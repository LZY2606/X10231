"""波束几何：把每条 beam 的方向显式参数化，几何本身进入派生指纹。

默认是 RDI 式凸 Janus（convex）四波束：
  beam1 船首向, beam2 船尾向, beam3 右舷, beam4 左舷；
  beam1/3 波束速度对“朝向换能器”的流动为正（RDI 惯例），beam2/4 符号相反。
几何用 beam1 方位角 + 相邻波束方位间隔生成，便于把“波束几何版本”做成独立来源。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    beam_angle_deg: float          # 与 instrument z 轴（向下）夹角，典型 20
    first_azimuth_deg: float       # beam1 在水平面的方位（自 x 轴向 y 轴为正）
    azimuth_step_deg: float        # 相邻波束方位间隔（凸 Janus=90）
    convention: str                # "RDI_CONVEX"，决定奇数/偶数波束径向符号
    n_beams: int = 4
    version: str = "geom-rdi-convex-20deg-v1"

    def directions(self) -> np.ndarray:
        """返回 (n_beams, 3) 单位方向，instrument 系：x 前、y 右、z 下。"""
        theta = math.radians(self.beam_angle_deg)
        dirs = []
        for i in range(self.n_beams):
            az = math.radians(self.first_azimuth_deg + i * self.azimuth_step_deg)
            dirs.append([
                math.sin(theta) * math.cos(az),
                math.sin(theta) * math.sin(az),
                math.cos(theta),
            ])
        return np.asarray(dirs, dtype=float)

    def beam_matrix(self) -> np.ndarray:
        """测得径向速度 b = s_i * (d_i . v_instrument) 构成的矩阵 B，b = B @ v。"""
        dirs = self.directions()
        if self.convention != "RDI_CONVEX":
            raise ValueError("未知波束惯例: %r" % self.convention)
        signs = [1.0, -1.0, 1.0, -1.0][: self.n_beams]
        return dirs * np.asarray(signs)[:, None]

    def fingerprint_basis(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "beam_angle_deg": self.beam_angle_deg,
            "first_azimuth_deg": self.first_azimuth_deg,
            "azimuth_step_deg": self.azimuth_step_deg,
            "convention": self.convention,
            "n_beams": self.n_beams,
        }

    @staticmethod
    def from_config(cfg: Dict[str, object]) -> "BeamGeometry":
        return BeamGeometry(
            beam_angle_deg=float(cfg.get("beam_angle_deg", 20.0)),
            first_azimuth_deg=float(cfg.get("first_azimuth_deg", 0.0)),
            azimuth_step_deg=float(cfg.get("azimuth_step_deg", 90.0)),
            convention=str(cfg.get("convention", "RDI_CONVEX")),
            n_beams=int(cfg.get("n_beams", 4)),
            version=str(cfg.get("version", "geom-rdi-convex-20deg-v1")),
        )


def solve(matrix: np.ndarray, beams: np.ndarray) -> Tuple[np.ndarray, float]:
    """最小二乘解 v 与残差（4 波束时残差即 RDI error velocity 同量级量）。"""
    v, residuals, rank, _ = np.linalg.lstsq(matrix, beams, rcond=None)
    resid = float(np.linalg.norm(matrix @ v - beams))
    return v, resid
