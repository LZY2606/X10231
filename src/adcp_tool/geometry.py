"""波束几何与方向余弦。

每条波束由 *方位角 azimuth*（仪器 x 轴起算、绕仪器 z 轴）与 *顶角 zenith*
（波束与仪器 +z 轴夹角）描述。径向速度 b_i 与仪器坐标速度 v 的关系为：

    b = D @ v,  D[i] = (sin(zenith_i) cos(azimuth_i),
                        sin(zenith_i) sin(azimuth_i),
                        cos(zenith_i))

四波束时 D 为 (4,3)，其左零空间 n（Dᵀn=0, ‖n‖=1）定义 error velocity
e = n·b；三波束时 D 为 (3,3) 可逆，无 error velocity，三波束解与四波束解
因此是不同的结果，不混合为同一值。

默认方位角对应 RDI 风格 Janus 四波束（凸阵），可在换能器配置中替换；
波束几何版本进入派生指纹。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    azimuth_deg: Tuple[float, ...]      # 每条波束方位角（度）
    zenith_deg: Tuple[float, ...]       # 每条波束顶角（度）
    convention: str = "RDI_JANUS"       # 几何命名（进入指纹）

    @property
    def n_beams(self) -> int:
        return len(self.azimuth_deg)

    def direction_cosines(self) -> np.ndarray:
        rows = []
        for az, zn in zip(self.azimuth_deg, self.zenith_deg):
            a = math.radians(az)
            t = math.radians(zn)
            rows.append([
                math.sin(t) * math.cos(a),
                math.sin(t) * math.sin(a),
                math.cos(t),
            ])
        return np.asarray(rows, dtype=float)

    def D(self, beams: Sequence[int]) -> np.ndarray:
        """选定波束（0 基索引）对应的方向余弦矩阵。"""
        full = self.direction_cosines()
        return full[list(beams), :]

    def null_vector(self, beams: Sequence[int]) -> np.ndarray:
        """选定波束集合下的左零空间单位向量。

        4 波束返回长度 4 的左零空间向量；3 波束无零空间（返回零向量占位，
        但三波束路径不会使用它）。
        """
        D = self.D(beams)
        if D.shape[0] <= D.shape[1]:
            return np.zeros(D.shape[0])
        # D (m x 3), 左零空间 = D 行空间在 R^m 中的正交补
        u, s, vh = np.linalg.svd(D, full_matrices=True)
        null_mask = np.ones(D.shape[0], dtype=bool)
        null_mask[: D.shape[1]] = False
        n = u[:, null_mask].flatten()
        return n / np.linalg.norm(n)

    def solve(self, b: np.ndarray, beams: Sequence[int]) -> Tuple[np.ndarray, float]:
        """由径向速度解仪器坐标速度。

        返回 (v_xyz, error_velocity)。三波束时 error_velocity 为 NaN
        （该解没有四波束冗余定义）。
        """
        D = self.D(beams)
        b = np.asarray(b, dtype=float)
        if D.shape[0] == D.shape[1]:
            v = np.linalg.solve(D, b)
            return v, float("nan")
        v, *_ = np.linalg.lstsq(D, b, rcond=None)
        n = self.null_vector(beams)
        err = float(n @ b)
        return v, err

    def fingerprint_input(self) -> dict:
        return {
            "convention": self.convention,
            "azimuth_deg": [float(a) for a in self.azimuth_deg],
            "zenith_deg": [float(z) for z in self.zenith_deg],
        }


def default_rd_geometry(n_beams: int = 4, zenith_deg: float = 20.0) -> BeamGeometry:
    """RDI 风格默认几何：波束 1..4 方位角 45/135/225/315 度。"""
    azimuths = tuple(45.0 + 90.0 * i for i in range(n_beams))
    return BeamGeometry(
        azimuth_deg=azimuths,
        zenith_deg=tuple(zenith_deg for _ in range(n_beams)),
        convention=f"RDI_JANUS_B{n_beams}",
    )
