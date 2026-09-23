"""换能器波束几何：radial beam velocity <-> instrument 坐标速度。

采用 RDI Janus 凸阵约定（beam angle θ 从竖直量起，波束径向速度
正方向为“远离换能器”）：

    b1 = (-s, 0, +c)      b2 = (0, -s, +c)
    b3 = (+s, 0, +c)      b4 = (0, +s, +c)

于是在理想对称几何下：
    u = (b3 - b1) / (2 sin θ)
    v = (b4 - b2) / (2 sin θ)
    w = (b1+b2+b3+b4) / (4 cos θ)
    error velocity = ((b1+b3)-(b2+b4)) / (4 cos θ)

三波束解：对剩余三行方向矩阵做精确 3x3 求解，无错误速度自由度。
四波束解：最小二乘 + 错误速度。二者从不合并为同一个值
（见 engine 中 candidate 的独立身份）。
凹阵 (convex=False) 仅翻转竖直分量，作为几何指纹的一部分。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

BEAMS = 4


@dataclass(frozen=True)
class BeamGeometry:
    beam_angle_deg: float = 20.0
    convex: bool = True
    name: str = "rdi_janus"

    @property
    def theta(self) -> float:
        return np.deg2rad(self.beam_angle_deg)

    @property
    def directions(self) -> np.ndarray:
        """4x3 单位波束方向矩阵（instrument 系）。"""
        s = np.sin(self.theta)
        c = np.cos(self.theta) * (1.0 if self.convex else -1.0)
        return np.array([
            [-s, 0.0, c],
            [0.0, -s, c],
            [+s, 0.0, c],
            [0.0, +s, c],
        ], dtype=float)

    def matrix(self, valid=None) -> np.ndarray:
        g = self.directions
        if valid is None:
            return g
        return g[np.asarray(valid, dtype=bool)]

    def forward(self, vuw: np.ndarray) -> np.ndarray:
        """instrument (u,v,w) -> 4 个波束径向速度（用于案例正演）。"""
        return self.directions @ np.asarray(vuw, dtype=float)

    def solve_four(self, radials: np.ndarray) -> dict:
        """四波束最小二乘解 + 错误速度。"""
        r = np.asarray(radials, dtype=float)
        sol, residuals, rank, _ = np.linalg.lstsq(self.directions, r, rcond=None)
        s = np.sin(self.theta)
        c = np.cos(self.theta)
        error = ((r[0] + r[2]) - (r[1] + r[3])) / (4.0 * c)
        fitted = self.directions @ sol
        return {
            "uvw": sol,
            "error_velocity": float(error),
            "residual": float(np.linalg.norm(r - fitted)),
            "rank": int(rank),
            "kind": "four_beam",
        }

    def solve_three(self, radials: np.ndarray, missing: int) -> dict:
        """指定缺失波束序号 (0-based) 的精确三波束解。"""
        valid = np.array([i != missing for i in range(BEAMS)])
        g3 = self.directions[valid]
        r3 = np.asarray(radials, dtype=float)[valid]
        sol = np.linalg.solve(g3, r3)
        # 用缺失波束的预测值反算，便于与四波束解对比
        predicted_missing = float(self.directions[missing] @ sol)
        return {
            "uvw": sol,
            "missing_beam": missing,
            "predicted_missing_radial": predicted_missing,
            "residual": 0.0,
            "rank": 3,
            "kind": "three_beam",
        }

    def fingerprint(self) -> str:
        payload = {
            "geom": self.name,
            "beam_angle_deg": self.beam_angle_deg,
            "convex": self.convex,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
