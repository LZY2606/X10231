"""换能器波束几何与 beam velocity -> instrument 坐标求解。

约定（在 geometry 配置中显式记录，避免与厂商手册的符号差异混淆）：
- instrument 坐标：x 向前(forward)，y 向右舷(starboard)，z 向上(up)。
- 凸型 Janus 四个换能器向下偏斜，波束方向单位向量 d_i 指向“离开换能器”。
- 径向速度 b_i 以“沿波束离开换能器”为正，因此 b_i = -v · d_i，即 b = -G v。
- 四波束凸 Janus 存在零空间 n=(+1,-1,+1,-1)/2；其投影为 error velocity。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .units import DEG


@dataclass(frozen=True)
class BeamGeometry:
    geometry_id: str
    name: str
    beam_angle_deg: float          # 波束与仪器 z 轴夹角（凸角），RDI 常见 20°
    azimuth_deg: Tuple[float, ...]  # 各波束水平方位（从 +x 朝 +y 计）
    sign_convention: str = "away_positive"
    n_beams: int = 4

    @property
    def directions(self) -> np.ndarray:
        th = self.beam_angle_deg * DEG
        s, c = np.sin(th), np.cos(th)
        az = np.asarray(self.azimuth_deg, dtype=float) * DEG
        z = -np.ones(self.n_beams) * c  # 向下看换能器：波束指向 -z
        return np.column_stack([s * np.cos(az), s * np.sin(az), z])

    @property
    def G(self) -> np.ndarray:
        # b = -G v；G 的行是波束方向。
        return -self.directions


# 标准 RDI 凸型 Janus：波束 1..4 方位 45/135/225/315
RDI_CONVEX_JANUS_20 = BeamGeometry(
    geometry_id="rdi-convex-janus-20-v1",
    name="RDI convex Janus 4-beam, 20 deg",
    beam_angle_deg=20.0,
    azimuth_deg=(45.0, 135.0, 225.0, 315.0),
)

GEOMETRIES: Dict[str, BeamGeometry] = {RDI_CONVEX_JANUS_20.geometry_id: RDI_CONVEX_JANUS_20}

# 四波束误差速度零空间（|n|=1）
ERROR_NULL = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0


@dataclass
class BeamSolution:
    method: str                       # "four_beam_lsq" | "three_beam_omit{i}"
    velocity: Tuple[float, float, float]  # instrument 坐标 (m/s)
    used_beams: Tuple[int, ...]
    error_velocity: Optional[float] = None
    resolvable: bool = True
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "velocity": list(self.velocity),
            "used_beams": list(self.used_beams),
            "error_velocity": self.error_velocity,
            "resolvable": self.resolvable,
            "note": self.note,
        }


def solve_beams(radials: Sequence[float], good: Sequence[bool],
                geometry: BeamGeometry) -> Dict[str, object]:
    """给定各波束径向速度(m/s)与好坏标志，返回四波束解与全部合法三波束候选。

    绝不把三波束候选合并成单一值；候选数量与可解性都显式返回。
    """
    b = np.asarray(radials, dtype=float)
    ok = np.asarray(good, dtype=bool)
    G = geometry.G
    good_idx = [i for i in range(geometry.n_beams) if ok[i]]

    candidates: List[BeamSolution] = []
    four: Optional[BeamSolution] = None

    if len(good_idx) == 4:
        v, *_ = np.linalg.lstsq(G, b, rcond=None)
        err = float(ERROR_NULL @ b)
        four = BeamSolution("four_beam_lsq", tuple(np.round(v, 9)), (0, 1, 2, 3), err)
        candidates.append(four)

    if len(good_idx) >= 3:
        # 每个由 3 个好波束构成的子集都是一个合法（但互不相同）候选。
        from itertools import combinations

        for combo in combinations(good_idx, 3):
            idx = list(combo)
            G3 = G[idx, :]
            if np.linalg.matrix_rank(G3) < 3:
                candidates.append(BeamSolution(
                    f"three_beam_omit{_omitted(idx, geometry.n_beams)}",
                    (float("nan"),) * 3, tuple(idx), None,
                    resolvable=False, note="rank deficient: 3 beams do not span instrument frame"))
                continue
            v = np.linalg.solve(G3, b[idx])
            omitted = _omitted(idx, geometry.n_beams)
            cand = BeamSolution(f"three_beam_omit{omitted}", tuple(np.round(v, 9)), tuple(idx))
            candidates.append(cand)

    unresolvable = len(good_idx) < 3
    return {
        "good_beams": good_idx,
        "n_good": len(good_idx),
        "four_beam": four.to_dict() if four else None,
        "candidates": [c.to_dict() for c in candidates],
        "resolvable": not unresolvable,
        "diagnostic": None if not unresolvable else
        f"only {len(good_idx)} good beam(s); instrument velocity underdetermined",
    }


def _omitted(used: Sequence[int], n: int) -> int:
    for i in range(n):
        if i not in used:
            return i
    return -1
