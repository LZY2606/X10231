"""波束几何: beam velocity <-> instrument 坐标。

红线语义:
1. 四束解与三束解**绝不混为一值**: 三束是欠定系统 (3 方程, 4 未知),
   只有一个连续解族; 本模块返回全部合法候选与显式假设, 不压成单一答案。
2. 波束几何 (方位角/倾角/符号约定/顺序) 是规则版本的一部分, 进入派生指纹。

约定 (与 RD Instruments 4 束 Janus convex 配置一致, 版本固化在 GEOMETRY_RULE):
   beam1 azimuth= 45deg, beam2=135deg, beam3=225deg, beam4=315deg
   入射角 theta=20deg; beam 径向速度正方向 = 朝向换能器 (sign=-1)。
   行顺序 [instrument_x, instrument_y, instrument_z, error]:
       b1:  sinθ·cos45,  sinθ·sin45, -cosθ, +1
       b2: -sinθ·cos45,  sinθ·sin45, -cosθ, +1
       b3: -sinθ·cos45, -sinθ·sin45, -cosθ, +1
       b4:  sinθ·cos45, -sinθ·sin45, -cosθ, +1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GEOMETRY_RULE = "rdi-janus-convex/v1"


@dataclass(frozen=True)
class BeamGeometry:
    """换能器波束几何配置 (安装修订会换版本, 不改原始记录)。"""

    azimuths_deg: Tuple[float, ...] = (45.0, 135.0, 225.0, 315.0)
    beam_angle_deg: float = 20.0
    sign: int = -1  # beam velocity 正向 = 朝向换能器
    rule: str = GEOMETRY_RULE

    def forward_matrix(self) -> np.ndarray:
        """B: instrument 分量 [u, v, w, e] -> 4 个 beam 径向速度。"""
        theta = math.radians(self.beam_angle_deg)
        s, c = math.sin(theta), math.cos(theta)
        rows = []
        for az in self.azimuths_deg:
            a = math.radians(az)
            rows.append([
                self.sign * s * math.cos(a),
                self.sign * s * math.sin(a),
                self.sign * -c,
                1.0,
            ])
        return np.array(rows, dtype=float)

    def as_dict(self) -> Dict[str, object]:
        return {
            "azimuths_deg": list(self.azimuths_deg),
            "beam_angle_deg": self.beam_angle_deg,
            "sign": self.sign,
            "rule": self.rule,
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "BeamGeometry":
        return BeamGeometry(
            azimuths_deg=tuple(d.get("azimuths_deg", (45.0, 135.0, 225.0, 315.0))),
            beam_angle_deg=float(d.get("beam_angle_deg", 20.0)),
            sign=int(d.get("sign", -1)),
            rule=str(d.get("rule", GEOMETRY_RULE)),
        )


@dataclass(frozen=True)
class FourBeamSolution:
    """四束唯一解; error velocity 是可观测的冗余坐标, 不是坏值。"""

    u: float
    v: float
    w: float
    e: float
    quality: str  # "ok" | "near_singular"

    def vector(self) -> Tuple[float, float, float]:
        return self.u, self.v, self.w


@dataclass(frozen=True)
class ThreeBeamCandidate:
    """三束解族中的一个候选; assumption 明示它牺牲了什么。"""

    assumption: str
    u: float
    v: float
    w: float
    e: float
    note: str = ""


def solve_four_beams(beams: Sequence[Optional[float]],
                     geom: BeamGeometry = BeamGeometry(),
                     singular_threshold: float = 1e-9
                     ) -> FourBeamSolution:
    """四束 -> instrument, 唯一解。缺束调用方必须改走 solve_three_beams。"""
    if len(beams) != 4 or any(x is None for x in beams):
        raise ValueError("solve_four_beams requires all four good beams")
    inv = np.linalg.inv(geom.forward_matrix())
    sol = inv @ np.array(beams, dtype=float)
    quality = "ok"
    if abs(np.linalg.det(geom.forward_matrix())) < singular_threshold:
        quality = "near_singular"
    return FourBeamSolution(float(sol[0]), float(sol[1]),
                            float(sol[2]), float(sol[3]), quality)


def solve_three_beams(beams: Sequence[Optional[float]],
                      geom: BeamGeometry = BeamGeometry()
                      ) -> Tuple[List[ThreeBeamCandidate], int]:
    """三束 -> instrument 解族。

    返回 (候选列表, 失效波束 0-based 索引)。
    欠定系统 B_3 x = b (3x4) 的通解 = x_p(t) + t * null。
    我们同时给出:
      - zero_error:    假设 error velocity = 0 (经典 RDI 三波束解)
      - minimum_norm:  最小范数解 (信息最少的无偏候选)
      - zero_vertical: 假设 w = 0 (浅水均匀流候选)
    三者一般不同, 全部保留; null_direction/null_scale 描述连续解族,
    UI 与指纹层据此知道"无法唯一决定", 而不是偷偷选一个。
    """
    good = [i for i, x in enumerate(beams) if x is not None]
    if len(good) != 3:
        raise ValueError("solve_three_beams requires exactly three good beams")
    missing = next(i for i in range(4) if i not in good)
    b = np.array([beams[i] for i in good], dtype=float)
    b3 = geom.forward_matrix()[good, :]
    # 最小二乘特解 (B_3 行满秩)
    xp, *_ = np.linalg.lstsq(b3, b, rcond=None)
    # 零空间: rank 1, 用 SVD 取最后一个右奇异向量
    _, _, vh = np.linalg.svd(b3)
    null = vh[-1]

    def at(t: float) -> np.ndarray:
        return xp + t * null

    # e = x[3] = 0
    t_e = -xp[3] / null[3] if abs(null[3]) > 1e-12 else 0.0
    xe = at(t_e)
    # w = x[2] = 0
    t_w = -xp[2] / null[2] if abs(null[2]) > 1e-12 else 0.0
    xw = at(t_w)

    candidates = [
        ThreeBeamCandidate(
            assumption="zero_error_velocity",
            u=float(xe[0]), v=float(xe[1]), w=float(xe[2]), e=0.0,
            note="classic RDI three-beam solution; assumes error velocity 0",
        ),
        ThreeBeamCandidate(
            assumption="minimum_norm_family_member",
            u=float(xp[0]), v=float(xp[1]), w=float(xp[2]), e=float(xp[3]),
            note="minimum-norm member of the underdetermined solution family",
        ),
        ThreeBeamCandidate(
            assumption="zero_vertical_velocity",
            u=float(xw[0]), v=float(xw[1]), w=0.0, e=float(xw[3]),
            note="shallow-water homogeneous candidate; assumes w 0",
        ),
    ]
    return candidates, missing


def family_parameterization(beams: Sequence[Optional[float]],
                            geom: BeamGeometry = BeamGeometry()
                            ) -> Dict[str, object]:
    """暴露连续解族参数化, 供证据界面显示"多解不是三个点而是一条线"。"""
    good = [i for i, x in enumerate(beams) if x is not None]
    b = np.array([beams[i] for i in good], dtype=float)
    b3 = geom.forward_matrix()[good, :]
    xp, *_ = np.linalg.lstsq(b3, b, rcond=None)
    _, _, vh = np.linalg.svd(b3)
    null = vh[-1]
    return {
        "particular_solution": [float(x) for x in xp],
        "null_direction": [float(x) for x in null],
        "free_parameter": "t",
        "components": ["u", "v", "w", "error"],
    }


def good_beam_mask(beam_flags: Sequence[bool],
                   correlations: Sequence[Optional[float]],
                   amplitudes: Sequence[Optional[float]],
                   correlation_floor: float = 64.0
                   ) -> Tuple[bool, ...]:
    """质量层级: 坏 beam 标志 OR 低相关 OR 无振幅 -> 该 beam 不可用。

    判定结果逐 beam 落事件, 被排除 beam 不参与后续坐标变换。
    """
    mask = []
    for i, flagged in enumerate(beam_flags):
        ok = not flagged
        corr = correlations[i] if i < len(correlations) else None
        amp = amplitudes[i] if i < len(amplitudes) else None
        if corr is None or corr < correlation_floor:
            ok = False
        if amp is None:
            ok = False
        mask.append(ok)
    return tuple(mask)
