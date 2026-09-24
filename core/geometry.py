"""波束几何与 beam -> instrument 速度解算。

坐标系约定（与 RD Instruments 文档同源的右手系）：
- instrument x：指向换能器头部“前方”（beam 3 一侧）
- instrument y：右舷方向（starboard，beam 4 一侧）
- instrument z：沿换能器轴向下（toward transducer face）

典型 4 波束 Janus 配置俯视几何：
    beam1 右舷, beam2 船中(fore, x+), beam3 左舷, beam4 船尾(aft, x-)

每条波束的单位指向向量（速度离开换能器为正径向速度）：
    beam1: ( 0,  s, c)   beam2: ( s, 0, c)
    beam3: ( 0, -s, c)   beam4: (-s, 0, c)
其中 s=sin(theta), c=cos(theta)，theta 为波束与换能器轴(z)的夹角（通常 20°）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .units import Quantity, quantity, require

BEAM_COUNT = 4


@dataclass(frozen=True)
class TransducerConfig:
    """换能器几何配置（与安装角分开记录）。"""

    beam_angle_deg: float
    config_code: str = "convex_down"
    beam_count: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_angle_deg": self.beam_angle_deg,
            "config_code": self.config_code,
            "beam_count": self.beam_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransducerConfig":
        return cls(
            beam_angle_deg=float(data["beam_angle_deg"]),
            config_code=str(data.get("config_code", "convex_down")),
            beam_count=int(data.get("beam_count", 4)),
        )


def beam_unit_vectors(config: TransducerConfig) -> np.ndarray:
    """返回 4x3 矩阵，行 i 为 beam i+1 在 instrument 系的单位指向。"""
    theta = np.deg2rad(config.beam_angle_deg)
    s, c = np.sin(theta), np.cos(theta)
    return np.array([
        [0.0, +s, c],
        [+s, 0.0, c],
        [0.0, -s, c],
        [-s, 0.0, c],
    ], dtype=float)


def forward_matrix(config: TransducerConfig) -> np.ndarray:
    """径向速度 = A @ [u, v, w, e]；e 为 error velocity（4 波束观测冗余量）。

    采用 RDI 凸形向下配置的标准符号约定：
        b1 =  s*v + c*w + e
        b2 =  s*u + c*w - e
        b3 = -s*v + c*w + e
        b4 = -s*u + c*w - e
    """
    theta = np.deg2rad(config.beam_angle_deg)
    s, c = np.sin(theta), np.cos(theta)
    return np.array([
        [0.0, +s, c, 1.0],
        [+s, 0.0, c, -1.0],
        [0.0, -s, c, 1.0],
        [-s, 0.0, c, -1.0],
    ], dtype=float)


def inverse_matrix(config: TransducerConfig) -> np.ndarray:
    """[u, v, w, e] = A^-1 @ beam。数值求逆并做正反映射自检。"""
    a = forward_matrix(config)
    inv = np.linalg.inv(a)
    if not np.allclose(inv @ a, np.eye(4), atol=1e-12):
        raise ValueError("beam matrix inversion failed self-check")
    return inv


@dataclass(frozen=True)
class BeamSolution:
    """一个 depth cell 的 instrument 系解算结果。"""

    good_beams: tuple[int, ...]
    kind: str  # "four-beam" | "three-beam" | "unsolvable"
    candidates: tuple[dict[str, Any], ...]
    note: str | None = None


def radial_to_instrument(
    beams: list[Quantity],
    config: TransducerConfig,
    *,
    good_mask: list[bool] | None = None,
    w_reference: float | None = None,
) -> BeamSolution:
    """把 4 条 beam 径向速度解算为 instrument 系 (u, v, w)。

    - 4 条好波束：解唯一，附带 error velocity e（4 波束冗余量），
      e 与 (u, v, w) 明确分开，三波束解绝不会与四波束解混为同一值。
    - 恰好 3 条好波束：欠定，保留全部合法候选：
        assumption="error_zero"        令 e=0
        assumption="w_reference"       令 w=w_reference（需要外部参考，如剖面/底跟踪）
        assumption="minimum_norm"      最小范数伪逆（无外部约束的数学候选）
      每个候选携带其假设；不自动采纳。
    - 好波束 <3：不可解，候选为空。
    """
    if len(beams) != BEAM_COUNT:
        raise ValueError("expected four beam radials")
    mask = list(good_mask) if good_mask is not None else [q.convertible for q in beams]
    good = tuple(i for i, ok in enumerate(mask) if ok and beams[i].convertible)
    inv = inverse_matrix(config)
    if len(good) == 4:
        b = require(beams, "four-beam radial")
        u, v, w, e = inv @ b
        return BeamSolution(
            good_beams=(0, 1, 2, 3),
            kind="four-beam",
            candidates=({
                "assumption": "four-beam-exact",
                "uvw": [float(u), float(v), float(w)],
                "error_velocity": float(e),
            },),
        )
    if len(good) == 3:
        a = forward_matrix(config)
        ag = a[good, :]
        bg = np.array([beams[i].value for i in good], dtype=float)
        candidates: list[dict[str, Any]] = []

        # 候选 1：e = 0（经典单波束失效约定）
        sub = ag[:, :3]
        u0, v0, w0 = np.linalg.solve(sub, bg)
        candidates.append({
            "assumption": "error_zero",
            "uvw": [float(u0), float(v0), float(w0)],
            "error_velocity": 0.0,
        })

        # 候选 2：w 被外部参考约束（同一 ping 其余四波束 cell 的 w 中位数 / 底跟踪）
        if w_reference is not None:
            m3 = ag[:, [0, 1, 3]]
            u2, v2, e2 = np.linalg.solve(m3, bg - ag[:, 2] * w_reference)
            candidates.append({
                "assumption": "w_reference",
                "uvw": [float(u2), float(v2), float(w_reference)],
                "error_velocity": float(e2),
                "w_reference": float(w_reference),
            })

        # 候选 3：最小范数解（不引入外部假设；与上面两个候选是不同的值）
        uvw, *_ = np.linalg.lstsq(ag, bg, rcond=None)
        candidates.append({
            "assumption": "minimum_norm",
            "uvw": [float(uvw[0]), float(uvw[1]), float(uvw[2])],
            "error_velocity": float(uvw[3]),
        })
        return BeamSolution(good_beams=good, kind="three-beam", candidates=tuple(candidates),
                            note="underdetermined; all legal candidates retained")
    return BeamSolution(good_beams=good, kind="unsolvable", candidates=(),
                        note=f"only {len(good)} good beams (<3)")


def instrument_speeds_from_raw_mm_s(
    raw_beams: list[Any],
    config: TransducerConfig,
    good_mask: list[bool] | None = None,
    w_reference: float | None = None,
) -> BeamSolution:
    """便捷入口：把原始 mm/s 波束速度规范化后解算。"""
    beams = [quantity(v, "mm/s", "speed", frame="beam") for v in raw_beams]
    return radial_to_instrument(beams, config, good_mask=good_mask, w_reference=w_reference)
