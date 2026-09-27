"""波束几何与坐标旋转（instrument -> ship -> earth）。

坐标系：
* 仪器/船水平系 FSD（F=船首 forward，S=右舷 starboard，D=垂直向下）；
* 地球系 ENU（E=东，N=北，U=上）。

海洋（RDI）约定：heading 为相对真北的顺时针方位角，H=0 时船首指北，
H=90° 时船首指东。水平变换把 (F,S,D) 映到 ENU，因此二维航向矩阵是
海洋惯例矩阵（F→N 随 H 顺时针转向 E）：

    M_H = [[ sin H,  cos H, 0],
           [ cos H, -sin H, 0],
           [ 0,      0,     1]]

安装角 α（仪器 x 轴相对船首向右偏为正）使用同形矩阵。俯仰/横摇使用
右手旋转矩阵。默认变换顺序 "RPH" 给出 M = M_H · R_P · R_R。
变换顺序是派生指纹的一部分，非零倾斜时不同顺序得到不同地球向量。

标准 convex Janus 四波束在仪器 FSD 系的单位方向：
    b1=( sinθ,0,cosθ)  b2=(-sinθ,0,cosθ)
    b3=(0, sinθ,cosθ)  b4=(0,-sinθ,cosθ)
波束速度为流速沿波束指向（向下为正）的投影。

四波束解：
    F=(v1-v2)/(2 sinθ)  S=(v3-v4)/(2 sinθ)
    D=(v1+v2+v3+v4)/(4 cosθ)
    e=(v1+v2-v3-v4)/(4 cosθ)           # RDI 误差速度
三波束解在三元组上最小二乘求解 (F,S,D)，带 e=0 假设，因此三波束值
带假设、与四波束值身份不同，绝不混合。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def beam_directions(beam_angle_deg: float, convex: bool = True) -> np.ndarray:
    """返回 4x3 波束单位向量矩阵（仪器 FSD 系，按 beam1..4）。"""
    theta = math.radians(beam_angle_deg)
    s, c = math.sin(theta), math.cos(theta)
    if not convex:
        s = -s
    return np.array([
        [ s, 0.0,  c],
        [-s, 0.0,  c],
        [0.0,  s,  c],
        [0.0, -s,  c],
    ], dtype=float)


def solve_four_beams(beam_vels: Sequence[float], beam_angle_deg: float,
                     convex: bool = True):
    """四波束解 -> (F, S, D, e)。"""
    if len(beam_vels) != 4:
        raise ValueError("four-beam solution requires 4 beam velocities")
    v1, v2, v3, v4 = [float(v) for v in beam_vels]
    theta = math.radians(beam_angle_deg)
    s, c = math.sin(theta), math.cos(theta)
    if not convex:
        s = -s
    f = (v1 - v2) / (2 * s)
    s_vel = (v3 - v4) / (2 * s)
    d = (v1 + v2 + v3 + v4) / (4 * c)
    e = (v1 + v2 - v3 - v4) / (4 * c)
    return f, s_vel, d, e


def solve_three_beams(beam_vels: Sequence[float], used_beams: Sequence[int],
                      beam_angle_deg: float, convex: bool = True):
    """三波束最小二乘解 -> (F, S, D, residual)，带 e=0 假设。"""
    if len(used_beams) != 3 or len(beam_vels) != 3:
        raise ValueError("three-beam solution requires exactly 3 beams")
    dirs = beam_directions(beam_angle_deg, convex)
    idx = [b - 1 for b in used_beams]
    a = dirs[idx, :]
    y = np.asarray(beam_vels, dtype=float)
    x, *_ = np.linalg.lstsq(a, y, rcond=None)
    residual = float(np.linalg.norm(a @ x - y))
    return float(x[0]), float(x[1]), float(x[2]), residual


# ---------------------------------------------------------------------------
# 姿态与安装角旋转（海洋 RDI 约定）
# ---------------------------------------------------------------------------

def heading_matrix(h_rad: float) -> np.ndarray:
    """航向矩阵：水平 FSD -> ENU（H=0 指北，顺时针）。"""
    c, s = math.cos(h_rad), math.sin(h_rad)
    return np.array([
        [ s,  c, 0.0],
        [ c, -s, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def mount_matrix(alpha_deg: float) -> np.ndarray:
    """安装角矩阵：仪器 FSD -> 船 FSD（与航向同形的海洋惯例旋转）。"""
    return heading_matrix(math.radians(alpha_deg))


def _roll_matrix(r_rad: float) -> np.ndarray:
    c, s = math.cos(r_rad), math.sin(r_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s],
        [0.0, -s,  c],
    ], dtype=float)


def _pitch_matrix(p_rad: float) -> np.ndarray:
    c, s = math.cos(p_rad), math.sin(p_rad)
    return np.array([
        [ c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [ s, 0.0,  c],
    ], dtype=float)


_MATRICES = {"H": heading_matrix, "P": _pitch_matrix, "R": _roll_matrix}


def ship_to_earth_matrix(heading_rad: float, pitch_rad: float,
                         roll_rad: float, order: str = "RPH") -> np.ndarray:
    """水平 FSD -> 地球 ENU。

    order 由 H/P/R 三字母组成（左->右为施加顺序），默认 "RPH" 得
    M_H·R_P·R_R。顺序进入派生指纹。
    """
    if sorted(order) != ["H", "P", "R"]:
        raise ValueError(f"invalid transform order {order!r}")
    vals = {"H": heading_rad, "P": pitch_rad, "R": roll_rad}
    mat = np.eye(3)
    for letter in order:
        mat = _MATRICES[letter](vals[letter]) @ mat
    return mat


def instrument_to_ship(vec_fsd: Sequence[float], mount_angle_deg: float):
    """仪器 FSD -> 船 FSD：绕垂直轴施加安装角。"""
    return mount_matrix(mount_angle_deg) @ np.asarray(vec_fsd, dtype=float)


def apply_transform(vec_fsd, instrument_level_fsd, heading_rad, pitch_rad,
                    roll_rad, order="RPH"):
    """仪器系向量 -> 船系 -> 地球系。

    instrument_level_fsd 为安装角旋转后的船系向量；返回 (ship, earth)。
    """
    ship = np.asarray(instrument_level_fsd, dtype=float)
    earth = ship_to_earth_matrix(heading_rad, pitch_rad, roll_rad, order) @ ship
    return ship, earth
