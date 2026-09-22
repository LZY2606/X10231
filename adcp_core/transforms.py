"""逐级坐标旋转：instrument -> ship -> earth(ENU)。

框架约定（全部右手系）：
  instrument: x 前、y 右舷、z 下
  ship:       x 前、y 右舷、z 下，相对 instrument 仅差安装角
  earth:      ENU（x 东、y 北、z 上）
heading 为顺时针、自真北起算；pitch 正=船首抬；roll 正=右舷下沉。
坐标转换顺序（安装角旋转序、先姿态后艏向 vs 先艏向后姿态）进入派生指纹。
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

DEG = math.pi / 180.0


def rot_x(a_deg: float) -> np.ndarray:
    a = a_deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a_deg: float) -> np.ndarray:
    a = a_deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a_deg: float) -> np.ndarray:
    a = a_deg * DEG
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def mount_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float,
                 order: str = "ZYX") -> np.ndarray:
    """instrument -> ship。order 决定旋转序，是安装角来源指纹的一部分。"""
    rx, ry, rz = rot_x(roll_deg), rot_y(pitch_deg), rot_z(yaw_deg)
    if order == "ZYX":
        return rx @ ry @ rz
    if order == "XYZ":
        return rz @ ry @ rx
    raise ValueError("未知安装角旋转序: %r（允许 ZYX / XYZ）" % order)


def _level_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ship -> level（x 前、y 右、z 下，仅无姿态倾斜）。

    行向量是 level 基在 ship 系中的表达，由 pitch/roll 物理定义直接构造，
    零姿态为单位阵；纯 pitch/纯 roll 的行为由测试物理用例锁定。
    """
    # = (R_x(roll) @ R_y(pitch))^T，行向量是 level 基在 ship 系中的表达。
    p, r = pitch_deg * DEG, roll_deg * DEG
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([
        [cp, sr * sp, cr * sp],
        [0.0, cr, sr],
        [sp, -sr * cp, cr * cp],
    ], dtype=float)


def _heading_matrix(true_heading_deg: float) -> np.ndarray:
    """level -> ENU。h=0 时前=北、右=东；h=90 时前=东、右=南。"""
    h = true_heading_deg * DEG
    s, c = math.sin(h), math.cos(h)
    return np.array([
        [s,  c, 0.0],
        [c, -s, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)


def wrap_360(angle_deg: float) -> float:
    """把角度规范到 [0, 360)。"""
    return angle_deg % 360.0


def anglediff(a_deg: float, b_deg: float) -> float:
    """有向最小角差 a-b，结果在 (-180, 180]。"""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def ship_to_earth(v_ship: np.ndarray, heading_deg: float,
                  pitch_deg: float, roll_deg: float,
                  chain_order: str = "TILT_THEN_HEADING") -> np.ndarray:
    """ship(FSD) -> earth(ENU)。

    TILT_THEN_HEADING：先水平化再按艏向投影（物理正确）。
    HEADING_THEN_TILT：先按艏向旋转再水平化（错误次序，仅作反例对照）。
    """
    c = _level_matrix(pitch_deg, roll_deg)
    h = _heading_matrix(wrap_360(heading_deg))
    v = np.asarray(v_ship, dtype=float)
    if chain_order == "TILT_THEN_HEADING":
        return h @ c @ v
    if chain_order == "HEADING_THEN_TILT":
        return c @ h @ v
    raise ValueError("未知坐标转换顺序: %r" % chain_order)


def instrument_to_ship(v_inst: np.ndarray, mount: Dict[str, float]) -> np.ndarray:
    m = mount_matrix(mount["yaw_deg"], mount["pitch_deg"], mount["roll_deg"],
                     mount.get("rotation_order", "ZYX"))
    return m @ np.asarray(v_inst, dtype=float)


def instrument_to_earth(v_inst: np.ndarray, heading_magnetic_deg: float,
                        declination_deg: float, pitch_deg: float, roll_deg: float,
                        mount: Dict[str, float],
                        chain_order: str = "TILT_THEN_HEADING") -> np.ndarray:
    heading_true = wrap_360(heading_magnetic_deg + declination_deg)
    v_ship = instrument_to_ship(v_inst, mount)
    return ship_to_earth(v_ship, heading_true, pitch_deg, roll_deg, chain_order)
