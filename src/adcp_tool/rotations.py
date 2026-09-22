"""欧拉角旋转矩阵。

约定：角度一律以弧度传入；旋转为 *intrinsic*（绕当前坐标系轴），
轴序字符串如 ``"zyx"`` 表示先绕固定/初始 z、再 y、再 x（等价经典航向-俯仰-
横滚）。坐标变换顺序进入派生指纹，因此不同轴序产生不同结果与不同指纹。

    R = R_z(h) @ R_y(p) @ R_x(r)          # ship -> earth（轴序 zyx）
    v_earth = R @ v_ship

安装角（instrument -> ship）单独成一组欧拉角，默认全零，可配置轴序
（默认 zyx）。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_AXES = ("x", "y", "z")


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


_ELEM = {"x": _rx, "y": _ry, "z": _rz}


def rotation_matrix(angles_rad: Sequence[float], order: str = "zyx") -> np.ndarray:
    """intrinsic 旋转矩阵，角度顺序与 ``order`` 对应。

    order='zyx'、angles=[h,p,r] 时返回 Rz(h)·Ry(p)·Rx(r)。
    """
    order = order.lower()
    if len(order) != len(angles_rad) or any(a not in _AXES for a in order):
        raise ValueError(f"非法轴序 {order!r}")
    R = np.eye(3)
    # intrinsic：按 order 顺序依次左乘
    for axis, ang in zip(order, angles_rad):
        R = _ELEM[axis](ang) @ R
    return R


def install_matrix(angles_rad: Sequence[float], order: str = "zyx") -> np.ndarray:
    """instrument -> ship 安装角矩阵。"""
    return rotation_matrix(angles_rad, order)
