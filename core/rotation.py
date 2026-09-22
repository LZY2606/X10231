"""逐级旋转：instrument -> ship -> earth(ENU)。

约定
----
- ship 坐标：x 向船首，y 向右舷，z 向上。
- earth 坐标输出顺序：(east, north, up)，单位 m/s。
- heading 顺时针、从正北量起；pitch 船首向上为正；roll 右舷向下为正。
- 磁偏角 declination 加在 heading 上（东偏为正）。
默认顺序 ``rdi_canonical`` 采用 Teledyne RDI 坐标变换手册的规范矩阵；
另提供 ``heading_then_level`` 顺序，乘法次序不同、数值不同，
二者是独立的派生指纹（坐标转换顺序进入指纹，不互相覆盖）。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .units import DEG


def _trigs(h: float, p: float, r: float):
    h, p, r = h * DEG, p * DEG, r * DEG
    return math.cos(h), math.sin(h), math.cos(p), math.sin(p), math.cos(r), math.sin(r)


def rdi_xform(h: float, p: float, r: float) -> np.ndarray:
    """RDI 规范矩阵，输入船坐标系，输出 (east, north, up)。"""
    ch, sh, cp, sp, cr, sr = _trigs(h, p, r)
    return np.array([
        [ch * cr + sh * sp * sr,  sr * cp, -sh * cr + ch * sp * sr],
        [-ch * sr + sh * sp * cr, cr * cp,  sr * sh + ch * sp * cr],
        [sh * cp,                 -sp,      ch * cp],
    ], dtype=float)


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def heading_then_level(h: float, p: float, r: float) -> np.ndarray:
    """替代顺序：先水平化(roll,pitch)，再绕天向轴转 heading。

    在 (east,north,up) 数学系中，ship 的 (forward,starboard,up) 对应
    水平基 (north,east,up)，通过置换 B 映射后再级联。
    """
    # ship(fwd,star,up) -> math(east,north,up) 基置换
    B = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
    level = rotx(-r * DEG) @ roty(p * DEG)      # 反向补偿 roll/pitch
    head = rotz(h * DEG)                        # ENU 数学系绕 up 旋转
    return head @ B @ level


TRANSFORM_ORDERS = ("rdi_canonical", "heading_then_level")


def instrument_to_ship(v_inst, mount_h: float, mount_p: float, mount_r: float) -> np.ndarray:
    """安装角（换能器相对船体系）固定矩阵。"""
    return rdi_xform(mount_h, mount_p, mount_r) @ np.asarray(v_inst, float)


def ship_to_earth(v_ship, heading: float, pitch: float, roll: float,
                  declination: float, order: str = "rdi_canonical") -> np.ndarray:
    h = heading + declination
    if order == "rdi_canonical":
        M = rdi_xform(h, pitch, roll)
    elif order == "heading_then_level":
        M = heading_then_level(h, pitch, roll)
    else:
        raise ValueError(f"unknown transform order: {order!r}")
    return M @ np.asarray(v_ship, float)


def transform_chain(radials, good, geometry, attitude, mount, order: str):
    """完整逐级链，返回每一级向量，便于 UI 逐级展示与证据回溯。

    返回 dict：beam 候选、instrument（采纳候选）、ship、earth。
    ``attitude``=(heading,pitch,roll)，``mount``=(mh,mp,mr)，均为度。
    """
    from .geometry import solve_beams

    sol = solve_beams(radials, good, geometry)
    chosen = sol["four_beam"]
    if chosen is None:
        # 三波束时不擅自合并：默认采纳下标最小的缺波束候选，其它候选保留。
        three = [c for c in sol["candidates"] if c["resolvable"]]
        chosen = three[0] if three else None
    out = {"beam": sol, "instrument": None, "ship": None, "earth": None}
    if chosen is None:
        return out
    vi = np.array(chosen["velocity"], float)
    mh, mp, mr = mount
    vs = instrument_to_ship(vi, mh, mp, mr)
    h, p, r = attitude
    ve = ship_to_earth(vs, h, p, r, 0.0, order)
    out["instrument"] = tuple(round(float(x), 9) for x in vi)
    out["ship"] = tuple(round(float(x), 9) for x in vs)
    out["earth"] = tuple(round(float(x), 9) for x in ve)
    out["chosen_method"] = chosen["method"]
    return out
