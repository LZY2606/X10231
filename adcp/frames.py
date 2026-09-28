"""波束几何与逐级坐标变换。

坐标基准（所有派生向量都携带 basis）：
  FRAME_BEAM       沿波束方向的径向速度
  FRAME_INSTRUMENT 仪器坐标 x=前 y=右舷 z=下
  FRAME_SHIP       船体坐标（经安装角旋转，未做姿态）
  FRAME_EARTH      大地坐标 ENU（x=东 y=北 z=上，做姿态+磁偏角）

姿态约定（页面与指纹中显式记录，见 geometry.transform_order）：
  航向 h 为磁罗经北顺时针角；pitch 正值船首抬升；roll 正值右舷下沉。
  默认内禀顺序 hpr：先绕 z 航向(-h)，再绕新 y 纵摇(+p)，再绕新 x 横摇(+r)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FRAME_BEAM = "beam"
FRAME_INSTRUMENT = "instrument"
FRAME_SHIP = "ship"
FRAME_EARTH = "earth_enu"

Vec3 = np.ndarray


def rot_axis_angle(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """绕世界单位轴 axis、右手正角 angle_deg 的 Rodrigues 旋转矩阵。"""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    t = math.radians(angle_deg)
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def _rotate_basis(basis: List[np.ndarray], axis_index: int, angle_deg: float) -> List[np.ndarray]:
    r = rot_axis_angle(basis[axis_index], angle_deg)
    return [r @ v for v in basis]


# (基础轴索引, 带符号角度)
_ANGLE_SPEC = {"h": (2, -1.0), "p": (1, +1.0), "r": (0, +1.0)}


def intrinsic_rotation(h: float, p: float, r: float, order: str = "hpr",
                       base: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """按内禀顺序构造旋转矩阵，列向量即旋转后的三个基轴。

    base 给定时为初始基轴（ENU 基准传入 N/E/-U），否则为标准正交基。
    order 是 'h','p','r' 的排列，如 'hpr' 或 'rph'；顺序不同结果不同，
    因此 transform_order 必须进入派生指纹。
    """
    if base is None:
        basis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 0.0, 1.0])]
    else:
        basis = [np.asarray(v, dtype=float) for v in base]
    values = {"h": h, "p": p, "r": r}
    for name in order:
        axis_index, sign = _ANGLE_SPEC[name]
        basis = _rotate_basis(basis, axis_index, sign * values[name])
    return np.column_stack(basis)


_ENU_BASE = [np.array([0.0, 1.0, 0.0]),   # 航向 0 时船首指北
             np.array([1.0, 0.0, 0.0]),   # 右舷指东
             np.array([0.0, 0.0, -1.0])]  # 仪器下方 -> ENU 的 up 反向


def ship_to_earth(h: float, p: float, r: float, order: str = "hpr") -> np.ndarray:
    """船体坐标 -> ENU。v_earth = R @ v_ship。"""
    return intrinsic_rotation(h, p, r, order=order, base=_ENU_BASE)


def instrument_to_ship(heading_alignment: float, mount_pitch: float, mount_roll: float,
                       order: str = "hpr") -> np.ndarray:
    """仪器坐标 -> 船体（安装角；安装角的旋转顺序单独版本化）。"""
    return intrinsic_rotation(heading_alignment, mount_pitch, mount_roll, order=order)


def apply_declination(heading_mag_deg: float, declination_east_deg: float) -> float:
    """磁航向 -> 真航向（东磁偏角为正）。磁偏角版本进入指纹。"""
    return heading_mag_deg + declination_east_deg


# ----------------------------------------------------------------波束几何

def beam_direction(beam_angle_deg: float, azimuth_deg: float) -> np.ndarray:
    """波束指向单位向量（仪器坐标）。方位角自 x 轴（船首）向 y（右舷）顺时针。"""
    theta = math.radians(beam_angle_deg)
    az = math.radians(azimuth_deg)
    return np.array([math.sin(theta) * math.cos(az),
                     math.sin(theta) * math.sin(az),
                     math.cos(theta)])


def beam_matrix(geom: Dict) -> np.ndarray:
    """H：b = H v。b 为波束径向速度（沿波束远离换能器为正）。"""
    sign = 1.0 if geom.get("beam_velocity_sign", "away") == "away" else -1.0
    rows = []
    for az in geom["azimuths"]:
        d = beam_direction(geom["beam_angle"], az)
        rows.append(-sign * d)
    return np.vstack(rows)


def solve_four_beam(H: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """四波束最小二乘解与 error velocity（第四约束残差，单位同 b）。

    四波束解是一个独立结果，绝不与三波束候选混为同一个值。
    """
    b = np.asarray(b, dtype=float)
    v, *_ = np.linalg.lstsq(H, b, rcond=None)
    residual = float(b[3] - float(H[3] @ v))
    return v, residual


def solve_three_beam(H: np.ndarray, b: np.ndarray, beam_ids: Sequence[int]) -> np.ndarray:
    """指定三个波束（beam_ids 为 0 基波束号）求精确解。"""
    H3 = H[list(beam_ids), :]
    b3 = np.asarray(b, dtype=float)[list(beam_ids)]
    return np.linalg.solve(H3, b3)


def three_beam_candidates(H: np.ndarray, b: np.ndarray,
                          good_beams: Sequence[int]) -> List[Dict]:
    """所有合法三波束候选；每个候选注明被省略的波束，彼此是不同的值。"""
    out = []
    good = list(good_beams)
    for omitted in good:
        ids = [i for i in good if i != omitted]
        if len(ids) != 3:
            continue
        v = solve_three_beam(H, b, ids)
        out.append({
            "kind": "three_beam",
            "omitted_beam": omitted,
            "used_beams": ids,
            "v": [float(x) for x in v],
        })
    return out
