"""ADCP 坐标变换：beam -> instrument -> ship -> earth。

语义红线：
- 三波束解与四波束解不混为一值；
- 姿态插值不跨缺口；
- 坐标转换顺序、波束几何、磁偏角版本都进入派生指纹（见 pipeline）。
"""
from __future__ import annotations

import math

import numpy as np


def beam_matrix(beam_angle_deg: float) -> np.ndarray:
    """4 波束 Janus 几何的前向矩阵：b = M @ [u, v, w, e]。"""
    s = math.sin(math.radians(beam_angle_deg))
    c = math.cos(math.radians(beam_angle_deg))
    return np.array(
        [
            [s, 0.0, c, c],
            [-s, 0.0, c, c],
            [0.0, s, c, -c],
            [0.0, -s, c, -c],
        ]
    )


def beam_to_instrument_four(beams, beam_angle_deg: float):
    """四波束解。返回 (vec[u,v,w], error_velocity)。"""
    s = math.sin(math.radians(beam_angle_deg))
    c = math.cos(math.radians(beam_angle_deg))
    b1, b2, b3, b4 = beams
    u = (b1 - b2) / (2 * s)
    v = (b3 - b4) / (2 * s)
    w = (b1 + b2 + b3 + b4) / (4 * c)
    e = (b1 + b2 - b3 - b4) / (4 * c)
    return np.array([u, v, w]), e


def beam_to_instrument_three(beams, dead_beam: int, beam_angle_deg: float,
                             mode: str, w_prior: float | None = None):
    """三波束解（某一波束失效）。返回 (vec, note)。

    三波束只有 3 个方程、4 个未知数（u,v,w,e），存在多解。本函数按 mode
    给出一个合法候选，绝不把候选平均成一个值：
    - mode="zero_error":     假设误差速度 e = 0；
    - mode="vertical_prior": 假设 w 取自邻 ping 先验。
    """
    M = beam_matrix(beam_angle_deg)
    keep = [i for i in range(4) if (i + 1) != dead_beam]
    A = M[keep, :]  # 3x4
    b = np.array([beams[i] for i in keep], dtype=float)
    if mode == "zero_error":
        A2 = A[:, :3]
        sol, *_ = np.linalg.lstsq(A2, b, rcond=None)
        return np.array([sol[0], sol[1], sol[2]]), "assume error velocity e=0"
    if mode == "vertical_prior":
        if w_prior is None:
            return None, "no vertical prior available; candidate indeterminate"
        A2 = A[:, [0, 1, 3]]
        b2 = b - A[:, 2] * w_prior
        sol, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        return np.array([sol[0], sol[1], w_prior]), f"assume w from neighbor prior ({w_prior:.4f} m/s)"
    raise ValueError(f"unknown three-beam mode {mode!r}")


def rot_x(deg):
    r = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])


def rot_y(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])


def rot_z(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])


def instrument_to_ship(vec, mounting_angle_deg: float):
    """绕垂直轴按安装角旋转（instrument -> ship）。"""
    return rot_z(mounting_angle_deg) @ np.asarray(vec, dtype=float)


def ship_to_earth(vec, heading_deg, pitch_deg, roll_deg, order: str = "HPR"):
    """ship -> earth。旋转顺序由 order 指定（如 "HPR"、"RPH"），
    顺序不同结果不同，因此顺序必须进入派生指纹。"""
    mats = {"H": rot_z(heading_deg), "P": rot_y(pitch_deg), "R": rot_x(roll_deg)}
    M = np.eye(3)
    for ch in order:
        M = mats[ch] @ M
    return M @ np.asarray(vec, dtype=float)


def apply_declination(heading_mag_deg: float, declination_deg: float) -> float:
    """磁航向 + 磁偏角 -> 真航向（西偏为负）。"""
    return (heading_mag_deg + declination_deg) % 360.0


def unwrap_headings(headings_deg):
    """跨越 360° 时先展开再插值，避免 359.5->0.5 被插成 180。"""
    return np.degrees(np.unwrap(np.radians(np.asarray(headings_deg, dtype=float))))


def interpolate_attitude(times, headings, pitches, rolls, t, max_gap_s):
    """在姿态时间序列上插值。返回 (dict|None, diagnostic|None)。

    姿态插值不跨缺口：若 bracketing 样本间隔超过 max_gap_s，返回 None 与诊断。
    """
    times = list(times)
    if not times:
        return None, {"code": "attitude_missing", "message": "no attitude samples at all"}
    if t < times[0] or t > times[-1]:
        return None, {"code": "attitude_missing", "message": f"t={t} outside attitude series"}
    t0, t1 = None, None
    for a, b in zip(times[:-1], times[1:]):
        if a <= t <= b:
            t0, t1 = a, b
            break
    if t0 is None:
        t0 = t1 = times[-1]
    if t1 - t0 > max_gap_s:
        return None, {
            "code": "attitude_gap",
            "message": f"attitude gap {t1 - t0:.1f}s > {max_gap_s}s; interpolation across gap refused",
        }
    idx0 = times.index(t0)
    idx1 = times.index(t1)
    unw = unwrap_headings(headings)
    frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    h = unw[idx0] + frac * (unw[idx1] - unw[idx0])
    p = pitches[idx0] + frac * (pitches[idx1] - pitches[idx0])
    r = rolls[idx0] + frac * (rolls[idx1] - rolls[idx0])
    return {"heading": h % 360.0, "pitch": p, "roll": r}, None
