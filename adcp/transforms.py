"""ADCP 坐标变换核心规则:beam -> instrument -> ship -> earth。

坐标基准:
- beam:    各波束方向上的径向速度(m/s,沿波束向下为正)
- instrument: x=换能器艏向, y=右舷, z=垂直向下 (m/s)
- ship:    经安装角对齐后的船体坐标 (m/s)
- earth:   x=东, y=北, z=下 (ENU-NED 中本工具采用东/北/下, m/s)

姿态航向来自磁罗经(磁航向),磁偏角单独成源,在 earth 阶段应用。
"""
from __future__ import annotations

import bisect

import numpy as np


def beam_matrix(beam_angle_deg, azimuths_deg):
    """Janus 波束几何矩阵 G: b = G @ [u, v, w]_instrument。

    beam_angle_deg: 波束与垂直轴夹角;azimuths_deg: 各波束相对 x 轴方位角。
    """
    theta = np.deg2rad(beam_angle_deg)
    rows = []
    for phi_deg in azimuths_deg:
        phi = np.deg2rad(phi_deg)
        rows.append(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )
    return np.asarray(rows, dtype=float)


def beam_to_instrument(beam_vels, geom, drop=()):
    """波束速度 -> instrument 坐标。

    四波束:最小二乘解 + error velocity 诊断;
    三波束:丢弃一个波束后 3x3 精确解,error velocity 不可得(None)。
    两种解以不同 kind 返回,调用方不得合并为单一值。
    """
    idx = [
        i
        for i in range(len(beam_vels))
        if i not in drop and beam_vels[i] is not None
    ]
    if len(idx) < 3:
        return None
    b = np.array([beam_vels[i] for i in idx], dtype=float)
    M = geom[idx]
    dropped = [i for i in range(len(beam_vels)) if i not in idx]
    if len(idx) == 4:
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        resid = M @ sol - b
        # 四波束冗余残差:误差速度量级诊断(非真值,四波束不一致的证据)
        err = float(np.sqrt(float(resid @ resid)) / 2.0)
        return {
            "kind": "four_beam",
            "vec": [float(x) for x in sol],
            "error_velocity": err,
            "dropped": dropped,
        }
    sol = np.linalg.solve(M, b)
    return {
        "kind": "three_beam",
        "vec": [float(x) for x in sol],
        "error_velocity": None,
        "dropped": dropped,
    }


def candidate_id(cand) -> str:
    if cand["dropped"]:
        return f"three_beam:drop{','.join(str(i) for i in cand['dropped'])}"
    return "four_beam"


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def instrument_to_ship(vec, mounting_angle_deg):
    """安装角(绕 z 轴的换能器-船体偏角)把 instrument 对齐到 ship。"""
    v = rot_z(np.deg2rad(mounting_angle_deg)) @ np.asarray(vec, dtype=float)
    return [float(x) for x in v]


def ship_to_earth(vec, heading_true_deg, pitch_deg, roll_deg, order="roll,pitch,yaw"):
    """船体 -> 地球坐标。

    order 决定旋转连乘顺序(先出现的步骤先作用于向量),
    例如 "roll,pitch,yaw" 对应 Rz(yaw) Ry(pitch) Rx(roll) v。
    顺序本身是派生输入,进入指纹;换顺序即得不同派生。
    """
    R = np.eye(3)
    for step in (s.strip() for s in order.split(",") if s.strip()):
        if step == "roll":
            R = rot_x(np.deg2rad(roll_deg)) @ R
        elif step == "pitch":
            R = rot_y(np.deg2rad(pitch_deg)) @ R
        elif step == "yaw":
            R = rot_z(np.deg2rad(heading_true_deg)) @ R
        else:
            raise ValueError(f"未知旋转步骤: {step}")
    out = R @ np.asarray(vec, dtype=float)
    return [float(x) for x in out]


def magnetic_to_true(heading_mag_deg, declination_deg):
    """磁航向 + 磁偏角 -> 真航向(结果模 360)。"""
    return (heading_mag_deg + declination_deg) % 360.0


def unwrap_headings(headings_deg):
    """对航向序列解卷(跨越 0/360 时保持连续),供插值使用。"""
    arr = np.asarray(headings_deg, dtype=float)
    return np.rad2deg(np.unwrap(np.deg2rad(arr))).tolist()


def interp_series(times, values, t, gap_threshold):
    """线性插值;样本间隔大于 gap_threshold 视为缺口,绝不跨缺口插值。

    返回 (插值或None, 诊断或None)。缺口与越界都不产生值,
    调用方必须把诊断留在原位置。
    """
    times = list(times)
    values = list(values)
    if not times:
        return None, "no_attitude_samples"
    if t < times[0] or t > times[-1]:
        return None, "outside_attitude_range"
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] == t:
        return values[j], None
    t0, t1 = times[j - 1], times[j]
    if t1 - t0 > gap_threshold:
        return None, "attitude_gap"
    f = (t - t0) / (t1 - t0)
    return values[j - 1] + f * (values[j] - values[j - 1]), None


def bracket_wraps(times, headings_raw, t, gap_threshold):
    """判断 t 所在的姿态样本对是否跨越 0/360。"""
    j = bisect.bisect_left(times, t)
    if j == 0 or j >= len(times):
        return None
    t0, t1 = times[j - 1], times[j]
    if t1 - t0 > gap_threshold:
        return None
    h0, h1 = headings_raw[j - 1], headings_raw[j]
    if abs(h1 - h0) > 180.0:
        return {"left_t": t0, "right_t": t1, "raw_left": h0, "raw_right": h1}
    return None
