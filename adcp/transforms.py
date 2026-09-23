"""ADCP 坐标变换:beam -> instrument -> ship -> earth。

语义守则:
- 三波束解与四波束解不混为一值;无法唯一决定时保留全部合法候选。
- 姿态插值先对 heading 做 360 度解卷绕,且不跨越时间缺口。
- 坐标转换顺序、波束几何、磁偏角版本都进入派生指纹(见 engine.fingerprint)。
"""
from __future__ import annotations

import numpy as np

# 误差速度阈值:超过则认为四波束自洽性存疑,需要给出全部三波束候选
ERROR_VELOCITY_MAX = 0.05

TRANSFORM_ORDERS = {
    "roll-pitch-yaw": ("x", "y", "z"),
    "yaw-pitch-roll": ("z", "y", "x"),
}


def beam_unit_vectors(beam_angle_deg: float) -> np.ndarray:
    """四波束 Janus 几何下的波束单位向量(instrument 系,x 前 y 右舷 z 上,波束指向下)。"""
    t = np.deg2rad(beam_angle_deg)
    s, c = np.sin(t), np.cos(t)
    return np.array([
        [0.0, s, -c],   # beam 1:+y 倾
        [0.0, -s, -c],  # beam 2:-y 倾
        [s, 0.0, -c],   # beam 3:+x 倾
        [-s, 0.0, -c],  # beam 4:-x 倾
    ])


def beam_to_instrument(radial, beam_angle_deg: float = 20.0) -> dict:
    """radial: 4 个径向速度(None/NaN 表示该 beam 失效)。

    返回候选字典,键为候选名:
    - "four_beam": 四波束最小二乘解(含 error_velocity)
    - "three_beam:dropK": 去掉第 K 束的三波束解(假设误差速度为 0)
    当四波束误差速度超阈时,同时给出全部四个三波束候选并标记 ambiguous。
    """
    b = np.array([np.nan if v is None else float(v) for v in radial], dtype=float)
    U = beam_unit_vectors(beam_angle_deg)
    valid = ~np.isnan(b)
    out: dict = {}

    def three_beam(mask):
        sol, *_ = np.linalg.lstsq(U[mask], b[mask], rcond=None)
        dropped = int(np.where(~mask)[0][0]) + 1
        return {
            "velocity": [float(x) for x in sol],
            "assumed_error_velocity": 0.0,
            "dropped_beam": dropped,
            "frame": "instrument",
            "unit": "m/s",
            "status": "three_beam",
        }

    if valid.sum() == 4:
        sol, *_ = np.linalg.lstsq(U, b, rcond=None)
        err = float(np.linalg.norm(b - U @ sol) / 2.0)
        out["four_beam"] = {
            "velocity": [float(x) for x in sol],
            "error_velocity": err,
            "frame": "instrument",
            "unit": "m/s",
            "status": "ok" if err <= ERROR_VELOCITY_MAX else "suspect",
        }
        if err > ERROR_VELOCITY_MAX:
            # 四波束不自洽:无法唯一决定,保留全部合法三波束候选
            for k in range(4):
                mask = np.ones(4, dtype=bool)
                mask[k] = False
                out[f"three_beam:drop{k + 1}"] = three_beam(mask)
    elif valid.sum() == 3:
        out[f"three_beam:drop{int(np.where(~valid)[0][0]) + 1}"] = three_beam(valid)
    return out


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def instrument_to_ship(v, roll_deg: float, pitch_deg: float, mounting_deg: float,
                       order: str = "roll-pitch-yaw"):
    """instrument -> ship。安装角作为绕 z 的固定偏置;旋转顺序由 order 决定。"""
    if order not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order: {order}")
    angles = {"x": roll_deg, "y": pitch_deg, "z": mounting_deg}
    mats = {"x": rot_x, "y": rot_y, "z": rot_z}
    m = np.eye(3)
    for axis in TRANSFORM_ORDERS[order]:
        m = mats[axis](angles[axis]) @ m
    return (m @ np.asarray(v, dtype=float)).tolist()


def ship_to_earth(v, heading_mag_deg: float, declination_deg: float):
    """ship -> earth。真航向 = 磁航向 + 磁偏角(东偏为正)。"""
    heading_true = heading_mag_deg + declination_deg
    return (rot_z(heading_true) @ np.asarray(v, dtype=float)).tolist()


def unwrap_headings(headings_deg):
    """对航向序列做 360 度解卷绕,返回连续角度(度)。"""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(headings_deg, dtype=float)))).tolist()


def interpolate_attitude(samples, t_query: float, gap_threshold_s: float) -> dict:
    """在给定时刻插值姿态。

    samples: [(t, heading_deg, pitch_deg, roll_deg), ...] 按时间排序。
    返回 {status, heading_deg, pitch_deg, roll_deg, wrapped, bracket}:
    - status="ok": 正常插值;wrapped 表示该区间发生了 360 度解卷绕
    - status="gap": 相邻样本间隔超过阈值,拒绝跨缺口插值
    - status="out_of_range": 查询时刻在样本范围外
    """
    if not samples:
        return {"status": "out_of_range"}
    ts = [s[0] for s in samples]
    hs = unwrap_headings([s[1] for s in samples])
    ps = [s[2] for s in samples]
    rs = [s[3] for s in samples]
    if t_query < ts[0] or t_query > ts[-1]:
        return {"status": "out_of_range"}
    idx = 0
    for i in range(len(ts) - 1):
        if ts[i] <= t_query <= ts[i + 1]:
            idx = i
            break
    t0, t1 = ts[idx], ts[idx + 1]
    if t1 - t0 > gap_threshold_s:
        return {"status": "gap", "bracket": [t0, t1], "gap_s": t1 - t0}
    frac = 0.0 if t1 == t0 else (t_query - t0) / (t1 - t0)
    h = hs[idx] + frac * (hs[idx + 1] - hs[idx])
    p = ps[idx] + frac * (ps[idx + 1] - ps[idx])
    r = rs[idx] + frac * (rs[idx + 1] - rs[idx])
    wrapped = abs(hs[idx] - samples[idx][1]) > 1e-9 or abs(hs[idx + 1] - samples[idx + 1][1]) > 1e-9
    return {
        "status": "ok",
        "heading_deg": h % 360.0,
        "pitch_deg": p,
        "roll_deg": r,
        "wrapped": wrapped,
        "bracket": [t0, t1],
        "unit": "deg",
    }
