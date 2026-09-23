"""ADCP 坐标变换核心规则:beam -> instrument -> ship -> earth。

约定(角度一律为度,速度一律为 m/s):
- beam 坐标:沿波束方向,正号远离换能器;
- instrument 坐标:x 在 3/4 波束平面,y 在 1/2 波束平面,z 向上;
- ship 坐标:x 右舷,y 船艏,z 向上;安装角为 instrument 到 ship 的艏向旋转;
- earth 坐标:x 东,y 北,z 向上;heading 为磁艏向,须加磁偏角。
"""
from __future__ import annotations

import math

import numpy as np


def beam_to_instrument_four(beam_vels, beam_angle_deg):
    """四波束 Janus 解,返回 (instrument 向量, 误差速度)。"""
    sin_a = math.sin(math.radians(beam_angle_deg))
    cos_a = math.cos(math.radians(beam_angle_deg))
    b1, b2, b3, b4 = beam_vels
    u = (b3 - b4) / (2.0 * sin_a)
    v = (b1 - b2) / (2.0 * sin_a)
    w = (b1 + b2 + b3 + b4) / (4.0 * cos_a)
    err = (b1 + b2 - b3 - b4) / (4.0 * cos_a)
    return np.array([u, v, w]), err


def beam_to_instrument_three(beam_vels, beam_angle_deg, error_candidates):
    """三波束解:误差速度未知,对每个假定误差速度给出一个合法候选。

    返回 [(vec, assumed_error_velocity), ...],不塌缩、不平均、不与四波束解混值。
    """
    present = {i + 1: v for i, v in enumerate(beam_vels) if v is not None}
    if len(present) != 3:
        raise ValueError("三波束解要求恰好 3 个有效波束")
    sin_a = math.sin(math.radians(beam_angle_deg))
    cos_a = math.cos(math.radians(beam_angle_deg))
    out = []
    for err in error_candidates:
        if 1 in present and 2 in present:
            v = (present[1] - present[2]) / (2.0 * sin_a)
            w = (present[1] + present[2]) / (2.0 * cos_a) - err
        else:
            u = (present[3] - present[4]) / (2.0 * sin_a)
            w = (present[3] + present[4]) / (2.0 * cos_a) + err
        if 3 in present and 4 in present:
            u = (present[3] - present[4]) / (2.0 * sin_a)
        else:
            b = present[3] if 3 in present else present[4]
            sign = 1.0 if 3 in present else -1.0
            u = sign * (b - (w - err) * cos_a) / sin_a
        if 1 in present and 2 in present:
            v = (present[1] - present[2]) / (2.0 * sin_a)
        else:
            b = present[1] if 1 in present else present[2]
            sign = 1.0 if 1 in present else -1.0
            v = sign * (b - (w + err) * cos_a) / sin_a
        out.append((np.array([u, v, w]), err))
    return out


def circular_lerp(a, b, frac):
    """角度环插值:359.9 -> 0.1 经过 0.0,不经过 180。"""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return (a + frac * diff) % 360.0


def interpolate_attitude(samples, t, gap_threshold_s):
    """姿态插值。samples: [(t, {"heading","pitch","roll"}, material_id), ...] 已按 t 排序。

    缺口超过阈值时返回 status="gap",绝不跨缺口插值。
    """
    before = None
    after = None
    for s in samples:
        if s[0] <= t:
            before = s
        if s[0] >= t:
            after = s
            break
    if before is None or after is None:
        return {"status": "no_data", "diag": "姿态样本不覆盖该 ping", "used_ids": []}
    t0, v0, id0 = before
    t1, v1, id1 = after
    gap = t1 - t0
    if gap > gap_threshold_s:
        return {"status": "gap", "gap_s": round(gap, 3), "used_ids": [id0, id1],
                "diag": "姿态缺口 %.1fs 超过阈值 %.1fs,不跨缺口插值" % (gap, gap_threshold_s)}
    frac = 0.0 if gap == 0 else (t - t0) / gap
    wrapped = abs(v1["heading"] - v0["heading"]) > 180.0
    return {"status": "ok",
            "heading": circular_lerp(v0["heading"], v1["heading"], frac),
            "pitch": v0["pitch"] + frac * (v1["pitch"] - v0["pitch"]),
            "roll": v0["roll"] + frac * (v1["roll"] - v0["roll"]),
            "h0": v0["heading"], "h1": v1["heading"], "wrapped": wrapped,
            "used_ids": [id0, id1], "diag": None}


def rot_x(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def instrument_to_ship(vec, mounting_angle_deg):
    return rot_z(mounting_angle_deg) @ np.asarray(vec, dtype=float)


def ship_to_earth(vec, heading_deg, pitch_deg, roll_deg, order, declination_deg):
    """坐标转换顺序是规则的一部分:HPR 与 RPH 给出不同结果,且都进入派生指纹。"""
    h = heading_deg + declination_deg
    if order == "HPR":
        m = rot_z(h) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    elif order == "RPH":
        m = rot_x(roll_deg) @ rot_y(pitch_deg) @ rot_z(h)
    else:
        raise ValueError("未知坐标转换顺序: %r" % (order,))
    return m @ np.asarray(vec, dtype=float)
