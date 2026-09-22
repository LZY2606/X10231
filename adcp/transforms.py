"""姿态旋转与逐级坐标变换。

约定
----
instrument 轴：X=fore, Y=starboard, Z=down（右手系，z 向下为正）。
ship 轴同定义；earth 内部用 (east, north, down) = datum "earth:END"，
展示 ENU 时 z 取负。

旋转（在 fore/starboard/down 右手系内的标准旋转）
  roll  正=右舷下沉：Rx
  pitch 正=船首上抬：Ry
  heading：H 把 fore/starboard 旋到 east/north，
           heading=0(北) 时 fore->north、starboard->east。

effective 角 = 传感器角 + 安装角（mount offset）；磁航向 + 磁偏角 -> 真航向。
磁偏角东偏为正，其版本进入派生指纹。

坐标转换顺序由 order 显式给出（如 "H,P,R" 或 "R,P,H"），进入指纹：
v_earth = M[order0] @ M[order1] @ M[order2] @ v_inst
（向量最右端矩阵先作用，故 H,P,R 表示先 roll、再 pitch、最后 heading。）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def heading_matrix(h_deg):
    h = math.radians(h_deg)
    s, c = math.sin(h), math.cos(h)
    return np.array([[s, c, 0.0], [c, -s, 0.0], [0.0, 0.0, 1.0]], float)


@dataclass(frozen=True)
class Attitude:
    heading_deg: float   # 磁航向 0..360
    pitch_deg: float
    roll_deg: float
    heading_unwrapped: float = None  # 连续化航向（跨 360 插值用）


@dataclass(frozen=True)
class MountAngles:
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    version: str = "mount:initial:v1"


@dataclass(frozen=True)
class Declination:
    east_deg: float = 0.0
    version: str = "decl:initial:v1"


def effective_attitude(att: Attitude, mount: MountAngles, decl: Declination):
    """安装角与磁偏角只在此处（派生层）应用。"""
    true_heading = (att.heading_deg + mount.heading_deg + decl.east_deg) % 360.0
    return true_heading, att.pitch_deg + mount.pitch_deg, att.roll_deg + mount.roll_deg


def rotation_matrix(h_deg, p_deg, r_deg, order="H,P,R"):
    mats = {"H": heading_matrix(h_deg), "P": _rot_y(math.radians(p_deg)),
            "R": _rot_x(math.radians(r_deg))}
    labels = [t.strip() for t in order.split(",")]
    M = np.eye(3)
    for label in labels:
        M = M @ mats[label]
    return M


def inst_to_ship(v_inst, pitch_deg, roll_deg, order_tilt="P,R"):
    mats = {"P": _rot_y(math.radians(pitch_deg)),
            "R": _rot_x(math.radians(roll_deg))}
    labels = [t.strip() for t in order_tilt.split(",")]
    M = np.eye(3)
    for label in labels:
        M = M @ mats[label]
    return M @ np.asarray(v_inst, float)


def ship_to_earth(v_ship, h_deg):
    return heading_matrix(h_deg) @ np.asarray(v_ship, float)


def transform_all(v_inst, att: Attitude, mount: MountAngles, decl: Declination,
                  order="H,P,R"):
    """返回逐级结果（instrument->ship->earth）与所用有效角/矩阵。"""
    h, p, r = effective_attitude(att, mount, decl)
    # tilt 子序取 order 中 P,R 的相对顺序
    labels = [t.strip() for t in order.split(",")]
    tilt_order = ",".join([t for t in labels if t in ("P", "R")]) or "P,R"
    v_ship = inst_to_ship(v_inst, p, r, tilt_order)
    v_earth = ship_to_earth(v_ship, h)
    full = rotation_matrix(h, p, r, order)
    return {
        "instrument": np.asarray(v_inst, float),
        "ship": v_ship,
        "earth_END": v_earth,
        "earth_ENU": np.array([v_earth[0], v_earth[1], -v_earth[2]]),
        "effective": {"heading_true_deg": h, "pitch_deg": p, "roll_deg": r},
        "matrix": full,
        "order": order,
        "declination_version": decl.version,
        "mount_version": mount.version,
    }


def unwrap_headings(headings):
    """把 0..360 航向连续化（处理跨越 360）。"""
    out = []
    prev = None
    for h in headings:
        if h is None or (isinstance(h, float) and math.isnan(h)):
            out.append(None)
            continue
        if prev is None:
            out.append(float(h))
        else:
            base = out[-1] if out[-1] is not None else prev
            cand = h
            while cand - base > 180:
                cand -= 360
            while cand - base < -180:
                cand += 360
            out.append(float(cand))
        prev = h
    return out


def segments_without_gaps(times, max_gap_s):
    """按时间缺口切分连续段；姿态插值绝不跨段。返回 [(i0,i1),...]。"""
    if not times:
        return []
    segs, start = [], 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap_s:
            segs.append((start, i - 1))
            start = i
    segs.append((start, len(times) - 1))
    return segs


def interpolate_in_segment(t, seg_times, seg_values):
    """仅在同一连续段内线性插值；越界/缺值返回 None。"""
    if t < seg_times[0] or t > seg_times[-1]:
        return None
    for i in range(len(seg_times) - 1):
        t0, t1 = seg_times[i], seg_times[i + 1]
        if t0 <= t <= t1:
            v0, v1 = seg_values[i], seg_values[i + 1]
            if v0 is None or v1 is None:
                return None
            if t1 == t0:
                return float(v0)
            frac = (t - t0) / (t1 - t0)
            return float(v0 + frac * (v1 - v0))
    return None
