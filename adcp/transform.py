"""ADCP 坐标变换核心数学:beam -> instrument -> ship -> earth。

守住的不变量:
- 三波束解与四波束解分别记录,绝不合并为单一值。
- 姿态插值先对 heading 解卷绕(允许跨越 360°),且绝不跨越时间缺口。
- 坐标转换顺序、波束几何、磁偏角版本都进入派生指纹。
- 无法唯一决定时保留全部合法候选,不吞掉多解或不确定状态。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

COND_LIMIT = 1e8


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class BeamGeometry:
    """Janus 四波束几何:波束角 + 各波束方位角。"""

    beam_angle_deg: float = 25.0
    azimuths_deg: tuple = (45.0, 135.0, 225.0, 315.0)

    def fingerprint(self) -> str:
        return stable_hash(
            {"beam_angle_deg": self.beam_angle_deg, "azimuths_deg": list(self.azimuths_deg)}
        )


def design_matrix(geom: BeamGeometry, beams) -> np.ndarray:
    """每行把 instrument 坐标 (u,v,w) 投影到对应 beam(远离换能器为正)。"""
    sin_b = math.sin(math.radians(geom.beam_angle_deg))
    cos_b = math.cos(math.radians(geom.beam_angle_deg))
    rows = []
    for b in beams:
        phi = math.radians(geom.azimuths_deg[b])
        rows.append([sin_b * math.cos(phi), sin_b * math.sin(phi), cos_b])
    return np.array(rows, dtype=float)


def solve_four_beam(geom: BeamGeometry, bvals) -> dict:
    """四波束最小二乘解 + 误差流速。与三波束解分开记录。"""
    mat = design_matrix(geom, (0, 1, 2, 3))
    sol, *_ = np.linalg.lstsq(mat, np.asarray(bvals, float), rcond=None)
    err = float(np.dot(np.asarray(bvals, float), [1.0, -1.0, 1.0, -1.0]) / 4.0)
    return {"u": float(sol[0]), "v": float(sol[1]), "w": float(sol[2]), "err": err}


def solve_three_beam(geom: BeamGeometry, beams, bvals):
    """三波束精确解。几何奇异时返回 (None, 诊断),绝不伪装成四波束值。"""
    mat = design_matrix(geom, beams)
    cond = float(np.linalg.cond(mat))
    if not math.isfinite(cond) or cond > COND_LIMIT:
        return None, f"三波束几何奇异 cond={cond:.3e},解不唯一"
    sol = np.linalg.solve(mat, np.asarray(bvals, float))
    return {"u": float(sol[0]), "v": float(sol[1]), "w": float(sol[2])}, None


def rot_x(a):
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def rot_y(a):
    return np.array(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    )


def rot_z(a):
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


def attitude_matrix(heading_deg, pitch_deg, roll_deg, order="HPR"):
    """按给定顺序组合姿态旋转。order 是 'H'/'P'/'R' 的排列,顺序不同结果不同。"""
    mat = np.eye(3)
    for ch in order:
        if ch == "H":
            mat = mat @ rot_z(math.radians(heading_deg))
        elif ch == "P":
            mat = mat @ rot_y(math.radians(pitch_deg))
        elif ch == "R":
            mat = mat @ rot_x(math.radians(roll_deg))
        else:
            raise ValueError(f"未知转换顺序分量: {ch!r}")
    return mat


def instrument_to_ship(vec, install):
    """instrument -> ship:按安装角(heading/pitch/roll 偏移)旋转。"""
    mat = attitude_matrix(
        install["heading_deg"], install["pitch_deg"], install["roll_deg"], "HPR"
    )
    return mat @ np.asarray(vec, float)


def ship_to_earth(vec, att, order, declination_deg):
    """ship -> earth:姿态旋转(顺序敏感),再按磁偏角转到真北基准。"""
    mat = rot_z(math.radians(declination_deg)) @ attitude_matrix(
        att["heading"], att["pitch"], att["roll"], order
    )
    return mat @ np.asarray(vec, float)


def unwrap_headings(headings_deg):
    """对 heading 序列解卷绕,使 359° -> 1° 的插值经过 0° 而不是 180°。"""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(headings_deg, float)))).tolist()


def _lerp(t0, v0, t1, v1, t):
    return v0 + (v1 - v0) * ((t - t0) / (t1 - t0))


def interp_attitude(samples, t, max_gap):
    """在姿态样本间插值。

    samples: 按 t 排序的 dict,含 t/heading(已解卷绕)/pitch/roll。
    样本间隔超过 max_gap 时返回 None —— 绝不跨时间缺口插值。
    """
    if not samples:
        return None
    if t < samples[0]["t"] or t > samples[-1]["t"]:
        return None
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if a["t"] <= t <= b["t"]:
            if b["t"] - a["t"] > max_gap:
                return None
            if b["t"] == a["t"]:
                return {"t": t, "heading": a["heading"] % 360.0, "pitch": a["pitch"], "roll": a["roll"]}
            return {
                "t": t,
                "heading": _lerp(a["t"], a["heading"], b["t"], b["heading"], t) % 360.0,
                "pitch": _lerp(a["t"], a["pitch"], b["t"], b["pitch"], t),
                "roll": _lerp(a["t"], a["roll"], b["t"], b["roll"], t),
            }
    last = samples[-1]
    return {"t": t, "heading": last["heading"] % 360.0, "pitch": last["pitch"], "roll": last["roll"]}


def derivation_fingerprint(*, transform_order, geom, declination, install, source_fps):
    """派生指纹:坐标转换顺序、波束几何、磁偏角版本全部进入。"""
    return stable_hash(
        {
            "transform_order": transform_order,
            "beam_geometry": geom.fingerprint(),
            "declination": {"version": declination["version"], "degrees": declination["degrees"]},
            "install_angle": install,
            "sources": source_fps,
        }
    )
