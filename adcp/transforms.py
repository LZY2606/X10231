"""姿态旋转：instrument -> ship -> earth；磁偏角/安装角；姿态插值。

RDI 标准约定（右手系，x 船首、y 右舷、z 向下；heading 顺时针自北）：
    v_ship  = R_roll @ R_pitch @ v_inst        （先 pitch 后 roll）
    v_earth = R_heading @ v_ship
    heading_eff = wrap360(heading + heading_align + declination)

变换顺序本身是规则版本的一部分，进入派生指纹：
    "rdi_pitch_roll" ：先 pitch 后 roll（标准）
    "roll_pitch"     ：先 roll 后 pitch（反例分支，结果不同）

姿态插值：pitch/roll 线性插值；heading 用单位圆插值（正确跨越 360°）。
若 ping 两侧姿态间隔超过 max_gap_s，或位于覆盖范围之外，拒绝插值
（不跨缺口），返回 None + 诊断。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .units import wrap360, wrap180


def r_pitch(deg: float) -> np.ndarray:
    p = np.deg2rad(deg)
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def r_roll(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def r_heading(deg: float) -> np.ndarray:
    h = np.deg2rad(deg)
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class MountAngles:
    """安装角：heading 对齐（方位）+ pitch/roll 固定偏置。"""

    heading_align_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0

    def fingerprint(self) -> str:
        blob = json.dumps({
            "ha": round(self.heading_align_deg, 9),
            "po": round(self.pitch_offset_deg, 9),
            "ro": round(self.roll_offset_deg, 9),
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Declination:
    """磁偏角：度，东偏为正；携带模型版本（进入指纹）。"""

    value_deg: float
    model: str = "WMM"
    epoch: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps({
            "dec": round(self.value_deg, 9),
            "model": self.model, "epoch": self.epoch,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Attitude:
    heading_deg: float
    pitch_deg: float
    roll_deg: float


def effective_heading(heading_deg: float, mount: MountAngles,
                      decl: Declination) -> float:
    return wrap360(heading_deg + mount.heading_align_deg + decl.value_deg)


def instrument_to_ship(v_inst, attitude: Attitude, mount: MountAngles,
                       order: str = "rdi_pitch_roll") -> np.ndarray:
    p = attitude.pitch_deg + mount.pitch_offset_deg
    r = attitude.roll_deg + mount.roll_offset_deg
    v = np.asarray(v_inst, dtype=float)
    if order == "rdi_pitch_roll":
        return r_roll(r) @ r_pitch(p) @ v
    if order == "roll_pitch":
        return r_pitch(p) @ r_roll(r) @ v
    raise ValueError(f"unknown transform order: {order}")


def ship_to_earth(v_ship, heading_eff_deg: float) -> np.ndarray:
    return r_heading(heading_eff_deg) @ np.asarray(v_ship, dtype=float)


def to_earth(v_inst, attitude: Attitude, mount: MountAngles,
             decl: Declination, order: str = "rdi_pitch_roll"):
    """返回 (earth, ship, heading_eff) 的逐级链路。"""
    ship = instrument_to_ship(v_inst, attitude, mount, order)
    heading_eff = effective_heading(attitude.heading_deg, mount, decl)
    earth = ship_to_earth(ship, heading_eff)
    return earth, ship, heading_eff


def lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def interp_heading(h0: float, h1: float, f: float) -> float:
    """单位圆插值，天然正确跨越 0/360。"""
    d = wrap180(h1 - h0)
    return wrap360(h0 + d * f)


def interpolate_attitude(times, headings, pitches, rolls, query_t: float,
                         max_gap_s: float):
    """在不跨缺口的前提下把姿态插值到 query_t。

    返回 (Attitude|None, diag|None)。缺口/越界时姿态为 None。
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 1:
        return None, "no attitude samples"
    order_idx = np.argsort(times)
    times = times[order_idx]
    h = np.asarray(headings, dtype=float)[order_idx]
    p = np.asarray(pitches, dtype=float)[order_idx]
    r = np.asarray(rolls, dtype=float)[order_idx]

    if query_t < times[0] - 1e-9 or query_t > times[-1] + 1e-9:
        return None, f"t={query_t:g} outside attitude coverage"

    j = int(np.searchsorted(times, query_t))
    if j < len(times) and abs(times[j] - query_t) < 1e-9:
        return Attitude(wrap360(h[j]), p[j], r[j]), None
    i = j - 1
    gap = float(times[j] - times[i])
    if gap > max_gap_s + 1e-9:
        return None, (f"attitude gap {gap:g}s > max_gap {max_gap_s:g}s; "
                      "interpolation refused across gap")
    f = (query_t - times[i]) / gap
    return Attitude(interp_heading(h[i], h[j], f),
                    lerp(p[i], p[j], f), lerp(r[i], r[j], f)), None


def rule_fingerprint(order: str, mount: MountAngles, decl: Declination) -> str:
    blob = json.dumps({
        "order": order,
        "mount": mount.fingerprint(),
        "decl": decl.fingerprint(),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
