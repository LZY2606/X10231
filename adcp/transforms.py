"""ADCP 坐标变换核心：beam -> instrument -> ship -> earth。

语义红线：
- 三波束解与四波束解不混为一值：候选解分别保留，绝不平均。
- 坐标转换顺序、波束几何和磁偏角版本都进入派生指纹。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

RULE_VERSION = "adcp-rules/1.3.0"

# 换能器几何：Janus 四波束，波束角自垂直方向量起，下视。
# 波束方位：beam1=0°, beam2=180°, beam3=90°, beam4=270°（仪器坐标系 x 前 y 右 z 上）。
GEOMETRIES = {
    "janus4-25-convex": {
        "beam_angle_deg": 25.0,
        "azimuths_deg": [0.0, 180.0, 90.0, 270.0],
        "n_beams": 4,
        "orientation": "down",
        "convex": True,
    },
    "janus4-20-convex": {
        "beam_angle_deg": 20.0,
        "azimuths_deg": [0.0, 180.0, 90.0, 270.0],
        "n_beams": 4,
        "orientation": "down",
        "convex": True,
    },
}

TRANSFORM_ORDERS = ("HPR", "HRP", "PHR", "PRH", "RHP", "RPH")


def geometry_fingerprint(geometry_id: str) -> str:
    geometry = GEOMETRIES[geometry_id]
    blob = json.dumps({"id": geometry_id, **geometry}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def beam_unit_vectors(geometry_id: str) -> np.ndarray:
    """返回 (n_beams, 3) 的波束单位向量（instrument 坐标系）。"""
    geometry = GEOMETRIES[geometry_id]
    theta = np.deg2rad(geometry["beam_angle_deg"])
    rows = []
    for azimuth_deg in geometry["azimuths_deg"]:
        azimuth = np.deg2rad(azimuth_deg)
        rows.append([
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
            -np.cos(theta),
        ])
    return np.array(rows)


def solve_beams(radials: list[float | None], geometry_id: str,
                suspect_beams: list[int] | None = None) -> dict:
    """由波束径向速度求 instrument 坐标流速。

    radials: 4 个波束的径向速度（m/s，远离换能器为正），None 表示该波束不可用。
    suspect_beams: 状态存疑但未确认失效的波束 -> 产生多个合法候选解。
    返回 {"candidates": [...], "diagnostics": [...]}；候选永不合并为单一值。
    """
    matrix = beam_unit_vectors(geometry_id)
    suspect_beams = suspect_beams or []
    present = [i for i, radial in enumerate(radials) if radial is not None]
    candidates = []
    diagnostics = []

    if len(present) < 3:
        diagnostics.append(f"insufficient_beams:{len(present)}<3")
        return {"candidates": [], "diagnostics": diagnostics}

    if len(present) == 4:
        r = np.array([radials[i] for i in present])
        e = matrix[present, :]
        v, residuals, *_ = np.linalg.lstsq(e, r, rcond=None)
        residual = r - e @ v
        # 四波束误差速度：径向残差在垂直方向的投影，衡量四波束自洽性。
        error_velocity = float(residual.sum() / (4.0 * np.cos(
            np.deg2rad(GEOMETRIES[geometry_id]["beam_angle_deg"]))))
        candidates.append({
            "kind": "four_beam_lsq",
            "vector": [float(x) for x in v],
            "unit": "m/s",
            "frame": "instrument",
            "error_velocity": error_velocity,
            "used_beams": present,
        })
        for beam in suspect_beams:
            subset = [i for i in present if i != beam]
            if len(subset) == 3:
                candidates.append(_three_beam_exact(radials, subset, matrix))
    elif len(present) == 3:
        candidates.append(_three_beam_exact(radials, present, matrix))

    return {"candidates": candidates, "diagnostics": diagnostics}


def _three_beam_exact(radials, subset: list[int], matrix: np.ndarray) -> dict:
    missing = [i for i in range(matrix.shape[0]) if i not in subset]
    r = np.array([radials[i] for i in subset])
    e = matrix[subset, :]
    v = np.linalg.solve(e, r)
    return {
        "kind": "three_beam_exact",
        "vector": [float(x) for x in v],
        "unit": "m/s",
        "frame": "instrument",
        "error_velocity": None,
        "used_beams": list(subset),
        "excluded_beam": missing[0] if missing else None,
    }


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])


def instrument_to_ship(vector, mounting_angle_deg: float):
    """安装角修正：绕 z 轴把仪器坐标旋到船体坐标。"""
    return rot_z(mounting_angle_deg) @ np.asarray(vector, dtype=float)


def ship_to_earth(vector, heading_deg: float, pitch_deg: float, roll_deg: float,
                  declination_deg: float, order: str = "HPR"):
    """船体坐标 -> 大地坐标（ENU）。

    heading 为磁航向，先加磁偏角得真航向。旋转顺序由 order 指定（如 HPR），
    顺序本身进入派生指纹——不同顺序是不同结果，不得混用。
    """
    if order not in TRANSFORM_ORDERS:
        raise ValueError(f"unknown transform order: {order}")
    heading_true = heading_deg + declination_deg
    rotations = {"H": rot_z(heading_true), "P": rot_y(pitch_deg), "R": rot_x(roll_deg)}
    matrix = np.eye(3)
    for letter in order:
        matrix = matrix @ rotations[letter]
    return matrix @ np.asarray(vector, dtype=float)


def derived_fingerprint(*, rule_version: str, transform_order: str,
                        geometry_id: str, declination_version: str,
                        mounting_revision: int, dep_hashes: dict) -> str:
    """派生指纹：转换顺序、波束几何、磁偏角版本、安装角版本与依赖哈希全部入内。"""
    blob = json.dumps({
        "rule_version": rule_version,
        "transform_order": transform_order,
        "geometry": {"id": geometry_id, "hash": geometry_fingerprint(geometry_id)},
        "declination_version": declination_version,
        "mounting_revision": mounting_revision,
        "deps": sorted(dep_hashes.items()),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
