"""仓库内离线领域案例包：流束换算台——ADCP 波束速度、姿态旋转与质量层级。

不是通用表格：所有数值围绕 Janus ADCP 的
  波束径向速度 -> instrument -> ship -> earth
逐级换算和质量层级手工编排，并在基线中埋入边界条件：

  P3  单 beam 坏标志（确定性三波束解）
  P5  无效深度 cell（盲区 / 超底跟踪深度）
  P6  一个波束速度 "NA"（不可转换值留原位附诊断）；姿态缺失需段内插值
  P4/P5 航向跨越 359.4 -> 0.6（连续化）
  P7 -> PG -> P8 时间缺口；PG 无姿态，插值不得跨缺口
  P9  beam1 correlation 临界：四波束解 + 四个三波束候选并存（多解/未定）
  P10 bottom track 无效
所有领域量携带单位/基准；部分速度以 cm/s 记录，导入保留原始表示。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .geometry import BeamGeometry
from .transforms import heading_matrix, _rot_x, _rot_y

CASE_ID = "adcp-janus-quality-ladder"

# 质量门槛（派生规则；阈值版本也进入指纹）
CONFIG = {
    "max_attitude_gap_s": 60.0,
    "correlation_min_count": 70,
    "correlation_hard_min": 60,
    "error_velocity_max_mps": 0.15,
    "blanking_min_m": 2.0,
    "rotation_order": "H,P,R",
    "rules_version": "rules:v1",
}

DECL = {"value": "8.5", "unit": "deg", "version": "decl:mag2024a:v1",
        "note": "magnetic declination, east positive"}
MOUNT = {"heading": "0", "pitch": "0", "roll": "0", "unit": "deg",
         "version": "mount:initial:v1"}

# (ping_index, offset_seconds_from_P0, heading_mag, pitch, roll, attitude_present)
_PINGS = [
    (0,    0,  12.5,  1.00, -0.40, True),
    (1,   20,  12.6,  0.95, -0.42, True),
    (2,   40,  12.4,  1.05, -0.38, True),
    (3,   60,  12.7,  1.02, -0.41, True),
    (4,   80, 359.4,  0.90, -0.50, True),
    (5,  100,   0.6,  0.88, -0.52, True),
    (6,  120,   None, None,  None, False),   # 姿态缺失：段内插值
    (7,  140,   1.1,  0.85, -0.55, True),
    (8,  300,  None, None, None, False),    # 时间缺口中的 ping，无姿态
    (9,  460,   1.4,  0.82, -0.57, True),   # 与 P7 间隔 320s：缺口
    (10, 480,   1.2,  0.80, -0.58, True),
    (11, 500,   1.0,  0.78, -0.60, True),
    (12, 520,   0.9,  0.76, -0.61, True),
]

# cell 中心深度（m）；0 号在盲区内
CELL_DEPTHS = [1.2, 3.0, 5.0, 7.0, 9.0, 19.0]
T0 = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)


def _true_earth_current(z):
    """合成真值剖面（earth END, m/s）。"""
    return np.array([0.30 - 0.004 * z, 0.10 + 0.002 * z, 0.015], float)


def _inst_from_earth(v_end, h_true, pitch, roll, order="H,P,R"):
    M = heading_matrix(h_true)
    labels = [t.strip() for t in order.split(",")]
    R = np.eye(3)
    mats = {"P": _rot_y(math.radians(pitch)), "R": _rot_x(math.radians(roll))}
    for label in labels:
        if label in mats:
            R = R @ mats[label]
    full = M @ R
    return full.T @ np.asarray(v_end, float)


def _fmt(v, unit):
    if unit == "m/s":
        return f"{v:.4f}"
    return f"{v * 100.0:.2f}"  # cm/s


def build_case() -> dict:
    geom = BeamGeometry()
    decl_east = float(DECL["value"])
    pings = []
    rng = np.random.default_rng(20260501)

    for idx, off, hmag, pitch, roll, has_att in _PINGS:
        t = T0 + timedelta(seconds=off)
        attitude = None
        h_true_for_sim = None
        if has_att:
            attitude = {
                "time": t.isoformat().replace("+00:00", "Z"),
                "heading": f"{hmag:.1f}", "pitch": f"{pitch:.2f}",
                "roll": f"{roll:.2f}", "unit": "deg",
                "heading_datum": "magnetic",
            }
            h_true_for_sim = hmag + decl_east
        elif idx == 6:  # noqa
            # 用插值真值（0.85 磁航向）生成波束，但删掉姿态记录，强迫系统插值
            h_true_for_sim = 0.85 + decl_east
            pitch, roll = 0.865, -0.535
        else:
            h_true_for_sim = 1.25 + decl_east
            pitch, roll = 0.8, -0.6

        vel_unit = "cm/s" if idx == 2 else "m/s"
        cells = []
        for c, z in enumerate(CELL_DEPTHS):
            v_end = _true_earth_current(z)
            v_inst = _inst_from_earth(v_end, h_true_for_sim, pitch, roll)
            radial = geom.beam_unit_vectors() @ v_inst
            radial += rng.normal(0, 0.002, 4)
            beams = [_fmt(x, vel_unit) for x in radial]
            corr = [205, 198, 211, 196]
            amp = [118, 121, 119, 117]
            bad = ["0", "0", "0", "0"]

            if idx == 3:                       # 单 beam 失效
                bad[2] = "1"
            if idx == 9:                       # beam1 临界相关
                corr[1] = 65
            if idx == 6 and c == 3:            # 不可转换值留在原位
                beams[2] = "NA"

            cells.append({
                "cell": c,
                "depth": f"{z:.1f}", "depth_unit": "m",
                "beam_velocity": beams, "velocity_unit": vel_unit,
                "correlation": [str(v) for v in corr],
                "amplitude": [str(v) for v in amp],
                "bad_beam": bad,
            })

        # bottom track
        vbt_end = np.array([0.285, 0.095, 0.0])
        vbt_inst = _inst_from_earth(vbt_end, h_true_for_sim, pitch, roll)
        bt_radial = geom.beam_unit_vectors() @ vbt_inst
        bt_range = [20.1, 19.9, 20.0, 20.2]
        if idx == 5:
            bt_range = [17.0, 16.9, 17.1, 17.0]   # 浅：最深 cell 越界
        bt = {
            "range": [f"{r:.1f}" for r in bt_range], "range_unit": "m",
            "beam_velocity": [f"{x:.4f}" for x in bt_radial],
            "velocity_unit": "m/s",
            "correlation": ["210", "208", "212", "209"],
            "bad_beam": ["0", "0", "0", "0"],
            "valid": "0" if idx == 10 else "1",
        }

        pings.append({
            "ping": idx,
            "time": t.isoformat().replace("+00:00", "Z"),
            "attitude": attitude,
            "cells": cells,
            "bottom_track": bt,
        })

    return {
        "case_id": CASE_ID,
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "description": (
            "Janus ADCP 离线案例：beam velocity/correlation/amplitude/bad flag/"
            "heading/pitch/roll/磁偏角/安装角分开记录，逐级重算 instrument/ship/"
            "earth 流速，覆盖单波束失效、航向跨 360、时间缺口、三波束多解、"
            "安装角修订、变换顺序与无效深度 cell。"),
        "schema_version": 1,
        "config": CONFIG,
        "beam_geometry": {
            "beam_angle_deg": 20.0, "convex": True,
            "orientation": "rdi_janus", "version": "geom:janus20:v1",
        },
        "declination": DECL,
        "mount": MOUNT,
        "pings": pings,
    }


def write_fixture(path):
    case = build_case()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=2, sort_keys=False)
    return path
