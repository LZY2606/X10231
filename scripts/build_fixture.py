"""离线领域案例包：流束换算台（ADCP 波束速度、姿态旋转与质量层级）。

不使用任何通用表格替代：该案例包专为“波束速度 -> instrument -> ship -> earth”
链路构造，初始版本即把波束测量、correlation、amplitude、坏 beam 标志、
heading/pitch/roll、磁偏角、安装角分开记录。随后通过确定性步骤覆盖：
  S1 单 beam 失效（三波束多解）
  S2 heading 跨越 360°（359 -> 1，展开后连续）
  S3 时间缺口（姿态插值不得跨缺口）
  S4 三波束多解（全部合法候选保留，不混同四波束值）
  S5 安装角修订（仅影响下游安装角/坐标节点）
  S6 坐标变换顺序（HPR vs HRP，进入派生指纹）
  S7 无效深度 cell（显式标志 + 越界深度，保留证据不丢弃）

所有输出确定可重复（无随机数）。运行：.venv/bin/python -m scripts.build_fixture
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from core.geometry import TransducerConfig, forward_matrix

BEAM_ANGLE_DEG = 20.0
N_PINGS = 8
N_CELLS = 6
DT = 1.0


def _true_uvw(ping: int, cell: int) -> list[float]:
    """合成真值：随深度变化的剪切流 + 微弱波动。"""
    z = 1.0 + 0.5 * cell
    u = 0.35 - 0.03 * cell + 0.02 * np.sin(ping / 2.0)
    v = -0.20 + 0.01 * cell + 0.015 * np.cos(ping / 3.0)
    w = 0.02 * np.sin(0.7 * ping + 0.5 * cell) - 0.002 * z
    return [round(float(u), 5), round(float(v), 5), round(float(w), 6)]


def _depth(ping: int, cell: int) -> float:
    return round(1.0 + 0.5 * cell + 0.001 * ping, 3)


def _heading(ping: int) -> float:
    # ping0..2 正常；ping3/4 跨越 360（359.x -> 0.x -> 1.x）
    table = {0: 12.0, 1: 12.4, 2: 13.0, 3: 359.2, 4: 0.6, 5: 1.2, 6: 1.8, 7: 2.4}
    return table[ping]


def _ping_time(ping: int) -> float:
    # ping5 与 ping6 之间制造 6 秒姿态缺口（见 attitude 采样）
    return float(ping)


def _attitude_samples():
    # 正常每秒一采；在 t=4.6 后下一条直接跳到 11.0 -> ping6(t=6)/ping7(t=7) 落缺口
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 4.6, 11.0, 12.0]
    heading = [12.0, 12.4, 13.0, 359.2, 359.8, 0.6, 2.0, 2.4]
    pitch = [1.0 + 0.1 * np.sin(t) for t in times]
    roll = [-0.5 + 0.08 * np.cos(t) for t in times]
    return {"times": times, "heading_deg": heading,
            "pitch_deg": [round(x, 4) for x in pitch],
            "roll_deg": [round(x, 4) for x in roll], "unit": "deg"}


def _beam_row(u: float, v: float, w: float, kill_beam: int | None = None,
              noise: float = 0.0) -> list[float | str]:
    a = forward_matrix(TransducerConfig(BEAM_ANGLE_DEG))
    b = a @ np.array([u, v, w, 0.01])
    b = b + noise
    row = [round(float(x) * 1000.0, 2) for x in b]  # m/s -> mm/s
    if kill_beam is not None:
        row[kill_beam] = "NaN"
    return row


def build_baseline() -> dict:
    """初始版本：来源分开记录，全部高质量，无缺口外的异常。"""
    pings = []
    for p in range(N_PINGS):
        rows, corr, amp, bad, cells = [], [], [], [], []
        for c in range(N_CELLS):
            u, v, w = _true_uvw(p, c)
            rows.append(_beam_row(u, v, w))
            corr.append([180, 175, 178, 170])
            amp.append([120, 118, 122, 115])
            bad.append([False, False, False, False])
            cells.append({"depth_raw": _depth(p, c), "depth_unit": "m", "invalid": False})
        pings.append({
            "ping": p, "time_s": _ping_time(p),
            "beam_velocity_raw": rows, "velocity_unit": "mm/s",
            "correlation": corr, "amplitude": amp, "bad_beam": bad, "cells": cells,
            "bottom_track_raw": _beam_row(0.25, -0.1, 0.0, noise=0.002),
            "bottom_track_unit": "mm/s",
        })
    return {
        "case_pack": "adcp-beam-stage",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "version": "v1-baseline",
        "pings": pings,
        "attitude": _attitude_samples(),
        "declination": {"value_deg": -7.5, "unit": "deg", "east_positive": True},
        "mounting": {"heading_alignment_deg": 0.0, "pitch_alignment_deg": 0.0,
                     "roll_alignment_deg": 0.0},
        "geometry": {"beam_angle_deg": BEAM_ANGLE_DEG, "config_code": "convex_down",
                     "beam_count": 4},
        "quality_rules": {"min_correlation": 64, "min_amplitude_counts": 40,
                          "max_error_velocity_mps": 0.5, "min_depth_m": 0.3,
                          "max_depth_m": 30.0},
        "transform_order": "HPR",
        "attitude_max_gap_s": 2.0,
    }


def apply_scenario(base: dict, code: str) -> dict:
    """只改动一个直接来源（或一项规则），返回变体；基线永不被修改。"""
    import copy
    d = copy.deepcopy(base)
    d["version"] = f"v1-{code}"
    d["scenario"] = code

    if code == "S1-single-beam-fail":
        # ping2 cell3 的 beam1 失效：bad flag + 低相关/幅度 + 不可转换速度
        cell = d["pings"][2]["cells"] and d["pings"][2]
        cell["beam_velocity_raw"][3][1] = "NaN"
        cell["bad_beam"][3][1] = True
        cell["correlation"][3][1] = 12
        cell["amplitude"][3][1] = 8

    elif code == "S2-heading-wrap":
        # 基线已含 359.2 -> 0.6；这里再加密一圈：ping4 航向改为 359.95 -> 0.05
        d["attitude"]["heading_deg"][4] = 359.95
        d["attitude"]["heading_deg"][5] = 0.05

    elif code == "S3-time-gap":
        # 扩大缺口：移除 t=4.6 样本，使 ping5(t=5) 之后直到 11s 无样本
        d["attitude"]["times"] = [0.0, 1.0, 2.0, 3.0, 4.0, 11.0, 12.0]
        d["attitude"]["heading_deg"] = [12.0, 12.4, 13.0, 359.2, 359.8, 2.0, 2.4]
        d["attitude"]["pitch_deg"] = d["attitude"]["pitch_deg"][:5] + d["attitude"]["pitch_deg"][6:]
        d["attitude"]["roll_deg"] = d["attitude"]["roll_deg"][:5] + d["attitude"]["roll_deg"][6:]

    elif code == "S4-three-beam-multi":
        # 两个不同 ping/cell 的单波束失效，用于并排比较三波束候选与四波束值
        for p, c, b in [(1, 2, 2), (6, 4, 0)]:
            cell = d["pings"][p]
            cell["beam_velocity_raw"][c][b] = "NaN"
            cell["bad_beam"][c][b] = True
            cell["correlation"][c][b] = 9
            cell["amplitude"][c][b] = 6

    elif code == "S5-mounting-revision":
        # 安装角修订：heading alignment 0 -> +1.5°（与几何/磁偏角独立）
        d["mounting"]["heading_alignment_deg"] = 1.5
        d["mounting"]["pitch_alignment_deg"] = 0.4

    elif code == "S6-transform-order":
        # 坐标变换顺序反例：HPR -> HRP
        d["transform_order"] = "HRP"

    elif code == "S7-invalid-cell":
        # 无效深度 cell：ping3 cell5 显式标志；ping4 cell0 深度越界且无法转换
        d["pings"][3]["cells"][5]["invalid"] = True
        d["pings"][4]["cells"][0]["depth_raw"] = "N/A"
        d["pings"][4]["cells"][0]["depth_unit"] = "m"

    else:
        raise KeyError(code)
    return d


SCENARIOS = ["S1-single-beam-fail", "S2-heading-wrap", "S3-time-gap",
             "S4-three-beam-multi", "S5-mounting-revision", "S6-transform-order",
             "S7-invalid-cell"]


def write_all(out_dir: pathlib.Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_baseline()
    paths = {}
    base_path = out_dir / "baseline.json"
    base_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["baseline"] = str(base_path)
    for code in SCENARIOS:
        v = apply_scenario(base, code)
        pth = out_dir / f"{code}.json"
        pth.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[code] = str(pth)
    return paths


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    print(json.dumps(write_all(here), ensure_ascii=False, indent=2))
