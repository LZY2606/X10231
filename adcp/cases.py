"""离线领域案例包: 流束换算台 —— ADCP 波束速度、姿态旋转与质量层级。

不是通用表格: 数据由波束几何正演生成 (B @ [u,v,w,e]), 带领域单位/基准,
基线把波束测量 / correlation / amplitude / 坏 beam 标志 / heading,pitch,roll /
磁偏角 / 安装角拆成独立来源, 重放脚本依次走过:
  S1 单 beam 失效 -> 三波束多解族
  S2 姿态跨越 360 度 (unwrap 正确, 不产生 180 度假转向)
  S3 时间缺口 (姿态插值不跨缺口)
  S4 安装角修订 (只影响 ship/earth, instrument 层 reused)
  S5 坐标转换顺序修订 (进入指纹, 作为独立分支保留)
  S6 无效深度 cell (留在排除证据, 不为画图丢弃)
  S7 并发人工决定冲突 (两位使用者同一基线分叉, 必须显式解决)
  S8 接受一个三波束候选 (其余候选保留, 剖面不混合)
所有值都是确定性的, 重放脚本可重复执行。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .attitude import ATTITUDE_RULE
from .geometry import BeamGeometry
from .transforms import TRANSFORM_RULE

CASEPACK_VERSION = "adcp-flowbench/v1"
N_PINGS = 12
N_CELLS = 6
GEOM = BeamGeometry()

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "casepack")


def _q(value: Any, unit: str, kind: str, frame: str = "none",
       reference: str = "none") -> Dict[str, Any]:
    from .units import q
    return q(value, unit, kind, frame=frame, reference=reference).as_dict()


def _true_earth_profile(cell_index: int) -> Tuple[float, float, float]:
    """真剖面: 随深度变化的东向流 + 北向剪切, w 小。"""
    depth = 3.0 + cell_index * 2.0
    u = 0.60 - 0.02 * cell_index + 0.05 * math.sin(cell_index / 2.0)
    v = 0.10 + 0.03 * cell_index
    w = 0.01 * (cell_index - 2)
    return u, v, w


def _heading_deg(ping_index: int) -> float:
    # ping 6->7 从 352 跨到 8 度: 真实航向连续前进 16 度穿过 0/360
    base = (10.0 + 16.0 * ping_index) % 360.0
    return base


def _synthesize_beam_cell(u: float, v: float, w: float,
                          geom: BeamGeometry = GEOM) -> List[Dict[str, Any]]:
    beams = geom.forward_matrix() @ np.array([u, v, w, 0.0])
    return [_q(round(float(x), 6), "m/s", "velocity", frame="beam")
            for x in beams]


def _build_beam_sources() -> List[Dict[str, Any]]:
    sources = []
    for p in range(N_PINGS):
        cells = []
        for c in range(N_CELLS):
            u, v, w = _true_earth_profile(c)
            # 基线里 instrument ~ earth 的差由姿态小角度体现;
            # 波束正演直接用真剖面, 旋转在数值上自洽即可 (姿态整体有小偏置)。
            beams = _synthesize_beam_cell(u, v, w)
            cells.append({"cell_index": c,
                          "depth_m": _q(3.0 + c * 2.0, "m", "depth"),
                          "beams": beams,
                          "raw_line": "P%02dC%02d,BEAM,%s"
                                      % (p, c, ",".join(
                                          b["raw_repr"] for b in beams))})
        sources.append({
            "id": "src-beam-p%02d" % p,
            "kind": "beam_measurement",
            "scope": "default",
            "payload": {"ping_id": "ping%02d" % p, "ping_index": p,
                        "time_s": 10.0 + 5.0 * p,
                        "cells": cells,
                        "raw_format": "m/s, beam frame, sign=toward-transducer"}})
    return sources


def _build_quality_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                      List[Dict[str, Any]]]:
    corr, amp, flag = [], [], []
    for p in range(N_PINGS):
        corr.append({"id": "src-corr-p%02d" % p, "kind": "correlation",
                     "scope": "default",
                     "payload": {"ping_id": "ping%02d" % p,
                                 "correlations": [180, 175, 170, 178],
                                 "raw_format": "0-255 counts"}})
        amp.append({"id": "src-amp-p%02d" % p, "kind": "amplitude",
                    "scope": "default",
                    "payload": {"ping_id": "ping%02d" % p,
                                "amplitudes": [120, 125, 118, 122],
                                "raw_format": "counts"}})
        flag.append({"id": "src-flag-p%02d" % p, "kind": "bad_beam_flag",
                     "scope": "default",
                     "payload": {"ping_id": "ping%02d" % p,
                                 "bad_flags": [False, False, False, False],
                                 "raw_format": "boolean per beam"}})
    return corr, amp, flag


def _build_attitude_samples(*, gap: bool = False,
                            override: Optional[Dict[int, float]] = None
                            ) -> List[Dict[str, Any]]:
    """姿态采样比 ping 密一倍; gap=True 时删掉 ping 9 附近两个样本造成缺口。
    override: ping_index(每 2.5 秒采样索引) -> heading 覆盖。
    """
    samples = []
    k = 0
    t = 10.0
    while t <= 10.0 + 5.0 * (N_PINGS - 1) + 0.1:
        idx_f = (t - 10.0) / 5.0
        h = _heading_deg(round(idx_f)) if abs(idx_f - round(idx_f)) < 1e-9 \
            else (_heading_deg(math.floor(idx_f))
                  + ((_heading_deg(math.ceil(idx_f))
                      - _heading_deg(math.floor(idx_f))) % 360.0)
                     * (idx_f - math.floor(idx_f))) % 360.0
        if override and k in override:
            h = override[k]
        samples.append({"t": round(t, 3), "heading_deg": round(h % 360.0, 4),
                        "pitch_deg": 1.20, "roll_deg": -0.80,
                        "raw_format": "degrees, heading magnetic"})
        t += 2.5
        k += 1
    if gap:
        # 删除覆盖 ping09 (t=55s) 的包围样本, 留下 >2s 的空洞 (52.5 与 57.5 都删)
        samples = [s for s in samples if s["t"] not in (52.5, 57.5)]
    return samples


def baseline_sources() -> List[Dict[str, Any]]:
    """案例包初始版本: 所有来源分类独立。"""
    beams = _build_beam_sources()
    corr, amp, flag = _build_quality_sources()
    sources: List[str] and List[Dict[str, Any]] = []
    sources.extend(beams)
    sources.extend(corr)
    sources.extend(amp)
    sources.extend(flag)
    sources.append({
        "id": "src-attitude-001", "kind": "heading_pitch_roll", "scope": "default",
        "payload": {"samples": _build_attitude_samples(),
                    "raw_format": "degrees", "gap_tolerance_s": 2.0}})
    sources.append({
        "id": "src-decl-001", "kind": "declination", "scope": "default",
        "payload": {"version": "decl-IGRF-2020/v1", "declination_deg": -7.5,
                    "reference": "magnetic->true, east-positive",
                    "raw_repr": "7d30W"}})
    sources.append({
        "id": "src-mount-001", "kind": "mount_config", "scope": "default",
        "payload": {"version": "mount-survey-2025/v1",
                    "angles": {"heading": 0.0, "pitch": 0.0, "roll": 0.0},
                    "raw_repr": "0.0,0.0,0.0 deg"}})
    sources.append({
        "id": "src-xdcr-001", "kind": "transducer_config", "scope": "default",
        "payload": {"version": "xdcr-janus-20deg/v1",
                    "geometry": GEOM.as_dict(),
                    "beam_count": 4, "frequency_khz": 300}})
    for p in range(N_PINGS):
        sources.append({
            "id": "src-btm-p%02d" % p, "kind": "bottom_track", "scope": "default",
            "payload": {"ping_id": "ping%02d" % p,
                        "range_m": _q(15.5 + 0.4 * math.sin(p), "m", "length"),
                        "valid_depth_m": 14.0,
                        "bt_velocity": _q([0.02 * p, -0.01, 0.0], "m/s",
                                          "velocity", frame="earth",
                                          reference="true"),
                        "raw_format": "m / m s-1"}})
    return sources


def fixture_files() -> Dict[str, str]:
    """写出仓库内 fixture 文件 (测试只使用这些文件, 不联网)。"""
    sources = baseline_sources()
    manifest = {
        "casepack": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "casepack_version": CASEPACK_VERSION,
        "attitude_rule": ATTITUDE_RULE,
        "transform_rule": TRANSFORM_RULE,
        "geometry_rule": GEOM.rule,
        "n_pings": N_PINGS, "n_cells": N_CELLS,
        "description": "ADCP beam velocity / attitude / bottom-track evidence "
                       "bench fixture with separated source kinds",
        "source_files": sorted({s["id"] for s in sources}),
    }
    return {
        "baseline.json": json.dumps(
            {"manifest": manifest, "sources": sources},
            ensure_ascii=False, indent=2, sort_keys=True),
        "manifest.json": json.dumps(manifest, ensure_ascii=False,
                                    indent=2, sort_keys=True),
    }
