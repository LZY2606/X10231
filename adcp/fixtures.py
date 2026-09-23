"""离线领域案例包：流束换算台 —— ADCP 波束速度、姿态旋转与质量层级。

不是通用表格：fixtures/case_adcp 下保存基线事实（分开记录的
波束测量/correlation/amplitude/坏 beam 标志/heading/pitch/roll/
磁偏角/安装角）以及 steps.json（确定性重放脚本）。

正演：用与引擎相同的波束几何，从设定的 earth 流速 + 姿态生成
径向波束速度，因此四波束解应回到设定值（自洽可校验）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .geometry import BeamGeometry
from .model import make_fact
from .transforms import (Attitude, MountAngles, Declination,
                         instrument_to_ship, r_heading)

CASE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "case_adcp"
N_PINGS = 8
N_CELLS = 6


def _earth_profile(ping, cell):
    """设定 earth 剖面（m/s，东/北/垂向向下为正）。"""
    east = 0.30 + 0.02 * cell + 0.03 * np.sin(ping * 0.9)
    north = 0.15 - 0.01 * cell + 0.02 * np.cos(ping * 0.7)
    up_down = 0.02 + 0.002 * cell
    return np.array([east, north, up_down])


def _attitude_of(ping):
    return Attitude(
        heading_deg=float((5.0 * ping + 350.0) % 360.0),  # 跨越 360
        pitch_deg=float(1.0 * np.sin(ping * 0.8)),
        roll_deg=float(0.8 * np.cos(ping * 0.6)))


def generate_baseline(*, with_unconvertible: bool = True):
    """生成基线事实列表。所有原始量带单位/基准；导入保留原始表示。"""
    rng = np.random.default_rng(20260924)
    geom = BeamGeometry(20.0, True)
    mount = MountAngles(0.0)
    decl = Declination(7.25, "WMM", "2025.0")
    facts = []

    facts.append(make_fact(
        "geoconfig", payload={
            "beam_angle_deg": 20.0, "convex": True,
            "name": "rdi_janus", "blanking_m": 0.88},
        fact_id="cfg_geom"))
    facts.append(make_fact(
        "qconfig", payload={
            "corr_trusted": 80, "corr_marginal": 40, "amp_floor": 30,
            "error_velocity_limit_mps": 0.20,
            "cell_depth_min_m": 0.5, "cell_depth_max_m": 60.0,
            "max_attitude_gap_s": 2.0},
        fact_id="cfg_quality"))
    facts.append(make_fact(
        "declination", raw=7.25, raw_unit="deg", datum="true_minus_mag",
        payload={"value_deg": 7.25, "model": "WMM", "epoch": "2025.0"},
        fact_id="cfg_decl"))
    facts.append(make_fact(
        "mount", payload={"heading_align_deg": 0.0,
                          "pitch_offset_deg": 0.0,
                          "roll_offset_deg": 0.0},
        fact_id="cfg_mount"))
    facts.append(make_fact(
        "transform_order", raw="rdi_pitch_roll",
        fact_id="cfg_order"))

    # heading / pitch / roll 三者分开记录（每个采样时刻一条）
    for p in range(N_PINGS):
        att = _attitude_of(p)
        t = float(p * 2)  # 每 2s 一 ping
        facts.append(make_fact(
            "heading", raw=att.heading_deg, raw_unit="deg",
            datum="magnetic", ping=p, time_s=t,
            fact_id=f"h_p{p}"))
        facts.append(make_fact(
            "pitch", raw=att.pitch_deg, raw_unit="deg", ping=p,
            time_s=t, fact_id=f"pi_p{p}"))
        facts.append(make_fact(
            "roll", raw=att.roll_deg, raw_unit="deg", ping=p,
            time_s=t, fact_id=f"ro_p{p}"))

    h_eff = lambda att: (att.heading_deg + mount.heading_align_deg
                         + decl.value_deg) % 360.0

    for p in range(N_PINGS):
        t = float(p * 2)
        att = _attitude_of(p)
        for c in range(N_CELLS):
            depth = round(2.0 + c * 1.0, 2)
            facts.append(make_fact(
                "celldepth", raw=depth, raw_unit="m", datum="transducer",
                ping=p, cell=c, fact_id=f"d_p{p}_c{c}"))
            earth = _earth_profile(p, c)
            v_inst = _earth_to_instrument(earth, att, mount, h_eff(att))
            radials = geom.forward(v_inst)
            for b in range(4):
                if with_unconvertible and p == 7 and c == 5 and b == 2:
                    raw, unit = "BAD", "m/s"
                else:
                    raw, unit = round(float(radials[b]), 5), "m/s"
                facts.append(make_fact(
                    "beam", raw=raw, raw_unit=unit, datum="radial_positive",
                    ping=p, cell=c, beam=b, time_s=t,
                    fact_id=f"v_p{p}_c{c}_b{b}"))
                # 故意布置低质量波束（供后续步骤打标/观察，不在这里置坏）
                corr = int(rng.integers(88, 99))
                amp = int(rng.integers(95, 130))
                if p == 4 and c == 2 and b == 1:
                    corr = 55  # marginal
                facts.append(make_fact(
                    "correlation", raw=corr, raw_unit="counts",
                    ping=p, cell=c, beam=b, fact_id=f"cor_p{p}_c{c}_b{b}"))
                facts.append(make_fact(
                    "amplitude", raw=amp, raw_unit="dB",
                    ping=p, cell=c, beam=b, fact_id=f"amp_p{p}_c{c}_b{b}"))
                facts.append(make_fact(
                    "badflag", raw=0, raw_unit="flag",
                    ping=p, cell=c, beam=b, fact_id=f"bad_p{p}_c{c}_b{b}"))

        # bottom track：船静止近似，earth=(0,0,0)
        bt_rad = geom.forward(np.zeros(3))
        for b in range(4):
            facts.append(make_fact(
                "bt_beam", raw=round(float(bt_rad[b]), 5), raw_unit="m/s",
                ping=p, beam=b, time_s=t, fact_id=f"btv_p{p}_b{b}"))
            facts.append(make_fact(
                "bt_correlation", raw=95, raw_unit="counts", ping=p,
                beam=b, fact_id=f"btcor_p{p}_b{b}"))
            facts.append(make_fact(
                "bt_amplitude", raw=120, raw_unit="dB", ping=p,
                beam=b, fact_id=f"btamp_p{p}_b{b}"))
            facts.append(make_fact(
                "bt_badflag", raw=0, raw_unit="flag", ping=p, beam=b,
                fact_id=f"btbad_p{p}_b{b}"))

    return facts


def _earth_to_instrument(earth, att, mount, h_eff):
    ship = r_heading(h_eff).T @ earth
    from .transforms import r_pitch, r_roll
    inst = r_pitch(att.pitch_deg + mount.pitch_offset_deg).T @ (
        r_roll(att.roll_deg + mount.roll_offset_deg).T @ ship)
    return inst


STEPS = [
    {"id": "S00", "title": "基线导入：七类来源分开记录",
     "mutation": "baseline",
     "expect": {"gate": "four_beam_unique", "pings": N_PINGS,
                "cells": N_CELLS}},
    {"id": "S01", "title": "单 beam 失效：p1/c2/b1 坏标志",
     "mutation": "single_beam_failure",
     "params": {"ping": 1, "cell": 2, "beam": 0},
     "expect": {"gate": "three_beam_unique", "chosen_kind": "three_beam"}},
    {"id": "S02", "title": "姿态跨越 360°：基线航向 350→355→0→5…",
     "mutation": "check_heading_wrap",
     "expect": {"wrap_seen": True}},
    {"id": "S03", "title": "时间缺口：删除 t=8s 姿态采样（缺口 4s）",
     "mutation": "attitude_gap",
     "params": {"time_s": 8.0},
     "expect": {"ping_uncomputable": 4, "neighbors_ok": [3, 5]}},
    {"id": "S04", "title": "三波束多解：p2/c1/b2 降为 marginal",
     "mutation": "marginal_beam",
     "params": {"ping": 2, "cell": 1, "beam": 1, "corr": 50},
     "expect": {"gate": "four_beam_ambiguous",
                "candidate_cids": ["C4", "C3_m1", "C3_m2", "C3_m3",
                                   "C3_m4"]}},
    {"id": "S05", "title": "人工裁定：否决 C4，接受 C3_m2（其余候选保留）",
     "mutation": "adjudicate",
     "params": {"ping": 2, "cell": 1, "reject": "C4", "accept": "C3_m2"},
     "expect": {"accepted": "C3_m2", "retained_unaccepted": True}},
    {"id": "S06", "title": "安装角修订：heading align 0°→+5°",
     "mutation": "mount_revision",
     "params": {"heading_align_deg": 5.0},
     "expect": {"all_earth_recomputed": True}},
    {"id": "S07", "title": "坐标变换顺序分支：roll_pitch（反例）",
     "mutation": "order_revision",
     "params": {"order": "roll_pitch"},
     "expect": {"earth_differs_from_standard": True}},
    {"id": "S08", "title": "无效深度 cell：p3/c0 深度改为 0.20m（盲区）",
     "mutation": "invalid_cell",
     "params": {"ping": 3, "cell": 0, "depth": 0.20},
     "expect": {"node_status": "excluded_cell", "profile_keeps_evidence": True}},
    {"id": "S09", "title": "局部源数据改动：仅 p5/c3/b0 径向速度修订",
     "mutation": "local_fact_correction",
     "params": {"ping": 5, "cell": 3, "beam": 0, "delta": 0.04},
     "expect": {"only_subtree_recomputed": True}},
    {"id": "S10", "title": "磁偏角模型版本修订：WMM2025 7.25→WMM2026 7.60",
     "mutation": "declination_revision",
     "params": {"value_deg": 7.60, "model": "WMM", "epoch": "2026.0"},
     "expect": {"all_earth_recomputed": True}},
    {"id": "S11", "title": "并发保存：两位使用者基于同一 HEAD 编辑并解决冲突",
     "mutation": "concurrent_conflict",
     "expect": {"conflict_raised": True, "conflict_resolved": True}},
]


def build_pack(force: bool = False) -> Path:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    facts = generate_baseline()
    baseline_path = CASE_DIR / "baseline.json"
    steps_path = CASE_DIR / "steps.json"
    manifest_path = CASE_DIR / "manifest.json"
    if force or not baseline_path.exists():
        baseline_path.write_text(json.dumps(
            {"case": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
             "facts": facts}, ensure_ascii=False, indent=2))
        steps_path.write_text(json.dumps(
            {"steps": STEPS}, ensure_ascii=False, indent=2))
        manifest_path.write_text(json.dumps(
            {"name": "case_adcp", "n_pings": N_PINGS, "n_cells": N_CELLS,
             "units": {"velocity": "m/s (raw preserved)",
                       "angle": "deg (raw preserved)",
                       "depth": "m, datum=transducer"},
             "note": "领域专用离线案例包，非通用表格"},
            ensure_ascii=False, indent=2))
    return CASE_DIR


def load_pack(case_dir: Path = CASE_DIR):
    case_dir = Path(case_dir)
    baseline = json.loads((case_dir / "baseline.json").read_text())
    steps = json.loads((case_dir / "steps.json").read_text())
    return baseline["facts"], steps["steps"]


if __name__ == "__main__":
    p = build_pack(force=True)
    print("case pack written to", p)
PY_PLACEHOLDER = None
