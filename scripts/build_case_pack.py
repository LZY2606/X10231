"""Build the offline domain case pack:
``流束换算台：ADCP 波束速度、姿态旋转与质量层级``.

The pack is *not* a generic spreadsheet: it is generated through the same
forward beam-geometry / attitude model that the engine inverts, so every raw
beam velocity is physically consistent with a known true ENU profile. The
replay script then mutates one direct source at a time.

Run:  .venv/bin/python scripts/build_case_pack.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adcp.math3d import BeamGeometry, rz, ship_to_earth


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fixtures", "case_pack.json")

CELL_SIZE = 1.0
BLANKING = 0.5
N_CELLS = 12
GAP_MAX_SPACING = 6.0

PING_TIMES = [0, 4, 8, 12, 16, 20, 24, 28, 40, 44]
PING_IDS = [f"p{i:02d}" for i in range(1, 11)]
# Declared gap in the *ping* stream (the attitude stream keeps sampling
# through it - the two gaps are independent evidence).
PING_GAPS = [[28.0, 40.0]]

ATT_TIMES = list(range(-2, 43, 4))  # -2,2,...,42

GEOM_V1 = {"geometry_id": "geom-v1", "beam_angle_deg": 20.0}
DECL_V1 = {"version_id": "decl-v1", "angle_deg": 7.0,
           "convention": "true = magnetic + declination(east-positive)"}
MOUNT_V1 = {"version_id": "mount-v1", "angle_deg": 0.0}
TH_V1 = {"threshold_id": "qc-v1", "corr_min": 64,
         "corr_marginal": 80, "amp_min_db": 6.0}


def true_heading(t: float) -> float:
    """Magnetic heading in the pack crosses 360 deg near t=10s."""
    return (340.0 + 2.0 * t) % 360.0


def pitch_roll(t: float) -> tuple[float, float]:
    return 2.1 + 0.4 * math.sin(t / 9.0), -1.3 + 0.2 * math.cos(t / 7.0)


def earth_truth(cell: int) -> np.ndarray:
    c = cell - 1
    return np.array([0.32 - 0.012 * c, 0.10 + 0.004 * c, 0.002 + 0.001 * c])


def build() -> dict:
    geom = BeamGeometry(**GEOM_V1)
    bm = geom.beam_matrix()

    attitude = []
    for k, t in enumerate(ATT_TIMES):
        h = true_heading(t)
        p, r = pitch_roll(t)
        attitude.append({
            "att_id": f"a{k:02d}",
            "t_s": float(t),
            "heading_raw": f"{h:.2f}", "heading_unit": "deg",
            "heading_datum": "magnetic",
            "pitch_raw": f"{p:.3f}", "pitch_unit": "deg",
            "roll_raw": f"{r:.3f}", "roll_unit": "deg",
        })

    pings = []
    rng = np.random.default_rng(20260924)
    for pid, t in zip(PING_IDS, PING_TIMES):
        h_true = (true_heading(t) + DECL_V1["angle_deg"]) % 360.0
        pitch, roll = pitch_roll(t)
        # v_ship from v_earth with mount 0:
        m = ship_to_earth(np.eye(3), h_true, pitch, roll, "hpr-v1")
        cells = []
        for cell in range(1, N_CELLS + 1):
            ve = earth_truth(cell)
            vs = np.linalg.inv(m) @ ve
            vb = bm @ vs
            beam_mm_s = [int(round(x * 1000)) for x in vb]
            corr = [int(95 + 18 * rng.uniform()) for _ in range(4)]
            amp = [round(92.0 - 3.1 * (cell - 1) + rng.uniform(-0.6, 0.6), 1)
                   for _ in range(4)]
            cells.append({
                "cell_no": cell,
                "centre_m": float(BLANKING + (cell - 0.5) * CELL_SIZE),
                "beam_velocity": {"values": [str(v) for v in beam_mm_s],
                                  "unit": "mm/s"},
                "correlation": {"values": [str(v) for v in corr],
                                "unit": "counts"},
                "amplitude": {"values": [str(v) for v in amp],
                              "unit": "dB"},
                "bad_flag": {"values": ["0", "0", "0", "0"]},
            })
        # Bottom track: deep water, stationary bottom, zero boat motion.
        depth = 20.0
        slant = depth / math.cos(math.radians(GEOM_V1["beam_angle_deg"]))
        bt = {
            "range": {"values": [f"{slant:.2f}"] * 4, "unit": "m"},
            "range_bad_flag": {"values": ["0"] * 4},
            "beam_velocity": {"values": ["0", "0", "0", "0"],
                              "unit": "mm/s"},
            "velocity_bad_flag": {"values": ["0"] * 4},
        }
        pings.append({"ping_id": pid, "t_s": float(t),
                      "cells": cells, "bottom_track": bt})

    pack = {
        "case_id": "beam-reduction-bench-v1",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "version": "1.0.0",
        "config": {
            "cell_size_m": CELL_SIZE,
            "blanking_m": BLANKING,
            "n_cells": N_CELLS,
            "gap_max_spacing_s": GAP_MAX_SPACING,
        },
        "sources": {
            "geometry": GEOM_V1,
            "mount_angle": MOUNT_V1,
            "declination": DECL_V1,
            "quality_thresholds": TH_V1,
            "transform_order": {"version_id": "order-v1", "order": "hpr-v1"},
            "ping_gaps": [{"gap_id": "pgap-v1", "t0_s": a, "t1_s": b}
                          for a, b in PING_GAPS],
            "attitude_gaps": [],
            "attitude": attitude,
            "pings": pings,
        },
        "script": [
            {"op": "flag_bad_beam", "ping": "p03", "beam": 4,
             "label": "单 beam 失效（p03 波束4 硬件坏标志）",
             "expect": {"code": "single-beam-failure",
                        "contained_to_pings": ["p03"]}},
            {"op": "attitude_outage", "gap": [14.0, 26.0],
             "remove_att": ["a04", "a05"],
             "label": "姿态缺口（14–26s），插值不得跨缺口",
             "expect": {"code": "attitude-gap-refused",
                        "contained_to_pings": ["p05", "p06", "p07"]}},
            {"op": "set_correlation", "ping": "p08", "beam": 4,
             "value_counts": 72,
             "label": "三波束多解（p08 波束4 相关度落入可疑带）",
             "expect": {"code": "three-beam-ambiguity",
                        "candidate_kinds": ["4-beam", "3-beam"],
                        "contained_to_pings": ["p08"]}},
            {"op": "shallow_bottom", "ping": "p09",
             "ranges_m": [6.0, 6.0, 5.9, 6.0],
             "bad_amplitude": {"cell": 6, "beam": 2, "raw": "ovf"},
             "label": "无效深度 cell + 不可解析原始值（p09）",
             "expect": {"code": "invalid-depth-cell",
                        "contained_to_pings": ["p09"]}},
            {"op": "revise_mount", "angle_deg": 2.5,
             "version_id": "mount-v2",
             "label": "安装角修订 0°→2.5°（全局传播）",
             "expect": {"code": "mount-revision",
                        "recompute": ["ship", "earth"],
                        "unchanged": ["beam", "instrument", "bottom"]}},
            {"op": "branch_transform_order", "order": "hpr-v2",
             "version_id": "order-v2",
             "label": "坐标转换顺序分支 hpr-v2（指纹不同）",
             "expect": {"code": "transform-order-branch"}},
            {"op": "review_branch", "decision": "reject",
             "label": "比对后否决 hpr-v2，保留未选分支",
             "expect": {"code": "branch-rejected"}},
            {"op": "local_source_edit", "ping": "p10", "cell": 2,
             "beam": 1, "field": "correlation", "value_counts": 55,
             "label": "局部源数据改动（仅 p10/c2 重算）",
             "expect": {"code": "local-source-edit",
                        "contained_to": ["p10", 2]}},
            {"op": "decide", "ping": "p08", "choice": "3-beam",
             "actor": "hydrologist-a",
             "label": "使用者 A 采纳 p08 三波束解",
             "expect": {"code": "decision-recorded"}},
            {"op": "decide_conflict", "ping": "p08", "choice": "4-beam",
             "actor": "reviewer-b",
             "label": "使用者 B 基于同一版本改选四波束 → 可解决冲突",
             "expect": {"code": "decision-conflict",
                        "resolution": "keep-a",
                        "retained": ["3-beam", "4-beam"]}},
        ],
    }
    return pack


def main() -> None:
    pack = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(pack, fh, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"wrote {OUT}: {len(pack['sources']['pings'])} pings, "
          f"{len(pack['sources']['attitude'])} attitude samples, "
          f"{len(pack['script'])} script ops")


if __name__ == "__main__":
    main()
