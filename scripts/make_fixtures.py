"""生成“流束换算台：ADCP 波束速度、姿态旋转与质量层级”离线案例包。

确定性生成（无随机数），落盘 fixtures/。包含：
  pings：6 个 ping x 3 个深度 cell x 4 波束，径向速度用 mm/s 原始单位
         （派生层才规范化为 m/s）；correlation(count)、amplitude(dB)、
         坏 beam 标志分开；
  attitude：独立时间序列，含跨 360°（359.x -> 3.x）与一次 6.5s 时间缺口；
  bottom_track：四波束速度 + 沿波束测距；
  基线内置：一个无法换算单位的波束（附诊断）、一个超过 bottom track 的 cell。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from adcp_tool.geometry import default_rd_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")

N_BEAMS = 4
ZENITH = 20.0


def beam_radials(v_inst):
    g = default_rd_geometry(N_BEAMS, ZENITH)
    return (g.direction_cosines() @ np.asarray(v_inst, dtype=float)) * 1000.0  # mm/s


def healthy_beam(radial_mm_s):
    return {
        "velocity": round(float(radial_mm_s), 3),
        "velocity_unit": "mm/s",
        "correlation": 128,
        "correlation_unit": "count",
        "amplitude": 72,
        "amplitude_unit": "dB",
        "bad_flag": 0,
    }


def main():
    os.makedirs(FIX, exist_ok=True)
    g = default_rd_geometry(N_BEAMS, ZENITH)

    # 每个 ping 的“真值”仪器速度 (m/s)
    true_velocities = {
        0: [0.30, -0.20, 0.05],
        1: [0.31, -0.18, 0.04],
        2: [0.28, -0.22, 0.06],
        3: [0.26, -0.25, 0.03],
        4: [0.24, -0.27, 0.02],
        5: [0.22, -0.30, 0.01],
    }

    pings = []
    times = [10.0, 11.0, 12.0, 13.0, 16.5, 21.0]
    # bottom 深度：大部分 8m，ping6 浅底 2.5m -> 2.9m 的 bin3 超过底深
    bottoms_m = [8.0, 8.0, 8.0, 8.0, 8.0, 2.5]
    bin_depths = [1.0, 2.0, 2.9]

    for pi, t in enumerate(times):
        pid = f"case-A/ping/{pi+1}"
        rad = beam_radials(true_velocities[pi])
        cells = []
        for bi, depth in enumerate(bin_depths):
            cid = f"{pid}/bin/{bi+1}"
            beams = [healthy_beam(rad[b]) for b in range(N_BEAMS)]
            # 基线异常：ping2/bin3/beam4 速度单位无法换算 -> 保留原值+诊断
            if pi == 1 and bi == 2:
                beams[3]["velocity_unit"] = "furlong/fortnight"
            cells.append({
                "cell_id": cid,
                "bin_index": bi + 1,
                "depth_raw": depth,
                "depth_unit": "m",
                "beams": beams,
            })
        # bottom track
        bt_rad = beam_radials([-0.05, 0.02, -0.01])
        bottom = bottoms_m[pi]
        rng = bottom * math.cos(math.radians(ZENITH))
        bt = []
        for b in range(N_BEAMS):
            e = healthy_beam(bt_rad[b])
            e["range"] = round(rng, 4)
            e["range_unit"] = "m"
            bt.append(e)
        pings.append({
            "ping_id": pid,
            "seq": pi + 1,
            "t_s": t,
            "raw_time": f"2026-09-23T00:00:{t:06.3f}Z",
            "time_unit": "s",
            "cells": cells,
            "bottom_track": bt,
        })

    attitude = [
        {"seq": 1, "t_s": 10.0, "heading_deg": 10.0, "pitch_deg": 0.5,
         "roll_deg": -0.5, "heading_unit": "deg", "heading_datum": "MAGNETIC",
         "pitch_unit": "deg", "roll_unit": "deg"},
        {"seq": 2, "t_s": 11.0, "heading_deg": 12.0, "pitch_deg": 0.6,
         "roll_deg": -0.4},
        {"seq": 3, "t_s": 12.0, "heading_deg": 359.0, "pitch_deg": 0.7,
         "roll_deg": -0.6},
        {"seq": 4, "t_s": 12.5, "heading_deg": 359.5, "pitch_deg": 0.7,
         "roll_deg": -0.6},
        {"seq": 5, "t_s": 13.5, "heading_deg": 3.5, "pitch_deg": 0.8,
         "roll_deg": -0.7},  # 跨 360：ping4(t=13) 插值得 ~1.5°
        {"seq": 6, "t_s": 20.0, "heading_deg": 30.0, "pitch_deg": 0.4,
         "roll_deg": -0.3},  # 与上一样本间隔 6.5s > 2s：ping5 缺口
        {"seq": 7, "t_s": 21.0, "heading_deg": 32.0, "pitch_deg": 0.4,
         "roll_deg": -0.3},
    ]

    case = {
        "case_id": "adcp-beam-console-v1",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "kind": "adcp_case",
        "encoding": "fixture/mm-s+count+dB+deg-MAGNETIC",
        "batch_id": "case-A",
        "pings": pings,
        "attitude": attitude,
    }
    with open(os.path.join(FIX, "adcp_case.json"), "w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=2, sort_keys=False)

    configs = {
        "geometry": {
            "v1": {
                "version": "v1",
                "geometry": {
                    "convention": g.convention,
                    "azimuth_deg": list(g.azimuth_deg),
                    "zenith_deg": list(g.zenith_deg),
                },
            },
            "v2-revised": {
                "version": "v2-revised",
                "geometry": {
                    "convention": f"{g.convention}-REVISED",
                    "azimuth_deg": list(g.azimuth_deg),
                    "zenith_deg": [21.0, 21.0, 21.0, 21.0],
                },
            },
        },
        "install": {
            "v1": {"version": "v1", "heading_offset_deg": 0.0,
                   "pitch_offset_deg": 0.0, "roll_offset_deg": 0.0,
                   "order": "zyx"},
            "v2-mount-revision": {"version": "v2-mount-revision",
                                  "heading_offset_deg": 1.5,
                                  "pitch_offset_deg": 0.2,
                                  "roll_offset_deg": -0.3,
                                  "order": "zyx"},
        },
        "declination": {
            "v1": {"version": "v1", "declination_deg": -7.0,
                   "source": "fixture-manual"},
            "v2-survey": {"version": "v2-survey", "declination_deg": -7.5,
                          "source": "fixture-survey-office"},
        },
        "transform": {
            "v1": {"version": "v1", "ship_earth_order": "hpr"},
            "v2-order": {"version": "v2-order", "ship_earth_order": "hrp"},
        },
        "qc": {"v1": {"min_correlation_count": 64.0, "min_amplitude_db": 40.0,
                      "corr_unit": "count", "amp_unit": "dB"}},
        "attitude": {"v1": {"method": "linear", "max_gap_s": 2.0}},
        "depth": {"v1": {"blank_m": 0.5}},
    }
    with open(os.path.join(FIX, "configs.json"), "w", encoding="utf-8") as fh:
        json.dump(configs, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print("wrote fixtures/adcp_case.json and fixtures/configs.json")


if __name__ == "__main__":
    main()
