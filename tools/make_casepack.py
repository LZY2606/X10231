"""Generate the committed offline ADCP case package.

Domain story: a hydrographic team re-derives instrument/ship/earth
velocities from raw beam velocities, attitude, bottom track and
transducer configuration.  The data are forward-modelled from a known
answer, with the boundary situations required by the replay embedded in
the fixture itself (360-degree heading seam, attitude time gap, one bad
beam, invalid depth cells, an unconvertible amplitude).

Run:  .venv/bin/python tools/make_casepack.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from adcp.transforms import beam_matrix, rot_x, rot_y, rot_z

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "casepack"

GEO = {"geometry_id": "rdi-wh600-janus20",
       "beam_angle_deg": 20.0,
       "azimuths_deg": [45.0, 135.0, 225.0, 315.0],
       "cells": 8, "cell_size_m": 0.5, "blanking_cells": 1}
DECLINATION = -2.0
MOUNTING = 0.0
ATT_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 6.2, 7.2]
PINGS = [1, 2, 3, 4, 5, 6]
MAX_ATT_GAP = 1.5


def ping_time(ping: int) -> float:
    return ping - 0.5


def heading_mag(t: float) -> float:
    return (350.0 + 3.4 * t) % 360.0


def pitch(t: float) -> float:
    return 1.5 * math.sin(0.9 * t)


def roll(t: float) -> float:
    return 2.0 * math.cos(0.6 * t)


def water_earth(cell: int) -> np.ndarray:
    return np.array([0.30 + 0.02 * cell,
                     0.10 - 0.005 * cell, 0.01])


def ship_motion(t: float) -> np.ndarray:
    # ship frame: x=starboard, y=forward (bottom track)
    return np.array([0.05, 1.0 + 0.02 * math.sin(t), 0.0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    D = beam_matrix(GEO["beam_angle_deg"], GEO["azimuths_deg"])

    beam_velocity, bottom_track = {}, {}
    for ping in PINGS:
        t = ping_time(ping)
        h_true = heading_mag(t) + DECLINATION
        motion_earth = rot_z(h_true) @ ship_motion(t)
        per_beam = [[] for _ in range(4)]
        for cell in range(GEO["cells"]):
            total_earth = water_earth(cell) + motion_earth
            v_instr = (rot_z(-MOUNTING) @ rot_x(-roll(t)) @ rot_y(-pitch(t))
                       @ rot_z(-h_true) @ total_earth)
            v_beam = D @ v_instr + np.array([1, -1, 1, -1]) * 0.003
            v_beam = v_beam + rng.normal(0.0, 0.002, 4)
            for beam in range(4):
                per_beam[beam].append(round(float(v_beam[beam]) * 1000.0, 1))
        beam_velocity[str(ping)] = per_beam
        motion = ship_motion(t)
        bottom_track[str(ping)] = {"ship:stbd_ms": round(float(motion[0]), 4),
                                   "ship:fore_ms": round(float(motion[1]), 4)}

    n_cells, n_beams = GEO["cells"], 4
    correlation = {str(p): [[90.0] * n_cells for _ in range(n_beams)]
                   for p in PINGS}
    amplitude = {str(p): [[120.0] * n_cells for _ in range(n_beams)]
                 for p in PINGS}
    bad_beam = {str(p): [False] * n_beams for p in PINGS}

    # boundary: low correlation invalidates depth cell 7 of ping 2
    for beam in range(n_beams):
        correlation["2"][beam][7] = 25.0
    # boundary: beam 2 of ping 4 is marginal on correlation and is later
    # flagged bad by the source-data revision (single-beam failure)
    correlation["4"][1][1:] = [55.0] * (n_cells - 1)

    baseline = {
        "case": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "case_id": "liushu-adcp-2026",
        "description": "offline replay package; forward modelled, no generic tables",
        "source": "casepack://liushu-adcp-2026/baseline",
        "config": GEO,
        "params": {"transform_order": "standard",
                    "min_correlation_pct": 40.0,
                    "blanking_cells": 1,
                    "max_att_gap_s": MAX_ATT_GAP,
                    "err_bound_ms": 0.05},
        "pings": [{"ping": p, "t": ping_time(p)} for p in PINGS],
        "attitude_samples": [
            {"t": t, "heading_mag_deg": heading_mag(t),
             "pitch_deg": round(pitch(t), 4),
             "roll_deg": round(roll(t), 4)} for t in ATT_TIMES],
        "magnetic_declination": {"value_deg": DECLINATION,
                                 "version": "WMM2020@2026.5"},
        "mounting_angle": {"value_deg": MOUNTING,
                           "version": "dock-align-2026-04"},
        "beam_velocity": {"unit": "mm/s", "frame": "beam",
                          "values": beam_velocity},
        "correlation": {"unit": "%", "frame": "beam",
                        "values": correlation},
        "amplitude": {"unit": "counts", "frame": "beam",
                      "values": amplitude,
                      "overrides": [
                          # unconvertible value: stays raw with a diagnostic,
                          # never dropped for plotting
                          {"ping": 6, "beam": 1, "cell": 3,
                           "value_raw": "overflow", "unit": "dB?"}]},
        "bad_beam_flag": {"unit": "bool", "values": bad_beam},
        "bottom_track": {"unit": "m/s", "frame": "ship",
                         "values": bottom_track},
    }

    steps = {
        "case_id": "liushu-adcp-2026",
        "steps": [
            {"id": "revise-mounting", "kind": "param_revision",
             "title": "安装角修订 (0.0° -> 1.7°)",
             "record": {"id": "mnt:1", "kind": "mounting_angle",
                        "value_raw": "1.7", "unit": "deg",
                        "frame": "ship",
                        "source": "dock-align-2026-05-rev1"},
             "expect_affected": sorted(
                 [f"ship/p{p}" for p in (1, 2, 3, 4)]
                 + [f"earth/p{p}" for p in (1, 2, 3, 4)]
                 + [f"profile/p{p}" for p in (1, 2, 3, 4)])},
            {"id": "beam-fail", "kind": "source_change",
             "title": "只改一个直接来源：ping4 beam2 坏波束标志",
             "supersede": "flg:p4:b2",
             "record": {"id": "flg:p4:b2#r4", "kind": "bad_beam_flag",
                        "ping": 4, "beam": 2, "cell": None,
                        "t": ping_time(4), "value_raw": "true",
                        "unit": "bool", "frame": "beam",
                        "source": "field-flag-revision-2026-05-02"},
             "expect_affected": ["beamq/p4", "instr/p4", "ship/p4",
                                 "earth/p4", "profile/p4"]},
            {"id": "conflict-decision", "kind": "decision_conflict",
             "title": "两位使用者基于同一版本对三波束候选做冲突决定",
             "node": "instr/p4",
             "users": ["hydro-a", "hydro-b"],
             "choices": [0, 2],
             "resolution": "theirs",
             "expect_versions_after": 6},
        ],
    }

    (OUT / "baseline.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "steps.json").write_text(
        json.dumps(steps, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT/'baseline.json'} and {OUT/'steps.json'}")


if __name__ == "__main__":
    main()
