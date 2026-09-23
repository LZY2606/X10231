"""Build the offline ADCP case pack (domain material, not a generic table).

The hydrology scenario: raw beam velocities + correlation + amplitude + bad
beam flags + HPR + magnetic declination + transducer alignment are recorded
separately, then recomputed to instrument/ship/earth velocity.

Water beam velocities are synthesised by a *forward* beam projection
(earth current + ship motion rotated into the beam frame), so a clean ping
reproduces the truth with ~zero beam error velocity - the round-trip is the
self-check of the geometry and rotation code.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adcp_tool.core import geometry as geo
from adcp_tool.core import rotation as rot

CASE_PACK_VERSION = "adcp-flowbench-v1"
N_PINGS = 6
N_CELLS = 5
DT = 1.0
BEAM_ANGLE_DEG = 20.0
BT_OFFSET_S = 0.5


def _measured(raw, raw_unit):
    return {"raw": raw, "raw_unit": raw_unit}


def _earth_current(t, depth_m):
    # Smooth sheared current in ENU (m/s).
    east = 0.45 + 0.04 * depth_m + 0.03 * math.sin(0.7 * t)
    north = 0.20 - 0.02 * depth_m + 0.02 * math.cos(0.5 * t)
    up = 0.01 * math.sin(0.9 * t + depth_m * 0.3)
    return np.array([east, north, up], dtype=float)


def _ship_earth(t):
    # Ship velocity in ENU (m/s), mostly east with slight north drift.
    return np.array([0.60 + 0.05 * math.sin(0.4 * t), 0.10, -0.02], dtype=float)


def _attitude(t):
    # Headings deliberately cross 360 deg between ping 0 and ping 1.
    table = [358.6, 2.4, 6.1, 10.0, 13.8, 17.6]
    i = min(int(round(t)), N_PINGS - 1)
    return {"heading_deg": float(table[i]),
            "pitch_deg": float(1.0 + 0.2 * math.sin(0.6 * t)),
            "roll_deg": float(0.5 + 0.15 * math.cos(0.8 * t))}


def _water_beams(t, depth_m, geom, cfg):
    cur_enu = _earth_current(t, depth_m)
    ship_enu = _ship_earth(t)
    rel_enu = cur_enu - ship_enu
    hpr = _attitude(t)
    true_h = hpr["heading_deg"] + cfg["declination_deg"]
    ship_frd = rot.FRD_TO_ENU.T @ rel_enu
    r_frd = rot.frd_rotation(
        math.radians(true_h), math.radians(hpr["pitch_deg"]),
        math.radians(hpr["roll_deg"]), cfg["transform_order"],
    )
    # rel_enu = FRD_TO_ENU @ R @ instr  =>  instr = R^T @ FRD_TO_ENU^T @ rel_enu
    instr = r_frd.T @ rot.FRD_TO_ENU.T @ rel_enu
    beams = geom.matrix() @ instr
    return beams, cur_enu


def _bottom_track(t, geom, cfg):
    # Bottom is fixed earth; instrument measures -ship motion in its own frame.
    ship_enu = _ship_earth(t)
    hpr = _attitude(t)
    true_h = hpr["heading_deg"] + cfg["declination_deg"]
    r_frd = rot.frd_rotation(
        math.radians(true_h), math.radians(hpr["pitch_deg"]),
        math.radians(hpr["roll_deg"]), cfg["transform_order"],
    )
    instr = r_frd.T @ rot.FRD_TO_ENU.T @ (-ship_enu)
    return geom.matrix() @ instr


def build_baseline():
    cfg = {
        "min_correlation": 64.0,
        "min_amplitude_db": 30.0,
        "error_velocity_threshold": 0.05,
        "declination_deg": -7.5,
        "declination_version": "wmm-2024-v1",
        "declination_basis": "magnetic",
        "mount_heading_deg": 0.0,
        "mount_pitch_deg": 0.0,
        "mount_roll_deg": 0.0,
        "transform_order": "hpr-v1",
        "gap_factor": 1.5,
        "beam_angle_deg": BEAM_ANGLE_DEG,
        "azimuths_deg": list(geo.DEFAULT_AZIMUTHS_DEG),
        "geometry_version": geo.GEOMETRY_VERSION,
    }
    geom = geo.BeamGeometry(cfg["beam_angle_deg"], tuple(cfg["azimuths_deg"]),
                            cfg["geometry_version"])

    attitude_samples = []
    pings = []
    truth = []
    for i in range(N_PINGS):
        t = float(i * DT)
        hpr = _attitude(t)
        attitude_samples.append({"time": t, **hpr, "raw_unit": "deg"})
        cells = {}
        for c in range(N_CELLS):
            depth = 2.0 + c * 2.0
            beams, cur = _water_beams(t, depth, geom, cfg)
            cell_id = "c%02d" % c
            cells[cell_id] = {
                "depth_m": _measured(depth, "m"),
                "valid": True,
                "beam_velocity": [_measured(round(float(v * 100.0), 4), "cm/s") for v in beams],
                "correlation": [_measured(180 - 3 * c - b, "count") for b in range(4)],
                "amplitude_db": [_measured(round(72.0 - 1.8 * c - 0.4 * b, 2), "dB") for b in range(4)],
                "bad_beam_flag": [False, False, False, False],
            }
            truth.append({"ping": "p%03d" % i, "cell": cell_id,
                          "depth_m": depth, "earth_ENU": [float(x) for x in cur]})
        bt_beams = _bottom_track(t + BT_OFFSET_S, geom, cfg)
        bt = {
            "time": t + BT_OFFSET_S,
            "range_m": [_measured(12.0 + 0.3 * b, "m") for b in range(4)],
            "beam_velocity": [_measured(round(float(v * 100.0), 4), "cm/s") for v in bt_beams],
            "correlation": [_measured(190, "count") for _ in range(4)],
            "amplitude_db": [_measured(85.0, "dB") for _ in range(4)],
            "bad_beam_flag": [False] * 4,
        }
        pings.append({"id": "p%03d" % i, "time": t, "cells": cells, "bottom_track": bt})

    # One attitude sample on the far side of the gap demonstrates refusal to
    # interpolate across it (baseline stays gap-free; gap fork moves p004).
    attitude_samples.append({"time": float(N_PINGS - 1) * DT + 28.0,
                             "heading_deg": 40.0, "pitch_deg": 1.0, "roll_deg": 0.5,
                             "raw_unit": "deg"})

    return {
        "schema": CASE_PACK_VERSION,
        "case_id": "flowbench-baseline",
        "title": "Flow Bench: ADCP beam velocities, attitude rotation and quality tiers",
        "description": "Janus 4-beam convex ADCP, separate raw records, forward-projected truth.",
        "config": cfg,
        "attitude_samples": attitude_samples,
        "attitude_gap_flags": [],
        "pings": pings,
        "truth": truth,
    }


def write_pack(out_dir):
    baseline = build_baseline()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "flowbench_baseline.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = write_pack(here)
    print("wrote", p)
