"""Offline case pack: "流束换算台 — beam velocity / attitude / quality tiers".

Deterministic synthetic data (not a generic table): a 4-beam convex Janus
head, ship on an eastward 0.10 m/s track, ENU current profile
U(z)=0.50+0.02z, 12 depth cells (m), attitude logged at 20 s cadence.

Baseline pings p01..p10 are fully separable source records. Counterexample
branches change exactly one direct source each. Everything is seeded with
stable string ids so export/delete/import preserves identity and order.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import rdi_janus, TransducerConfig, project
from .transforms import ship_to_earth_matrix

GEOMETRY_V1 = "geom:v1"
DECL_V1 = "decl:v1"
DECL_V2 = "decl:v2"
INSTALL_V1 = "install:v1"
INSTALL_V2 = "install:v2"
ORDER_V1 = "RDI_PR"
ORDER_V2 = "ALT_RP"
POLICY_V1 = "policy:v1"
CELLS_V1 = "cells:v1"

N_BEAMS = 4
N_CELLS = 12
DEPTHS = [3.0 + 1.0 * i for i in range(N_CELLS)]   # 3..14 m
GAP_THRESHOLD_S = 45.0
T0 = 1_800_000_000.0


def config_v1() -> TransducerConfig:
    return rdi_janus(GEOMETRY_V1, beam_angle_deg=20.0)


def _earth_truth(depth_m: float) -> tuple[float, float, float]:
    return (0.50 + 0.02 * depth_m, 0.0, -0.05)  # east, north, up (slight down)


def attitude_log() -> list[dict]:
    # 20 s cadence from t=0; pings sit midpoint between samples.
    samples = []
    for i in range(11):
        t = T0 + 20.0 * i
        samples.append({
            "sample_id": f"a{i + 1:02d}", "t": t,
            "heading": 80.0 + 2.0 * i, "pitch": 1.0 + 0.1 * i, "roll": -0.5,
            "heading_unit": "deg", "pitch_unit": "deg", "roll_unit": "deg",
            "source": "attitude_log",
        })
    return samples


def ping_times() -> dict[str, float]:
    # Each ping is midpoint of two attitude samples (t=20i+10), so the
    # baseline itself exercises wrap-aware interpolation.
    return {f"p{i + 1:02d}": T0 + 20.0 * i + 10.0 for i in range(10)}


def _ship_velocity_si() -> tuple[float, float, float]:
    return (0.10, 0.0, 0.0)  # east, north, up: ship steaming east


def _true_heading_deg(i: int) -> float:
    # matches attitude log interpolated heading at the ping time t=20i+10
    return 81.0 + 2.0 * i


def synth_ping(idx: int) -> dict:
    """Return raw per-beam arrays plus metadata for ping p(idx+1).

    Forward model uses true earth current + ship motion, rotates earth->ship
    with the known heading, then projects onto the beams.
    """
    ping_id = f"p{idx + 1:02d}"
    t = T0 + 20.0 * idx + 10.0
    cfg = config_v1()
    heading_deg = _true_heading_deg(idx)
    pitch = 1.0 + 0.1 * (idx + 0.5)
    roll = -0.5
    # water velocity relative to ship = current - ship velocity, earth frame
    M = np.asarray(ship_to_earth_matrix(heading_deg))
    ship_e = np.array(_ship_velocity_si())
    cells = []
    for depth in DEPTHS:
        current_e = np.array(_earth_truth(depth))
        rel_e = current_e - ship_e
        # earth -> ship is inverse (transpose) of ship -> earth
        rel_ship = M.T @ rel_e
        # pitch/roll zero order: instrument == ship here
        beams_mps = project(cfg, rel_ship)
        cells.append({"depth_m": depth, "beams_mps": beams_mps})
    # bottom track: instrument sees negative ship velocity in ship frame
    bt_rel_ship = M.T @ (-ship_e)
    bt_beams = project(cfg, bt_rel_ship)
    return {
        "ping_id": ping_id, "t": t,
        "heading_true_synth": heading_deg, "pitch_synth": pitch, "roll_synth": roll,
        "cells": cells,
        "bt": {"range_m": 16.5, "beams_mps": bt_beams},
    }


def build_baseline() -> dict:
    """Produce the separable baseline source records, rules and pings."""
    cfg = config_v1()
    pings = [synth_ping(i) for i in range(10)]
    sources: dict[str, list[dict]] = {
        "ping": [], "beam_velocity": [], "correlation": [], "amplitude": [],
        "bad_flag": [], "attitude": attitude_log(), "bt_velocity": [],
        "bt_range": [], "bt_flag": [], "depth_cell_bounds": [],
    }
    for p in pings:
        sources["ping"].append({"record_key": p["ping_id"], "version": 1, "payload": {
            "ping_id": p["ping_id"], "t": p["t"], "time_unit": "s",
            "source": "adcp_raw",
        }})
        sources["bt_range"].append({"record_key": p["ping_id"], "version": 1, "payload": {
            "ping_id": p["ping_id"], "range": {"value": p["bt"]["range_m"], "unit": "m"},
            "source": "bottom_track"}})
        sources["bt_flag"].append({"record_key": p["ping_id"], "version": 1, "payload": {
            "ping_id": p["ping_id"], "flags": [False] * N_BEAMS,
            "source": "bottom_track"}})
        for b in range(N_BEAMS):
            sources["bad_flag"].append({
                "record_key": f"{p['ping_id']}:b{b + 1}", "version": 1,
                "payload": {"ping_id": p["ping_id"], "beam": b + 1,
                            "bad": False, "source": "quality_flag"}})
        for ci, cell in enumerate(p["cells"]):
            for b in range(N_BEAMS):
                v = cell["beams_mps"][b]
                sources["beam_velocity"].append({
                    "record_key": f"{p['ping_id']}:c{ci + 1:02d}:b{b + 1}",
                    "version": 1,
                    "payload": {
                        "ping_id": p["ping_id"], "cell_idx": ci, "beam": b + 1,
                        "q": {"value": round(v * 100.0, 4), "unit": "cm/s",
                              "raw": f"{round(v * 100.0, 4):.4f} cm/s"},
                        "source": "beam_velocity"}})
                sources["correlation"].append({
                    "record_key": f"{p['ping_id']}:c{ci + 1:02d}:b{b + 1}",
                    "version": 1,
                    "payload": {"ping_id": p["ping_id"], "cell_idx": ci, "beam": b + 1,
                                "corr": {"value": 200, "unit": "counts_255", "raw": "200"},
                                "source": "correlation"}})
                # p05 cell 4 beam 2 amplitude is an unconvertible raw token.
                if p["ping_id"] == "p05" and ci == 3 and b == 1:
                    amp = {"value": None, "unit": "counts", "raw": "N/A"}
                else:
                    amp = {"value": 130, "unit": "counts", "raw": "130"}
                sources["amplitude"].append({
                    "record_key": f"{p['ping_id']}:c{ci + 1:02d}:b{b + 1}",
                    "version": 1,
                    "payload": {"ping_id": p["ping_id"], "cell_idx": ci, "beam": b + 1,
                                "amp": amp, "source": "amplitude"}})
        for b in range(N_BEAMS):
            v = p["bt"]["beams_mps"][b]
            sources["bt_velocity"].append({
                "record_key": f"{p['ping_id']}:b{b + 1}", "version": 1,
                "payload": {"ping_id": p["ping_id"], "beam": b + 1,
                            "q": {"value": round(v * 100.0, 4), "unit": "cm/s",
                                  "raw": f"{round(v * 100.0, 4):.4f} cm/s"},
                            "source": "bottom_track"}})
    sources["depth_cell_bounds"].append({"record_key": "profile", "version": 1, "payload": {
        "cell_depths_m": DEPTHS, "blanking_m": 0.5, "depth_unit": "m",
        "source": "transducer_config"}})

    rules = {
        "geometry": {"version_id": GEOMETRY_V1,
                     "payload": {"type": "rdi_janus", "beam_angle_deg": 20.0,
                                 "azimuths_deg": [45.0, 135.0, 225.0, 315.0],
                                 "radial_sign": 1.0}},
        "declination": {"version_id": DECL_V1,
                        "payload": {"declination_deg": -7.5, "unit": "deg",
                                    "name": "WMM v1"}},
        "install": {"version_id": INSTALL_V1,
                    "payload": {"heading_offset_deg": 0.0,
                                "pitch_offset_deg": 0.0, "roll_offset_deg": 0.0,
                                "unit": "deg"}},
        "transform_order": {"version_id": ORDER_V1, "payload": {"order": ORDER_V1}},
        "quality_policy": {"version_id": POLICY_V1,
                           "payload": {"corr_min_counts": 80, "amp_min_counts": 60,
                                       "err_velocity_max": 0.05,
                                       "gap_threshold_s": GAP_THRESHOLD_S}},
        "cells": {"version_id": CELLS_V1,
                  "payload": {"cell_depths_m": DEPTHS, "blanking_m": 0.5}},
    }
    return {"sources": sources, "rules": rules, "ping_times": ping_times()}
