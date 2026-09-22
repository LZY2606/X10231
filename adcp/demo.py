"""Offline domain case package: 流束换算台 ADCP 波束速度、姿态旋转与质量层级.

This is not a generic spreadsheet: pings carry beam velocities forward-modeled
from a known NED current, Janus beam geometry and a documented attitude stream.
All quantities keep their imported representation and unit/datum; one cell even
imports an unconvertible depth so it must stay in place with a diagnostic.

Replay steps are machine-readable, idempotent events. Each event changes exactly
one direct source or one rule/config version and the engine then propagates only
to the affected dependency subgraph.
"""
from __future__ import annotations

import math
import copy
from typing import Any, Dict, List

import numpy as np

from .geometry import (
    Geometry, ORDER_HPR, ORDER_XYZ, attitude_rotation, mount_rotation,
)
from .quantity import speed, angle, length, count

CASE_ID = "adcp-bench-001"
NBINS = 5
NPING = 8
DT = 1.0
BEAM_ANGLE = 20.0
CELL_DEPTHS_M = [2.5, 3.5, 4.5, 5.5, 6.5]   # last cell below bottom for ping 3
BOTTOM_DEPTH_M = 5.9


def _dirs():
    return Geometry(beam_angle_deg=BEAM_ANGLE).directions()


def _beam_velocities(isf: np.ndarray) -> List[float]:
    return [float(x) for x in _dirs() @ isf]


def _qbeam_ms(values_ms):
    """Beam velocities are generated in m/s but imported/recorded in cm/s."""
    return [speed(v, "m/s").to_json() for v in values_ms]


def _good_cell(depth_raw, depth_unit, isf: np.ndarray):
    beams = _beam_velocities(isf)
    return {
        "depth_q": length(depth_raw, depth_unit).to_json(),
        "velocity_q": _qbeam_ms(beams),
        "correlation_q": [count(180 + 5 * ((b + 1) % 3), "count").to_json()
                          for b in range(4)],
        "amplitude_q": [count(120 + 4 * b, "count").to_json() for b in range(4)],
        "bad_beam": [False, False, False, False],
    }


def _heading_true_deg(pi: int) -> float:
    """Magnetic heading stream unwrapped: 356,357,358,359,360(=0)..."""
    return 356.0 + float(pi)


def true_earth_ned(pi: int, ci: int) -> np.ndarray:
    """Known water velocity in earth NED (the value recomputation must recover)."""
    north = 0.30 + 0.01 * ci
    east = -0.10 + 0.005 * ((pi - 3) / 3.0)
    down = 0.02 + 0.002 * ci
    return np.array([north, east, down])


def _true_isf(pi: int, ci: int) -> np.ndarray:
    """Forward model: earth NED current rotated into ISF using the same chain."""
    heading_true = math.radians(_heading_true_deg(pi) - 7.5)
    r_earth = attitude_rotation(heading_true, math.radians(1.2 + 0.1 * pi),
                                math.radians(-0.8 + 0.05 * pi), ORDER_HPR)
    return r_earth.T @ true_earth_ned(pi, ci)


def _attitude_stream():
    # Times and magnetic headings deliberately crossing 359 -> 1.
    times = [float(i) for i in range(NPING + 2)]
    headings = [356.0 + float(i) for i in range(5)] + [3.0, 4.0, 5.0, 6.0, 7.0]
    pitches = [1.2 + 0.1 * i for i in range(len(times))]
    rolls = [-0.8 + 0.05 * i for i in range(len(times))]
    samples = []
    for t, h, p, r in zip(times, headings, pitches, rolls):
        samples.append({
            "time": t,
            "heading_q": angle(h, "deg", "magnetic").to_json(),
            "pitch_q": angle(p, "deg").to_json(),
            "roll_q": angle(r, "deg").to_json(),
        })
    return samples


def _bottom_track(pi: int):
    # Bottom moves relative to earth with a known small NED vector.
    heading_true = math.radians(_heading_true_deg(pi) - 7.5)
    r_earth = attitude_rotation(heading_true, math.radians(1.2 + 0.1 * pi),
                                math.radians(-0.8 + 0.05 * pi), ORDER_HPR)
    isf = r_earth.T @ np.array([-0.05, 0.02, -0.01])
    beams = _beam_velocities(isf)
    return {
        "depth_q": length(BOTTOM_DEPTH_M, "m").to_json(),
        "velocity_q": _qbeam_ms(beams),
        "correlation_q": [count(190, "count").to_json() for _ in range(4)],
        "amplitude_q": [count(150, "count").to_json() for _ in range(4)],
        "bad_beam": [False, False, False, False],
    }


def baseline_bundle() -> Dict[str, Any]:
    pings = []
    for pi in range(NPING):
        cells = []
        for ci in range(NBINS):
            depth_raw = CELL_DEPTHS_M[ci]
            depth_unit = "m"
            # Import raw representation is preserved: ping 0 depths recorded in cm.
            if pi == 0:
                depth_raw, depth_unit = CELL_DEPTHS_M[ci] * 100.0, "cm"
            cell = _good_cell(depth_raw, depth_unit, _true_isf(pi, ci))
            cells.append(cell)
        # Ping 3: deepest cell is below bottom (invalid depth cell evidence).
        # Ping 4: one imported depth cannot be normalized and must stay put.
        if pi == 4:
            cells[2]["depth_q"] = {
                "raw": "n/a", "unit": "m", "datum": None, "si_value": None,
                "si_unit": "m", "convertible": False,
                "diagnostic": "unconvertible-length", "meta": {"note": "logger fault"},
            }
        pings.append({
            "ping_id": "ping-%02d" % pi,
            "time": float(pi * DT),
            "cells": cells,
            "bottom_track": _bottom_track(pi),
        })

    return {
        "case_id": CASE_ID,
        "pings": pings,
        "attitude": _attitude_stream(),
        "geometry": Geometry(beam_angle_deg=BEAM_ANGLE).to_json(),
        "mount": {
            "mount_version": "mount-1",
            "heading_offset_rad": 0.0,
            "pitch_offset_rad": 0.0,
            "roll_offset_rad": 0.0,
            "note": "factory aligned",
        },
        "declination": {
            "declination_version": "decl-1",
            "angle_rad": math.radians(-7.5),
            "model": "WMM-adjacent-synthetic",
        },
        "policy": {"min_correlation": 60.0, "min_amplitude": 30.0,
                   "error_tolerance": 0.05, "policy_version": "qpol-1"},
        "transform_order": ORDER_HPR,
        "gap_factor": 2.5,
    }


# ---------------------------------------------------------------------------
# Single-source mutations. Each touches exactly one direct source/rule input.
# ---------------------------------------------------------------------------

def apply_step(bundle: Dict[str, Any], step: str) -> Dict[str, Any]:
    """Return a deep-copied bundle with one named replay mutation applied."""
    b = copy.deepcopy(bundle)
    if step == "single-beam-fail":
        # Ping 2, cell 1, beam 3: hardware bad flag (whole beam evidence stays).
        b["pings"][2]["cells"][1]["bad_beam"][3] = True
        b["pings"][2]["cells"][1]["correlation_q"][3]["raw"] = 12
        b["pings"][2]["cells"][1]["correlation_q"][3]["si_value"] = 12.0
    elif step == "heading-wrap":
        # Already present in baseline; force a steeper wrap near a ping time.
        b["attitude"][3]["heading_q"]["raw"] = 359.7
        b["attitude"][3]["heading_q"]["si_value"] = math.radians(359.7)
        b["attitude"][4]["heading_q"]["raw"] = 0.3
        b["attitude"][4]["heading_q"]["si_value"] = math.radians(0.3)
    elif step == "time-gap":
        # Remove attitude samples for times 4 and 5: creates a >2.5x gap.
        b["attitude"] = [s for s in b["attitude"] if s["time"] not in (4.0, 5.0)]
    elif step == "three-beam-ambiguous":
        # Inject a strong beam-1 spike at ping 5/cell 0: error velocity blows up,
        # four-beam is flagged inconsistent while all 3-beam candidates persist.
        q = b["pings"][5]["cells"][0]["velocity_q"][0]
        q["raw"] = q["raw"] + 45.0
        q["si_value"] = q["si_value"] + 0.45
    elif step == "mount-revision":
        b["mount"] = {
            "mount_version": "mount-2",
            "heading_offset_rad": math.radians(2.0),
            "pitch_offset_rad": math.radians(0.5),
            "roll_offset_rad": math.radians(-0.3),
            "note": "field alignment revision 2026 survey",
        }
    elif step == "transform-order-alt":
        b["transform_order"] = ORDER_XYZ
    elif step == "declination-revision":
        b["declination"] = {
            "declination_version": "decl-2",
            "angle_rad": math.radians(-8.2),
            "model": "WMM-revised-synthetic",
        }
    elif step == "invalid-cell":
        # Ping 6 cell 4 raw depth cannot be parsed: keep in place + diagnostic.
        b["pings"][6]["cells"][4]["depth_q"] = {
            "raw": "--", "unit": "m", "datum": None, "si_value": None,
            "si_unit": "m", "convertible": False,
            "diagnostic": "unconvertible-length", "meta": {}}
    else:
        raise ValueError("unknown replay step: %s" % step)
    return b


REPLAY_STEPS = [
    ("E1", "single-beam-fail", "单 beam 失效（ping2/cell1/beam3 坏标志）"),
    ("E2", "heading-wrap", "姿态跨越三百六十度（359.7°→0.3°）"),
    ("E3", "time-gap", "时间缺口（删除 4s、5s 姿态采样）"),
    ("E4", "three-beam-ambiguous", "三波束多解（beam1 异常，error velocity 超差）"),
    ("E5", "mount-revision", "安装角修订（mount-1 → mount-2）"),
    ("E6", "transform-order-alt", "坐标变换顺序替换（保留 HPR 候选）"),
    ("E7", "invalid-cell", "无效深度 cell（深度无法规范化，原位保留诊断）"),
]
