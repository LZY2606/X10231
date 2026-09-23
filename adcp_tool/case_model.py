"""Normalise an imported case pack into the internal graph input.

Import keeps every raw representation untouched (``raw``/``raw_unit``); the
normalised SI value lives beside it as ``norm``/``norm_unit``.  Values that
cannot be converted stay in place with a ``diagnostic`` - they are never
silently dropped just to keep a plot alive.
"""
from __future__ import annotations

import copy

from .core.units import (
    Measured,
    identity,
    to_mps,
    to_rad,
    UNIT_DB,
    UNIT_M,
    UNIT_COUNT,
)


def _measured_from(obj):
    return Measured(raw=copy.deepcopy(obj["raw"]), raw_unit=obj["raw_unit"])


def _norm_counts(m):
    return identity(m)


def _norm_velocity(m):
    return to_mps(m)


def _norm_db(m):
    if m.raw_unit == UNIT_DB:
        return identity(m)
    m.diagnostic = "unsupported amplitude unit: %s" % m.raw_unit
    return None


def _norm_depth(m):
    if m.raw_unit == UNIT_M:
        return identity(m)
    m.diagnostic = "unsupported depth unit: %s" % m.raw_unit
    return None


def normalize_case(raw_case):
    case = {
        "schema": raw_case.get("schema"),
        "case_id": raw_case["case_id"],
        "title": raw_case.get("title", ""),
        "description": raw_case.get("description", ""),
        "config": copy.deepcopy(raw_case["config"]),
        "attitude_gap_flags": list(raw_case.get("attitude_gap_flags", [])),
        "pings": [],
        "attitude_samples": [],
        "source_ids": {
            "beam": {}, "correlation": {}, "amplitude": {}, "badflag": {},
            "attitude": {}, "bt_beam": {}, "bt_badflag": {}, "bt_attitude": {},
        },
    }

    for s in raw_case["attitude_samples"]:
        sm = {"time": float(s["time"]),
              "heading_deg": float(s["heading_deg"]),
              "pitch_deg": float(s["pitch_deg"]),
              "roll_deg": float(s["roll_deg"]),
              "raw_unit": s.get("raw_unit", "deg"),
              "raw": {"heading_deg": _angle(s["heading_deg"], s.get("raw_unit", "deg")),
                      "pitch_deg": _angle(s["pitch_deg"], s.get("raw_unit", "deg")),
                      "roll_deg": _angle(s["roll_deg"], s.get("raw_unit", "deg"))}}
        sm["usable"] = all(v.usable for v in sm["raw"].values())
        if not sm["usable"]:
            sm["diagnostic"] = ";".join(
                "%s:%s" % (k, v.diagnostic) for k, v in sm["raw"].items() if not v.usable
            )
        case["attitude_samples"].append(sm)

    for p in raw_case["pings"]:
        pid = p["id"]
        cells = {}
        for key in ("beam", "correlation", "amplitude", "badflag"):
            case["source_ids"][key][pid] = {}
        for cell_id, cell in p["cells"].items():
            depth = _measured_from(cell["depth_m"])
            _norm_depth(depth)
            beams = [_measured_from(v) for v in cell["beam_velocity"]]
            corrs = [_measured_from(v) for v in cell["correlation"]]
            amps = [_measured_from(v) for v in cell["amplitude_db"]]
            for m in beams:
                _norm_velocity(m)
            for m in corrs:
                _norm_counts(m)
            for m in amps:
                _norm_db(m)
            cells[cell_id] = {
                "depth_m": depth.as_dict(),
                "valid": cell.get("valid", True),
                "reason": cell.get("reason"),
                "beam_velocity": [m.as_dict() for m in beams],
                "correlation": [m.as_dict() for m in corrs],
                "amplitude_db": [m.as_dict() for m in amps],
                "bad_beam_flag": [bool(x) for x in cell["bad_beam_flag"]],
            }
            case["source_ids"]["beam"][pid][cell_id] = f"src:beam:{pid}:{cell_id}"
            case["source_ids"]["correlation"][pid][cell_id] = f"src:corr:{pid}:{cell_id}"
            case["source_ids"]["amplitude"][pid][cell_id] = f"src:amp:{pid}:{cell_id}"
            case["source_ids"]["badflag"][pid][cell_id] = f"src:badflag:{pid}:{cell_id}"

        bt = None
        if p.get("bottom_track"):
            btb = [_measured_from(v) for v in p["bottom_track"]["beam_velocity"]]
            btc = [_measured_from(v) for v in p["bottom_track"]["correlation"]]
            bta = [_measured_from(v) for v in p["bottom_track"]["amplitude_db"]]
            for m in btb:
                _norm_velocity(m)
            for m in btc:
                _norm_counts(m)
            for m in bta:
                _norm_db(m)
            bt = {
                "time": float(p["bottom_track"]["time"]),
                "range_m": [_norm_depth_dict(v) for v in p["bottom_track"]["range_m"]],
                "beam_velocity": [m.as_dict() for m in btb],
                "correlation": [m.as_dict() for m in btc],
                "amplitude_db": [m.as_dict() for m in bta],
                "bad_beam_flag": [bool(x) for x in p["bottom_track"]["bad_beam_flag"]],
            }
            case["source_ids"]["bt_beam"][pid] = f"src:bt-beam:{pid}"
            case["source_ids"]["bt_badflag"][pid] = f"src:bt-badflag:{pid}"
            case["source_ids"]["bt_attitude"][pid] = f"src:bt-attitude:{pid}"

        case["pings"].append({"id": pid, "time": float(p["time"]),
                              "cells": cells, "bottom_track": bt})

    for i, s in enumerate(case["attitude_samples"]):
        nearest = _nearest_ping(case, s["time"])
        if nearest:
            case["source_ids"]["attitude"].setdefault(nearest, f"src:attitude:{nearest}")
    return case


def _angle(value, unit):
    m = Measured(raw=value, raw_unit=unit)
    to_rad(m)
    return m


def _norm_depth_dict(obj):
    m = _measured_from(obj)
    _norm_depth(m)
    return m.as_dict()


def _nearest_ping(case, t):
    best, best_dt = None, None
    for p in case["pings"]:
        d = abs(p["time"] - t)
        if best_dt is None or d < best_dt:
            best, best_dt = p["id"], d
    return best
