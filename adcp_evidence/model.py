"""Typed source records. Each material kind is recorded separately.

All imported cells go through :func:`units.parse_quantity`, so raw text, unit
and datum survive into storage and diagnostics attach to the original cell.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from . import units


def q(raw: Any, kind: str, unit: str, datum: Optional[str] = None,
      bounds: Optional[tuple[float, float]] = None) -> dict:
    return units.parse_quantity(raw, kind, unit, datum, bounds).to_dict()


def _header_unit(header: str, default: str) -> str:
    if "[" in header and header.endswith("]"):
        return header[header.index("[") + 1 : -1]
    return default


def _base(header: str) -> str:
    return header.split("[", 1)[0]


def _read_raw_header(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {_base(h): h for h in next(csv.reader(handle))}


@dataclass
class CaseMaterials:
    transducer: dict
    mount: dict
    declination: dict
    rules: dict
    pings: dict[str, dict] = field(default_factory=dict)
    attitude: dict[str, dict] = field(default_factory=dict)

    def record_items(self) -> list[tuple[str, dict]]:
        items = [
            ("doc:transducer", self.transducer),
            ("doc:mount", self.mount),
            ("doc:declination", self.declination),
            ("doc:rules", self.rules),
        ]
        items += sorted(self.pings.items(), key=lambda kv: kv[1]["t_s"])
        items += sorted(self.attitude.items(), key=lambda kv: kv[1]["t_s"])
        return items


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {_base(key): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def load_case_pack(case_dir: str,
                   mount_file: str = "mount_v1.json",
                   decl_file: str = "declination_v1.json",
                   rules_file: str = "rules_v1.json") -> CaseMaterials:
    with open(os.path.join(case_dir, "transducer.json"), encoding="utf-8") as f:
        transducer = json.load(f)
    with open(os.path.join(case_dir, mount_file), encoding="utf-8") as f:
        mount = json.load(f)
    with open(os.path.join(case_dir, decl_file), encoding="utf-8") as f:
        declination = json.load(f)
    with open(os.path.join(case_dir, rules_file), encoding="utf-8") as f:
        rules = json.load(f)

    vel_rows = _read_csv(os.path.join(case_dir, "beam_velocities.csv"))
    corr_rows = _read_csv(os.path.join(case_dir, "correlations.csv"))
    amp_rows = _read_csv(os.path.join(case_dir, "amplitudes.csv"))
    flag_rows = _read_csv(os.path.join(case_dir, "bad_beam_flags.csv"))
    bt_rows = _read_csv(os.path.join(case_dir, "bottom_track.csv"))
    # Per-column units survive on the imported quantities (one beam column is
    # authored in cm/s in the fixture to prove normalization happens only in
    # the derivation layer).
    raw_vel_header = _read_raw_header(os.path.join(case_dir, "beam_velocities.csv"))
    vel_units = {
        f"bv{i}": _header_unit(raw_vel_header[f"bv{i}"], "m/s")
        for i in range(1, 5)
    }

    corr_map = {(r["ping_id"], int(r["cell"])): r for r in corr_rows}
    amp_map = {(r["ping_id"], int(r["cell"])): r for r in amp_rows}
    flag_map = {(r["ping_id"], int(r["cell"])): r for r in flag_rows}
    pings: dict[str, dict] = {}
    for row in vel_rows:
        pid = row["ping_id"]
        cell_idx = int(row["cell"])
        ping = pings.setdefault(
            pid, {"record_kind": "ping", "ping_id": pid, "t_s": float(row["t_s"]),
                  "cells": {}}
        )
        c = corr_map[(pid, cell_idx)]
        a = amp_map[(pid, cell_idx)]
        fl = flag_map[(pid, cell_idx)]
        ping["cells"][str(cell_idx)] = {
            "index": cell_idx,
            "center_m": q(row.get("cell_center_m", ""), "depth", "m",
                          bounds=(0.0, 10000.0)),
            "configured": row.get("configured", "1").strip() not in ("0", "false"),
            "beam_velocity": [
                q(row[f"bv{i}"], "velocity", vel_units[f"bv{i}"])
                for i in range(1, 5)
            ],
            "correlation": [
                q(c[f"c{i}"], "correlation", "count",
                  bounds=(0.0, 255.0)) for i in range(1, 5)
            ],
            "amplitude": [
                q(a[f"a{i}"], "amplitude", "count",
                  bounds=(0.0, 255.0)) for i in range(1, 5)
            ],
            "bad_flag": [
                str(fl[f"f{i}"]).strip() == "1" for i in range(1, 5)
            ],
        }

    for row in bt_rows:
        pid = row["ping_id"]
        pings[pid]["bottom_track"] = {
            "depth_m": q(row["depth_m"], "depth", "m",
                         bounds=(0.0, 10000.0)),
            "beam_velocity": [
                q(row[f"bt{i}"], "velocity", "m/s") for i in range(1, 5)
            ],
        }
    for ping in pings.values():
        ping["cells"] = [ping["cells"][k] for k in sorted(ping["cells"], key=int)]

    att_rows = _read_csv(os.path.join(case_dir, "attitude.csv"))
    attitude: dict[str, dict] = {}
    for row in att_rows:
        attitude[row["sample_id"]] = {
            "record_kind": "attitude",
            "sample_id": row["sample_id"],
            "t_s": float(row["t_s"]),
            "heading": q(row["heading_deg"], "angle", "deg",
                         datum=units.MAGNETIC_HEADING, bounds=(0.0, 360.0)),
            "pitch": q(row["pitch_deg"], "angle", "deg", bounds=(-90.0, 90.0)),
            "roll": q(row["roll_deg"], "angle", "deg", bounds=(-180.0, 180.0)),
        }

    return CaseMaterials(transducer, mount, declination, rules, pings, attitude)
