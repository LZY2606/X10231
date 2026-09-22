"""Deterministic generator for the beam_bench offline case pack.

The baseline keeps every source class in its own channel:
beam radials, correlation, amplitude, bad-beam flags, heading/pitch/roll
samples, magnetic declination and transducer mounting are recorded
separately.  Raw values carry their original units; normalization happens
only in the derivation layer.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict

N_PINGS = 10
N_CELLS = 6
N_BEAMS = 4
T0 = 1000.0
DT = 1.0
ATT_DT = 0.5
THETA = 20.0
AZ = (45.0, 135.0, 225.0, 315.0)

HERE = os.path.dirname(__file__)


def beam_row(vx, vy, vz, az_deg, theta_deg=THETA):
    a = math.radians(az_deg)
    t = math.radians(theta_deg)
    gx = math.sin(t) * math.cos(a)
    gy = math.sin(t) * math.sin(a)
    gz = -math.cos(t)
    return vx * gx + vy * gy + vz * gz


def qty(value, unit, dim, kind=None, datum=None):
    canonical = {"m": "m", "m/s": "m/s", "deg": "deg",
                 "count": None, "dB": None}.get(unit)
    return {"kind": kind or "quantity", "datum": datum,
            "raw_quantity": True,
            "raw": {"raw": ("%g" % value) if isinstance(value, float)
                    else str(value), "unit": unit,
                    "dim": dim, "value": value, "valid": True,
                    "canonical_unit": canonical}}


def make_baseline() -> dict:
    sources: Dict[str, dict] = {}
    sources["geometry"] = {
        "kind": "config_geometry", "datum": "transducer",
        "raw": {"pattern": "convex_janus", "beam_angle_deg": THETA,
                "azimuths_deg": list(AZ)}}
    sources["mounting"] = {
        "kind": "config_mounting", "datum": "ship",
        "raw": {"heading_offset_deg": 0.0, "tilt_deg": 0.0,
                "recipe": "ALPHA_BETA"}}
    sources["declination"] = {
        "kind": "config_declination", "datum": "WMM",
        "raw": {"value_deg": 7.5, "convention": "east_positive",
                "epoch": "WMM-2025"}}
    sources["rules"] = {
        "kind": "config_rules", "datum": "quality",
        "raw": {"min_correlation": 64.0, "min_amplitude": 30.0,
                "max_error_velocity": 0.5,
                "min_depth_m": 0.5, "max_depth_m": 120.0,
                "gap_seconds": 5.0}}

    n_att = N_PINGS * 2 + 1
    for k in range(n_att):
        t = T0 + k * ATT_DT
        sid = "attitude.s%02d" % k
        sources[sid + ".time"] = {
            "kind": "time", "datum": "unix", "raw_quantity": True,
            "raw": {"raw": "%.3f" % t, "unit": "unix_s", "dim": "time",
                    "value": t, "valid": True, "canonical_unit": "unix_s"}}
        heading = (10.0 + 4.0 * k) % 360.0
        pitch = 2.0 * math.sin(k / 3.0)
        roll = 1.5 * math.cos(k / 4.0)
        for name, val in (("heading", heading), ("pitch", pitch),
                          ("roll", roll)):
            sources["%s.%s" % (sid, name)] = {
                "kind": "att_" + name, "datum": "MRU/deg",
                "raw_quantity": True,
                "raw": {"raw": "%.3f" % val, "unit": "deg", "dim": "angle",
                        "value": val, "valid": True,
                        "canonical_unit": "deg"}}

    for p in range(N_PINGS):
        ping = "ping-%02d" % p
        t = T0 + p * DT
        sources["%s.time" % ping] = {
            "kind": "time", "datum": "unix", "raw_quantity": True,
            "raw": {"raw": "%.3f" % t, "unit": "unix_s", "dim": "time",
                    "value": t, "valid": True, "canonical_unit": "unix_s"}}
        for c in range(N_CELLS):
            scope = "%s.cell.%d" % (ping, c)
            depth = 4.0 + c * 2.0
            vx = 0.30 + 0.02 * c + 0.01 * p
            vy = -0.20 - 0.01 * c
            vz = 0.01 * math.sin(p + c)
            sources["%s.cell.depth" % scope] = qty(depth, "m", "length")
            for b in range(1, N_BEAMS + 1):
                rad = beam_row(vx, vy, vz, AZ[b - 1])
                corr = 180.0
                amp = 90.0
                bad = False
                bsid = "%s.beam.%d" % (scope, b)
                if p == 0 and c == 1 and b == 2:
                    sources[bsid + ".amp"] = {
                        "kind": "beam_amp", "datum": "ADCP/dB",
                        "raw_quantity": True,
                        "raw": {"raw": "N/R", "unit": "dB", "dim": None,
                                "value": None, "valid": False,
                                "diagnostic": "unparseable_value:N/R",
                                "canonical_unit": None}}
                else:
                    sources[bsid + ".amp"] = qty(
                        amp, "dB", None, kind="beam_amp", datum="ADCP/dB")
                sources[bsid + ".radial"] = qty(
                    rad, "m/s", "velocity", kind="beam_radial",
                    datum="ADCP/beam%d" % b)
                sources[bsid + ".corr"] = qty(
                    corr, "count", None, kind="beam_corr",
                    datum="ADCP/count")
                sources[bsid + ".bad"] = {
                    "kind": "beam_bad", "datum": "ADCP/flag", "raw": bad}
        for b in range(1, N_BEAMS + 1):
            rad = beam_row(0.05, -0.02, 0.0, AZ[b - 1])
            bsid = "%s.bt.beam.%d" % (ping, b)
            sources[bsid + ".radial"] = qty(
                rad, "m/s", "velocity", kind="bt_radial",
                datum="ADCP/bt-beam%d" % b)
            sources[bsid + ".corr"] = qty(
                200.0, "count", None, kind="bt_corr", datum="ADCP/count")
            sources[bsid + ".amp"] = qty(
                120.0, "dB", None, kind="bt_amp", datum="ADCP/dB")
            sources[bsid + ".bad"] = {
                "kind": "bt_bad", "datum": "ADCP/flag", "raw": False}

    return {
        "case_id": "beam_bench",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "n_pings": N_PINGS, "n_cells": N_CELLS, "n_beams": N_BEAMS,
        "t0": T0, "dt": DT,
        "sources": sources,
        "notes": ["heading series crosses the 360 degree boundary",
                  "ping-00 cell-1 beam-2 amplitude is unparseable"]}


def write_case_pack():
    baseline = make_baseline()
    path = os.path.join(HERE, "baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1, sort_keys=True)
    return path


if __name__ == "__main__":
    print(write_case_pack())
