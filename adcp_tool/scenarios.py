"""Offline domain case pack:
"流束换算台：ADCP 波束速度、姿态旋转与质量层级".

This is NOT a generic spreadsheet. It synthesises a physically consistent
RDI-style 4-beam ADCP deployment (water profile + vessel motion + bottom
track), records every input stream separately with its original units, and
scripts the counterexample situations the reviewer must be able to replay:

single beam failure, heading crossing 360, time gap, three-beam
multi-solution, mounting-angle revision, transform order change, invalid
depth cell, plus a crash, a legacy migration and a concurrent save.

Replay emits machine-readable judgment events; both adopted and non-adopted
candidates/decisions are retained.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import DEFAULT_JANUS

N_PINGS = 24
N_CELLS = 6
SEED = 20260928


def _rot(heading, pitch, roll):
    c, s = math.cos, math.sin
    h, p, r = map(math.radians, (heading, pitch, roll))
    Rz = np.array([[c(h), s(h), 0], [-s(h), c(h), 0], [0, 0, 1]])
    Ry = np.array([[c(p), 0, -s(p)], [0, 1, 0], [s(p), 0, c(p)]])
    Rx = np.array([[1, 0, 0], [0, c(r), s(r)], [0, -s(r), c(r)]])
    return Rz @ Ry @ Rx


def build_dataset() -> dict:
    rng = np.random.default_rng(SEED)
    geom = DEFAULT_JANUS
    G = geom.radial_matrix()
    M = geom.matrix4()
    decl = 6.5  # magnetic declination deg (east positive), WMM-style version

    pings = list(range(N_PINGS))
    # time base with a deliberate gap: ping 10 -> 11 spans 9 s (gap > 2.5)
    times, t = [], 0.0
    for i in pings:
        times.append(round(t, 3))
        t += 9.0 if i == 10 else 1.0

    headings, pitches, rolls = [], [], []
    for i in pings:
        # heading rises through the 359/0 branch cut around ping 17-18
        headings.append((346.0 + 1.6 * i) % 360.0)
        pitches.append(2.0 * math.sin(i / 4.0))
        rolls.append(-1.5 * math.cos(i / 5.0))

    # vessel earth-frame motion
    boat = np.array([[0.35 * math.sin(i / 6.0),
                      0.18 + 0.06 * math.cos(i / 3.0),
                      0.02 * math.sin(i / 8.0)] for i in pings])

    def water_true(i, j):
        # depth varying east/north current (sheared profile)
        depth = 3.0 + j * 2.0
        return np.array([0.55 - 0.03 * j + 0.04 * math.sin(i / 5.0 + j),
                         0.20 + 0.025 * j,
                         -0.03 + 0.01 * math.cos(i / 7.0 + j / 2.0)])

    streams = {k: [] for k in
               ("beam_velocity", "correlation", "amplitude", "bad_beam_flag",
                "heading", "pitch", "roll", "cell_config", "bottom_track")}

    def add(stream, **kw):
        streams[stream].append(kw)

    for j in range(N_CELLS):
        depth_m = 3.0 + j * 2.0
        add("cell_config", seq=j, cell=j, depth_m=depth_m,
            min_valid_depth_m=2.5, blank_m=1.0)

    for i in pings:
        pid = f"p{i+1:02d}"
        Re = _rot(headings[i], pitches[i], rolls[i])
        Rinvs = Re.T
        for j in range(N_CELLS):
            v_water_rel = water_true(i, j) - boat[i]
            v_inst = Rinvs @ v_water_rel
            beam = G @ v_inst
            beam += rng.normal(0, 0.004, size=4)  # mm/s-level noise
            for k, b in enumerate(beam):
                add("beam_velocity", ping=i, cell=j, beam=k,
                    t=times[i], raw=f"{b:.4f}", unit="m/s",
                    basis="beam-radial")
            base_corr = 150 - 6 * j + int(rng.integers(-4, 5))
            base_amp = 130 - 9 * j + int(rng.integers(-5, 6))
            for k in range(4):
                add("correlation", ping=i, cell=j, beam=k,
                    value=int(np.clip(base_corr + int(rng.integers(-3, 4)),
                                      40, 200)))
                add("amplitude", ping=i, cell=j, beam=k,
                    value=int(np.clip(base_amp + int(rng.integers(-3, 4)),
                                      20, 200)))
                add("bad_beam_flag", ping=i, cell=j, beam=k, value=0)

        # bottom track: beam radial of boat motion in instrument frame
        vbt_inst = Rinvs @ boat[i]
        bt = G @ vbt_inst
        for k, b in enumerate(bt):
            add("bottom_track", ping=i, beam=k, t=times[i],
                raw=f"{b:.4f}", unit="m/s", kind="beam_velocity")
        for k in range(4):
            add("bottom_track", ping=i, beam=k, t=times[i],
                value=150 + int(rng.integers(-4, 4)),
                kind="correlation")
            add("bottom_track", ping=i, beam=k, t=times[i],
                value=140 + int(rng.integers(-4, 4)),
                kind="amplitude")
            add("bottom_track", ping=i, beam=k, t=times[i], value=0,
                kind="bad_beam_flag")

    # attitude recorded separately (original representation kept)
    for i in pings:
        add("heading", ping=i, t=times[i], raw=f"{headings[i]:.3f}",
            unit="deg", basis="magnetic")
        add("pitch", ping=i, t=times[i], raw=f"{pitches[i]:.3f}", unit="deg")
        add("roll", ping=i, t=times[i], raw=f"{rolls[i]:.3f}", unit="deg")

    return {
        "dataset_id": "adcp-janus-demo",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "n_pings": N_PINGS, "n_cells": N_CELLS,
        "geometry": {
            "id": DEFAULT_JANUS.id, "beam_angle_deg": 20.0,
            "azimuths_deg": list(DEFAULT_JANUS.azimuths_deg),
            "convention": DEFAULT_JANUS.convention,
            "coord_frame": DEFAULT_JANUS.coord_frame},
        "declination": {"id": "wmm-2020", "declination_deg": decl,
                        "unit": "deg", "basis": "true-north"},
        "mount": {"id": "mount-v1", "x_deg": 0.0, "y_deg": 0.0,
                  "heading_alignment_deg": 0.0},
        "quality_policy": {"id": "qpolicy-v1", "correlation_floor": 90,
                           "correlation_hard": 64, "amplitude_min": 40,
                           "amplitude_bad": 20},
        "transform": {"tilt_order": "roll-pitch", "apply_declination": True},
        "max_attitude_gap_s": 2.5,
        "streams": streams,
    }


def script_events() -> list[dict]:
    """Ordered replay script. Every step carries expectation metadata."""
    E = []

    def ev(seq, kind, payload, check=None):
        e = {"seq": seq, "kind": kind, "payload": payload}
        if check:
            e["check"] = check
        E.append(e)

    ev(1, "ingest", {"source": "fixture", "name": "adcp-janus-demo"},
       {"id": "baseline-separate-streams",
        "expect": "all input streams recorded separately with units/basis"})

    # 1) single beam failure: hard flag on beam 2 at ping 6, deep cell 3
    ev(2, "source_change",
       {"stream": "bad_beam_flag", "ping": 5, "cell": 2, "beam": 1,
        "value": 1, "reason": "single-beam-failure"},
       {"id": "single-beam-failure-local",
        "expect": "ping6 cell3 drops to 3-beam; neighbours stay 4-beam",
        "assert": {"scope_ping": 5, "scope_cell": 2,
                   "mode": "3-beam", "affected_only": True}})

    # 2) heading crosses 360 around ping 17-18 (already in baseline); check it
    ev(3, "check", {"id": "heading-crosses-360", "ping": 17,
                    "expect": "interpolated heading travels short way 359->1"})

    # 3) attitude time gap: delete attitude sample at ping 11 (t=11)
    ev(4, "source_gap",
       {"stream": "attitude", "ping": 11, "reason": "missing-attitude-sample"},
       {"id": "no-interp-across-gap",
        "expect": "attitude at gap not interpolated; instrument stays valid",
        "assert": {"ping": 11, "attitude_status": "gap",
                   "instrument_kept": True, "ship_earth": None}})

    # 4) three-beam multi-solution: make beam 2 marginal (not bad) at ping 14 c2
    ev(5, "source_change",
       {"stream": "correlation", "ping": 13, "cell": 2, "beam": 1,
        "value": 80, "reason": "marginal-beam-multi-solution"},
       {"id": "three-beam-multiple-solutions",
        "expect": "4-beam solution plus 3-beam candidates retained, not merged",
        "assert": {"ping": 13, "cell": 2, "min_candidates": 4,
                   "distinct_modes": ["4-beam", "3-beam"]}})

    # 5) mounting-angle revision (heading alignment) from ping 12 on
    ev(6, "source_change",
       {"stream": "mount", "id": "mount-v2", "x_deg": 0.5, "y_deg": -0.5,
        "heading_alignment_deg": 1.5, "valid_from_ping": 11,
        "reason": "installation-survey-revision"},
       {"id": "mount-revision-fingerprint",
        "expect": "earth velocities after ping12 change; instrument do not",
        "assert": {"instrument_unchanged": True, "fingerprint_changes": True}})

    # 6) coordinate transform order variant on an experiment branch
    ev(7, "branch", {"name": "xorder-pitch-roll",
                     "tilt_order": "pitch-roll"},
       {"id": "transform-order-in-fingerprint",
        "expect": "different order -> different deriv fingerprints/ids"})

    # 7) invalid depth cell: invalidate shallow cell 1 at ping 3
    ev(8, "source_change",
       {"stream": "cell_invalid", "ping": 2, "cell": 0,
        "reason": "side-lobe-invalid-depth-cell"},
       {"id": "invalid-depth-cell-excluded",
        "expect": "cell marked excluded with reason, evidence retained",
        "assert": {"ping": 2, "cell": 0, "status": "cell-invalid"}})

    # 8) conflicting human decisions on two branches
    ev(9, "decision",
       {"branch": "main", "ping": 13, "cell": 2,
        "action": "select-candidate", "candidate": "b1234",
        "label": "adopt-4beam"},
       {"id": "conflicting-human-decisions-a"})
    ev(10, "decision",
       {"branch": "xorder-pitch-roll", "ping": 13, "cell": 2,
        "action": "select-candidate", "candidate": "b124",
        "label": "adopt-3beam-b124"},
       {"id": "conflicting-human-decisions-b",
        "expect": "merging raises a resolvable conflict, nothing overwritten"})

    # 9) un-adopted candidates must still be readable
    ev(11, "check", {"id": "rejected-candidates-retained",
                     "expect": "non-selected 3-beam candidates remain present"})

    # 10) provenance triad
    ev(12, "check", {"id": "derivation-fingerprint-triple",
                     "expect": "order, beam geometry, declination version in fp"})
    return E


def write_case_pack(out_dir: str = "fixtures") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset()
    (out / "adcp_case.json").write_text(
        json.dumps(ds, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "replay_script.json").write_text(
        json.dumps(script_events(), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return {"dataset": str(out / "adcp_case.json"),
            "script": str(out / "replay_script.json")}


if __name__ == "__main__":
    paths = write_case_pack()
    print(json.dumps(paths, ensure_ascii=False, indent=2))
