"""Generate the in-repo offline domain case pack.

Stream-desk calibration deployment "stream_desk_v1":
12 pings, 6 depth cells, explicit attitude sample stream with a gap and a
359.4 -> 0.6 degree heading wrap. Files keep original representations, units
and datums; derivation is the only layer that normalizes.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adcp_evidence.geometry import attitude_matrix_frd_to_enu

CASE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                        "cases", "stream_desk_v1")

N_PINGS = 12
N_CELLS = 6
FIRST_T = 1000.0
DT = 2.0
GAP_BETWEEN = 5  # pings index 4 and 5: interval becomes 12 s
CELL_SPACING_M = 1.0
CELL_START_M = 1.5


def ping_time(i: int) -> float:
    t = FIRST_T + i * DT
    if i >= GAP_BETWEEN + 1:
        t += 10.0
    return t


def write(name: str, text: str) -> None:
    with open(os.path.join(CASE_DIR, name), "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    os.makedirs(CASE_DIR, exist_ok=True)
    write("MANIFEST.json", json.dumps({
        "case_id": "stream_desk_v1",
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "beam_velocity_convention": "positive = water toward transducer face",
        "frames": {
            "instrument_xyz": "right handed x=fwd,y=stbd,z=down",
            "ship": "mount rotation then attitude; FRD internally, FSU shown",
            "earth": "ENU; heading 0 -> forward points north"
        },
        "units_on_import": {
            "beam_velocity": "m/s (one row deliberately carries cm/s units)",
            "correlation": "counts 0..255",
            "amplitude": "counts 0..255 (one cell carries an unparseable token)",
            "attitude": "degrees; heading datum magnetic",
            "depth": "meters"
        },
        "scenario_index": {
            "P02": "single beam failure (beam 2 flagged, low correlation)",
            "P04->P05": "time gap; attitude sample across the gap removed",
            "P03-P04": "heading wrap 359.4 -> 0.6 degrees",
            "P06": "three-beam multiple solution (beam 4 weak, not flagged)",
            "P09": "invalid depth cell (below sidelobe-safe depth)",
            "P10": "bottom track evidence",
            "P11": "local-only source correction in replay"
        }
    }, ensure_ascii=False, indent=2))

    write("transducer.json", json.dumps({
        "record_kind": "transducer",
        "version": "transducer-v1",
        "beam_angle_deg": 20.0,
        "azimuths_deg": [45.0, 315.0, 225.0, 135.0],
        "frequency_khz": 600,
        "cell_spacing_m": CELL_SPACING_M,
        "cell_start_m": CELL_START_M,
    }, indent=2))

    write("mount_v1.json", json.dumps({
        "record_kind": "mount",
        "version": "mount-v1",
        "alpha_deg": 0.0,
        "beta_deg": 0.0,
        "gamma_deg": 0.0,
        "note": "as-installed survey; revised in replay",
    }, indent=2))
    write("mount_v2.json", json.dumps({
        "record_kind": "mount",
        "version": "mount-v2",
        "alpha_deg": 0.0,
        "beta_deg": 0.35,
        "gamma_deg": -1.10,
        "note": "post-deployment alignment revision",
    }, indent=2))

    write("declination_v1.json", json.dumps({
        "record_kind": "declination",
        "version": "declination-v1",
        "declination_deg": 7.2,
        "model": "IGRF-13 2025.0 local sheet",
    }, indent=2))
    write("declination_v2.json", json.dumps({
        "record_kind": "declination",
        "version": "declination-v2",
        "declination_deg": 7.4,
        "model": "IGRF-14 2026.0 local sheet",
    }, indent=2))

    write("rules_v1.json", json.dumps({
        "record_kind": "rules",
        "version": "rules-v1",
        "correlation_min": 64,
        "amplitude_min": 40,
        "attitude_max_gap_s": 5.0,
        "sidelobe_margin_m": 0.5,
        "transform_order": "v1_separated",
    }, indent=2))

    # ---- beam velocities: synthetic coherent flow ----
    headings_deg = [
        358.0, 358.8, 359.4, 0.6, 1.2,
        8.0, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8
    ]
    # Attitude samples: the sample bracketing the P04->P05 gap is removed.
    attitude_rows = ["sample_id,t_s,heading_deg,pitch_deg,roll_deg"]
    for i in range(N_PINGS):
        if i == GAP_BETWEEN + 1:
            continue
        attitude_rows.append(
            f"A{i:02d},{ping_time(i):.1f},{headings_deg[i]:.1f},"
            f"{(-0.4 + 0.1 * i):.2f},{(0.2 * math.sin(i)):.2f}"
        )
    write("attitude.csv", "\n".join(attitude_rows) + "\n")

    vel_header = (
        "ping_id,t_s,cell,cell_center_m,configured,"
        "bv1[m/s],bv2[m/s],bv3[cm/s],bv4[m/s]"
    )
    corr_header = "ping_id,cell,c1,c2,c3,c4"
    amp_header = "ping_id,cell,a1,a2,a3,a4"
    flag_header = "ping_id,cell,f1,f2,f3,f4"
    vel_rows = [vel_header]
    corr_rows = [corr_header]
    amp_rows = [amp_header]
    flag_rows = [flag_header]
    bt_rows = ["ping_id,depth_m[m],bt1[m/s],bt2[m/s],bt3[m/s],bt4[m/s]"]

    theta = math.radians(20.0)
    az = [math.radians(a) for a in (45, 315, 225, 135)]
    for i in range(N_PINGS):
        pid = f"P{i+1:02d}"
        t = ping_time(i)
        h = math.radians(headings_deg[i] + 7.2)
        p = math.radians(-0.4 + 0.1 * i)
        r = math.radians(0.2 * math.sin(i))
        # FRD basis at this attitude (same routine used by the engine).
        basis = attitude_matrix_frd_to_enu(
            math.degrees(h), math.degrees(p), math.degrees(r)
        )

        for c in range(N_CELLS):
            center = CELL_START_M + c * CELL_SPACING_M
            # Earth ENU current varying slightly with depth: east 0.45,
            # north -0.2, up weak. Convert to FRD velocity:
            ve, vn, vu = 0.45 - 0.01 * c, -0.20 + 0.005 * c, 0.0
            earth_vec = (ve, vn, vu)
            frd_vec = basis.T @ np.asarray(earth_vec)
            beams = []
            for b in range(4):
                look = (
                    math.sin(theta) * math.cos(az[b]),
                    math.sin(theta) * math.sin(az[b]),
                    math.cos(theta),
                )
                along = float(np.dot(look, frd_vec))
                beams.append(-along + (0.002 * (b + 1) * (c - 2)))
            configured = "1"
            if i == 8 and c == 5:  # P09 deepest cell marked unconfigured
                configured = "0"
            vel_rows.append(
                f"{pid},{t:.1f},{c},{center:.2f},{configured},"
                f"{beams[0]:.4f},{beams[1]:.4f},{100*beams[2]:.2f},"
                f"{beams[3]:.4f}"
            )
            corr = [190, 185, 188, 182]
            amp = [120, 118, 121, 116]
            flags = [0, 0, 0, 0]
            if i == 1 and c in (2, 3):  # P02 single beam failure
                corr[1] = 28
                amp[1] = 18
                flags[1] = 1
            if i == 5 and c == 3:  # P06 beam 4 weak -> ambiguity
                corr[3] = 61
                amp[3] = 43
            if i == 8 and c == 4:  # unparseable amplitude kept on an
                amp[2] = None       # excluded-depth cell so it is diagnosed
            corr_rows.append(
                f"{pid},{c},{corr[0]},{corr[1]},{corr[2]},{corr[3]}"
            )
            amp_tokens = ["NA" if v is None else str(v) for v in amp]
            if i == 8 and c == 4:
                amp_tokens[2] = "??"
            amp_rows.append(
                f"{pid},{c},{amp_tokens[0]},{amp_tokens[1]},"
                f"{amp_tokens[2]},{amp_tokens[3]}"
            )
            flag_rows.append(f"{pid},{c},{flags[0]},{flags[1]},{flags[2]},"
                             f"{flags[3]}")

        bottom_depth = 6.4 if i != 8 else 4.3
        # Bottom track velocities: bed is stationary, expect ~0 earth velocity.
        bt = [-0.02 + 0.005 * i, 0.01 - 0.002 * i,
              -0.02 + 0.005 * i, 0.01 - 0.002 * i]
        bt_rows.append(
            f"{pid},{bottom_depth:.2f},{bt[0]:.3f},{bt[1]:.3f},"
            f"{bt[2]:.3f},{bt[3]:.3f}"
        )

    write("beam_velocities.csv", "\n".join(vel_rows) + "\n")
    write("correlations.csv", "\n".join(corr_rows) + "\n")
    write("amplitudes.csv", "\n".join(amp_rows) + "\n")
    write("bad_beam_flags.csv", "\n".join(flag_rows) + "\n")
    write("bottom_track.csv", "\n".join(bt_rows) + "\n")

    write("scenarios.json", json.dumps({
        "replay_steps": [
            {"id": "S0", "label": "baseline import", "kind": "import"},
            {"id": "S1", "label": "single beam failure evidence (fixture)",
             "kind": "inspect", "focus": "P02"},
            {"id": "S2", "label": "heading wrap across 360",
             "kind": "inspect", "focus": ["P03", "P04"]},
            {"id": "S3", "label": "time gap: no interpolation across it",
             "kind": "inspect", "focus": ["P04", "P05"]},
            {"id": "S4", "label": "three-beam multi-solution retained",
             "kind": "inspect", "focus": "P06"},
            {"id": "S5", "label": "mount angle revision v1 -> v2",
             "kind": "source_revision", "doc": "doc:mount",
             "file": "mount_v2.json"},
            {"id": "S6", "label": "wrong transform order branch",
             "kind": "branch", "order": "v2_fused_yaw"},
            {"id": "S7", "label": "invalid depth cell evidence",
             "kind": "inspect", "focus": "P09"},
            {"id": "S8", "label": "declination revision v1 -> v2",
             "kind": "source_revision", "doc": "doc:declination",
             "file": "declination_v2.json"},
            {"id": "S9", "label": "local-only source correction (P11 beam 1)",
             "kind": "source_correction", "focus": "P11"},
            {"id": "S10", "label": "conflicting human decisions",
             "kind": "conflict"},
        ]
    }, indent=2))
    print("wrote", CASE_DIR)


if __name__ == "__main__":
    main()
