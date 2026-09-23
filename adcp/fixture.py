"""仓库内置离线案例包：流束换算台 ADCP 波束速度、姿态旋转与质量层级。

不是通用表格：所有材料是面向 beam->instrument->ship->earth 重放的领域量，
导入时保留原始字符串表示与单位，派生层才规范化。
固定随机种子，结果可重复。
"""
from __future__ import annotations

import math

import numpy as np

from .beam import BeamGeometry

PINGS = [f"ping-{t:02d}" for t in (0, 5, 10, 15, 20, 30, 40, 45, 50, 60)]
TIMES = [int(p.split("-")[1]) for p in PINGS]
CELL_DEPTHS_M = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def _water_profile(t: float) -> list[np.ndarray]:
    """每层真实 earth 速度（北、东、向下），随 ping 缓变。"""
    out = []
    for k, d in enumerate(CELL_DEPTHS_M):
        north = 0.40 - 0.02 * k + 0.03 * math.sin(t / 20.0)
        east = 0.10 + 0.01 * k + 0.02 * math.cos(t / 25.0)
        down = 0.01 * math.sin(t / 30.0)
        out.append(np.array([north, east, down]))
    return out


def _attitude_samples() -> list[dict]:
    # 0..30 每 5s 一个；40..60 每 5s 一个；30->40 是 10s（阈值内），
    # 但 ping-45 两侧样本 40 与 50 间隔 10s 且无样本在 45？实际 45 落在样本上。
    # 为制造“时间缺口”，把 40s 样本放到 42s 前的空档：样本序列在 30 与 42 间断 12s。
    rows = []
    for t, h in [(0, 10.0), (5, 12.0), (10, 14.0), (15, 16.0), (20, 18.0), (30, 22.0),
                 (42, 355.0), (50, 3.0), (60, 8.0)]:
        rows.append({"record_id": f"att-{t:02d}", "time_s": t, "heading_deg": h,
                     "pitch_deg": 1.0 + 0.1 * math.sin(t / 17.0),
                     "roll_deg": -0.8 + 0.1 * math.cos(t / 19.0)})
    return rows


def _earth_to_beams(geom: BeamGeometry, v_earth: np.ndarray, heading_deg: float,
                    pitch_deg: float, roll_deg: float, declination_deg: float,
                    mount: tuple[float, float, float], order: str = "rdi_xyz") -> np.ndarray:
    from .attitude import instrument_to_ship, ship_to_earth
    r_se = ship_to_earth((heading_deg + declination_deg) % 360.0, pitch_deg, roll_deg, order)
    r_is = instrument_to_ship(*mount)
    v_inst = np.linalg.inv(r_is) @ (r_se.T @ v_earth)
    return geom.matrix() @ v_inst


def build_baseline() -> dict:
    rng = np.random.default_rng(20260923)
    geom = BeamGeometry.rdi_convex(beam_angle_deg=20.0)
    declination_deg = -7.0
    mount = (0.5, 0.0, 0.0)
    order = "rdi_xyz"
    att = _attitude_samples()
    att_at = {a["time_s"]: a for a in att}

    def heading_at(t: float) -> float:
        if t in att_at:
            return att_at[t]["heading_deg"]
        return float(np.interp(t, [a["time_s"] for a in att],
                               [a["heading_deg"] for a in att]))

    bottom_depths = {0: 13.0, 5: 13.0, 10: 13.0, 15: 11.0, 20: 13.0, 30: 13.0,
                     40: 13.0, 45: 13.0, 50: 13.0, 60: 13.0}

    records: list[dict] = []
    records.append({"stream_id": "geom-rdi20", "kind": "geom", "scope": "global",
                    "note": "4-beam convex Janus, RDI beam ordering",
                    "payload": {"beam_angle_deg": geom.beam_angle_deg,
                                "azimuths_deg": list(geom.azimuths_deg), "n_beams": 4}})
    records.append({"stream_id": "grid-main", "kind": "cellgrid", "scope": "global",
                    "payload": {"depths_m": CELL_DEPTHS_M, "blanking_m": 1.0, "min_depth_m": 1.5,
                                "gap_threshold_s": 10.0, "corr_threshold_pct": 60.0,
                                "errvel_threshold_mps": 0.05, "transform_order": order,
                                "depth_frame": "transducer_down"}})
    records.append({"stream_id": "decl-2026", "kind": "declination", "scope": "global",
                    "note": "magnetic declination, versioned independently",
                    "payload": {"angle_deg": declination_deg, "datum_in": "magnetic",
                                "datum_out": "true_north"}})
    records.append({"stream_id": "mount-alpha", "kind": "mount", "scope": "global",
                    "payload": {"heading_deg": mount[0], "pitch_deg": mount[1], "roll_deg": mount[2],
                                "convention": "instrument_to_ship"}})

    for a in att:
        records.append({"stream_id": f'stm-{a["record_id"]}', "kind": "attitude", "scope": "global",
                        "payload": {"record_id": a["record_id"], "time_s": a["time_s"],
                                    "time_unit": "s",
                                    "heading_raw": str(a["heading_deg"]), "heading_unit": "deg",
                                    "pitch_raw": f'{a["pitch_deg"]:.4f}', "pitch_unit": "deg",
                                    "roll_raw": f'{a["roll_deg"]:.4f}', "roll_unit": "deg"}})

    for ping_id, t in zip(PINGS, TIMES):
        h = heading_at(t)
        p = 1.0
        r = -0.8
        for k, depth in enumerate(CELL_DEPTHS_M):
            cell_id = f"{ping_id}-cell{int(round(depth*10)):03d}"
            v = _water_profile(t)[k]
            beams = _earth_to_beams(geom, v, h, p, r, declination_deg, mount, order)
            beams = beams + rng.normal(0, 2e-4, 4)
            beams = np.round(beams, 5)
            corr = [88 + (k % 3) * 2, 85, 90 - (k % 2) * 3, 87]
            amp = [132 + k, 128 + k, 140 - k, 130 + k]
            flags = [0, 0, 0, 0]
            raw_b = [f"{x:.5f}" for x in beams]
            # ping-20 最深 cell：beam0 硬失效（坏标志 + 缺测）——单 beam 失效
            if ping_id == "ping-20" and depth == 12.0:
                raw_b[0] = ""
                flags[0] = 1
                corr[0] = 12
            # ping-30 中层 cell：四波束看似齐全但误差速度超阈（三波束多解）
            if ping_id == "ping-30" and depth == 6.0:
                raw_b[1] = f"{float(raw_b[1]) + 0.12:.5f}"
            # ping-05 一层 amplitude 无法解析（保留原始表示，附诊断，不丢数据）
            if ping_id == "ping-05" and depth == 4.0:
                amp[2] = "N/A"
            records.append({"stream_id": f"bv:{cell_id}", "kind": "beam_vel", "scope": ping_id,
                            "payload": {"ping_id": ping_id, "cell_id": cell_id,
                                        "beams": raw_b, "unit": "m/s"}})
            records.append({"stream_id": f"cor:{cell_id}", "kind": "correlation", "scope": ping_id,
                            "payload": {"ping_id": ping_id, "cell_id": cell_id,
                                        "values": [str(x) for x in corr], "unit": "pct"}})
            records.append({"stream_id": f"amp:{cell_id}", "kind": "amplitude", "scope": ping_id,
                            "payload": {"ping_id": ping_id, "cell_id": cell_id,
                                        "values": [str(x) for x in amp], "unit": "count"}})
            records.append({"stream_id": f"bf:{cell_id}", "kind": "badflag", "scope": ping_id,
                            "payload": {"ping_id": ping_id, "cell_id": cell_id, "flags": flags}})

        # bottom track：船对地速度，恒定向后 0.2 m/s（north 系）
        bt_v = np.array([-0.20, 0.0, 0.0])
        bt_beams = np.round(_earth_to_beams(geom, bt_v, h, p, r, declination_deg, mount, order), 5)
        records.append({"stream_id": f"bt:{ping_id}", "kind": "bottom_track", "scope": ping_id,
                        "payload": {"bt_id": f"{ping_id}-bt", "ping_id": ping_id,
                                    "beams": [f"{x:.5f}" for x in bt_beams], "unit": "m/s",
                                    "correlation": ["95", "94", "96", "93"], "correlation_unit": "pct",
                                    "amplitude": ["200", "198", "202", "199"], "amplitude_unit": "count",
                                    "bad_flags": [0, 0, 0, 0],
                                    "range_m": bottom_depths[t]}})
        records.append({"stream_id": f"bd:{ping_id}", "kind": "bottom_depth", "scope": ping_id,
                        "payload": {"ping_id": ping_id, "depth_m": bottom_depths[t], "unit": "m"}})

    return {"records": records, "geometry": geom.canonical(),
            "landmarks": {
                "single_beam_failure": "ping-20 / 12m cell: beam0 flagged+missing -> unique 3-beam solution",
                "three_beam_multiplicity": "ping-30 / 6m cell: error velocity 0.12 m/s over 0.05 threshold -> 4 candidates retained",
                "heading_wrap_360": "attitude 42s=355deg -> 50s=3deg; ping-45 interpolation crosses 360",
                "time_gap": "no contiguous attitude sample at ping-40 (gap 30s->42s = 12s > 10s threshold) -> earth blocked",
                "invalid_depth_cell": "ping-15 / 12m cell is below bottom (bottom=11m) -> excluded below_bottom",
                "unconvertible_amplitude": "ping-05 / 4m cell beam2 amplitude 'N/A' retained with diagnostic",
                "mount_revision": "mutate mount-alpha heading 0.5 -> 1.2 deg",
                "transform_order": "switch grid-main rdi_xyz -> hpr changes every earth fingerprint",
                "declination_revision": "mutate decl-2026 -7.0 -> -6.5 deg",
            }}
