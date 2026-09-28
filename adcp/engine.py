"""依赖图重算核心：beam -> instrument -> ship -> earth 的逐级推导。

每个节点都带 lineage（内容指纹）：
  * 波束几何（beam_angle / azimuths / sign）
  * 磁偏角版本（declination revision + east_deg）
  * 安装角版本（mount revision + 三个角 + 顺序）
  * 坐标变换顺序（processing.transform_order）
  * 直接来源内容 ID
全部进入指纹；任何一项变化都会产生新的派生 ID。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import frames, quality, units
from .attitude import interp_attitude
from .identity import content_id


def _q(raw, unit, dim, basis=None):
    return units.normalize(raw, unit, dim, basis)


def _velocity_vec(vs: List[Optional[float]], basis: str, stage: str,
                  unit: str = "m/s") -> Dict:
    return {"stage": stage, "basis": basis, "unit": unit,
            "components": {"x": vs[0], "y": vs[1], "z": vs[2]}}


def normalize_snapshot(sources: Dict[Tuple[str, str], Dict]) -> Dict:
    """把分开记录的直接来源规范化为计算快照，原始表示同时保留。"""
    def st(source_type):
        return {k.split(":", 1)[1]: v for (t, k), v in sources.items() if t == source_type}

    geom = st("transducer_geometry")["default"]["payload"]
    proc = st("processing_config")["default"]["payload"]
    decl = st("declination")["default"]["payload"]
    mount = st("mount")["default"]["payload"]
    cells_raw = st("depth_cells")["default"]["payload"]["cells"]

    pings = {}
    for key, rec in sorted(st("ping").items()):
        pings[int(key)] = rec["payload"]
    times = {i: pings[i]["time_s"] for i in pings}

    cell_diag = []
    cells = []
    for idx, c in enumerate(cells_raw):
        qd = _q(c.get("depth"), c.get("depth_unit", "m"), "length")
        qt = _q(c.get("thickness"), c.get("thickness_unit", "m"), "length")
        check = quality.depth_cell_valid({
            "depth_m": qd.value, "thickness_m": qt.value})
        cells.append({"index": idx, "depth": qd.to_dict(), "thickness": qt.to_dict(),
                      "valid": check["valid"], "reasons": check["reasons"]})
        cell_diag.append((idx, check))

    return {"geom": geom, "proc": proc, "decl": decl, "mount": mount,
            "cells": cells, "pings": pings, "times": times}


def _lineage_inputs(snap: Dict, ping: int, cell: int, source_ids: Dict,
                    extra: Optional[Dict] = None) -> Dict:
    line = {
        "rule_version": snap["proc"].get("_rule_version", ""),
        "geom_id": source_ids[("transducer_geometry", "default")],
        "geom": {"beam_angle": snap["geom"]["beam_angle"],
                 "azimuths": snap["geom"]["azimuths"],
                 "beam_velocity_sign": snap["geom"].get("beam_velocity_sign", "away")},
        "declination_id": source_ids[("declination", "default")],
        "declination_rev": snap["decl"]["revision"],
        "east_deg": snap["decl"]["east_deg"],
        "mount_id": source_ids[("mount", "default")],
        "mount_rev": snap["mount"]["revision"],
        "mount": {"heading_alignment": snap["mount"]["heading_alignment_deg"],
                  "pitch": snap["mount"]["pitch_deg"], "roll": snap["mount"]["roll_deg"],
                  "order": snap["mount"].get("order", "hpr")},
        "transform_order": snap["proc"].get("transform_order", "hpr"),
        "ping": ping, "cell": cell,
    }
    if extra:
        line.update(extra)
    return line


def _candidate_ids(geom_line: Dict, bvals: List[float], candidates: List[Dict],
                   four_v) -> Tuple[Optional[str], List[str]]:
    four_id = None
    if four_v is not None:
        four_id = content_id("cand", {"solution": "four_beam", "geom": geom_line,
                                      "beam_velocities": bvals,
                                      "v": [round(float(x), 9) for x in four_v]})
    ids = []
    for c in candidates:
        ids.append(content_id("cand", {"solution": "three_beam", "geom": geom_line,
                                       "beam_velocities": bvals,
                                       "omitted_beam": c["omitted_beam"],
                                       "v": [round(float(x), 9) for x in c["v"]]}))
    return four_id, ids


def derive_node(snap: Dict, source_ids: Dict, ping: int, cell: int) -> Dict:
    """计算单个 (ping, cell) 的完整节点图，返回 node_key -> node dict。"""
    p = snap["pings"][ping]
    geom = snap["geom"]
    proc = snap["proc"]
    H = frames.beam_matrix(geom)

    braw = p["beam_velocity_mps"][cell]
    corr = p["correlation_counts"][cell]
    amp = p["amplitude_counts"][cell]
    flags = p["bad_beam_flags"][cell]

    bq = [_q(v, p.get("velocity_unit", "m/s"), "velocity", frames.FRAME_BEAM)
          for v in braw]
    classes = quality.classify_beams(corr, amp, flags, proc.get("thresholds"))
    cell_info = snap["cells"][cell]

    node_key = f"ping:{ping}:cell:{cell}"
    base_line = _lineage_inputs(snap, ping, cell, source_ids, {
        "beam_velocity_raw": [q.to_dict() for q in bq],
        "correlation": list(corr), "amplitude": list(amp),
        "bad_beam_flags": list(flags),
        "depth_cell": {"index": cell, "depth": cell_info["depth"],
                       "thickness": cell_info["thickness"]},
    })

    quality_result = {"beam_status": classes,
                      "correlation": list(corr), "amplitude": list(amp),
                      "bad_beam_flags": list(flags),
                      "beam_velocity_raw": [q.to_dict() for q in bq]}

    # ---- 阻塞条件：无效 cell / 单位不可转换
    blocked = []
    if not cell_info["valid"]:
        blocked.extend(cell_info["reasons"])
    bad_units = [i for i, q in enumerate(bq) if not q.ok]
    if bad_units:
        blocked.append(f"unconvertible_beam_velocity:{bad_units}")

    instrument = None
    candidates_block = []
    four_v = error_vel = None
    three = []
    if not blocked:
        bvals = np.array([q.value for q in bq], dtype=float)
        good = [i for i, c in enumerate(classes) if c["status"] != "bad"]
        four_v, error_vel = frames.solve_four_beam(H, bvals)
        three = frames.three_beam_candidates(H, bvals, good)
        geom_line = {k: base_line[k] for k in ("geom",)}
        four_id, three_ids = _candidate_ids(base_line["geom"],
                                            [float(x) for x in bvals], three, four_v)
        # suspect 波束使四波束解降级为“多解待裁定”，但四波束值仍然保留
        ambiguous = any(c["status"] == "suspect" for c in classes)
        ambiguous |= abs(error_vel) > proc.get("error_velocity_limit_mps", 0.20)
        for cid, c in zip(three_ids, three):
            c["candidate_id"] = cid
        instrument = {
            "four_beam": {
                "candidate_id": four_id,
                "basis": frames.FRAME_INSTRUMENT, "unit": "m/s",
                "v": [float(x) for x in four_v],
                "error_velocity_mps": error_vel,
            },
            "three_beam": three,
            "ambiguous": ambiguous,
            "good_beams": good,
        }

    status = "ok"
    if blocked:
        status = "blocked"
        instrument = {"blocked_reasons": blocked}
    elif instrument["ambiguous"]:
        status = "multi"

    # ---- 姿态：缺口感知插值，绝不跨缺口
    att = snap["pings"][ping].get("attitude", {})
    hq = _q(att.get("heading"), att.get("heading_unit", "deg"), "angle")
    pq = _q(att.get("pitch"), att.get("pitch_unit", "deg"), "angle")
    rq = _q(att.get("roll"), att.get("roll_unit", "deg"), "angle")
    attitude_ok = all(q.ok for q in (hq, pq, rq))
    interp_diag = att.get("_interp_diag")

    # ---- 逐级向量
    ship = earth = None
    stage_block = list(blocked)
    if not blocked:
        R_is = frames.instrument_to_ship(
            snap["mount"]["heading_alignment_deg"], snap["mount"]["pitch_deg"],
            snap["mount"]["roll_deg"], snap["mount"].get("order", "hpr"))
        v_ship = R_is @ four_v
        ship = _velocity_vec([float(x) for x in v_ship], frames.FRAME_SHIP, "ship",
                             )
        ship["matrix"] = [[round(float(a), 9) for a in row] for row in R_is]
        ship["from_candidate_id"] = instrument["four_beam"]["candidate_id"]
        if not attitude_ok:
            stage_block.append("attitude_unavailable")
        else:
            heading_true = frames.apply_declination(hq.value, snap["decl"]["east_deg"])
            order = proc.get("transform_order", "hpr")
            R_se = frames.ship_to_earth(heading_true, pq.value, rq.value, order)
            v_earth = R_se @ v_ship
            earth = _velocity_vec([float(x) for x in v_earth], frames.FRAME_EARTH, "earth")
            earth["matrix"] = [[round(float(a), 9) for a in row] for row in R_se]
            earth["heading_true_deg"] = heading_true
            earth["heading_magnetic_deg"] = hq.value
            earth["order"] = order

    result = {
        "ping": ping, "cell": cell,
        "time_s": p["time_s"],
        "depth_cell": {"index": cell, "depth": cell_info["depth"],
                       "thickness": cell_info["thickness"],
                       "valid": cell_info["valid"], "reasons": cell_info["reasons"]},
        "quality": quality_result,
        "instrument": instrument,
        "ship": ship,
        "earth": earth,
        "attitude": {"heading": hq.to_dict(), "pitch": pq.to_dict(), "roll": rq.to_dict(),
                     "interp_diag": interp_diag},
        "blocked_reasons": stage_block,
    }
    return {"node_key": node_key, "lineage": base_line, "status": status,
            "result": result}


def derive_bottom(snap: Dict, source_ids: Dict, ping: int) -> Dict:
    """bottom track 是独立来源，单独推导为 instrument/ship/earth。"""
    p = snap["pings"][ping]
    bt = p.get("bottom_track")
    node_key = f"ping:{ping}:bottom"
    if not bt:
        return {"node_key": node_key,
                "lineage": {"ping": ping, "bottom": "absent"},
                "status": "blocked",
                "result": {"ping": ping, "blocked_reasons": ["bottom_track_absent"]}}
    H = frames.beam_matrix(snap["geom"])
    bq = [_q(v, bt.get("velocity_unit", "m/s"), "velocity", frames.FRAME_BEAM)
          for v in bt["beam_velocity_mps"]]
    flags = bt["bad_beam_flags"]
    good = [i for i, f in enumerate(flags) if not f]
    result = {"ping": ping, "beam_velocity_raw": [q.to_dict() for q in bq]}
    blocked = [i for i, q in enumerate(bq) if not q.ok]
    if len(good) < 3 or blocked:
        return {"node_key": node_key, "lineage": {"ping": ping},
                "status": "blocked",
                "result": {**result, "blocked_reasons":
                           [f"bottom_good_beams={len(good)}"] +
                           [f"unconvertible:{blocked}"] if blocked else
                           [f"bottom_good_beams={len(good)}"]}}
    bvals = np.array([q.value for q in bq])
    v, err = frames.solve_four_beam(H, bvals)
    return {"node_key": node_key,
            "lineage": {"ping": ping, "bottom": True,
                        "geom": {"beam_angle": snap["geom"]["beam_angle"],
                                 "azimuths": snap["geom"]["azimuths"]},
                        "mount_rev": snap["mount"]["revision"],
                        "declination_rev": snap["decl"]["revision"],
                        "transform_order": snap["proc"].get("transform_order", "hpr")},
            "status": "ok",
            "result": {**result, "instrument_v": [float(x) for x in v],
                       "error_velocity_mps": err, "good_beams": good}}


def build_all_nodes(snap: Dict, source_ids: Dict) -> List[Dict]:
    nodes = []
    for ping in sorted(snap["pings"]):
        for cell in range(len(snap["cells"])):
            nodes.append(derive_node(snap, source_ids, ping, cell))
        nodes.append(derive_bottom(snap, source_ids, ping))
    return nodes
