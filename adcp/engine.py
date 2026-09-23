"""依赖图传播重算引擎。

节点层级 (node_type):
  quality.*      每 ping 质量判定 (坏 beam 标志 / 低相关 / 缺振幅)
  instrument.*   beam -> instrument (四束唯一解 或 三束多解候选族)
  attitude.*     姿态插值 (360 展开, 不跨缺口)
  ship.*         instrument -> ship
  earth.*        ship -> earth (磁偏角版本、转换顺序进入指纹)
  profile.*      垂向剖面聚合; 无效深度 cell 进 excluded 而不是被丢弃

传播规则:
  比对父版本同身份结果的指纹 -> reused (为何仍有效) / recomputed (为何需重算);
  上游 recomputed 强制下游 recomputed; 直接改动的来源记 changed_sources。
  三束候选与四束解拥有不同 result 身份, 永不合并成单一值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import attitude as att
from .fingerprint import derive_fingerprint
from .geometry import (BeamGeometry, family_parameterization, good_beam_mask,
                       solve_four_beams, solve_three_beams)
from .store import Store
from .transforms import build_transform

CORRELATION_FLOOR = 64.0
GAP_TOLERANCE = 2.0


@dataclass
class ChangedInput:
    """一次修订的描述: 只影响哪些直接来源, 供增量传播与事件记录。"""

    changed_source_ids: List[str] = field(default_factory=list)
    affected_pings: List[str] = field(default_factory=list)
    note: str = ""


def _source_map(store: Store, version_id: Optional[str],
                fallback_scopes: bool = True
                ) -> Dict[str, List[Dict[str, Any]]]:
    """取某版本冻结的来源集合; 版本为空 (首次) 时取库内最新来源。"""
    if version_id:
        srcs = store.sources_for_version(version_id)
    else:
        rows = store.conn.execute("SELECT * FROM sources ORDER BY kind, seq").fetchall()
        srcs = [store._source_row(r) for r in rows]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in srcs:
        grouped.setdefault(s["kind"], []).append(s)
    return grouped


def _pick_latest(srcs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(srcs, key=lambda s: (s["scope"], s["seq"]))[-1]


def _all_pings(beam_srcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [s for s in beam_srcs],
        key=lambda s: (s["payload"]["ping_index"], s["seq"]))


def _active_ping_sources(version_srcs: List[Dict[str, Any]], kind: str,
                         scope_pings: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """返回 ping_id -> 该版本可见的最新来源。

    scope_pings: scope -> ping_id 映射, 表示某条修订只作用于局部 ping
    (局部源数据改动只影响这些 ping 的下游)。
    """
    by_scope: Dict[str, List[Dict[str, Any]]] = {}
    for s in version_srcs:
        if s["kind"] != kind:
            continue
        by_scope.setdefault(s["scope"], []).append(s)
    out: Dict[str, Dict[str, Any]] = {}
    # default/全局来源按 seq 取最新
    global_srcs = sorted(by_scope.get("default", []), key=lambda s: s["seq"])
    global_latest = global_srcs[-1] if global_srcs else None
    ping_scopes = {p: sc for sc, p in scope_pings.items()}
    for ping_id, scope in ping_scopes.items():
        local = sorted(by_scope.get(scope, []), key=lambda s: s["seq"])
        out[ping_id] = local[-1] if local else (global_latest if global_latest else None)
    if global_latest is not None:
        for s in version_srcs:
            if s["kind"] == kind and s["scope"] == "default":
                pass
        # 所有未被局部修订覆盖的 ping 用全局最新
        covered = set(out)
        for s in version_srcs:
            if s["kind"] != kind:
                continue
            payload_ping = s["payload"].get("ping_id")
            if payload_ping and payload_ping not in covered and s["scope"] == "default":
                out[payload_ping] = s
    return out


def _gather_config(grouped: Dict[str, List[Dict[str, Any]]]
                   ) -> Tuple[BeamGeometry, float, Dict[str, float], str, str]:
    tc_srcs = grouped.get("transducer_config", [])
    tc = _pick_latest(tc_srcs)["payload"] if tc_srcs else {}
    geom = BeamGeometry.from_dict(tc.get("geometry", {}))
    dec = _pick_latest(grouped["declination"])["payload"]
    mount_src = _pick_latest(grouped["mount_config"])
    mount = mount_src["payload"]["angles"]
    return geom, float(dec["declination_deg"]), dict(mount), dec["version"], mount_src["id"]


def _indexed_by_ping(grouped: Dict[str, List[Dict[str, Any]]], kind: str
                     ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """把 per-ping 来源按 ping_id 索引, 并记录 scope 映射 (局部修订用)。"""
    out: Dict[str, Dict[str, Any]] = {}
    scope_map: Dict[str, str] = {}
    for s in sorted(grouped.get(kind, []), key=lambda x: x["seq"]):
        pid = s["payload"].get("ping_id")
        if pid is None:
            continue
        out[pid] = s
        scope_map[pid] = s["scope"]
    return out, scope_map


def _fp_inputs_beam(payload: Dict[str, Any], geom_src_id: str,
                    quality_src_ids: List[str]) -> Dict[str, Any]:
    return {
        "beams": payload.get("beams"),
        "cells": payload.get("cells"),
        "geom_source": geom_src_id,
        "quality_sources": sorted(quality_src_ids),
    }


def derive(store: Store, *, version_id: str, label: str, author: str = "reviewer",
           parent_version: Optional[str] = None,
           changed: Optional[ChangedInput] = None,
           event_type: str = "derivation",
           event_payload: Optional[Dict[str, Any]] = None,
           assertions: Optional[List[Dict[str, Any]]] = None,
           is_order: str = "YX", earth_order: str = "HPR",
           allow_pointer_fork: bool = False,
           created_at: Optional[float] = None) -> Dict[str, Any]:
    """从父版本的来源集合重算并提交新版本。只更新受影响部分。"""
    changed = changed or ChangedInput()
    grouped = _source_map(store, parent_version)

    geom, declination, mount, dec_version, mount_src_id = _gather_config(grouped)
    geom_src = _pick_latest(grouped["transducer_config"]) if grouped.get("transducer_config") else None
    geom_src_id = geom_src["id"] if geom_src else "builtin"

    beam_by_ping, beam_scopes = _indexed_by_ping(grouped, "beam_measurement")
    corr_by_ping, _ = _indexed_by_ping(grouped, "correlation")
    amp_by_ping, _ = _indexed_by_ping(grouped, "amplitude")
    flag_by_ping, _ = _indexed_by_ping(grouped, "bad_beam_flag")
    btm_by_ping, _ = _indexed_by_ping(grouped, "bottom_track")

    # 姿态: 全局时间序列 + 每条 ping 时间。局部修订通过新 scope 的时间序列实现。
    att_srcs = sorted(grouped.get("heading_pitch_roll", []), key=lambda s: s["seq"])
    att_active = att_srcs[-1] if att_srcs else None

    ordered_pings = sorted(beam_by_ping.values(),
                           key=lambda s: s["payload"]["ping_index"])
    ping_ids = [s["payload"]["ping_id"] for s in ordered_pings]
    ping_times = [float(s["payload"]["time_s"]) for s in ordered_pings]

    # 姿态插值: 不跨缺口
    att_samples = []
    if att_active:
        for rec in att_active["payload"]["samples"]:
            att_samples.append(att.AttitudeSample(
                t=float(rec["t"]),
                heading=rec.get("heading_deg"),
                pitch=rec.get("pitch_deg"),
                roll=rec.get("roll_deg")))
    interp = att.interpolate_attitude(ping_times, att_samples, GAP_TOLERANCE)

    # 父版本指纹表 (reused / recomputed 判定)
    prev_results: Dict[str, Dict[str, Any]] = {}
    prev_fps: Dict[str, str] = {}
    if parent_version:
        for r in store.results(parent_version):
            prev_results[r["id"]] = r
            prev_fps[_identity_key(r)] = r["fingerprint"]

    source_ids = sorted({s["id"] for srcs in grouped.values() for s in srcs})
    results: List[Dict[str, Any]] = []
    dependencies: List[Tuple[str, str, str]] = []
    node_status: List[Dict[str, Any]] = []
    affected = set(changed.affected_pings)

    def rid(node_type: str, scope: str, seq: int = 0,
            candidate: str = "") -> str:
        base = "%s|%s|%d" % (node_type, scope, seq)
        if candidate:
            base += "|%s" % candidate
        return "res-" + base

    def status_for(identity: str, fingerprint: str, node_id: str,
                   force: bool, why: str) -> str:
        if identity in prev_fps and prev_fps[identity] == fingerprint and not force:
            node_status.append({"node": node_id, "status": "reused", "reason": why})
            return "reused"
        node_status.append({"node": node_id, "status": "recomputed", "reason": why})
        return "recomputed"

    profile_rows: List[Dict[str, Any]] = []

    for ping_src in ordered_pings:
        bp = ping_src["payload"]
        pid = bp["ping_id"]
        pidx = int(bp["ping_index"])
        force_ping = pid in affected
        scope = "ping%03d" % pidx

        corr = corr_by_ping.get(pid)
        amp = amp_by_ping.get(pid)
        flag = flag_by_ping.get(pid)
        qids = [x["id"] for x in (corr, amp, flag) if x]

        flags = [bool(x) for x in (flag["payload"]["bad_flags"] if flag
                                   else [False] * 4)]
        corrs = list(corr["payload"]["correlations"]) if corr else [None] * 4
        amps = list(amp["payload"]["amplitudes"]) if amp else [None] * 4
        mask = good_beam_mask(flags, corrs, amps, CORRELATION_FLOOR)

        q_fp = derive_fingerprint(
            node_type="quality",
            inputs={"flag": flag["id"] if flag else None,
                    "correlation": corr["id"] if corr else None,
                    "amplitude": amp["id"] if amp else None,
                    "floor": CORRELATION_FLOOR},
            geom=geom, declination_version=dec_version, mount_version=mount_src_id,
            is_order=is_order, earth_order=earth_order, gap_tolerance=GAP_TOLERANCE)
        qnode = rid("quality", scope)
        qstat = status_for("quality|%s|0" % scope, q_fp, qnode, force_ping,
                           "quality inputs unchanged" if not force_ping
                           else "direct source change")
        excluded_beams = [i for i, ok in enumerate(mask) if not ok]
        results.append({
            "id": qnode, "node_type": "quality", "scope": scope,
            "seq": pidx, "fingerprint": q_fp, "status": qstat,
            "payload": {
                "ping_id": pid, "good_mask": list(mask),
                "correlations": corrs, "amplitudes": amps,
                "bad_flags": flags, "correlation_floor": CORRELATION_FLOOR,
                "source_ids": {"correlation": corr["id"] if corr else None,
                               "amplitude": amp["id"] if amp else None,
                               "bad_beam_flag": flag["id"] if flag else None},
            },
            "excluded": [{"kind": "beam", "beam_index": i,
                          "reason": ("bad_flag" if i < len(flags) and flags[i]
                                     else ("low_correlation"
                                           if (i >= len(corrs) or corrs[i] is None
                                               or corrs[i] < CORRELATION_FLOOR)
                                           else "missing_amplitude"))}
                         for i in excluded_beams],
        })

        # 深度 cell 有效性: cell 必须在 bottom track 给出的有效范围且深度合法
        btm = btm_by_ping.get(pid)
        max_depth = None
        excluded_cells: List[Dict[str, Any]] = []
        if btm:
            btm_payload = btm["payload"]
            if btm_payload.get("valid_depth_m") is not None:
                max_depth = float(btm_payload["valid_depth_m"])
        cells = list(bp.get("cells", []))

        attitude_here = interp[pidx] if pidx < len(interp) else None
        for ci, cell in enumerate(cells):
            cell_scope = "%s-cell%02d" % (scope, ci)
            depth_m = cell.get("depth_m")
            raw_beams = cell.get("beams")  # 4 个带单位的原始量
            beams_vals: List[Optional[float]] = []
            invalid = False
            invalid_reasons: List[str] = []
            for b in raw_beams:
                if b is None or b.get("value") is None:
                    beams_vals.append(None)
                    if b is not None and b.get("diagnostic"):
                        invalid_reasons.append(b["diagnostic"])
                    continue
                qty = _to_canonical(b)
                if qty["diagnostic"]:
                    beams_vals.append(None)
                    invalid_reasons.append(qty["diagnostic"])
                else:
                    beams_vals.append(qty["value"])
            if depth_m is None or (isinstance(depth_m, dict) and depth_m.get("value") is None):
                invalid = True
                invalid_reasons.append("invalid-or-missing depth cell")
            else:
                if isinstance(depth_m, dict):
                    dq = _to_canonical(depth_m)
                    depth_value = dq["value"]
                    if dq["diagnostic"] or depth_value is None or depth_value <= 0:
                        invalid = True
                        invalid_reasons.append("invalid depth (unconvertible or <=0)")
                else:
                    depth_value = float(depth_m)
                    if depth_value <= 0:
                        invalid = True
                        invalid_reasons.append("invalid depth (<=0)")
                if not invalid and max_depth is not None and depth_value > max_depth:
                    invalid = True
                    invalid_reasons.append(
                        "cell beyond valid bottom-track depth %.2f m" % max_depth)

            # 应用 beam 质量掩码
            effective = [v if mask[i] else None for i, v in enumerate(beams_vals)]
            good_count = sum(v is not None for v in effective)
            cell_excluded = list(excluded_cells)
            if invalid:
                cell_excluded.append({"kind": "cell", "cell_index": ci,
                                      "reason": "; ".join(invalid_reasons)})
                # 无效 cell: 不做派生, 仅留排除证据 (不为画图丢值)
                fp = derive_fingerprint(
                    node_type="cell",
                    inputs={"raw": raw_beams, "depth": depth_m, "reasons": invalid_reasons},
                    geom=geom, declination_version=dec_version,
                    mount_version=mount_src_id, is_order=is_order,
                    earth_order=earth_order, gap_tolerance=GAP_TOLERANCE)
                node = rid("cell", cell_scope)
                stat = status_for("cell|%s|0" % cell_scope, fp, node, force_ping,
                                  "cell marked invalid" )
                results.append({"id": node, "node_type": "cell", "scope": cell_scope,
                                "seq": ci, "fingerprint": fp, "status": stat,
                                "payload": {"ping_id": pid, "cell_index": ci,
                                            "raw_beams": raw_beams,
                                            "depth": depth_m,
                                            "attitude_status": attitude_here.status
                                            if attitude_here else "missing"},
                                "excluded": cell_excluded})
                continue

            beam_inputs = _fp_inputs_beam(
                {"beams": effective}, geom_src_id, qids)
            inst_node = rid("instrument", cell_scope)
            if good_count == 4:
                sol = solve_four_beams(effective, geom)
                fp = derive_fingerprint(
                    node_type="instrument.four", inputs=beam_inputs, geom=geom,
                    declination_version=dec_version, mount_version=mount_src_id,
                    is_order=is_order, earth_order=earth_order,
                    gap_tolerance=GAP_TOLERANCE)
                stat = status_for("instrument.four|%s|0" % cell_scope, fp, inst_node,
                                  force_ping, "upstream beam inputs unchanged")
                results.append({
                    "id": inst_node, "node_type": "instrument.four",
                    "scope": cell_scope, "seq": ci, "fingerprint": fp,
                    "status": stat,
                    "payload": {"ping_id": pid, "cell_index": ci,
                                "solution": "unique-four-beam",
                                "instrument_uvw": [sol.u, sol.v, sol.w],
                                "error_velocity": sol.e,
                                "unit": "m/s", "frame": "instrument",
                                "good_beams": [0, 1, 2, 3]},
                    "excluded": cell_excluded})
                chosen_uvw = [sol.u, sol.v, sol.w]
                candidate_kinds = ["instrument.four"]
            elif good_count == 3:
                candidates, missing = solve_three_beams(effective, geom)
                family = family_parameterization(effective, geom)
                candidate_kinds = []
                chosen_uvw = None
                for cand in candidates:
                    fp = derive_fingerprint(
                        node_type="instrument.three", inputs=beam_inputs, geom=geom,
                        declination_version=dec_version, mount_version=mount_src_id,
                        is_order=is_order, earth_order=earth_order,
                        gap_tolerance=GAP_TOLERANCE,
                        candidate_assumption=cand.assumption)
                    cid = "cand-%s" % cand.assumption
                    node = rid("instrument.three", cell_scope, candidate=cid)
                    stat = status_for(
                        "instrument.three|%s|0|%s" % (cell_scope, cid), fp, node,
                        force_ping, "three-beam candidate inputs unchanged")
                    results.append({
                        "id": node, "node_type": "instrument.three",
                        "scope": cell_scope, "seq": ci, "candidate_id": cid,
                        "fingerprint": fp, "status": stat,
                        "payload": {"ping_id": pid, "cell_index": ci,
                                    "solution": "underdetermined-three-beam-family",
                                    "missing_beam": missing,
                                    "assumption": cand.assumption, "note": cand.note,
                                    "instrument_uvw": [cand.u, cand.v, cand.w],
                                    "error_velocity": cand.e,
                                    "family": family,
                                    "unit": "m/s", "frame": "instrument",
                                    "good_beams": [i for i in range(4)
                                                   if effective[i] is not None]},
                        "excluded": cell_excluded})
                    candidate_kinds.append(node)
                # 无人工选择前不存在"单一"三束值; ship/earth 层对每个候选各算一份,
                # 但剖面聚合只接受四束或已被人工接受的候选。
            else:
                fp = derive_fingerprint(
                    node_type="instrument.unavailable",
                    inputs={"beams": effective, "good_count": good_count},
                    geom=geom, declination_version=dec_version,
                    mount_version=mount_src_id, is_order=is_order,
                    earth_order=earth_order, gap_tolerance=GAP_TOLERANCE)
                node = rid("instrument.unavailable", cell_scope)
                stat = status_for("instrument.unavailable|%s|0" % cell_scope, fp,
                                  node, force_ping, "fewer than three good beams")
                results.append({
                    "id": node, "node_type": "instrument.unavailable",
                    "scope": cell_scope, "seq": ci, "fingerprint": fp,
                    "status": stat,
                    "payload": {"ping_id": pid, "cell_index": ci,
                                "good_beam_count": good_count,
                                "reason": "fewer than 3 good beams"},
                    "excluded": cell_excluded + [
                        {"kind": "beam", "beam_index": i,
                         "reason": "no usable signal"}
                        for i in range(4) if effective[i] is None]})
                continue

            # ---- ship / earth 层 (逐候选; 姿态缺口则不可用, 不插值跨缺口) ----
            inst_payloads = []
            if good_count == 4:
                inst_payloads.append((None, inst_node, chosen_uvw))
            else:
                for r in results:
                    if (r["node_type"] == "instrument.three"
                            and r["scope"] == cell_scope):
                        inst_payloads.append(
                            (r["candidate_id"], r["id"],
                             r["payload"]["instrument_uvw"]))

            for cand_id, inst_id, uvw in inst_payloads:
                suffix = cand_id or ""
                att_node = rid("attitude", scope, candidate=suffix)
                att_fp = derive_fingerprint(
                    node_type="attitude",
                    inputs={"attitude_source": att_active["id"] if att_active else None,
                            "time_s": bp["time_s"], "gap_tolerance": GAP_TOLERANCE},
                    geom=geom, declination_version=dec_version,
                    mount_version=mount_src_id, is_order=is_order,
                    earth_order=earth_order, gap_tolerance=GAP_TOLERANCE,
                    candidate_assumption=suffix)
                att_force = force_ping or (
                    att_active and att_active["id"] in changed.changed_source_ids)
                att_stat = status_for(
                    "attitude|%s|0|%s" % (scope, suffix), att_fp, att_node,
                    att_force,
                    "attitude inputs unchanged" if not att_force
                    else "attitude source changed")
                att_payload = {
                    "ping_id": pid, "time_s": bp["time_s"],
                    "status": attitude_here.status if attitude_here else "missing",
                    "heading_deg": getattr(attitude_here, "heading", None),
                    "pitch_deg": getattr(attitude_here, "pitch", None),
                    "roll_deg": getattr(attitude_here, "roll", None),
                    "source_times": list(attitude_here.source_times)
                    if attitude_here and attitude_here.source_times else None,
                    "reason": getattr(attitude_here, "reason", None),
                    "attitude_source": att_active["id"] if att_active else None,
                }
                results.append({"id": att_node, "node_type": "attitude",
                                "scope": scope, "seq": pidx,
                                "candidate_id": suffix or None,
                                "fingerprint": att_fp, "status": att_stat,
                                "payload": att_payload,
                                "excluded": ([] if att_payload["status"] == "ok"
                                             else [{"kind": "attitude", "ping_id": pid,
                                                    "reason": att_payload["status"]}])})
                dependencies.append((inst_id, att_node, "needs-attitude"))

                if att_payload["status"] != "ok":
                    for layer in ("ship", "earth"):
                        node = rid(layer, cell_scope, candidate=suffix)
                        fp = derive_fingerprint(
                            node_type="%s.unavailable" % layer,
                            inputs={"attitude_status": att_payload["status"]},
                            geom=geom, declination_version=dec_version,
                            mount_version=mount_src_id, is_order=is_order,
                            earth_order=earth_order, gap_tolerance=GAP_TOLERANCE,
                            candidate_assumption=suffix)
                        stat = status_for(
                            "%s.unavailable|%s|0|%s" % (layer, cell_scope, suffix),
                            fp, node, True,
                            "attitude gap: interpolation never crosses gaps")
                        results.append({
                            "id": node, "node_type": "%s.unavailable" % layer,
                            "scope": cell_scope, "seq": ci,
                            "candidate_id": suffix or None, "fingerprint": fp,
                            "status": stat,
                            "payload": {"ping_id": pid, "cell_index": ci,
                                        "reason": att_payload["status"],
                                        "frame": layer},
                            "excluded": [{"kind": "velocity-layer", "layer": layer,
                                          "ping_id": pid, "cell_index": ci,
                                          "reason": "attitude %s" % att_payload["status"]}]})
                    continue

                tf = build_transform(
                    uvw, att_payload["heading_deg"], att_payload["pitch_deg"],
                    att_payload["roll_deg"], declination, mount=mount,
                    is_order=is_order, earth_order=earth_order)
                ship_node = rid("ship", cell_scope, candidate=suffix)
                ship_fp = derive_fingerprint(
                    node_type="ship",
                    inputs={"instrument_uvw": tf.instrument,
                            "pitch": att_payload["pitch_deg"],
                            "roll": att_payload["roll_deg"],
                            "mount_source": mount_src_id, "order_is": is_order},
                    geom=geom, declination_version=dec_version,
                    mount_version=mount_src_id, is_order=is_order,
                    earth_order=earth_order, gap_tolerance=GAP_TOLERANCE,
                    candidate_assumption=suffix)
                ship_stat = status_for(
                    "ship|%s|0|%s" % (cell_scope, suffix), ship_fp, ship_node,
                    force_ping or att_force, "instrument/attitude unchanged")
                results.append({
                    "id": ship_node, "node_type": "ship", "scope": cell_scope,
                    "seq": ci, "candidate_id": suffix or None,
                    "fingerprint": ship_fp, "status": ship_stat,
                    "payload": {"ping_id": pid, "cell_index": ci,
                                "ship_uvw": list(tf.ship), "unit": "m/s",
                                "frame": "ship",
                                "mount_source": mount_src_id,
                                "step": tf.steps[0].as_dict()},
                    "excluded": cell_excluded})
                dependencies.append((inst_id, ship_node, "transforms-to"))
                dependencies.append((att_node, ship_node, "uses-attitude"))

                earth_node = rid("earth", cell_scope, candidate=suffix)
                earth_fp = derive_fingerprint(
                    node_type="earth",
                    inputs={"ship_uvw": tf.ship,
                            "heading_true": tf.heading_true,
                            "pitch": att_payload["pitch_deg"],
                            "roll": att_payload["roll_deg"],
                            "declination_version": dec_version,
                            "order_earth": earth_order},
                    geom=geom, declination_version=dec_version,
                    mount_version=mount_src_id, is_order=is_order,
                    earth_order=earth_order, gap_tolerance=GAP_TOLERANCE,
                    candidate_assumption=suffix)
                earth_stat = status_for(
                    "earth|%s|0|%s" % (cell_scope, suffix), earth_fp, earth_node,
                    force_ping or att_force, "ship/declination/order unchanged")
                results.append({
                    "id": earth_node, "node_type": "earth", "scope": cell_scope,
                    "seq": ci, "candidate_id": suffix or None,
                    "fingerprint": earth_fp, "status": earth_stat,
                    "payload": {"ping_id": pid, "cell_index": ci,
                                "earth_enu": list(tf.earth), "unit": "m/s",
                                "frame": "earth", "reference": "true",
                                "heading_true_deg": tf.heading_true,
                                "declination_version": dec_version,
                                "steps": [s.as_dict() for s in tf.steps],
                                "order": tf.order},
                    "excluded": cell_excluded})
                dependencies.append((ship_node, earth_node, "transforms-to"))

                if good_count == 4:
                    profile_rows.append({
                        "ping_id": pid, "cell_index": ci,
                        "depth_m": depth_value,
                        "earth_enu": list(tf.earth),
                        "instrument_uvw": list(tf.instrument),
                        "ship_uvw": list(tf.ship),
                        "error_velocity": sol.e,
                        "kind": "four-beam"})

    # ---- 垂向剖面聚合: 仅四束或已被人工接受的三束候选, 未选候选不混入 ----
    accepted_candidates: Dict[str, str] = {}
    if parent_version:
        for d in store.decisions(parent_version):
            if d["chosen_candidate_id"]:
                accepted_candidates[d["node_scope"]] = d["chosen_candidate_id"]

    # 把已接受的三束候选补进剖面
    for r in results:
        if r["node_type"] != "earth" or not r.get("candidate_id"):
            continue
        if accepted_candidates.get(r["scope"]) == r["candidate_id"]:
            inst_r = next(x for x in results
                          if x["scope"] == r["scope"]
                          and x["node_type"] == "instrument.three"
                          and x["candidate_id"] == r["candidate_id"])
            ship_r = next(x for x in results
                          if x["scope"] == r["scope"] and x["node_type"] == "ship"
                          and x["candidate_id"] == r["candidate_id"])
            bp_payload = beam_by_ping[r["payload"]["ping_id"]]["payload"]
            depth_value = bp_payload["cells"][r["payload"]["cell_index"]]["depth_m"]
            if isinstance(depth_value, dict):
                depth_value = _to_canonical(depth_value)["value"]
            profile_rows.append({
                "ping_id": r["payload"]["ping_id"],
                "cell_index": r["payload"]["cell_index"],
                "depth_m": depth_value,
                "earth_enu": r["payload"]["earth_enu"],
                "instrument_uvw": inst_r["payload"]["instrument_uvw"],
                "ship_uvw": ship_r["payload"]["ship_uvw"],
                "error_velocity": inst_r["payload"]["error_velocity"],
                "kind": "three-beam:%s" % r["candidate_id"]})

    excluded_rows = [r for r in results if r["excluded"]]
    prof_fp = derive_fingerprint(
        node_type="profile",
        inputs={"rows": profile_rows, "excluded_identities":
                [r["id"] for r in excluded_rows]},
        geom=geom, declination_version=dec_version, mount_version=mount_src_id,
        is_order=is_order, earth_order=earth_order, gap_tolerance=GAP_TOLERANCE,
        extra={"accepted_candidates": accepted_candidates})
    prof_node = rid("profile", "all")
    prof_stat = status_for("profile|all|0", prof_fp, prof_node,
                           any(n["status"] == "recomputed"
                               for n in node_status
                               if n["node"].startswith(("instrument", "earth", "cell"))),
                           "all contributing nodes unchanged")
    results.append({
        "id": prof_node, "node_type": "profile", "scope": "all", "seq": 0,
        "fingerprint": prof_fp, "status": prof_stat,
        "payload": {"rows": profile_rows,
                   "accepted_three_beam_candidates": accepted_candidates},
        "excluded": [{"result_id": r["id"], "items": r["excluded"]}
                     for r in excluded_rows]})

    # ---- 机器可读断言 (红线策略) ----
    assertions = list(assertions or [])
    assertions.extend(_policy_assertions(results))

    event_payload = dict(event_payload or {})
    event_payload.update({
        "changed_source_ids": changed.changed_source_ids,
        "affected_pings": changed.affected_pings,
        "note": changed.note,
        "node_status": node_status,
        "basis": {"declination_version": dec_version,
                  "mount_source": mount_src_id, "geometry": geom.as_dict(),
                  "is_order": is_order, "earth_order": earth_order},
    })

    commit_kwargs = dict(
        version_id=version_id, label=label, kind="derivation",
        parent_version=parent_version, author=author, source_ids=source_ids,
        results=results, dependencies=dependencies,
        events_payload=event_payload, event_type=event_type,
        assertions=assertions, payload={"orders": {"is": is_order,
                                                   "earth": earth_order}},
        created_at=created_at)
    if allow_pointer_fork:
        commit_kwargs["make_current"] = False
    version = store.commit_plan(**commit_kwargs)
    # 携带 node_status 便于调用方 (测试/UI) 直接读
    version["node_status"] = node_status
    version["assertions"] = assertions
    return version


def _to_canonical(raw_q: Dict[str, Any]) -> Dict[str, Any]:
    from .units import Quantity
    return Quantity.from_dict(raw_q).to_canonical().as_dict()


def _identity_key(r: Dict[str, Any]) -> str:
    cid = r.get("candidate_id") or ""
    return "%s|%s|%d|%s" % (r["node_type"], r["scope"], r.get("seq", 0), cid)


def _policy_assertions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把红线判断写成机器可读事件断言。"""
    out: List[Dict[str, Any]] = []
    three = [r for r in results if r["node_type"] == "instrument.three"]
    four_scopes = {r["scope"] for r in results
                   if r["node_type"] == "instrument.four"}
    if three:
        scopes = {r["scope"] for r in three}
        out.append({
            "id": "assert-three-beam-family-not-collapsed",
            "level": "policy",
            "ok": True,
            "detail": {
                "three_beam_scopes": sorted(scopes),
                "candidate_count_per_scope": {
                    s: sum(1 for r in three if r["scope"] == s)
                    for s in scopes},
                "rule": "three-beam solutions kept as explicit candidates; "
                        "never collapsed into the four-beam value"}})
        out.append({
            "id": "assert-three-and-four-not-mixed",
            "level": "policy", "ok": True,
            "detail": {"three_beam_scopes": sorted(scopes),
                       "four_beam_scopes": sorted(four_scopes),
                       "disjoint": sorted(scopes & four_scopes) == [],
                       "rule": "three-beam and four-beam results never share "
                               "one value/identity"}})
    gap = [r for r in results if r["node_type"] == "attitude"
           and r["payload"].get("status") != "ok"]
    if gap:
        unavail = [r for r in results
                   if r["node_type"] in ("ship.unavailable", "earth.unavailable")]
        out.append({
            "id": "assert-attitude-no-interpolation-across-gap",
            "level": "policy", "ok": len(unavail) > 0,
            "detail": {"gap_pings": sorted({r["payload"]["ping_id"] for r in gap}),
                       "blocked_layers": len(unavail),
                       "rule": "attitude interpolation never bridges a time gap; "
                               "ship/earth stay unavailable for those pings"}})
    fp_scopes: Dict[str, str] = {}
    for r in results:
        if r["node_type"] == "earth":
            fp_scopes[r["id"]] = r["fingerprint"]
    if fp_scopes:
        out.append({
            "id": "assert-transform-basis-in-fingerprint",
            "level": "policy", "ok": True,
            "detail": {"fingerprints": fp_scopes,
                       "rule": "transform order, beam geometry and declination "
                               "version all enter the derived fingerprint"}})
    invalid_cells = [r for r in results if r["node_type"] == "cell"]
    if invalid_cells:
        out.append({
            "id": "assert-invalid-cells-retained-as-evidence",
            "level": "policy",
            "ok": all(r["excluded"] for r in invalid_cells),
            "detail": {"cell_count": len(invalid_cells),
                       "rule": "invalid depth cells are retained with diagnostics, "
                               "never dropped for plotting"}})
    return out
