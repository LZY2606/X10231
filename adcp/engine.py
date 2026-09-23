"""依赖图重算引擎。

输入：某版本（revision）可见的事实快照 Snapshot。
输出：节点图 Nodes：beamq -> cell3d(候选) -> transform(候选链路)
      -> profile；另有 attitude / bottom track 分支。

- 每个节点带稳定身份 node_id 与 fingerprint；
- 候选（三波束/四波束）各自独立身份，从不合并为同一值；
- 无唯一解时保留全部合法候选，状态 candidate / uncomputable；
- 人工 decision 接受/否决只产生新状态，不删除未选候选；
- fingerprint 纳入：规则版本、波束几何、磁偏角版本、安装角、
  变换顺序、上游输入哈希。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from .geometry import BeamGeometry, BEAMS
from .quality import (QualityRules, classify_beam, beam_gate, valid_cell)
from .transforms import (Attitude, MountAngles, Declination, to_earth,
                         interpolate_attitude)
from .units import Quantity

ENGINE_VERSION = "engine-1.0.0"


def _fp(parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _vec(v, datum):
    v = np.asarray(v, dtype=float)
    return {"xyz": [float(x) for x in v], "unit": "m/s", "datum": datum}


def _node(node_id, kind, ping, cell=None, status="ok", inputs=None,
          fingerprint=None, value=None, candidates=None, diag=None,
          accepted=None, geom_hash=None):
    return {
        "node_id": node_id,
        "kind": kind,
        "ping": ping,
        "cell": cell,
        "status": status,
        "inputs": inputs or [],
        "fingerprint": fingerprint,
        "value": value,
        "candidates": candidates or [],
        "diag": diag,
        "accepted": accepted,
        "geom_hash": geom_hash,
    }


class Snapshot:
    """某版本可见的事实集合。事实在导入期保持原始表示。"""

    def __init__(self, facts):
        self.facts = list(facts)
        self._idx = {}
        for f in self.facts:
            key = (f["ftype"], f.get("ping"), f.get("cell"), f.get("beam"),
                   f.get("time_s"))
            self._idx.setdefault(key, []).append(f)

    def one(self, ftype, ping=None, cell=None, beam=None, time_s=None):
        arr = self._idx.get((ftype, ping, cell, beam, time_s))
        return arr[-1] if arr else None

    def all_of(self, ftype):
        return [f for f in self.facts if f["ftype"] == ftype]

    def pings_cells(self):
        pings = sorted({f["ping"] for f in self.facts
                        if f.get("ping") is not None})
        cells = sorted({f["cell"] for f in self.facts
                        if f.get("cell") is not None})
        return pings, cells


def _config(snap: Snapshot):
    geo_f = snap.one("geoconfig")
    gp = geo_f.get("payload", {}) if geo_f else {}
    geom = BeamGeometry(
        beam_angle_deg=float(gp.get("beam_angle_deg", 20.0)),
        convex=bool(gp.get("convex", True)),
        name=str(gp.get("name", "rdi_janus")))

    q_f = snap.one("qconfig")
    qp = q_f.get("payload", {}) if q_f else {}
    qrules = QualityRules(**{k: v for k, v in qp.items()
                             if k in QualityRules().__dataclass_fields__})
    max_gap = float(qp.get("max_attitude_gap_s", 2.0))

    dec_f = snap.one("declination")
    dp = dec_f.get("payload", {}) if dec_f else {}
    decl = Declination(float(dp.get("value_deg", 0.0)),
                       str(dp.get("model", "WMM")),
                       str(dp.get("epoch", "")))

    mt_f = snap.one("mount")
    mp = mt_f.get("payload", {}) if mt_f else {}
    mount = MountAngles(float(mp.get("heading_align_deg", 0.0)),
                        float(mp.get("pitch_offset_deg", 0.0)),
                        float(mp.get("roll_offset_deg", 0.0)))

    ord_f = snap.one("transform_order")
    order = (ord_f["raw"] if ord_f and ord_f.get("raw")
             else "rdi_pitch_roll")

    config_ids = [x["fact_id"] for x in (geo_f, q_f, dec_f, mt_f, ord_f) if x]
    config_hashes = [x["fhash"] for x in (geo_f, q_f, dec_f, mt_f, ord_f) if x]
    return {
        "geom": geom, "qrules": qrules, "decl": decl, "mount": mount,
        "order": str(order), "max_gap": max_gap,
        "config_ids": config_ids, "config_hashes": config_hashes,
        "dec_f": dec_f, "mt_f": mt_f, "ord_f": ord_f,
        "geo_f": geo_f, "q_f": q_f,
    }


def _radial(fact):
    """事实 -> Quantity（派生层才规范化）。"""
    if fact is None:
        return None
    return Quantity.from_raw(fact.get("raw"), fact.get("raw_unit"),
                             "velocity")


def _flag(fact):
    if fact is None:
        return 0
    raw = fact.get("raw")
    try:
        return 1 if int(raw) != 0 else 0
    except (TypeError, ValueError):
        return 1 if str(raw).strip().lower() in ("1", "true", "bad") else 0


def _scalar(fact, kind="angle"):
    if fact is None:
        return None
    return Quantity.from_raw(fact.get("raw"), fact.get("raw_unit"), kind,
                             datum=fact.get("datum"))


def _attitude_series(snap: Snapshot):
    """heading/pitch/roll 分开记录 -> 按 time_s 对齐的采样序列。

    支持两种来源：独立 heading/pitch/roll 事实（优先，体现“分开记录”），
    或单条 attitude 事实 payload 三字段。
    """
    hv = {f["time_s"]: f for f in snap.all_of("heading")}
    pv = {f["time_s"]: f for f in snap.all_of("pitch")}
    rv = {f["time_s"]: f for f in snap.all_of("roll")}
    times = sorted(set(hv) & set(pv) & set(rv))
    hq = [_scalar(hv[t]) for t in times]
    pq = [_scalar(pv[t]) for t in times]
    rq = [_scalar(rv[t]) for t in times]
    if not times:
        for a in snap.all_of("attitude"):
            t = a["time_s"]
            times.append(t)
            pay = a.get("payload", {})
            hq.append(Quantity.from_raw(pay.get("heading"),
                                        pay.get("heading_unit"), "angle",
                                        datum=pay.get("heading_datum")))
            pq.append(Quantity.from_raw(pay.get("pitch"),
                                        pay.get("pitch_unit"), "angle"))
            rq.append(Quantity.from_raw(pay.get("roll"),
                                        pay.get("roll_unit"), "angle"))
        order_idx = np.argsort(np.array(times, dtype=float))
        times = [times[i] for i in order_idx]
        hq = [hq[i] for i in order_idx]
        pq = [pq[i] for i in order_idx]
        rq = [rq[i] for i in order_idx]
    return times, hq, pq, rq


def _interp_for_ping(snap, cfg, ping_time, inputs_collector):
    times, hq, pq, rq = _attitude_series(snap)
    valid = [i for i in range(len(times)) if hq[i].ok and pq[i].ok and rq[i].ok]
    if not valid:
        return None, "no readable attitude samples", []
    for i in valid:
        inputs_collector.append({
            "t": times[i],
            "heading_fhash_hint": hq[i].diag or "ok",
        })
    att, diag = interpolate_attitude(
        [times[i] for i in valid],
        [hq[i].value for i in valid],
        [pq[i].value for i in valid],
        [rq[i].value for i in valid],
        ping_time, cfg["max_gap"])
    return att, diag, times


def _radials_and_quality(snap, cfg, ping, cell, is_bt=False):
    pref = "bt_" if is_bt else ""
    rads, states, beams_info, input_ids = [], [], [], []
    for b in range(BEAMS):
        vf = snap.one(pref + "beam", ping, None if is_bt else cell, b)
        cf = snap.one(pref + "correlation", ping,
                      None if is_bt else cell, b)
        af = snap.one(pref + "amplitude", ping,
                      None if is_bt else cell, b)
        bf = snap.one(pref + "badflag", ping,
                      None if is_bt else cell, b)
        for x in (vf, cf, af, bf):
            if x:
                input_ids.append(x["fact_id"])
        qty = _radial(vf)
        radial = qty.value if qty is not None and qty.ok else None
        corr = cf.get("raw") if cf else None
        amp = af.get("raw") if af else None
        state, reasons = classify_beam(_flag(bf), corr, amp, radial,
                                       cfg["qrules"])
        if qty is not None and not qty.ok:
            state, reasons = "excluded", reasons + ["radial_unconvertible"]
            radial = None
        rads.append(radial)
        states.append(state)
        beams_info.append({
            "beam": b + 1,
            "state": state,
            "reasons": reasons,
            "radial": None if radial is None else float(radial),
            "radial_unit": "m/s",
            "raw_radial": None if vf is None else vf.get("raw"),
            "raw_radial_unit": None if vf is None else vf.get("raw_unit"),
            "correlation": corr,
            "amplitude": amp,
            "bad_flag": _flag(bf),
            "fact_ids": [x["fact_id"] for x in (vf, cf, af, bf) if x],
        })
    return rads, states, beams_info, input_ids


def _candidates(geom, rads, states, gate):
    """生成候选集合。四波束与三波束候选身份永远分离。"""
    cands = []
    excluded = [b for b in range(BEAMS) if states[b] == "excluded"]
    if gate.startswith("four_beam"):
        sol4 = geom.solve_four(rads)
        cands.append({
            "cid": "C4", "kind": "four_beam",
            "uvw": [float(x) for x in sol4["uvw"]],
            "error_velocity": sol4["error_velocity"],
            "residual": sol4["residual"],
            "label": "4-beam least-squares",
        })
        # 即使四波束全可用，也保留每个三波束候选（它们不与四波束合并）
        for missing in range(BEAMS):
            sol3 = geom.solve_three(rads, missing)
            cands.append({
                "cid": f"C3_m{missing + 1}",
                "kind": "three_beam",
                "missing_beam": missing + 1,
                "uvw": [float(x) for x in sol3["uvw"]],
                "predicted_missing_radial": sol3["predicted_missing_radial"],
                "label": f"3-beam (drop beam {missing + 1})",
            })
    elif gate.startswith("three_beam"):
        for missing in excluded:
            sol3 = geom.solve_three(rads, missing)
            cands.append({
                "cid": f"C3_m{missing + 1}",
                "kind": "three_beam",
                "missing_beam": missing + 1,
                "uvw": [float(x) for x in sol3["uvw"]],
                "predicted_missing_radial": sol3["predicted_missing_radial"],
                "label": f"3-beam (drop beam {missing + 1})",
            })
    return cands


def _decision_map(snap):
    """人工决定：{scope_key: {cid: accept|reject}}，按 time_s 顺序应用。"""
    dmap = {}
    decisions = sorted(snap.all_of("decision"),
                       key=lambda f: (f.get("time_s") or 0, f["fact_id"]))
    for d in decisions:
        p = d.get("payload", {})
        scope = f"p{p.get('ping')}c{p.get('cell')}"
        dmap.setdefault(scope, {})[p.get("cid")] = p.get("action")
    return dmap


def _apply_decision(cands, scope, dmap):
    actions = dmap.get(scope, {})
    accepted = None
    for c in cands:
        act = actions.get(c["cid"])
        c["adjudication"] = act  # accept / reject / None
        if act == "accept":
            accepted = c["cid"]
        if act == "reject":
            c["status"] = "rejected"
    return accepted


def _chains(cands, cfg, att, node_inputs):
    """为每个候选构造 instrument->ship->earth 逐级链路。"""
    out = []
    for c in cands:
        chain = None
        heading_eff = None
        diag = None
        if att is None:
            diag = "attitude unavailable; transform not computed"
        else:
            earth, ship, heading_eff = to_earth(
                c["uvw"], att, cfg["mount"], cfg["decl"], cfg["order"])
            chain = {
                "instrument": _vec(c["uvw"], "instrument_frame"),
                "ship": _vec(ship, "ship_frame"),
                "earth": _vec(earth, "earth_frame_ENU_depth"),
            }
        out.append({
            "cid": c["cid"],
            "kind": c["kind"],
            "label": c["label"],
            "uvw": c["uvw"],
            "missing_beam": c.get("missing_beam"),
            "error_velocity": c.get("error_velocity"),
            "residual": c.get("residual"),
            "predicted_missing_radial": c.get("predicted_missing_radial"),
            "adjudication": c.get("adjudication"),
            "heading_eff_deg": heading_eff,
            "chain": chain,
            "diag": diag,
        })
    return out


def recompute(facts, prior=None):
    """对快照运行全量派生；与 prior 对比得到变更集（依赖图传播语义）。"""
    snap = Snapshot(facts)
    cfg = _config(snap)
    dmap = _decision_map(snap)
    pings, cells = snap.pings_cells()
    nodes = []

    cfg_fp = _fp({
        "engine": ENGINE_VERSION,
        "geom": cfg["geom"].fingerprint(),
        "qrules": cfg["qrules"].fingerprint(),
        "decl": cfg["decl"].fingerprint(),
        "mount": cfg["mount"].fingerprint(),
        "order": cfg["order"],
        "config_hashes": cfg["config_hashes"],
    })

    # ---- 姿态节点：每次 ping 在其采样时刻插值（拒绝跨缺口）----
    ping_times = {}
    for f in snap.facts:
        if f["ftype"] == "beam" and f.get("ping") is not None:
            if f.get("time_s") is not None:
                ping_times[f["ping"]] = f["time_s"]
    # 若波束事实没带 time_s，则 ping 序号即秒序（仅演示用）
    for p in pings:
        ping_times.setdefault(p, float(p))

    for p in pings:
        collector = []
        att, diag, tseries = _interp_for_ping(snap, cfg, ping_times[p],
                                              collector)
        att_fact_ids = [f["fact_id"] for f in snap.facts
                        if f["ftype"] in ("heading", "pitch", "roll",
                                          "attitude")]
        value = None
        status = "ok"
        if att is None:
            status = "uncomputable"
        else:
            value = {
                "heading_deg": att.heading_deg,
                "pitch_deg": att.pitch_deg,
                "roll_deg": att.roll_deg,
                "heading_eff_deg": (att.heading_deg
                                    + cfg["mount"].heading_align_deg
                                    + cfg["decl"].value_deg) % 360.0,
                "time_s": ping_times[p],
                "interpolated": not any(
                    abs(t - ping_times[p]) < 1e-9 for t in tseries),
                "heading_datum": cfg["decl"].model,
                "angle_unit": "deg",
            }
        nodes.append(_node(
            f"att_p{p}", "attitude", p, status=status,
            inputs=att_fact_ids + cfg["config_ids"],
            fingerprint=_fp([cfg_fp, att_fact_ids, "attitude", p]),
            value=value, diag=diag))

    # ---- 逐 ping/逐 cell：beamq -> 候选解 -> 逐级变换 ----
    for p in pings:
        att_node = next(n for n in nodes if n["kind"] == "attitude"
                        and n["ping"] == p)
        att, att_diag, _ = _interp_for_ping(snap, cfg, ping_times[p], [])
        for c in cells:
            rads, states, binfo, in_ids = _radials_and_quality(
                snap, cfg, p, c)
            gate, n_ok, gate_msg = beam_gate(states)
            nid_q = f"beampq_p{p}_c{c}"
            qfp = _fp([cfg_fp, in_ids, "beamq", p, c])
            nodes.append(_node(
                nid_q, "beamq", p, c,
                status="marginal" if gate == "four_beam_ambiguous"
                else ("unsolvable" if gate == "unsolvable" else "ok"),
                inputs=in_ids, fingerprint=qfp,
                value={"gate": gate, "usable_beams": n_ok,
                       "gate_msg": gate_msg, "beams": binfo}))

            depth_f = snap.one("celldepth", p, c)
            depth = _scalar(depth_f, "length")
            blank = cfg["geom"] and float(
                (cfg["geo_f"].get("payload", {}) if cfg["geo_f"] else {})
                .get("blanking_m", 0.0))
            depth_ok, depth_diag = (False, "no cell depth")
            if depth is not None and depth.ok:
                depth_ok, depth_diag = valid_cell(depth.value, blank,
                                                  cfg["qrules"])

            nid = f"cell3d_p{p}_c{c}"
            base_inputs = in_ids + cfg["config_ids"]
            if depth_f:
                base_inputs.append(depth_f["fact_id"])
            cfp = _fp([qfp, base_inputs, "cell3d", p, c])

            if not depth_ok:
                nodes.append(_node(
                    nid, "cell3d", p, c, status="excluded_cell",
                    inputs=base_inputs, fingerprint=cfp,
                    diag=f"invalid depth cell: {depth_diag}"))
                continue
            if gate == "unsolvable":
                nodes.append(_node(
                    nid, "cell3d", p, c, status="unsolvable",
                    inputs=base_inputs, fingerprint=cfp,
                    diag=gate_msg))
                continue

            cands = _candidates(cfg["geom"], rads, states, gate)
            scope = f"p{p}c{c}"
            accepted = _apply_decision(cands, scope, dmap)
            chains = _chains(cands, cfg, att, base_inputs)

            # 四波束一致性：error velocity 超阈值 -> 不自动唯一
            auto_cid = None
            if gate == "four_beam_unique":
                ev = abs(chains[0]["error_velocity"] or 0.0)
                if ev <= cfg["qrules"].error_velocity_limit_mps:
                    auto_cid = "C4"
            if gate == "three_beam_unique" and len(chains) == 1:
                auto_cid = chains[0]["cid"]
            if accepted:
                auto_cid = accepted
                status = "resolved"
            elif auto_cid:
                status = "resolved"
            else:
                status = "candidate"

            chosen = next((x for x in chains if x["cid"] == auto_cid), None)
            value = None
            if chosen and chosen.get("chain"):
                value = {
                    "cid": chosen["cid"],
                    "kind": chosen["kind"],
                    "instrument": chosen["chain"]["instrument"],
                    "ship": chosen["chain"]["ship"],
                    "earth": chosen["chain"]["earth"],
                    "heading_eff_deg": chosen["heading_eff_deg"],
                    "error_velocity": chosen.get("error_velocity"),
                }
            elif chosen and att is None:
                status = "uncomputable"
            diag = None if att is not None else att_diag
            nodes.append(_node(
                nid, "cell3d", p, c, status=status,
                inputs=base_inputs, fingerprint=cfp,
                value=value, candidates=chains, diag=diag,
                accepted=accepted, geom_hash=cfg["geom"].fingerprint()))

    # ---- bottom track 分支 ----
    for p in pings:
        att, att_diag, _ = _interp_for_ping(snap, cfg, ping_times[p], [])
        rads, states, binfo, in_ids = _radials_and_quality(
            snap, cfg, p, None, is_bt=True)
        if not any(snap.one("bt_beam", p, beam=b) for b in range(BEAMS)):
            continue
        gate, n_ok, gate_msg = beam_gate(states)
        qid = f"btq_p{p}"
        nodes.append(_node(qid, "btq", p, status=(
            "unsolvable" if gate == "unsolvable" else "ok"),
            inputs=in_ids,
            fingerprint=_fp([cfg_fp, in_ids, "btq", p]),
            value={"gate": gate, "usable_beams": n_ok,
                   "gate_msg": gate_msg, "beams": binfo}))
        chains = []
        status = "unsolvable"
        if gate != "unsolvable":
            cands = _candidates(cfg["geom"], rads, states, gate)
            chains = _chains(cands, cfg, att, in_ids + cfg["config_ids"])
            status = "resolved" if chains else "candidate"
            if att is None:
                status = "uncomputable"
        chosen = chains[0] if chains else None
        value = None
        if chosen and chosen.get("chain"):
            value = {"cid": chosen["cid"],
                     "instrument": chosen["chain"]["instrument"],
                     "ship": chosen["chain"]["ship"],
                     "earth": chosen["chain"]["earth"]}
        nodes.append(_node(
            f"bt_p{p}", "bt", p, status=status,
            inputs=in_ids + cfg["config_ids"],
            fingerprint=_fp([cfg_fp, in_ids, "bt", p]),
            value=value, candidates=chains,
            diag=None if att is not None else att_diag))

    # ---- 垂向剖面节点（聚合）----
    for p in pings:
        members = [n for n in nodes if n["kind"] == "cell3d"
                   and n["ping"] == p]
        pf_inputs = sorted({i for n in members for i in n["inputs"]})
        nodes.append(_node(
            f"prof_p{p}", "profile", p,
            status="ok" if members else "empty",
            inputs=pf_inputs,
            fingerprint=_fp([cfg_fp, sorted(n["fingerprint"]
                                            for n in members), "prof", p]),
            value={"member_nodes": [n["node_id"] for n in members],
                   "excluded": [n["node_id"] for n in members
                                if n["status"] == "excluded_cell"]}))

    result = {"nodes": nodes, "config_fingerprint": cfg_fp,
              "pings": pings, "cells": cells}
    if prior is not None:
        result["changes"] = propagate(prior, result)
    return result


def propagate(prior: dict, current: dict) -> dict:
    """依赖图传播：对比指纹，只标记受影响节点。

    返回 {"unchanged":[...], "recomputed":[...], "added":[...],
          "removed":[...]}，并给出上游变化来源。
    """
    old = {n["node_id"]: n for n in prior.get("nodes", [])}
    new = {n["node_id"]: n for n in current.get("nodes", [])}
    changed, unchanged, added, removed = [], [], [], []

    def upstream_changed(node, oldmap):
        sources = []
        for dep in node["inputs"]:
            # 直接事实哈希通过 inputs 体现；节点输入则查节点指纹
            dep_node = oldmap.get(dep)
            cur_node = new.get(dep)
            if dep_node and cur_node and dep_node["fingerprint"] != \
                    cur_node["fingerprint"]:
                sources.append(dep)
        return sources

    for nid, n in new.items():
        if nid not in old:
            added.append(nid)
        elif old[nid]["fingerprint"] != n["fingerprint"]:
            changed.append(nid)
        else:
            unchanged.append(nid)
    for nid in old:
        if nid not in new:
            removed.append(nid)

    return {"unchanged": sorted(unchanged), "recomputed": sorted(changed),
            "added": sorted(added), "removed": sorted(removed)}


def explain_validity(node: dict, prior: dict, current: dict) -> dict:
    """说明某节点为何仍有效或为何需要重算。"""
    nid = node["node_id"]
    old = {n["node_id"]: n for n in prior.get("nodes", [])}
    cur = {n["node_id"]: n for n in current.get("nodes", [])}
    if nid not in old:
        return {"node_id": nid, "state": "new",
                "reason": "new node in this revision"}
    if old[nid]["fingerprint"] == cur[nid]["fingerprint"]:
        return {"node_id": nid, "state": "still_valid",
                "reason": "fingerprint identical: rule version, geometry, "
                          "declination, mount, order and all source facts "
                          "unchanged"}
    changed_deps = [d for d in node["inputs"]
                    if d in old and d in cur and False]  # facts have no nodes
    return {
        "node_id": nid,
        "state": "recomputed",
        "old_fingerprint": old[nid]["fingerprint"],
        "new_fingerprint": cur[nid]["fingerprint"],
        "input_facts": node["inputs"],
        "reason": "one or more fingerprint inputs changed (source fact, "
                  "rule version, beam geometry, declination model, mount "
                  "angle or transform order)",
    }
