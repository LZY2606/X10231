"""Dependency-graph recompute engine.

Derived node order per ping:

    beamq    beam velocity + correlation + amplitude + bad-beam flag
    att      heading/pitch/roll interpolated at ping time (no cross-gap)
    instr    beam -> instrument, four-beam value or three-beam candidates
    ship     instrument -> ship (mounting angle, pitch, roll)
    earth    ship -> earth (heading + magnetic declination, bottom track)
    profile  vertical profile with explicit excluded-cell evidence

Each node stores a fingerprint and its concrete inputs.  When a new
version is built, a node whose fingerprint (and candidate adoption) is
unchanged is carried over from the parent version - so recompute touches
exactly the affected subgraph and every node can explain why it is still
valid or why it was recomputed.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import transforms
from .fingerprint import fingerprint
from .store import ConflictError, Store
from .units import normalize, velocity_to_ms

RULE_VERSION = "adcp-rules/1.3.0"

DEFAULT_PARAMS = {
    "transform_order": "standard",       # alternative: tilt-after-heading
    "min_correlation_pct": 40.0,
    "blanking_cells": 1,
    "max_att_gap_s": 1.5,
    "err_bound_ms": 0.05,
}

NODE_KINDS = ("beamq", "att", "instr", "ship", "earth", "profile")
ORDERS = ("standard", "tilt-after-heading")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def pings_at(store: Store, vid: int) -> List[int]:
    return sorted({r["ping"] for r in
                   store.effective_raw(vid, kind="beam_velocity")})


def geometry_at(store: Store, vid: int) -> Dict:
    recs = store.effective_raw(vid, kind="transducer_config")
    rec = recs[-1]
    geo = json.loads(rec["value_raw"])
    geo["record_id"] = rec["id"]
    return geo


def _latest(store: Store, vid: int, kind: str):
    recs = store.effective_raw(vid, kind=kind)
    return recs[-1] if recs else None


def _raw_fp(recs) -> List:
    return [[r["id"], r["value_raw"], r["unit"], r.get("source")] for r in recs]


def _node(ctx: Dict, key: str, fp: str, status: str, payload: Dict,
          diagnostics: List, inputs: Dict) -> Dict:
    return {"id": f"n:v{ctx['vid']}:{key}", "version_id": ctx["vid"],
            "node_key": key, "fingerprint": fp, "status": status,
            "payload": payload, "diagnostics": diagnostics, "inputs": inputs}


# ----------------------------------------------------------------------
# node builders
# ----------------------------------------------------------------------
def build_beamq(ctx: Dict, ping: int) -> Dict:
    store, vid, params = ctx["store"], ctx["vid"], ctx["params"]
    geo = ctx["geo"]
    recs: List[Dict] = []
    for kind in ("beam_velocity", "correlation", "amplitude",
                 "bad_beam_flag"):
        recs += store.effective_raw(vid, kind=kind, ping=ping)
    fp = fingerprint(rule=RULE_VERSION, node=f"beamq/p{ping}",
                     min_correlation_pct=params["min_correlation_pct"],
                     blanking_cells=params["blanking_cells"], raw=_raw_fp(recs))
    vel, cor, amp, flg = {}, {}, {}, {}
    diagnostics = []
    for r in recs:
        if r["kind"] == "beam_velocity":
            vel[(r["beam"], r["cell"])] = r
        elif r["kind"] == "correlation":
            cor[(r["beam"], r["cell"])] = r
        elif r["kind"] == "amplitude":
            amp[(r["beam"], r["cell"])] = r
        elif r["kind"] == "bad_beam_flag":
            flg[r["beam"]] = r
    beams = []
    for beam in range(1, 5):
        flag_rec = flg.get(beam)
        bad = bool(flag_rec and flag_rec["numeric"] == 1.0)
        cells = []
        for cell in range(geo["cells"]):
            rv, rc, ra = (vel.get((beam, cell)), cor.get((beam, cell)),
                          amp.get((beam, cell)))
            reasons = []
            if cell < params["blanking_cells"]:
                reasons.append("blanking")
            if bad:
                reasons.append("bad-beam-flag")
            if rc is None or rc["numeric"] is None or \
                    rc["numeric"] < params["min_correlation_pct"]:
                reasons.append("low-correlation")
            if ra is not None and ra["status"] != "ok":
                reasons.append("amplitude-unconverted")
                diagnostics.append({"raw_id": ra["id"],
                                    "diagnostic": ra["diagnostic"]})
            if rv is None or rv["status"] != "ok":
                reasons.append("velocity-missing")
            cells.append({
                "cell": cell,
                "velocity_raw_id": rv["id"] if rv else None,
                "velocity_mms": rv["numeric"] if rv else None,
                "correlation_pct": rc["numeric"] if rc else None,
                "amplitude_counts": ra["numeric"] if ra else None,
                "amplitude_unit": ra["unit"] if ra else None,
                "valid": not reasons,
                "reasons": reasons,
            })
        beams.append({"beam": beam, "bad_flag": bad,
                      "flag_raw_id": flag_rec["id"] if flag_rec else None,
                      "cells": cells})
    payload = {"ping": ping, "beams": beams}
    inputs = {"raw": _raw_fp(recs),
              "params": {"rule": RULE_VERSION,
                         "min_correlation_pct":
                             params["min_correlation_pct"],
                         "blanking_cells": params["blanking_cells"]}}
    return _node(ctx, f"beamq/p{ping}", fp, "ok", payload, diagnostics, inputs)


def build_att(ctx: Dict, ping: int) -> Dict:
    store, vid, params = ctx["store"], ctx["vid"], ctx["params"]
    recs = (store.effective_raw(vid, kind="heading")
            + store.effective_raw(vid, kind="pitch")
            + store.effective_raw(vid, kind="roll"))
    ping_time = store.effective_raw(vid, kind="beam_velocity", ping=ping)[0]["t"]
    fp = fingerprint(rule=RULE_VERSION, node=f"att/p{ping}",
                     max_att_gap_s=params["max_att_gap_s"],
                     ping_t=ping_time, raw=_raw_fp(recs))
    grouped: Dict[float, Dict] = {}
    for r in recs:
        grouped.setdefault(r["t"], {"t": r["t"]})[r["kind"]] = r["numeric"]
    samples = [values for _, values in sorted(grouped.items())
               if {"heading", "pitch", "roll"} <= set(values)]
    result = transforms.interpolate_attitude(
        samples, ping_time, params["max_att_gap_s"])
    status = "ok" if result["status"] == "ok" else "gap"
    diagnostics = []
    if status == "gap":
        diagnostics.append("姿态插值不跨缺口: " + result.get("reason", "gap"))
    return _node(ctx, f"att/p{ping}", fp, status,
                 {"ping": ping, **result}, diagnostics,
                 {"raw": _raw_fp(recs),
                  "params": {"rule": RULE_VERSION,
                             "max_att_gap_s": params["max_att_gap_s"]}})


def _instr_vector(cell: Dict):
    if cell["kind"] == "four":
        return [cell["u_ms"], cell["v_ms"], cell["w_ms"]]
    if cell["kind"] == "three" and cell.get("adopted") is not None:
        cand = cell["candidates"][cell["adopted"]]
        return [cand["u_ms"], cand["v_ms"], cand["w_ms"]]
    return None


def build_instr(ctx: Dict, ping: int) -> Dict:
    store, vid, params = ctx["store"], ctx["vid"], ctx["params"]
    geo = ctx["geo"]
    beamq = ctx["nodes"][f"beamq/p{ping}"]
    key = f"instr/p{ping}"
    parent = ctx["parent_nodes"].get(key)
    adopted = ctx["decisions"].get(key)
    if adopted is None and parent is not None:
        adopted = json.loads(parent["payload"]).get("adopted")
    fp = fingerprint(rule=RULE_VERSION, node=key,
                     beamq=beamq["fingerprint"],
                     beam_geometry_id=geo["geometry_id"],
                     geometry_record_id=geo["record_id"],
                     err_bound_ms=params["err_bound_ms"])
    D = transforms.beam_matrix(geo["beam_angle_deg"], geo["azimuths_deg"])
    cells = []
    for cell in range(geo["cells"]):
        good, velocities = [], []
        for beam in range(1, 5):
            info = beamq["payload"]["beams"][beam - 1]["cells"][cell]
            if info["valid"]:
                good.append(beam - 1)
                velocities.append(velocity_to_ms(info["velocity_mms"], "mm/s"))
        if len(good) == 4:
            V, err = transforms.solve_four(velocities, D)
            cells.append({"cell": cell, "kind": "four",
                          "u_ms": round(float(V[0]), 6),
                          "v_ms": round(float(V[1]), 6),
                          "w_ms": round(float(V[2]), 6),
                          "error_velocity_ms": round(err, 6)})
        elif len(good) == 3:
            candidates = transforms.three_beam_candidates(
                velocities, good, D, params["err_bound_ms"])
            cells.append({"cell": cell, "kind": "three",
                          "kept_beams": [b + 1 for b in good],
                          "candidates": candidates, "adopted": adopted})
        else:
            cells.append({"cell": cell, "kind": "none",
                          "good_beams": [b + 1 for b in good],
                          "reason": "insufficient-beams"})
    payload = {"ping": ping, "beam_geometry_id": geo["geometry_id"],
               "cells": cells, "adopted": adopted}
    diagnostics = []
    if any(c["kind"] == "three" for c in cells):
        diagnostics.append(
            "三波束解与四波束解不混为一值: 合法候选全部保留，等待人工裁定")
    inputs = {"nodes": {"beamq": beamq["fingerprint"]},
              "params": {"rule": RULE_VERSION,
                         "beam_geometry_id": geo["geometry_id"],
                         "geometry_record_id": geo["record_id"],
                         "err_bound_ms": params["err_bound_ms"]}}
    return _node(ctx, key, fp, "ok", payload, diagnostics, inputs)


def _blocked(ctx: Dict, key: str, reason: str, att: Dict,
             upstream: Dict, raw: List, detail=None) -> Dict:
    fp = fingerprint(rule=RULE_VERSION, node=key,
                     att=att["fingerprint"], upstream=upstream)
    payload = {"ping": att["payload"]["ping"], "status": "blocked",
               "reason": reason, "attitude": att["payload"]}
    if detail is not None:
        payload["detail"] = detail
    diagnostics = [f"{reason}; 上游未产出时不进行计算"]
    inputs = {"raw": _raw_fp(raw),
              "nodes": {"att": att["fingerprint"], **upstream},
              "params": {"rule": RULE_VERSION}}
    return _node(ctx, key, fp, "blocked", payload, diagnostics, inputs)


def build_ship(ctx: Dict, ping: int) -> Dict:
    store, vid, params = ctx["store"], ctx["vid"], ctx["params"]
    att = ctx["nodes"][f"att/p{ping}"]
    instr = ctx["nodes"][f"instr/p{ping}"]
    key = f"ship/p{ping}"
    upstream = {"instr": [instr["fingerprint"], instr["payload"]["adopted"]]}
    raw_att = [r for r in (store.effective_raw(vid, kind="heading")
                           + store.effective_raw(vid, kind="pitch")
                           + store.effective_raw(vid, kind="roll"))]
    if att["status"] == "gap":
        return _blocked(ctx, key, "attitude-gap", att, upstream, raw_att)
    mnt = _latest(store, vid, "mounting_angle")
    fp = fingerprint(rule=RULE_VERSION, node=key,
                     instr=upstream["instr"], att=att["fingerprint"],
                     mounting=[mnt["id"], mnt["value_raw"], mnt["source"]])
    pitch, roll = (att["payload"]["pitch_deg"],
                   att["payload"]["roll_deg"])
    cells = []
    for cell in instr["payload"]["cells"]:
        if cell["kind"] == "none":
            cells.append({"cell": cell["cell"], "status": "excluded",
                          "reason": cell["reason"]})
            continue
        vector = _instr_vector(cell)
        if vector is None:
            cells.append({"cell": cell["cell"], "status": "pending",
                          "reason": "three-beam-candidate-undecided",
                          "candidate_count": len(cell["candidates"])})
            continue
        v = transforms.instrument_to_ship(
            vector, pitch, roll, mnt["numeric"])
        cells.append({"cell": cell["cell"], "status": "ok",
                      "x_stbd_ms": round(float(v[0]), 6),
                      "y_fore_ms": round(float(v[1]), 6),
                      "z_up_ms": round(float(v[2]), 6)})
    payload = {"ping": ping, "mounting_angle_deg": mnt["numeric"],
               "mounting_source": mnt["source"],
               "mounting_record_id": mnt["id"],
               "pitch_deg": pitch, "roll_deg": roll, "cells": cells}
    inputs = {"raw": _raw_fp(raw_att + [mnt]),
              "nodes": {"att": att["fingerprint"],
                        "instr": upstream["instr"]},
              "params": {"rule": RULE_VERSION,
                         "mounting_record_id": mnt["id"],
                         "mounting_source": mnt["source"]}}
    return _node(ctx, key, fp, "ok", payload, [], inputs)


def build_earth(ctx: Dict, ping: int) -> Dict:
    store, vid, params = ctx["store"], ctx["vid"], ctx["params"]
    att = ctx["nodes"][f"att/p{ping}"]
    instr = ctx["nodes"][f"instr/p{ping}"]
    ship = ctx["nodes"][f"ship/p{ping}"]
    key = f"earth/p{ping}"
    upstream = {"ship": ship["fingerprint"]}
    raw_att = (store.effective_raw(vid, kind="heading")
               + store.effective_raw(vid, kind="pitch")
               + store.effective_raw(vid, kind="roll"))
    if att["status"] == "gap":
        return _blocked(ctx, key, "attitude-gap", att, upstream, raw_att)
    if ship["status"] != "ok":
        fp = fingerprint(rule=RULE_VERSION, node=key,
                         att=att["fingerprint"], ship=ship["fingerprint"])
        payload = {"ping": ping, "status": "blocked",
                   "reason": "ship-pending: " + ship["status"]}
        return _node(ctx, key, fp, "blocked", payload,
                     ["等待三波束候选裁定后才能计算 earth 速度"],
                     {"nodes": {"att": att["fingerprint"],
                                "ship": ship["fingerprint"]},
                      "params": {"rule": RULE_VERSION}})
    dec = _latest(store, vid, "magnetic_declination")
    mnt = _latest(store, vid, "mounting_angle")
    bt = store.effective_raw(vid, kind="bottom_track", ping=ping)
    order = params["transform_order"]
    fp = fingerprint(rule=RULE_VERSION, node=key,
                     ship=ship["fingerprint"], att=att["fingerprint"],
                     order=order,
                     declination=[dec["id"], dec["value_raw"], dec["source"]],
                     bottom_track=_raw_fp(bt))
    heading_true = (att["payload"]["heading_mag_deg"]
                    + dec["numeric"]) % 360.0
    bt_map = {r["frame"]: r["numeric"] for r in bt}
    motion_ship = [bt_map.get("ship:stbd", 0.0),
                   bt_map.get("ship:fore", 0.0), 0.0]
    motion_earth = transforms.rot_z(heading_true) @ motion_ship
    pitch, roll = att["payload"]["pitch_deg"], att["payload"]["roll_deg"]

    def rotate_cell(cell, order_name):
        if cell["kind"] == "none":
            return {"cell": cell["cell"], "status": "excluded",
                    "reason": cell["reason"]}
        vector = _instr_vector(cell)
        if vector is None:
            return {"cell": cell["cell"], "status": "pending",
                    "reason": "three-beam-candidate-undecided"}
        if order_name == "standard":
            ship_v = transforms.instrument_to_ship(
                vector, pitch, roll, mnt["numeric"])
            earth_v = transforms.ship_to_earth(ship_v, heading_true)
        else:
            earth_v = transforms.instrument_to_earth_alternative(
                vector, heading_true, pitch, roll, mnt["numeric"])
        earth_v = earth_v - motion_earth
        return {"cell": cell["cell"], "status": "ok",
                "east_ms": round(float(earth_v[0]), 6),
                "north_ms": round(float(earth_v[1]), 6),
                "up_ms": round(float(earth_v[2]), 6)}

    cells = [rotate_cell(c, order) for c in instr["payload"]["cells"]]
    other = "tilt-after-heading" if order == "standard" else "standard"
    alternatives = [{
        "order": other, "adopted": False,
        "cells": [rotate_cell(c, other) for c in instr["payload"]["cells"]],
    }]
    payload = {"ping": ping, "order": order,
               "heading_true_deg": round(heading_true, 6),
               "declination_version": dec["source"],
               "declination_record_id": dec["id"],
               "bottom_track_record_ids": [r["id"] for r in bt],
               "ship_motion_earth_ms": [round(float(x), 6) for x in motion_earth],
               "cells": cells, "alternatives": alternatives}
    inputs = {"raw": _raw_fp(raw_att + [dec, mnt] + bt),
              "nodes": {"att": att["fingerprint"],
                        "ship": ship["fingerprint"]},
              "params": {"rule": RULE_VERSION, "transform_order": order,
                         "declination_record_id": dec["id"],
                         "declination_version": dec["source"]}}
    return _node(ctx, key, fp, "ok", payload, [], inputs)


def build_profile(ctx: Dict, ping: int) -> Dict:
    geo = ctx["geo"]
    earth = ctx["nodes"][f"earth/p{ping}"]
    beamq = ctx["nodes"][f"beamq/p{ping}"]
    key = f"profile/p{ping}"
    fp = fingerprint(rule=RULE_VERSION, node=key,
                     earth=earth["fingerprint"], beamq=beamq["fingerprint"],
                     cell_size_m=geo["cell_size_m"])
    earth_cells = {c["cell"]: c for c in earth["payload"].get("cells", [])}
    cells, excluded = [], []
    for cell in range(geo["cells"]):
        beam_reasons = sorted({
            reason for beam in beamq["payload"]["beams"]
            for reason in beam["cells"][cell]["reasons"]
        })
        depth_m = round((cell + 1) * geo["cell_size_m"], 3)
        ec = earth_cells.get(cell)
        if earth["status"] != "ok" or ec is None or ec["status"] != "ok":
            reasons = list(beam_reasons)
            if ec is not None and ec["status"] != "ok" and ec.get("reason"):
                reasons.append(ec["reason"])
            if earth["status"] != "ok":
                reasons.append(earth["payload"].get("reason",
                                                    earth["status"]))
            excluded.append({"cell": cell, "depth_m": depth_m,
                             "reasons": reasons or ["unavailable"]})
            continue
        if beam_reasons:
            excluded.append({"cell": cell, "depth_m": depth_m,
                             "reasons": beam_reasons})
            continue
        cells.append({"cell": cell, "depth_m": depth_m,
                      "east_ms": ec["east_ms"],
                      "north_ms": ec["north_ms"],
                      "up_ms": ec["up_ms"]})
    status = "ok" if earth["status"] == "ok" else earth["status"]
    payload = {"ping": ping, "status": status,
               "order": earth["payload"].get("order"),
               "declination_version": earth["payload"].get(
                   "declination_version"),
               "cells": cells, "excluded": excluded}
    inputs = {"nodes": {"earth": earth["fingerprint"],
                        "beamq": beamq["fingerprint"]},
              "params": {"rule": RULE_VERSION,
                         "cell_size_m": geo["cell_size_m"]}}
    return _node(ctx, key, fp, status, payload,
                 list(earth.get("diagnostics", [])), inputs)


BUILDERS = {"beamq": build_beamq, "att": build_att, "instr": build_instr,
            "ship": build_ship, "earth": build_earth,
            "profile": build_profile}


def build_version(store: Store, parent_id: int, label: str,
                  param_updates: Optional[Dict] = None,
                  raw_changes: Optional[List[Dict]] = None,
                  decisions: Optional[Dict] = None,
                  events: Optional[List] = None,
                  _fail_at: Optional[str] = None) -> Dict:
    """Build a complete derived version, then switch the current pointer.

    ``_fail_at`` (test-only) injects a crash inside the transaction.
    Returns {"version": vid, "changed": [node_keys actually recomputed]}.
    """
    parent = store.get_version(parent_id)
    params = dict(parent["params"])
    params.update(param_updates or {})
    decisions = dict(decisions or {})
    parent_nodes = {n["node_key"]: n for n in store.nodes(parent_id)}
    with store.transaction():
        vid = store.insert_version(parent_id, label, params)
        if _fail_at == "after_version_row":
            raise RuntimeError("simulated crash after version row")
        for change in raw_changes or []:
            if change.get("supersede"):
                store.supersede_raw(change["supersede"],
                                    _prepare_raw(change["record"]), vid)
            else:
                store.insert_raw(
                    {**_prepare_raw(change["record"]), "valid_from": vid})
        if _fail_at == "after_raw":
            raise RuntimeError("simulated crash after raw change")
        ctx = {"store": store, "vid": vid, "params": params,
               "decisions": decisions, "parent_nodes": parent_nodes,
               "nodes": {}, "geo": geometry_at(store, vid)}
        changed = []
        for ping in pings_at(store, vid):
            for kind in NODE_KINDS:
                node = BUILDERS[kind](ctx, ping)
                key = node["node_key"]
                prior = parent_nodes.get(key)
                prior_adopted = (json.loads(prior["payload"]).get("adopted")
                                 if prior else None)
                carried = (
                    prior is not None
                    and prior["fingerprint"] == node["fingerprint"]
                    and prior_adopted == node["payload"].get("adopted")
                    and key not in decisions)
                if carried:
                    node["payload"] = json.loads(prior["payload"])
                    node["diagnostics"] = json.loads(prior["diagnostics"])
                    node["inputs"]["carried_from"] = prior["id"]
                else:
                    changed.append(key)
                store.insert_node(node)
                ctx["nodes"][key] = node
        if _fail_at == "before_pointer":
            raise RuntimeError("simulated crash before pointer switch")
        for kind, payload in events or []:
            store.insert_event(kind, vid, payload)
        store.complete_version(vid)
    return {"version": vid, "changed": sorted(changed)}


def diff_versions(store: Store, old_vid: int, new_vid: int) -> Dict:
    """Node-level diff used to present a resolvable conflict."""
    old = {n["node_key"]: n for n in store.nodes(old_vid)}
    new = {n["node_key"]: n for n in store.nodes(new_vid)}
    diff = {}
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a is None or b is None or a["fingerprint"] != b["fingerprint"] \
                or a["payload"].get("adopted") != b["payload"].get("adopted"):
            diff[key] = {
                "base": None if a is None else
                {"fingerprint": a["fingerprint"],
                 "adopted": a["payload"].get("adopted"),
                 "status": a["status"]},
                "current": None if b is None else
                {"fingerprint": b["fingerprint"],
                 "adopted": b["payload"].get("adopted"),
                 "status": b["status"]}}
    return diff


def save_decision(store: Store, session_id: str, node_key: str,
                  adopted: int) -> Dict:
    """Save a reviewer decision with optimistic-concurrency check."""
    session = store.get_session(session_id)
    current = store.current_version()
    if session["base_version"] != current:
        raise ConflictError(
            session, current,
            diff_versions(store, session["base_version"], current))
    payload = {"step_id": f"decision:{session['id']}:{node_key}:{adopted}",
               "actor": "reviewer", "user": session["user"],
               "node": node_key, "adopted_candidate": adopted,
               "rule": "three-beam-candidates-are-kept-distinct",
               "justification": "三波束解与四波束解不混为一值；"
               "未选候选继续保留，确认不是不可逆覆盖"}
    return build_version(
        store, current,
        f"decision:{session['user']}:{node_key}=cand{adopted}",
        decisions={node_key: adopted}, events=[("decision", payload)])


def explain_node(store: Store, vid: int, node_key: str) -> Dict:
    """Why a node is still valid, or why it had to be recomputed."""
    version = store.get_version(vid)
    node = store.node(vid, node_key)
    if node is None:
        raise KeyError(node_key)
    answer = {"node": node["id"], "node_key": node_key,
              "fingerprint": node["fingerprint"], "status": node["status"]}
    if "carried_from" in node["inputs"]:
        answer["verdict"] = "still-valid"
        answer["reason"] = ("指纹与父版本一致，输入未变化，结果沿用 "
                            + node["inputs"]["carried_from"])
        return answer
    if version["parent_id"] is None:
        answer["verdict"] = "new"
        answer["reason"] = "基线首次推导"
        return answer
    parent = store.node(version["parent_id"], node_key)
    if parent is None:
        answer["verdict"] = "new"
        answer["reason"] = "父版本无此节点"
        return answer
    answer["verdict"] = "recomputed"
    causes = []
    p_raw = {r[0] for r in parent["inputs"].get("raw", [])}
    n_raw = {r[0] for r in node["inputs"].get("raw", [])}
    for rid in sorted(n_raw - p_raw):
        causes.append(f"新增/替换来源记录 {rid}")
    p_params = parent["inputs"].get("params", {})
    n_params = node["inputs"].get("params", {})
    for key in sorted(set(p_params) | set(n_params)):
        if p_params.get(key) != n_params.get(key):
            causes.append(f"参数 {key}: {p_params.get(key)!r} -> "
                          f"{n_params.get(key)!r}")
    p_nodes = parent["inputs"].get("nodes", {})
    n_nodes = node["inputs"].get("nodes", {})
    for key in sorted(set(p_nodes) | set(n_nodes)):
        if p_nodes.get(key) != n_nodes.get(key):
            causes.append(f"上游节点 {key} 已变化")
    answer["reason"] = "; ".join(causes) or "指纹不一致"
    answer["parent_node"] = parent["id"]
    return answer
