"""Dependency-graph recomputation: raw beams -> instrument -> ship -> earth.

Every node is content addressed (uid + fingerprint). The runner is pure:
callers decide which unchanged nodes are reused by comparing fingerprints with
the parent commit, so only affected nodes are recomputed/replaced.

Counters (three- vs four-beam) are never merged into one value. When the
four-beam error velocity is too large, all legal triplet candidates stay
"unadopted" until a human decision event selects one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from adcp import geometry
from adcp.attitude import align
from adcp.fingerprint import config_block, fingerprint
from adcp.frames import mount_matrix, attitude_matrix
from adcp.quality import judge_cell
from adcp.bottom import correct_water_velocity
import numpy as np

DEFAULT_ERROR_VEL_LIMIT = 0.15  # m/s


def _q_ok(rec: Optional[dict]) -> bool:
    return bool(rec) and rec.get("ok", False) and rec.get("canonical") is not None


def _canon(rec: dict) -> float:
    return float(rec["canonical"])


def _node(uid: str, kind: str, deps: List[dict], refs: List[dict],
          value: Dict[str, Any], extra_parts: Optional[List[Any]] = None
          ) -> Dict[str, Any]:
    parts: List[Any] = [kind, uid, deps, refs, value]
    if extra_parts:
        parts.extend(extra_parts)
    return {"uid": uid, "kind": kind, "deps": deps, "refs": refs,
            "value": value, "fp": fingerprint(parts)}


def _config_node(dataset: dict) -> Dict[str, Any]:
    cfg = config_block(dataset)
    return _node("config", "config", [], [], {"config": cfg}), cfg


def _attitude_node(dataset: dict, ping: dict) -> Dict[str, Any]:
    att = dataset["attitude"]
    res = align(att["times"], [_canon(r) for r in att["heading"]],
                [_canon(r) for r in att["pitch"]],
                [_canon(r) for r in att["roll"]], float(ping["time"]),
                att.get("max_gap"))
    refs = [{"ref": att["heading"][i].get("ref"), "role": "heading_sample"}
            for i in res.get("bracket", [])]
    val = {"ping": ping["ping"], "time": ping["time"], **res}
    return _node(f"att-p{ping['ping']}", "attitude_alignment",
                 [{"kind": "config", "uid": "config"}], refs, val,
                 [att.get("version", "att-1")])


def _cell_beam_records(cell: dict) -> Dict[str, List]:
    return {"beam_velocity": cell["beam_velocity"],
            "correlation": cell["correlation"],
            "amplitude": cell["amplitude"],
            "bad_flags": cell["bad_flags"]}


def _refs_for_cell(cell: dict) -> List[dict]:
    refs: List[dict] = []
    for role in ("beam_velocity", "correlation", "amplitude"):
        for k, rec in enumerate(cell[role]):
            refs.append({"ref": rec.get("ref"), "role": f"{role}_b{k}"})
    return refs


def _solution_candidates(cell: dict, judged: dict, geom: dict,
                         err_limit: float,
                         decision: Optional[dict] = None) -> Dict[str, Any]:
    vel = [_canon(r) if _q_ok(r) else None for r in cell["beam_velocity"]]
    candidates: List[Dict[str, Any]] = []
    valid = judged["valid_beams"]
    resolution = "rejected"
    adopted: Optional[str] = None

    if len(valid) == 4:
        four = geometry.solve_four([_canon(r) for r in cell["beam_velocity"]],
                                   geom)
        fid = "four"
        candidates.append({"id": fid, **four, "adopted": False})
        ambiguous = abs(four["error_velocity"]) > err_limit
        for triplet in geometry.all_triplets():
            sol = geometry.solve_three(
                [_canon(r) for r in cell["beam_velocity"]], triplet, geom)
            candidates.append({"id": "t-" + "".join(str(k) for k in triplet),
                               **sol, "adopted": False})
        if ambiguous:
            resolution = "ambiguous_multi"
        else:
            resolution = "four"
            adopted = fid
    elif len(valid) == 3:
        sol = geometry.solve_three(
            [v if v is not None else 0.0 for v in vel], valid, geom)
        tid = "t-" + "".join(str(k) for k in valid)
        candidates.append({"id": tid, **sol, "adopted": True})
        resolution = "three"
        adopted = tid
    if decision is not None:
        chosen = decision.get("candidate_id")
        if any(c["id"] == chosen for c in candidates):
            resolution = "decided_" + chosen[:1]
            adopted = chosen
    for c in candidates:
        c["adopted"] = c["id"] == adopted
    return {"resolution": resolution, "adopted": adopted,
            "candidates": candidates, "valid_beams": valid,
            "decision": decision}


def run(dataset: dict) -> Dict[str, Any]:
    """Produce the full derived node graph for one dataset view."""
    cfg_node, cfg = _config_node(dataset)
    geom = cfg["geometry"]
    mount = cfg["mount"]
    decl = cfg["declination"]
    order = cfg["transform_order"]
    err_limit = float(dataset.get("error_velocity_limit",
                                  DEFAULT_ERROR_VEL_LIMIT))
    nodes: Dict[str, Dict[str, Any]] = {cfg_node["uid"]: cfg_node}
    excluded: List[dict] = []
    depth_cells = {c["cell"]: c for c in dataset["depth_cells"]}

    m_mount = mount_matrix(mount)
    decisions = {(d["ping"], d["cell"]): d
                 for d in dataset.get("decisions", [])}

    for ping in dataset["pings"]:
        pno = ping["ping"]
        att_node = _attitude_node(dataset, ping)
        nodes[att_node["uid"]] = att_node
        aligned = att_node["value"].get("aligned", False)
        attitude = None
        if aligned:
            attitude = {
                "heading_deg": att_node["value"]["heading_deg"],
                "pitch_deg": att_node["value"]["pitch_deg"],
                "roll_deg": att_node["value"]["roll_deg"],
                "declination_deg": float(decl["deg"]),
            }
            m_att = attitude_matrix(attitude["heading_deg"],
                                    attitude["pitch_deg"], attitude["roll_deg"],
                                    attitude["declination_deg"])

        # bottom track
        bt = ping.get("bottom_track")
        bt_node = _bottom_node(pno, bt, cfg["thresholds"], geom,
                               m_mount, attitude,
                               m_att if aligned else None, order)
        nodes[bt_node["uid"]] = bt_node

        for cell in ping["cells"]:
            cno = cell["cell"]
            dcell = depth_cells[cno]
            depth_rec = dcell["bin_depth"]
            base_refs = _refs_for_cell(cell)
            qdeps = [{"kind": "config", "uid": "config"}]
            judged = judge_cell(cell, cfg["thresholds"])
            depth_ok = _q_ok(depth_rec)
            qval = {"ping": pno, "cell": cno, **judged,
                    "bin_depth": depth_rec,
                    "depth_valid": depth_ok}
            qnode = _node(f"q-p{pno}-c{cno}", "beam_quality", qdeps,
                          base_refs, qval)
            nodes[qnode["uid"]] = qnode

            decision = decisions.get((pno, cno))
            sols = _solution_candidates(cell, judged, geom, err_limit,
                                        decision)
            decision_parts = [decision] if decision else []
            bnode = _node(
                f"b-p{pno}-c{cno}", "beam_solution",
                [{"kind": "beam_quality", "uid": qnode["uid"],
                  "fp": qnode["fp"]},
                 {"kind": "config", "uid": "config"}],
                base_refs,
                {"ping": pno, "cell": cno, **{
                    k: v for k, v in sols.items() if k != "decision"},
                 "error_velocity_limit": err_limit,
                 "decision": decision},
                [geom.get("version", "geom-1")] + decision_parts)
            nodes[bnode["uid"]] = bnode

            exclude_reasons: List[str] = []
            if not depth_ok:
                exclude_reasons.append("invalid_depth_cell")
            if judged["status"] == "rejected":
                exclude_reasons.append("fewer_than_three_beams")
            if sols["resolution"] == "ambiguous_multi":
                exclude_reasons.append("ambiguous_multi_candidate")

            adopted_cand = next((c for c in sols["candidates"]
                                 if c["id"] == sols["adopted"]), None)
            if adopted_cand and adopted_cand.get("velocity_instrument"):
                vi = np.asarray(adopted_cand["velocity_instrument"])
                inode = _node(
                    f"i-p{pno}-c{cno}", "instrument_velocity",
                    [{"kind": "beam_solution", "uid": bnode["uid"],
                      "fp": bnode["fp"]}],
                    base_refs,
                    {"ping": pno, "cell": cno,
                     "velocity_instrument": vi.tolist(),
                     "resolution": sols["resolution"],
                     "candidate_id": sols["adopted"],
                     "unit": "m/s", "frame": "instrument"})
                nodes[inode["uid"]] = inode

                vs = (m_mount @ vi)
                snode = _node(
                    f"s-p{pno}-c{cno}", "ship_velocity",
                    [{"kind": "instrument_velocity", "uid": inode["uid"],
                      "fp": inode["fp"]},
                     {"kind": "config", "uid": "config"}],
                    base_refs,
                    {"ping": pno, "cell": cno,
                     "velocity_ship": vs.tolist(),
                     "mount_version": mount.get("version"),
                     "unit": "m/s", "frame": "ship"})
                nodes[snode["uid"]] = snode

                if not aligned:
                    exclude_reasons.append(att_node["value"].get(
                        "reason", "attitude_unavailable"))
                else:
                    ve = m_att @ vs if order == "mount_first" \
                        else m_mount @ (m_att @ vi)
                    enode = _node(
                        f"e-p{pno}-c{cno}", "earth_velocity",
                        [{"kind": "ship_velocity", "uid": snode["uid"],
                          "fp": snode["fp"]},
                         {"kind": "attitude_alignment",
                          "uid": att_node["uid"], "fp": att_node["fp"]},
                         {"kind": "config", "uid": "config"}],
                        base_refs + att_node["refs"],
                        {"ping": pno, "cell": cno,
                         "velocity_earth": ve.tolist(),
                         "declination_version": decl.get("version"),
                         "true_heading_deg":
                             (attitude["heading_deg"]
                              + attitude["declination_deg"]) % 360.0,
                         "transform_order": order,
                         "unit": "m/s", "frame": "earth_ned"})
                    nodes[enode["uid"]] = enode

                    bt_ok = bt_node["value"].get("bottom_track_ok", False)
                    if not bt_ok:
                        exclude_reasons.append("bottom_track_unavailable")
                    else:
                        ship_earth = np.asarray(
                            bt_node["value"]["ship_velocity_earth"])
                        absv = np.asarray(ve) - ship_earth
                        anode = _node(
                            f"a-p{pno}-c{cno}", "absolute_velocity",
                            [{"kind": "earth_velocity", "uid": enode["uid"],
                              "fp": enode["fp"]},
                             {"kind": "bottom_track", "uid": bt_node["uid"],
                              "fp": bt_node["fp"]}],
                            base_refs + bt_node.get("refs", []),
                            {"ping": pno, "cell": cno,
                             "velocity_earth_absolute": absv.tolist(),
                             "ship_velocity_earth": ship_earth.tolist(),
                             "unit": "m/s", "frame": "earth_absolute"})
                        nodes[anode["uid"]] = anode

            if exclude_reasons:
                excluded.append({"ping": pno, "cell": cno,
                                 "reasons": sorted(set(exclude_reasons)),
                                 "quality_node": qnode["uid"],
                                 "solution_node": bnode["uid"],
                                 "attitude_node": att_node["uid"],
                                 "bottom_node": bt_node["uid"]})

    exnode = _node("excluded", "excluded_cells",
                   [{"kind": "config", "uid": "config"}], [],
                   {"excluded": excluded})
    nodes[exnode["uid"]] = exnode

    profile = _profile_node(dataset, nodes)
    nodes[profile["uid"]] = profile

    return {"nodes": list(nodes.values()), "excluded": excluded,
            "summary": {"node_count": len(nodes),
                        "excluded_count": len(excluded)}}


def _bottom_node(pno: int, bt: Optional[dict], thresholds: dict,
                 geom: dict, m_mount: np.ndarray, attitude: Optional[dict],
                 m_att: Optional[np.ndarray], order: str) -> Dict[str, Any]:
    deps = [{"kind": "config", "uid": "config"}]
    refs: List[dict] = []
    value: Dict[str, Any] = {"ping": pno, "bottom_track_ok": False}
    if not bt:
        value["reason"] = "no_bottom_track"
        return _node(f"bt-p{pno}", "bottom_track", deps, refs, value)
    judged = judge_cell(bt, thresholds)
    refs = _refs_for_cell(bt)
    value["beam_quality"] = judged
    if len(judged["valid_beams"]) < 3:
        value["reason"] = "fewer_than_three_beams"
        return _node(f"bt-p{pno}", "bottom_track", deps, refs, value)
    if len(judged["valid_beams"]) == 4:
        sol = geometry.solve_four([_canon(r) for r in bt["beam_velocity"]],
                                  geom)
    else:
        sol = geometry.solve_three(
            [_canon(r) for r in bt["beam_velocity"]],
            judged["valid_beams"], geom)
    vi = np.asarray(sol["velocity_instrument"])
    vs = m_mount @ vi
    value["velocity_instrument"] = vi.tolist()
    value["velocity_ship"] = vs.tolist()
    if attitude is None or m_att is None:
        value["reason"] = "attitude_unavailable"
        return _node(f"bt-p{pno}", "bottom_track", deps, refs, value)
    ve = m_att @ vs if order == "mount_first" else m_mount @ (m_att @ vi)
    value["ship_velocity_earth"] = ve.tolist()
    value["bottom_track_ok"] = True
    return _node(f"bt-p{pno}", "bottom_track", deps, refs, value)


def _profile_node(dataset: dict, nodes: Dict[str, dict]) -> Dict[str, Any]:
    by_cell: Dict[int, list] = {}
    abs_deps = []
    for n in nodes.values():
        if n["kind"] == "absolute_velocity":
            by_cell.setdefault(n["value"]["cell"], []).append(
                n["value"]["velocity_earth_absolute"])
            abs_deps.append({"kind": "absolute_velocity", "uid": n["uid"],
                             "fp": n["fp"]})
    cells = []
    for c in sorted(by_cell):
        arr = np.asarray(by_cell[c])
        cells.append({"cell": c,
                      "mean_velocity_earth_absolute": arr.mean(axis=0).tolist(),
                      "sample_count": len(arr),
                      "unit": "m/s"})
    return _node("profile", "vertical_profile", abs_deps, [],
                 {"cells": cells})
