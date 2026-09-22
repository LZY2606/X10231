"""Dependency-graph derivation: sources -> beam XYZ -> ship -> earth.

Propagation is fingerprint based: a node is recomputed only when its rule
version, beam geometry, transform order or a dependency source version
changes. Every node stores *why it is valid / why it was recomputed*, and
indeterminate states keep all legal candidates instead of collapsing to one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import fingerprint as fp
from . import geometry as geo
from . import transforms as tf
from .attitude import AttitudeSample, attitude_at
from .fingerprint import geometry_fingerprint, node_fingerprint
from .geometry import TransducerConfig

MAX_ATTITUDE_GAP_S = 2.5  # pings are 1 s apart; a missing span beyond this is a gap


def _src(versions: Dict[str, str], store, sid: str) -> Optional[dict]:
    vid = versions.get(sid)
    return store.get_source_version(vid) if vid else None


def _ping_ids(versions, store) -> List[str]:
    rec = _src(versions, store, "src:index:pings")
    if rec:
        return list(rec["payload"]["pings"])
    out = []
    for sid in sorted(versions):
        if sid.startswith("src:beam:"):
            out.append(sid.split(":", 2)[2])
    return out


def derive_set(store, base_versions: Dict[str, str], tag: str = "",
               branch: str = "main", parent_set_id: Optional[str] = None,
               transform_order: str = "RDI_HRP",
               crash_before_commit: bool = False) -> Tuple[str, List[dict], dict]:
    """Run the full DAG, write a set, commit and flip the branch pointer.

    Returns (set_id, nodes, summary). Reuses parent nodes whose fingerprint is
    unchanged so incremental source edits only propagate to affected nodes.
    """
    cfg_rec = _src(base_versions, store, "src:config:transducer")
    cfg = TransducerConfig.from_dict(cfg_rec["payload"]) if cfg_rec else TransducerConfig()
    geom_fp = geometry_fingerprint(cfg.to_dict())

    mount = _src(base_versions, store, "src:mount:angles")
    mount_p = mount["payload"] if mount else {}
    decl = _src(base_versions, store, "src:declination:site")
    decl_v = float(decl["payload"]["declination_deg"]) if decl else 0.0
    order_rec = _src(base_versions, store, "src:config:transform_order")
    order = order_rec["payload"]["order"] if order_rec else transform_order

    parent_nodes = {n["node_key"]: n for n in
                    store.list_nodes(parent_set_id)} if parent_set_id else {}
    changed_keys: List[str] = []
    nodes_out: List[dict] = []
    seq = 0
    counts = {"four": 0, "three": 0, "indeterminate": 0, "gap": 0,
              "invalid_bin": 0, "reused": 0}

    ping_ids = _ping_ids(base_versions, store)

    for pid in ping_ids:
        beams = _src(base_versions, store, "src:beam:" + pid)
        corr = _src(base_versions, store, "src:correlation:" + pid)
        amp = _src(base_versions, store, "src:amplitude:" + pid)
        flags = _src(base_versions, store, "src:badflag:" + pid)
        atti = _src(base_versions, store, "src:attitude:" + pid)
        bt = _src(base_versions, store, "src:bottomtrack:" + pid)
        binmeta = _src(base_versions, store, "src:bins:" + pid)
        if not beams:
            continue
        payload = beams["payload"]
        t = float(payload.get("t", 0.0))
        nbins = len(payload["radials_mm_s"])
        depth_m = (binmeta["payload"].get("bin_depth_m")
                   if binmeta else [None] * nbins)
        valid_mask = (binmeta["payload"].get("cell_valid")
                      if binmeta else [True] * nbins)
        bt_range_m = bt["payload"].get("bottom_range_m") if bt else None

        samples = [AttitudeSample.from_dict(s)
                   for s in atti["payload"]["samples"]] if atti else []
        att = attitude_at(samples, t, MAX_ATTITUDE_GAP_S)

        for b in range(nbins):
            base_deps = {
                "config": cfg_rec["version_id"] if cfg_rec else "",
                "mount": mount["version_id"] if mount else "",
                "declination": decl["version_id"] if decl else "",
                "transform_order": order_rec["version_id"] if order_rec else "default",
                "beam": beams["version_id"],
                "correlation": corr["version_id"] if corr else "",
                "amplitude": amp["version_id"] if amp else "",
                "badflag": flags["version_id"] if flags else "",
                "attitude": atti["version_id"] if atti else "",
                "bins": binmeta["version_id"] if binmeta else "",
            }
            radials = payload["radials_mm_s"][b]
            quality = geo.classify_beams(
                radials,
                corr["payload"]["correlation_counts"][b] if corr else [None] * 4,
                amp["payload"]["amplitude_counts"][b] if amp else [None] * 4,
                flags["payload"]["bad_beam"][b] if flags else [False] * 4,
                cfg)

            is_invalid_bin = (valid_mask is not None
                              and b < len(valid_mask) and not valid_mask[b])
            beyond_bottom = (bt_range_m is not None and depth_m is not None
                             and depth_m[b] is not None
                             and float(depth_m[b]) >= float(bt_range_m))
            excluded = bool(is_invalid_bin or beyond_bottom)
            exclude_reasons = []
            if is_invalid_bin:
                exclude_reasons.append("invalid_depth_cell")
            if beyond_bottom:
                exclude_reasons.append("beyond_bottom_track")

            good = quality.good_beams
            candidates: List[geo.XYZSolution] = []
            if len(good) == 4:
                candidates = [geo.solve_four(
                    [radials[i] for i in range(4)], cfg)]
                kind = "four"
            elif len(good) == 3:
                sparse = [radials[i] if i in good else None for i in range(4)]
                candidates = geo.solve_three_candidates(sparse, cfg)
                kind = "three"
            else:
                kind = "indeterminate"

            if att.status in ("gap", "outside"):
                earth_kind = "earth_unresolved"
            else:
                earth_kind = "earth"

            for cand in candidates:
                seq += 1
                cid = cand.candidate_id
                node_key = "%s/bin%02d/%s" % (pid, b, cid)
                deps = dict(base_deps,
                            candidate=cid,
                            bottomtrack=bt["version_id"] if bt else "")
                extra = {"kind_four": kind == "four",
                         "excluded": excluded,
                         "attitude_status": att.status}
                nfp = node_fingerprint("xyz", (pid, "bin%02d" % b), geom_fp,
                                       order, deps, extra)
                v_ship = v_earth = None
                steps: list = []
                if att.status not in ("gap", "outside"):
                    mh = float(mount_p.get("mount_heading_deg", 0.0))
                    mp = float(mount_p.get("mount_pitch_deg", 0.0))
                    mr = float(mount_p.get("mount_roll_deg", 0.0))
                    vs, ve, st_s, st_e = tf.chain(
                        np.array(cand.xyz_mm_s), mh, mp, mr,
                        att.heading_deg, att.pitch_deg or 0.0,
                        att.roll_deg or 0.0, decl_v, order)
                    v_ship = [float(x) for x in vs]
                    v_earth = [float(x) for x in ve]
                    steps = [st_s.to_dict(), st_e.to_dict()]
                node_payload = {
                    "ping": pid, "bin": b, "candidate_id": cid,
                    "frame_path": ["beam", "instrument", "ship", "earth"],
                    "xyz_mm_s": list(cand.xyz_mm_s),
                    "ship_mm_s": v_ship, "earth_mm_s": v_earth,
                    "error_velocity_mm_s": cand.error_velocity_mm_s,
                    "assumption": cand.assumption,
                    "quality": quality.to_dict(),
                    "attitude": att.to_dict(),
                    "depth_m": None if not depth_m or depth_m[b] is None
                                else float(depth_m[b]),
                    "bottom_range_m": bt_range_m,
                    "excluded": excluded,
                    "exclude_reasons": exclude_reasons,
                    "transform_steps": steps,
                    "transform_order": order,
                    "geometry_fp": geom_fp,
                    "t_s": t,
                    "unit": "mm/s",
                    "datum": "x=starboard/E,y=forward/N,z=up",
                }
                status = "ok"
                if excluded:
                    status = "excluded"
                elif att.status in ("gap", "outside"):
                    status = "attitude_" + att.status
                node = {"node_key": node_key, "kind": kind, "ping": pid,
                        "bin": b, "candidate_id": cid, "fingerprint": nfp,
                        "deps": deps, "payload": node_payload, "seq": seq,
                        "status": status}
                if node_key in parent_nodes and \
                        parent_nodes[node_key]["fingerprint"] == nfp:
                    node = parent_nodes[node_key]
                    counts["reused"] += 1
                else:
                    changed_keys.append(node_key)
                nodes_out.append(node)
                counts[kind] += 1
                if excluded:
                    counts["invalid_bin"] += 1
                if att.status in ("gap", "outside"):
                    counts["gap"] += 1

            if not candidates:
                seq += 1
                node_key = "%s/bin%02d/indeterminate" % (pid, b)
                deps = dict(base_deps, bottomtrack=bt["version_id"] if bt else "")
                nfp = node_fingerprint("indeterminate", (pid, "bin%02d" % b),
                                       geom_fp, order, deps)
                node = {"node_key": node_key, "kind": "indeterminate",
                        "ping": pid, "bin": b, "candidate_id": None,
                        "fingerprint": nfp, "deps": deps, "seq": seq,
                        "status": "indeterminate" if not excluded else "excluded",
                        "payload": {
                            "ping": pid, "bin": b,
                            "quality": quality.to_dict(),
                            "good_beams": list(good),
                            "reason": "fewer than 3 valid beams; no legal 3D fix",
                            "attitude": att.to_dict(),
                            "excluded": excluded,
                            "exclude_reasons": exclude_reasons,
                            "depth_m": None if not depth_m or depth_m[b] is None
                                        else float(depth_m[b]),
                            "candidates": [], "t_s": t,
                            "unit": "mm/s"}}
                if node_key in parent_nodes and \
                        parent_nodes[node_key]["fingerprint"] == nfp:
                    node = parent_nodes[node_key]
                    counts["reused"] += 1
                else:
                    changed_keys.append(node_key)
                nodes_out.append(node)
                counts["indeterminate"] += 1
                if excluded:
                    counts["invalid_bin"] += 1
                if att.status in ("gap", "outside"):
                    counts["gap"] += 1

    set_id = store.begin_set(base_versions, tag, branch, parent_set_id,
                             fail_before_commit=crash_before_commit)
    store.add_nodes_bulk(set_id, nodes_out)
    set_fp = fp.set_fingerprint(nodes_out, tag)
    store.commit_set(set_id, branch,
                     event_payload={"set_id": set_id, "set_fingerprint": set_fp,
                                    "changed_nodes": changed_keys,
                                    "transform_order": order,
                                    "summary": counts})
    counts["total"] = len(nodes_out)
    counts["changed"] = len(changed_keys)
    counts["set_id"] = set_id
    counts["set_fingerprint"] = set_fp
    return set_id, nodes_out, counts


def explain_node(store, set_id: str, node_key: str,
                 parent_set_id: Optional[str] = None) -> dict:
    """Explain why a node is still valid or why it had to be recomputed."""
    node = store.get_node(set_id, node_key)
    if not node:
        return {"node_key": node_key, "valid": False, "reason": "missing"}
    if not parent_set_id:
        return {"node_key": node_key, "valid": True,
                "reason": "first derivation", "fingerprint": node["fingerprint"]}
    old = store.get_node(parent_set_id, node_key)
    if not old:
        return {"node_key": node_key, "valid": True,
                "reason": "new node (no parent value)",
                "fingerprint": node["fingerprint"]}
    if old["fingerprint"] == node["fingerprint"]:
        changed_deps = {}
        return {"node_key": node_key, "valid": True, "reused": True,
                "reason": "fingerprint unchanged; dependencies identical",
                "fingerprint": node["fingerprint"],
                "deps": node["deps"]}
    changed_deps = {k: {"old": old["deps"].get(k), "new": node["deps"].get(k)}
                    for k in set(old["deps"]) | set(node["deps"])
                    if old["deps"].get(k) != node["deps"].get(k)}
    return {"node_key": node_key, "valid": True, "reused": False,
            "reason": "recomputed: changed dependencies",
            "fingerprint": node["fingerprint"],
            "changed_dependencies": changed_deps}
