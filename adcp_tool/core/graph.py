"""Build the derivation graph and propagate changes along it.

Graph (per ping, per depth cell)::

    sources(beam/corr/amp/badflag)
        -> quality -> beam-solution -> instrument -> ship -> earth
    attitude samples + gaps -> attitude(ping) -^
    geometry / mount / declination / transform-order (resource fingerprints) -^
    bottom track sources -> bt-quality -> bt-beam-solution -> bt-instrument
        -> bt-ship -> bt-earth  (uses its own attitude sample, never interpolated
        across a gap)
"""
from __future__ import annotations

import math

from . import attitude as att
from . import geometry as geo
from . import quality as qua
from . import rotation as rot
from .fingerprint import fingerprint
from .nodes import (
    BT_ATT_RULE,
    DECLINATION_RULE,
    MOUNT_RULE,
    quality_node,
    transform_nodes_for_cell,
)
from .units import FRAME_EARTH, FRAME_INSTRUMENT, FRAME_SHIP

CONFIG_RULE = "transform-config-v1"
GEOM_RESOURCE_RULE = geo.GEOMETRY_RULE + "#resource"


def normalize_config(raw_cfg):
    cfg = {
        "min_correlation": float(raw_cfg.get("min_correlation", 64.0)),
        "min_amplitude_db": float(raw_cfg.get("min_amplitude_db", 30.0)),
        "error_velocity_threshold": float(raw_cfg.get("error_velocity_threshold", 0.05)),
        "declination_deg": float(raw_cfg.get("declination_deg", 0.0)),
        "declination_version": raw_cfg.get("declination_version", "decl-initial-v1"),
        "declination_basis": raw_cfg.get("declination_basis", "magnetic"),
        "mount_heading_deg": float(raw_cfg.get("mount_heading_deg", 0.0)),
        "mount_pitch_deg": float(raw_cfg.get("mount_pitch_deg", 0.0)),
        "mount_roll_deg": float(raw_cfg.get("mount_roll_deg", 0.0)),
        "transform_order": raw_cfg.get("transform_order", "hpr-v1"),
        "gap_factor": float(raw_cfg.get("gap_factor", 1.5)),
        "beam_angle_deg": float(raw_cfg.get("beam_angle_deg", geo.DEFAULT_BEAM_ANGLE_DEG)),
        "azimuths_deg": list(raw_cfg.get("azimuths_deg", geo.DEFAULT_AZIMUTHS_DEG)),
        "geometry_version": raw_cfg.get("geometry_version", geo.GEOMETRY_VERSION),
    }
    cfg["mount"] = {
        "heading_deg": cfg["mount_heading_deg"],
        "pitch_deg": cfg["mount_pitch_deg"],
        "roll_deg": cfg["mount_roll_deg"],
    }
    if cfg["transform_order"] not in rot.ORDERS:
        raise ValueError("unsupported transform_order: %r" % cfg["transform_order"])
    return cfg


def _config_basis(cfg):
    return {k: cfg[k] for k in sorted(cfg) if k != "mount"}


def resource_nodes(cfg):
    geom = geo.BeamGeometry(cfg["beam_angle_deg"], tuple(cfg["azimuths_deg"]), cfg["geometry_version"])
    geom_fp = fingerprint(GEOM_RESOURCE_RULE, geom.basis())
    mount_fp = fingerprint(MOUNT_RULE, {
        "heading_deg": cfg["mount_heading_deg"],
        "pitch_deg": cfg["mount_pitch_deg"],
        "roll_deg": cfg["mount_roll_deg"],
        "order": cfg["transform_order"],
    })
    decl_fp = fingerprint(DECLINATION_RULE, {
        "declination_deg": cfg["declination_deg"],
        "version": cfg["declination_version"],
        "basis_name": cfg["declination_basis"],
    })
    order_fp = fingerprint(CONFIG_RULE, {
        "transform_order": cfg["transform_order"],
        "min_correlation": cfg["min_correlation"],
        "min_amplitude_db": cfg["min_amplitude_db"],
        "error_velocity_threshold": cfg["error_velocity_threshold"],
    })
    nodes = [
        {"node_id": "resource:geometry", "kind": "resource_geometry", "ping": None, "cell": None,
         "status": "ok", "rule": GEOM_RESOURCE_RULE, "upstream": [], "depends_on": [], "source_ids": [],
         "fingerprint": geom_fp, "value": geom.basis()},
        {"node_id": "resource:mount", "kind": "resource_mount", "ping": None, "cell": None,
         "status": "ok", "rule": MOUNT_RULE, "upstream": [], "depends_on": [], "source_ids": [],
         "fingerprint": mount_fp, "value": {"mount": cfg["mount"], "order": cfg["transform_order"]}},
        {"node_id": "resource:declination", "kind": "resource_declination", "ping": None, "cell": None,
         "status": "ok", "rule": DECLINATION_RULE, "upstream": [], "depends_on": [], "source_ids": [],
         "fingerprint": decl_fp,
         "value": {"declination_deg": cfg["declination_deg"],
                   "version": cfg["declination_version"], "basis": cfg["declination_basis"]}},
        {"node_id": "resource:order", "kind": "resource_order", "ping": None, "cell": None,
         "status": "ok", "rule": CONFIG_RULE, "upstream": [], "depends_on": [], "source_ids": [],
         "fingerprint": order_fp,
         "value": {"transform_order": cfg["transform_order"]}},
    ]
    return nodes, {"geom": geom, "geom_fp": geom_fp, "mount_fp": mount_fp,
                   "decl_fp": decl_fp, "order_fp": order_fp}


def _attitude_node(p, att_result, cfg, source_id):
    basis = {
        "query_time": p["time"],
        "result_keys": sorted(att_result.keys()),
        "heading_deg": att_result.get("heading_deg"),
        "pitch_deg": att_result.get("pitch_deg"),
        "roll_deg": att_result.get("roll_deg"),
        "status": att_result["status"],
        "gaps": att_result.get("gaps"),
        "bracket_times": att_result.get("bracket_times"),
        "evidence": att_result.get("evidence"),
        "rule": att.ATTITUDE_RULE,
    }
    fp = fingerprint(att.ATTITUDE_RULE + "#ping", basis)
    return {"node_id": "attitude:%s" % p["id"], "kind": "attitude", "ping": p["id"],
            "cell": None, "status": att_result["status"], "rule": att.ATTITUDE_RULE,
            "upstream": [source_id] if source_id else [],
            "depends_on": [], "source_ids": [source_id] if source_id else [],
            "fingerprint": fp, "value": att_result}


def _bottom_track(case, ping, cfg, resources):
    p = case["pings"][ping]
    bt = p.get("bottom_track")
    geom = resources["geom"]
    if not bt:
        return []
    qsource = [case["source_ids"]["bt_badflag"][p["id"]]]
    verdict = qua.evaluate_beams(
        [bt["correlation"][k]["norm"] for k in range(4)],
        [bt["amplitude_db"][k]["norm"] for k in range(4)],
        [bt["bad_beam_flag"][k] for k in range(4)],
        cfg["min_correlation"], cfg["min_amplitude_db"],
    )
    qfp = fingerprint(qua.QUALITY_RULE + "#bt", {"verdict": verdict})
    nodes = [{
        "node_id": "bt-quality:%s" % p["id"], "kind": "bt_quality", "ping": p["id"],
        "cell": None, "status": "ok", "rule": qua.QUALITY_RULE, "upstream": [],
        "depends_on": [], "source_ids": qsource, "fingerprint": qfp, "value": verdict,
    }]
    beam_mps = [bt["beam_velocity"][k]["norm"] for k in range(4)]
    sol = geo.solve_beams(beam_mps, verdict["valid"], geom, cfg["error_velocity_threshold"])
    sol_fp = fingerprint(geo.GEOMETRY_RULE + "#bt",
                         {"beam_values": beam_mps, "valid": verdict["valid"],
                          "geometry": geom.basis(), "threshold": cfg["error_velocity_threshold"]})
    nodes.append({
        "node_id": "bt-beam-solution:%s" % p["id"], "kind": "bt_beam_solution",
        "ping": p["id"], "cell": None, "status": sol["status"], "rule": geo.GEOMETRY_RULE,
        "frame": FRAME_INSTRUMENT, "upstream": [qfp, resources["geom_fp"]],
        "depends_on": ["bt-quality:%s" % p["id"], "resource:geometry"],
        "source_ids": [case["source_ids"]["bt_beam"][p["id"]]],
        "fingerprint": sol_fp, "value": sol,
    })
    if sol["status"] != "four_beam":
        nodes.append({
            "node_id": "bt-earth:%s" % p["id"], "kind": "bt_earth", "ping": p["id"],
            "cell": None, "status": "blocked:" + sol["status"], "rule": rot.ROTATION_RULE,
            "frame": FRAME_EARTH, "upstream": [sol_fp], "source_ids": [],
            "depends_on": ["bt-beam-solution:%s" % p["id"]],
            "fingerprint": fingerprint(BT_ATT_RULE + "#blocked", {"bt": sol_fp}),
            "value": {"diagnostic": "bottom track cannot reach earth: " + sol["status"],
                      "candidate_count": len(sol.get("candidates", []))},
        })
        return nodes

    # Bottom track uses the attitude sample taken at the BT ping time.
    flags = case.get("attitude_gap_flags")
    att_res = att.interpolate(case["attitude_samples"], bt["time"], flags, cfg["gap_factor"])
    att_fp = fingerprint(BT_ATT_RULE, {"time": bt["time"], "status": att_res["status"],
                                       "heading_deg": att_res.get("heading_deg")})
    mount = cfg["mount"]
    ship = rot.instrument_to_ship(sol["vector"], mount["heading_deg"], mount["pitch_deg"],
                                  mount["roll_deg"], cfg["transform_order"])
    blocked = att_res["status"] not in ("exact", "interpolated")
    true_h = (att_res.get("heading_deg", 0.0) + cfg["declination_deg"])
    earth = None
    if not blocked:
        earth = rot.ship_to_earth(ship, math.radians(true_h), att_res["pitch_rad"],
                                  att_res["roll_rad"], cfg["transform_order"])
    earth_fp = fingerprint(BT_ATT_RULE + "#earth",
                           {"bt": sol_fp, "attitude": att_fp, "mount": resources["mount_fp"],
                            "declination": resources["decl_fp"], "order": cfg["transform_order"],
                            "blocked": blocked})
    nodes.append({
        "node_id": "bt-earth:%s" % p["id"], "kind": "bt_earth", "ping": p["id"], "cell": None,
        "status": "blocked-attitude" if blocked else "ok", "rule": BT_ATT_RULE,
        "frame": FRAME_EARTH,
        "upstream": [sol_fp, att_fp, resources["mount_fp"], resources["decl_fp"], resources["order_fp"]],
        "depends_on": ["bt-beam-solution:%s" % p["id"], "resource:mount",
                       "resource:declination", "resource:order"],
        "source_ids": [case["source_ids"]["bt_attitude"][p["id"]]],
        "fingerprint": earth_fp,
        "value": ({
            "vector_ship": [float(x) for x in ship],
            "vector": [float(x) for x in earth],
            "frame": FRAME_EARTH,
            "attitude_status": att_res["status"],
            "attitude_evidence": att_res,
            "true_heading_deg": true_h,
        } if not blocked else {"diagnostic": "bottom track attitude " + att_res["status"],
                               "attitude_evidence": att_res}),
    })
    return nodes


def build_graph(case, raw_cfg):
    """Pure: return list of all nodes for ``case`` under config ``raw_cfg``."""
    cfg = normalize_config(raw_cfg)
    res_nodes, resources = resource_nodes(cfg)
    geom = resources["geom"]
    nodes = list(res_nodes)

    flags = case.get("attitude_gap_flags")
    for pi, p in enumerate(case["pings"]):
        att_src = case["source_ids"]["attitude"][p["id"]] if att_sample_exists(case, p["time"]) else None
        att_result = att.interpolate(case["attitude_samples"], p["time"], flags, cfg["gap_factor"])
        an = _attitude_node(p, att_result, cfg, att_src)
        nodes.append(an)

        for ci, cell_id in enumerate(["c%02d" % c for c in range(len(p["cells"]))]):
            raw = p["cells"][cell_id]
            qn = quality_node(case, pi, ci, cfg)
            nodes.append(qn)

            issues = _cell_issues(raw)
            if issues:
                nodes.append({
                    "node_id": "cell-exclusion:%s:%s" % (p["id"], cell_id),
                    "kind": "cell_exclusion", "ping": p["id"], "cell": cell_id,
                    "status": "excluded", "rule": "invalid-cell-v1",
                    "upstream": [qn["node_id"]], "depends_on": [qn["node_id"]],
                    "source_ids": [case["source_ids"]["beam"][p["id"]][cell_id]],
                    "fingerprint": fingerprint("invalid-cell-v1",
                                               {"issues": issues, "quality": qn["fingerprint"]}),
                    "value": {"diagnostics": issues,
                              "depth_m": (raw.get("depth_m") or {}).get("norm")},
                })
                continue

            beam_mps = [raw["beam_velocity"][k]["norm"] for k in range(4)]
            nodes.extend(transform_nodes_for_cell(
                case, pi, ci, cfg, qn["value"], qn["node_id"],
                att_result, an["fingerprint"], geom, resources["geom_fp"], beam_mps,
            ))

        nodes.extend(_bottom_track(case, pi, cfg, resources))
    return nodes


def att_sample_exists(case, time):
    return any(abs(s["time"] - time) < 1e-9 for s in case["attitude_samples"])


def _cell_issues(raw):
    issues = []
    if raw.get("valid", True) is False:
        issues.append("source marked cell invalid: %s" % raw.get("reason", "(no reason)"))
    depth = raw.get("depth_m")
    if depth is None or depth.get("norm") is None:
        issues.append("cell depth not convertible to metres: %r" % raw.get("depth_m"))
    for k in range(4):
        bv = raw["beam_velocity"][k]
        if bv.get("norm") is None:
            issues.append("beam %d velocity unconvertible: %s" % (k + 1, bv.get("diagnostic")))
    return issues


def changed_roots(node_ids_old_new):
    """Set of node ids whose content fingerprint changed (incl. new/removed)."""
    old, new = node_ids_old_new
    roots = set()
    for nid, fp in new.items():
        if old.get(nid) != fp:
            roots.add(nid)
    return roots


def index_nodes(nodes):
    return {n["node_id"]: n for n in nodes}
