"""Dependency-graph node construction (pure functions).

Node identity is *what* a node means; node content fingerprint is *which
version* it currently is.  The graph is built from the normalised case data,
and every derived node records the fingerprints of its direct inputs so the
storage layer can propagate invalidation along the graph.
"""
from __future__ import annotations

from . import attitude as att
from . import geometry as geo
from . import quality as qua
from . import rotation as rot
from .fingerprint import fingerprint
from .units import FRAME_EARTH, FRAME_INSTRUMENT, FRAME_SHIP

MOUNT_RULE = "mount-align-v1"
DECLINATION_RULE = "magnetic-declination-v1"
BT_ATT_RULE = "bottomtrack-attitude-v1"


def quality_node(case, ping, cell, cfg):
    p = case["pings"][ping]
    cell_id = "c%02d" % cell
    source_ids = [
        case["source_ids"]["beam"][p["id"]][cell_id],
        case["source_ids"]["correlation"][p["id"]][cell_id],
        case["source_ids"]["amplitude"][p["id"]][cell_id],
        case["source_ids"]["badflag"][p["id"]][cell_id],
    ]
    raw = case["pings"][ping]["cells"][cell]
    verdict = qua.evaluate_beams(
        [raw["correlation"][k]["norm"] for k in range(4)],
        [raw["amplitude_db"][k]["norm"] for k in range(4)],
        [raw["bad_beam_flag"][k] for k in range(4)],
        cfg["min_correlation"],
        cfg["min_amplitude_db"],
    )
    basis = {
        "verdict": verdict,
        "thresholds": {
            "min_correlation": cfg["min_correlation"],
            "min_amplitude_db": cfg["min_amplitude_db"],
        },
        "inputs": source_ids,
    }
    fp = fingerprint(qua.QUALITY_RULE, basis)
    return {
        "node_id": f"quality:{p['id']}:{cell_id}",
        "kind": "quality",
        "ping": p["id"],
        "cell": cell_id,
        "status": "ok",
        "rule": qua.QUALITY_RULE,
        "source_ids": source_ids,
        "upstream": source_ids,
        "depends_on": [],
        "fingerprint": fp,
        "value": verdict,
    }


def _invalid_cell_diagnostic(cell):
    issues = []
    if cell.get("valid", True) is False:
        issues.append("source marked cell invalid: %s" % cell.get("reason", "(no reason)"))
    depth = cell.get("depth_m")
    if depth is None or depth["norm"] is None:
        issues.append("cell depth not convertible to metres: %r" % depth)
    return issues


def _attitude_for_ping(case, ping, cfg):
    p = case["pings"][ping]
    samples = case["attitude_samples"]
    flags = case.get("attitude_gap_flags")
    res = att.interpolate(samples, p["time"], flags, cfg.get("gap_factor", 1.5))
    return res


def transform_nodes_for_cell(
    case, ping, cell, cfg, quality_value, quality_node_id, att_result, att_fp, geom, geom_fp, beam_mps
):
    p = case["pings"][ping]
    cell_id = "c%02d" % cell
    raw = p["cells"][cell_id]
    beam_source = case["source_ids"]["beam"][p["id"]][cell_id]

    sol = geo.solve_beams(beam_mps, quality_value["valid"], geom, cfg["error_velocity_threshold"])
    sol_basis = {
        "beam_values": beam_mps,
        "valid": quality_value["valid"],
        "geometry": geom.basis(),
        "threshold": cfg["error_velocity_threshold"],
        "upstream": [quality_node_id, geom_fp],
    }
    sol_fp = fingerprint(geo.GEOMETRY_RULE, sol_basis)

    nodes = [
        {
            "node_id": "beam-solution:%s:%s" % (p["id"], cell_id),
            "kind": "beam_solution",
            "ping": p["id"],
            "cell": cell_id,
            "status": sol["status"],
            "rule": geo.GEOMETRY_RULE,
            "frame": FRAME_INSTRUMENT,
            "upstream": [quality_node_id, geom_fp],
            "depends_on": [quality_node_id, "resource:geometry"],
            "source_ids": [beam_source],
            "fingerprint": sol_fp,
            "value": sol,
        }
    ]

    # Three/four beam non-mixing: downstream transforms are only built for the
    # unique four-beam value; candidates stay attached to the beam node.
    if sol["status"] == "four_beam":
        instrument_vec = sol["vector"]
        nodes.extend(
            _frame_nodes(
                case, p, cell_id, instrument_vec, att_result, att_fp, sol_fp, raw, cfg
            )
        )
    else:
        nodes.append(
            {
                "node_id": "instrument:%s:%s" % (p["id"], cell_id),
                "kind": "instrument",
                "ping": p["id"],
                "cell": cell_id,
                "status": sol["status"],
                "rule": geo.GEOMETRY_RULE,
                "frame": FRAME_INSTRUMENT,
                "upstream": [nodes[0]["node_id"]],
                "depends_on": [nodes[0]["node_id"]],
                "source_ids": [],
                "fingerprint": fingerprint(
                    geo.GEOMETRY_RULE + "#blocked",
                    {"status": sol["status"], "beam_solution": sol_fp},
                ),
                "value": {
                    "diagnostic": sol.get(
                        "diagnostic",
                        "instrument velocity not unique; %d candidate(s) retained" % len(sol["candidates"]),
                    ),
                    "candidate_count": len(sol["candidates"]),
                },
            }
        )
    return nodes


def _mount_basis(cfg):
    return {
        "heading_deg": cfg["mount_heading_deg"],
        "pitch_deg": cfg["mount_pitch_deg"],
        "roll_deg": cfg["mount_roll_deg"],
        "order": cfg["transform_order"],
    }


def _frame_nodes(case, p, cell_id, instrument_vec, att_result, att_fp, beam_fp, raw, cfg):
    mount = cfg["mount"] if "mount" in cfg else {
        "heading_deg": cfg["mount_heading_deg"],
        "pitch_deg": cfg["mount_pitch_deg"],
        "roll_deg": cfg["mount_roll_deg"],
    }
    order = cfg["transform_order"]
    mount_basis = _mount_basis(cfg)
    mount_fp = fingerprint(MOUNT_RULE, mount_basis)

    ship_vec = rot.instrument_to_ship(
        instrument_vec, mount["heading_deg"], mount["pitch_deg"], mount["roll_deg"], order
    )
    decl_basis = {
        "declination_deg": cfg["declination_deg"],
        "version": cfg["declination_version"],
        "basis_name": cfg.get("declination_basis", "magnetic"),
    }
    decl_fp = fingerprint(DECLINATION_RULE, decl_basis)

    if att_result["status"] not in ("exact", "interpolated"):
        return [
            _frame_blocked(
                p, cell_id, "instrument", instrument_vec, FRAME_INSTRUMENT, beam_fp,
                [beam_fp], "instrument vector available", extra={"frame": FRAME_INSTRUMENT},
            ),
            _frame_blocked(
                p, cell_id, "ship", ship_vec, FRAME_SHIP, mount_fp,
                [beam_fp, mount_fp], "ship vector available", extra={"frame": FRAME_SHIP},
            ),
            {
                "node_id": "earth:%s:%s" % (p["id"], cell_id),
                "kind": "earth",
                "ping": p["id"],
                "cell": cell_id,
                "status": "blocked-attitude",
                "rule": rot.ROTATION_RULE,
                "frame": FRAME_EARTH,
                "upstream": ["ship:%s:%s" % (p["id"], cell_id), att_fp, decl_fp, mount_fp],
                "depends_on": ["ship:%s:%s" % (p["id"], cell_id), "attitude:%s" % p["id"],
                               "resource:declination", "resource:mount", "resource:order"],
                "source_ids": [],
                "fingerprint": fingerprint(
                    rot.ROTATION_RULE + "#blocked",
                    {"attitude_status": att_result["status"], "attitude": att_fp, "declination": decl_fp},
                ),
                "value": {"diagnostic": "earth transform blocked: attitude " + att_result["status"],
                          "attitude_evidence": att_result},
            },
        ]

    true_heading_deg = att_result["heading_deg"] + cfg["declination_deg"]
    import math as _m

    earth_vec = rot.ship_to_earth(
        ship_vec,
        _m.radians(true_heading_deg),
        att_result["pitch_rad"],
        att_result["roll_rad"],
        order,
    )

    inst_fp = fingerprint(
        geo.GEOMETRY_RULE + "#instrument", {"vector": instrument_vec, "beam_solution": beam_fp}
    )
    ship_fp = fingerprint(
        rot.ROTATION_RULE + "#ship",
        {"instrument": inst_fp, "mount": mount_basis, "order": order},
    )
    earth_fp = fingerprint(
        rot.ROTATION_RULE + "#earth",
        {
            "ship": ship_fp,
            "attitude_rule": att.ATTITUDE_RULE,
            "attitude": att_fp,
            "declination": decl_basis,
            "order": order,
            "true_heading_deg": true_heading_deg,
        },
    )
    return [
        {
            "node_id": "instrument:%s:%s" % (p["id"], cell_id),
            "kind": "instrument", "ping": p["id"], "cell": cell_id,
            "status": "ok", "rule": geo.GEOMETRY_RULE, "frame": FRAME_INSTRUMENT,
            "upstream": ["beam-solution:%s:%s" % (p["id"], cell_id)],
            "depends_on": ["beam-solution:%s:%s" % (p["id"], cell_id)],
            "source_ids": [], "fingerprint": inst_fp,
            "value": {"vector": [float(x) for x in instrument_vec], "frame": FRAME_INSTRUMENT,
                      "solution_status": "four_beam"},
        },
        {
            "node_id": "ship:%s:%s" % (p["id"], cell_id),
            "kind": "ship", "ping": p["id"], "cell": cell_id,
            "status": "ok", "rule": rot.ROTATION_RULE, "frame": FRAME_SHIP,
            "upstream": ["instrument:%s:%s" % (p["id"], cell_id), mount_fp],
            "depends_on": ["instrument:%s:%s" % (p["id"], cell_id), "resource:mount"],
            "source_ids": [], "fingerprint": ship_fp,
            "value": {"vector": [float(x) for x in ship_vec], "frame": FRAME_SHIP,
                      "mount": mount_basis, "mount_fingerprint": mount_fp},
        },
        {
            "node_id": "earth:%s:%s" % (p["id"], cell_id),
            "kind": "earth", "ping": p["id"], "cell": cell_id,
            "status": "ok", "rule": rot.ROTATION_RULE, "frame": FRAME_EARTH,
            "upstream": ["ship:%s:%s" % (p["id"], cell_id), att_fp, decl_fp, mount_fp],
            "depends_on": ["ship:%s:%s" % (p["id"], cell_id), "attitude:%s" % p["id"],
                           "resource:declination", "resource:mount", "resource:order"],
            "source_ids": [], "fingerprint": earth_fp,
            "value": {
                "vector": [float(x) for x in earth_vec], "frame": FRAME_EARTH,
                "true_heading_deg": true_heading_deg,
                "magnetic_heading_deg": att_result["heading_deg"],
                "declination_deg": cfg["declination_deg"],
                "declination_version": cfg["declination_version"],
                "order": order,
                "attitude_status": att_result["status"],
                "attitude_evidence": att_result,
            },
        },
    ]


def _frame_blocked(p, cell_id, kind, vec, frame, fp, upstream, msg, extra=None):
    return {
        "node_id": "%s:%s:%s" % (kind, p["id"], cell_id),
        "kind": kind, "ping": p["id"], "cell": cell_id,
        "status": "ok" if kind in ("instrument", "ship") else "blocked",
        "rule": rot.ROTATION_RULE if kind == "ship" else geo.GEOMETRY_RULE,
        "frame": frame,
        "upstream": upstream, "source_ids": [],
        "depends_on": upstream,
        "fingerprint": fp,
        "value": ({"vector": [float(x) for x in vec], "frame": frame} if vec is not None else {"diagnostic": msg}),
    }
