"""Derivation pipeline: separated raw sources -> instrument/ship/earth.

Each derivation version is immutable. Nodes carry content fingerprints;
unchanged nodes are copied/reused from the parent derivation (so a local
source change only updates the affected part). 3-beam and 4-beam results are
stored as distinct candidates and never collapsed into one value. Transform
order, beam geometry and declination version all enter the fingerprint.
"""
from __future__ import annotations

import json
import math
import numpy as np

from . import adcp, quantities as Q, storage as S, graph as G
from .attitude import AttitudeSample, resolve_attitude
from .ident import make_id, node_id, dval_id, cand_id, fingerprint

RULE_VERSION = "rules-2026.1"
GAP_SECONDS = 5.0


def _f(raw):
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _json_value(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 9) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    return v


def parse_context(snap):
    geom = sorted(snap["geom"], key=lambda r: r["ord"])[-1]
    decl = sorted(snap["declination"], key=lambda r: r["ord"])[-1]
    inst = sorted(snap["installation"], key=lambda r: r["ord"])[-1]
    beam_angle = Q.angle(geom["beam_angle"], geom["beam_angle_unit"])
    declination = Q.angle(decl["value"], decl["unit"])
    mount = Q.angle(inst["heading_mount"], inst["unit"])
    return {
        "geom": geom, "decl": decl, "inst": inst,
        "beam_angle_deg": beam_angle.value if beam_angle.ok else 20.0,
        "declination_deg": declination.value if declination.ok else 0.0,
        "mount_deg": mount.value if mount.ok else 0.0,
        "geom_version": geom["version"], "decl_version": decl["version"],
        "install_version": inst["version"],
        "geom_diag": beam_angle.diagnostics + declination.diagnostics + mount.diagnostics,
    }


def _index4(rows):
    out = {}
    for r in rows:
        out.setdefault(r["ping_id"], {}) \
           .setdefault(r["channel"], {}) \
           .setdefault(r["cell_idx"] if r["cell_idx"] is not None else -1, {}) \
           [r["beam"]] = r
    return out


def derivation_fp(ctx, sver, order, base_event_id, branch_id):
    return fingerprint([
        RULE_VERSION, order, adcp.BEAM_GEOMETRY_ID,
        ctx["geom_version"], ctx["decl_version"], ctx["install_version"],
        sver, base_event_id or "root", branch_id])


def _enumerate_candidates(radial, tiers, beam_angle):
    good = [b for b in (1, 2, 3, 4) if tiers.get(b) == adcp.Q_GOOD and radial.get(b) is not None]
    suspect = [b for b in (1, 2, 3, 4) if tiers.get(b) == adcp.Q_SUSPECT and radial.get(b) is not None]
    usable = good + suspect
    sols = []
    if len(usable) == 4:
        sols += adcp.solve_beams(radial, beam_angle, tiers)  # 4-beam
        # every 3-beam subset obtained by dropping a suspect beam is also legal
        from itertools import combinations
        for drop in suspect:
            subset = tuple(b for b in (1, 2, 3, 4) if b != drop)
            B = adcp.beam_matrix(beam_angle, beams=subset)
            r = np.array([radial[b] for b in subset], dtype=float)
            vi, resid, rank, _ = np.linalg.lstsq(-B, r, rcond=None)
            if rank == 3:
                sols.append(adcp.BeamSolution(
                    f"3beam:missing{drop}", tuple(subset), vi,
                    error_velocity=None,
                    residual=float(resid[0]) if len(resid) else 0.0,
                    diagnostics=("suspect_beam_excluded",)))
    elif len(usable) >= 3:
        sols += adcp.solve_beams(radial, beam_angle, tiers)
    return sols, good, suspect


def _solve_cell(con, drv_id, fp_seed, pid, channel, cell_idx, bv, co, am, bd,
                ctx, ar, order, write_node, summary, depthq=None):
    ckey = cell_idx if cell_idx is not None else -1
    bv_c = bv.get(pid, {}).get(channel, {}).get(ckey, {})
    co_c = co.get(pid, {}).get(channel, {}).get(ckey, {})
    am_c = am.get(pid, {}).get(channel, {}).get(ckey, {})
    bd_c = bd.get(pid, {}).get(channel, {}).get(ckey, {})

    radial, tiers, qvals, qdiag = {}, {}, [], {}
    for b in (1, 2, 3, 4):
        bvr = bv_c.get(b)
        if not bvr:
            tiers[b] = adcp.Q_UNKNOWN
            continue
        q = Q.velocity(bvr["raw_value"], bvr["unit"])
        qdiag[b] = q.diagnostics
        qvals.append((f"beam{b}_radial", q))
        radial[b] = q.value if q.ok else None
        tier, _c, _a, flag = _tier_from_rows(bvr, co_c.get(b), am_c.get(b), bd_c.get(b))
        if not q.ok:
            tier = adcp.Q_BAD
        tiers[b] = tier

    bq_status = adcp.Q_BAD if any(tiers.get(b) == adcp.Q_BAD for b in (1, 2, 3, 4)) else \
        (adcp.Q_SUSPECT if any(tiers.get(b) == adcp.Q_SUSPECT for b in (1, 2, 3, 4))
         else adcp.Q_GOOD)
    bq_id, bq_fp, bq_reused = write_node(
        "beam_quality", pid, channel, cell_idx, bq_status,
        [fp_seed, "bq", pid, channel, ckey,
         {b: (radial.get(b), tiers.get(b), qdiag.get(b, ())) for b in (1, 2, 3, 4)}],
        {"tiers": {str(b): tiers.get(b) for b in (1, 2, 3, 4)}},
        values=qvals + [(f"beam{b}_tier", Q.enum(tiers.get(b))) for b in (1, 2, 3, 4)])

    sols, good, suspect = _enumerate_candidates(radial, tiers, ctx["beam_angle_deg"])
    has_four = any(s.method == "4beam" for s in sols)
    three = [s for s in sols if s.method.startswith("3beam")]

    # selection rule: never collapse 4-beam and 3-beam into one value
    selected_sol = None
    cell_status = "computed"
    if not sols:
        cell_status = "insufficient_beams"
    elif has_four:
        selected_sol = next(s for s in sols if s.method == "4beam")
    elif len(three) == 1:
        selected_sol = three[0]
    else:
        cell_status = "ambiguous"  # multiple 3-beam candidates, none auto chosen

    # nodes for the selected vector chain (instrument -> ship -> earth)
    prev_ids = [bq_id]
    selected = {"instrument": None, "ship": None, "earth": None}
    if selected_sol is not None:
        vi = selected_sol.velocity
        vs = adcp.instrument_to_ship(vi, ctx["mount_deg"])
        if ar.true_heading is None or ar.pitch is None or ar.roll is None:
            ve, earth_diag = None, ("attitude_unavailable",)
        else:
            if order == adcp.TRANSFORM_ORDER:
                ve = adcp.ship_to_earth_rdi(vs, ar.true_heading, ar.pitch, ar.roll)
            else:
                ve = adcp.ship_to_earth_alt_order(vs, ar.true_heading, ar.pitch, ar.roll)
            earth_diag = ("alt_transform_order",) if order != adcp.TRANSFORM_ORDER else ()
        frames = [("water_instrument" if channel == "water" else "bottom_instrument",
                   vi, "instrument", None),
                  ("water_ship" if channel == "water" else "bottom_ship",
                   vs, "ship", None),
                  ("water_earth" if channel == "water" else "bottom_earth",
                   ve, "earth", earth_diag)]
        for ntype, vecv, frame, ediag in frames:
            q = Q.vec(vecv, frame, None,
                      tuple(ar.diagnostics) + (ediag if ediag else ()))
            nid, nfp, reused = write_node(
                ntype, pid, channel, cell_idx,
                ar.status if frame != "instrument" else cell_status,
                [fp_seed, ntype, pid, channel, ckey,
                 _json_value(vecv) if vecv is not None else None,
                 ctx["install_version"] if frame == "ship" else "",
                 ctx["decl_version"] if frame == "earth" else "",
                 order if frame == "earth" else ""],
                {"frame": frame, "method": selected_sol.method},
                values=[("velocity", q)], frame=frame)
            for up in prev_ids:
                con.execute("INSERT OR REPLACE INTO edges VALUES(?,?,?,?)",
                            (drv_id, up, nid, "feeds"))
            prev_ids = [nid]
            selected[frame] = _json_value(vecv)

    # every legal candidate persisted with a stable identity (incl. unchosen)
    for sol in sols:
        vi = sol.velocity
        vs = adcp.instrument_to_ship(vi, ctx["mount_deg"])
        ve = None
        if ar.true_heading is not None and ar.pitch is not None and ar.roll is not None:
            if order == adcp.TRANSFORM_ORDER:
                ve = adcp.ship_to_earth_rdi(vs, ar.true_heading, ar.pitch, ar.roll)
            else:
                ve = adcp.ship_to_earth_alt_order(vs, ar.true_heading, ar.pitch, ar.roll)
        cid = cand_id(drv_id, pid, channel, ckey, sol.method, list(sol.beams))
        is_sel = 1 if (selected_sol is not None and sol is selected_sol) else 0
        diag = tuple(sol.diagnostics)
        if cell_status == "ambiguous":
            diag = diag + ("retained_unselected_candidate",)
        con.execute(
            "INSERT OR REPLACE INTO candidates(cand_id,drv_id,ping_id,channel,"
            "cell_idx,kind,method,beams,instrument_json,ship_json,earth_json,"
            "error_velocity,resolution,selected,status,diagnostics_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, drv_id, pid, channel, cell_idx,
             "four_beam" if sol.method == "4beam" else "three_beam",
             sol.method, json.dumps(list(sol.beams)),
             json.dumps(_json_value(vi)), json.dumps(_json_value(vs)),
             json.dumps(_json_value(ve) if ve is not None else None),
             sol.error_velocity, sol.residual, is_sel, cell_status,
             json.dumps(list(diag))))
        summary["candidates"] += 1

    if cell_status == "ambiguous":
        summary["ambiguous"] += 1


def recompute(con, *, sver, case_id, branch_id, changed_tables=None,
              changed_pings=None, transform_order=None, base_event_id=None):
    from . import repository as R
    snap = R.load_snapshot(con, sver)
    ctx = parse_context(snap)
    order = transform_order or adcp.TRANSFORM_ORDER
    pings = snap["pings"]

    bv = _index4(snap["beam_vel"])
    co = _index4(snap["correlation"])
    am = _index4(snap["amplitude"])
    bd = _index4(snap["badbeam"])
    att = {r["ping_id"]: r for r in snap["attitude"]}
    cells = {}
    for r in snap["cells"]:
        cells.setdefault(r["ping_id"], {})[r["cell_idx"]] = r

    prow = con.execute("SELECT head_drv FROM branches WHERE branch_id=?",
                       (branch_id,)).fetchone()
    parent_drv = prow["head_drv"] if prow and prow["head_drv"] else None

    affected_types = (G.affected_node_types(set(changed_tables))
                      if changed_tables is not None else None)
    scope_pings = set(changed_pings) if changed_pings is not None else None

    samples = []
    for p in pings:
        row = att.get(p["ping_id"])
        if row:
            samples.append(AttitudeSample(
                p["ping_id"], p["t"], _f(row["heading"]),
                _f(row["pitch"]), _f(row["roll"])))
    qtimes = {p["ping_id"]: p["t"] for p in pings}
    att_results = resolve_attitude(samples, qtimes, ctx["declination_deg"],
                                   gap_seconds=GAP_SECONDS)

    drv_seed = derivation_fp(ctx, sver, order, base_event_id, branch_id)
    basis_seed = fingerprint([
        RULE_VERSION, order, adcp.BEAM_GEOMETRY_ID, ctx["geom_version"],
        ctx["decl_version"], ctx["install_version"]])
    drv_id = make_id("deriv", drv_seed)

    summary = {"drv_id": drv_id, "parent_drv": parent_drv, "pings": 0,
               "nodes": 0, "reused": 0, "recomputed": 0, "candidates": 0,
               "ambiguous": 0, "excluded_cells": 0, "missing_attitude": 0,
               "diagnostics": list(ctx["geom_diag"])}
    scope = {"changed_tables": sorted(changed_tables) if changed_tables else None,
             "affected_node_types": sorted(affected_types) if affected_types else None,
             "changed_pings": sorted(scope_pings) if scope_pings else None}

    def in_scope(ntype, pid):
        if affected_types is None:
            return True
        if ntype not in affected_types:
            return False
        if scope_pings is None:
            return True
        return pid in scope_pings

    def write_node(ntype, pid, channel, cell_idx, status, fp_parts, payload,
                   values=(), frame=None):
        nfp = fingerprint([basis_seed] + list(fp_parts) + [ntype, channel, cell_idx])
        nid = node_id(drv_id, ntype, pid, channel, cell_idx)
        pnid = node_id(parent_drv, ntype, pid, channel, cell_idx) if parent_drv else None
        reused = False
        if pnid:
            pr = con.execute("SELECT fp FROM nodes WHERE node_id=?", (pnid,)).fetchone()
            if pr and pr["fp"] == nfp:
                reused = True
        con.execute(
            "INSERT INTO nodes(node_id,drv_id,ntype,ping_id,channel,cell_idx,frame,"
            "status,fp,reused_from,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (nid, drv_id, ntype, pid, channel, cell_idx, frame, status, nfp,
             pnid if reused else None, json.dumps(payload, ensure_ascii=False)))
        for name, qty in values:
            con.execute(
                "INSERT INTO dvalues(val_id,node_id,name,value_json,datum_json,"
                "raw_repr,diagnostics_json) VALUES(?,?,?,?,?,?,?)",
                (dval_id(nid, name), nid, name,
                 json.dumps(_json_value(qty.value)),
                 json.dumps(qty.datum.to_dict()), qty.raw_repr,
                 json.dumps(list(qty.diagnostics))))
        summary["nodes"] += 1
        summary["reused" if reused else "recomputed"] += 1
        return nid, nfp, reused

    for p in pings:
        pid = p["ping_id"]
        summary["pings"] += 1
        ar = att_results[pid]
        if ar.status == "missing":
            summary["missing_attitude"] += 1
        write_node(
            "attitude", pid, None, None, ar.status,
            ["attitude", pid, ar.heading, ar.pitch, ar.roll,
             ar.segment_id, ar.wrap_crossed, ctx["decl_version"]],
            {"status": ar.status, "segment": ar.segment_id,
             "wrap_crossed": ar.wrap_crossed, "true_heading": ar.true_heading},
            values=[
                ("heading_magnetic",
                 Q.angle(ar.heading, "deg", "magnetic") if ar.heading is not None
                 else Q.Quantity(None, Q.Datum("angle", "deg", None, "magnetic"),
                                 None, tuple(ar.diagnostics))),
                ("heading_true",
                 Q.angle(ar.true_heading, "deg", "true") if ar.true_heading is not None
                 else Q.Quantity(None, Q.Datum("angle", "deg", None, "true"),
                                 None, tuple(ar.diagnostics))),
                ("pitch", Q.angle(ar.pitch, "deg") if ar.pitch is not None
                 else Q.Quantity(None, Q.Datum("angle", "deg"), None, tuple(ar.diagnostics))),
                ("roll", Q.angle(ar.roll, "deg") if ar.roll is not None
                 else Q.Quantity(None, Q.Datum("angle", "deg"), None, tuple(ar.diagnostics)))],
            frame=None)

        for cell_idx, crow in sorted(cells.get(pid, {}).items(), key=lambda kv: kv[0]):
            depthq = Q.length(crow["depth_raw"], crow["depth_unit"])
            configured = bool(crow["configured"])
            valid_depth = configured and depthq.ok and depthq.value is not None and depthq.value > 0
            if not valid_depth:
                reason = ("not_configured",) if not configured else (
                    tuple(depthq.diagnostics) if depthq.diagnostics else ("invalid_depth",))
                write_node(
                    "cell_status", pid, "water", cell_idx, "excluded",
                    ["cell", pid, cell_idx, crow["depth_raw"], crow["configured"]],
                    {"configured": configured, "excluded": True, "reasons": list(reason)},
                    values=[("depth", Q.Quantity(
                        depthq.value, Q.Datum("length", crow["depth_unit"]),
                        crow["depth_raw"], tuple(reason)))], frame=None)
                summary["excluded_cells"] += 1
                continue
            _solve_cell(con, drv_id, basis_seed, pid, "water", cell_idx, bv, co,
                        am, bd, ctx, ar, order, write_node, summary)

        if bv.get(pid, {}).get("bottom"):
            _solve_cell(con, drv_id, basis_seed, pid, "bottom", None, bv, co,
                        am, bd, ctx, ar, order, write_node, summary)

    con.execute(
        "INSERT INTO derivations(drv_id,case_id,sver,rule_version,geom_version,"
        "decl_version,install_version,transform_order,beam_geometry,fp,status,"
        "parent_drv,base_event_id,created_seq,transform_order_alt) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (drv_id, case_id, sver, RULE_VERSION, ctx["geom_version"],
         ctx["decl_version"], ctx["install_version"], order,
         adcp.BEAM_GEOMETRY_ID, drv_seed, "pending", parent_drv,
         base_event_id, S.next_seq(con),
         0 if order == adcp.TRANSFORM_ORDER else 1))
    return drv_id, summary, scope
