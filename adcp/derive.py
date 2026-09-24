"""Dependency-graph derivation engine.

Each derived node declares its dependencies; a node is recomputed only when
one of its dependency hashes changes, otherwise the parent's payload is reused
with an explicit "still valid" reason. Every node payload carries `_deps`
so the UI can walk the source chain from any displayed value.
"""
from __future__ import annotations

import math

import numpy as np

from . import transforms
from .attitude import AttitudeSeries
from .fingerprint import pipeline_fingerprint, short_hash
from .units import normalize


def _content_hash(payload):
    return short_hash({k: v for k, v in payload.items() if k != "_deps"})


class Ctx:
    def __init__(self, store):
        self.store = store
        self.sources = store.sources()
        self.by_kind = {}
        for s in self.sources:
            self.by_kind.setdefault(s["kind"], []).append(s)
        _, geom = store.param("beam_geometry")
        self.geom = transforms.BeamGeometry(**geom)
        _, self.order = store.param("transform_order")
        self.decl_version, decl = store.param("magnetic_declination")
        self.decl_deg = decl["value_deg"]
        self.mount_version, mount = store.param("mounting_angle")
        self.mount_deg = mount["value_deg"]
        _, qc = store.param("qc")
        self.qc = qc
        _, ens = store.param("ensemble")
        self.ping_times = ens["ping_times"]
        self.cell_centers = ens["cell_centers_m"]
        _, self.err_thresh = store.param("error_velocity_threshold")
        _, gap = store.param("attitude_gap_threshold_s")
        headings = sorted(self.by_kind.get("heading", []), key=lambda s: s["payload"]["t"])
        pitches = {s["payload"]["t"]: s["payload"]["value"]
                   for s in self.by_kind.get("pitch", [])}
        rolls = {s["payload"]["t"]: s["payload"]["value"]
                 for s in self.by_kind.get("roll", [])}
        times = [s["payload"]["t"] for s in headings]
        self.attitude = AttitudeSeries(
            times,
            [s["payload"]["value"] for s in headings],
            [pitches[t] for t in times],
            [rolls[t] for t in times],
            gap_threshold_s=gap)
        self.decisions = {d["scope"]: d["choice"]
                          for d in store.decisions(status="applied")}
        self.n_pings = len(self.ping_times)
        self.n_cells = len(self.cell_cent_times) if False else len(self.cell_centers)
        self.n_beams = self.geom.n_beams

    def source_dep(self, kind, ping=None, cell=None, beam=None):
        for s in self.by_kind.get(kind, []):
            if ping is not None and s["ping"] != ping:
                continue
            if cell is not None and s["cell"] != cell:
                continue
            if beam is not None and s["beam"] != beam:
                continue
            return s
        return None

    def bt(self, ping):
        s = self.source_dep("bottom_track", ping=ping)
        return s["payload"] if s else None


def _dep_source(s):
    return {s["id"]: short_hash({"payload": s["payload"], "unit": s["unit"]})} if s else {}


def _param_dep(ctx, name, version):
    return {f"param:{name}@{version}": short_hash([name, version])}


# -- node computers ---------------------------------------------------------
# Each returns (payload, deps). payload['_deps'] is filled by the builder.

def compute_qc(ctx, p, c, b):
    corr = ctx.source_dep("correlation", p, c, b)
    amp = ctx.source_dep("amplitude", p, c, b)
    flag = ctx.source_dep("bad_beam_flag", p, b)
    deps = {}
    for s in (corr, amp, flag):
        deps.update(_dep_source(s))
    reasons = []
    cval = corr["payload"]["value"] if corr else None
    aval = amp["payload"]["value"] if amp else None
    fval = flag["payload"]["value"] if flag else 0
    if cval is None or cval < ctx.qc["correlation_min_counts"]:
        reasons.append(f"correlation {cval} < {ctx.qc['correlation_min_counts']} counts")
    if aval is None or not (ctx.qc["amplitude_min_counts"] <= aval
                            <= ctx.qc["amplitude_max_counts"]):
        reasons.append(f"amplitude {aval} outside "
                       f"[{ctx.qc['amplitude_min_counts']},{ctx.qc['amplitude_max_counts']}]")
    if fval:
        reasons.append("bad_beam_flag set")
    return ({"ping": p, "cell": c, "beam": b, "ok": not reasons,
             "correlation": cval, "amplitude": aval, "bad_flag": fval,
             "reasons": reasons}, deps)


def compute_instr(ctx, p, c, qc_nodes):
    deps = {}
    for b in range(ctx.n_beams):
        s = ctx.source_dep("beam_velocity", p, c, b)
        deps.update(_dep_source(s))
        deps[f"qc:ping={p}:cell={c}:beam={b}"] = _content_hash(qc_nodes[b]["payload"])
    deps.update(_param_dep(ctx, "beam_geometry", ctx.geom.version))
    deps.update(_param_dep(ctx, "error_velocity_threshold", "cfg"))
    scope = f"instr:ping={p}"
    if scope in ctx.decisions:
        deps[f"decision:{scope}"] = short_hash(ctx.decisions[scope])

    radials, bad, diags = [], [], []
    for b in range(ctx.n_beams):
        s = ctx.source_dep("beam_velocity", p, c, b)
        value, unit = (s["payload"]["value"], s["unit"]) if s else (None, "m/s")
        norm, _, diag = normalize(value, unit)
        if diag:
            diags.append({"beam": b, "diagnostic": diag, "raw": value, "unit": unit})
            bad.append(b)
            radials.append(0.0)
        else:
            radials.append(norm)
            if not qc_nodes[b]["payload"]["ok"]:
                bad.append(b)
    geom = ctx.geom
    candidates, status, value, method, err = [], "ok", None, None, None
    if len(bad) >= 2:
        status = "failed"
        diags.append({"diagnostic": f"{len(bad)} beams unusable; no unique solution "
                      "and none fabricated"})
    elif len(bad) == 1:
        candidates = transforms.three_beam_candidates(radials, geom, bad_beams=bad)
        value, method, status = candidates[0]["value"], candidates[0]["id"], "degraded"
    else:
        v, err = transforms.beam_to_instrument(radials, geom)
        cand4 = {"id": "4beam", "method": "4beam", "value": [float(x) for x in v],
                 "error_velocity": err, "unit": "m/s", "basis": "instrument"}
        candidates = [cand4]
        value, method = cand4["value"], "4beam"
        if abs(err) > ctx.err_thresh:
            # keep every legal candidate; never merge 3-beam and 4-beam values
            candidates += transforms.three_beam_candidates(radials, geom)
            status = "ambiguous"
    adjudicated = None
    if scope in ctx.decisions and candidates:
        choice = ctx.decisions[scope]
        for cand in candidates:
            if cand["id"] == choice:
                value, method, status = cand["value"], cand["id"], "adjudicated"
                adjudicated = choice
    payload = {"ping": p, "cell": c, "status": status, "method": method,
               "value": value, "unit": "m/s", "basis": "instrument",
               "error_velocity": err, "candidates": candidates,
               "adjudicated": adjudicated, "bad_beams": bad, "diagnostics": diags}
    return payload, deps


def compute_ship(ctx, p, c, instr_node):
    deps = {f"instr:ping={p}:cell={c}": _content_hash(instr_node["payload"])}
    for kind in ("heading", "pitch", "roll"):
        for s in ctx.by_kind.get(kind, []):
            deps.update(_dep_source(s))
    deps.update(_param_dep(ctx, "mounting_angle", ctx.mount_version))
    deps.update(_param_dep(ctx, "transform_order", "cfg"))
    deps.update(_param_dep(ctx, "attitude_gap_threshold_s", "cfg"))
    att = ctx.attitude.interpolate(ctx.ping_times[p])
    payload = {"ping": p, "cell": c, "basis": "ship", "unit": "m/s",
               "attitude": att.to_dict()}
    if instr_node["payload"]["value"] is None:
        payload.update(status="unavailable", value=None,
                       diagnostic="instrument velocity unavailable")
    elif att.status in ("gap", "out_of_range"):
        payload.update(status="unavailable", value=None,
                       diagnostic=f"attitude {att.status}: {att.diagnostic}")
    else:
        v = transforms.instrument_to_ship(instr_node["payload"]["value"], ctx.mount_deg)
        payload.update(status=instr_node["payload"]["status"],
                       value=[float(x) for x in v])
    return payload, deps


def compute_earth(ctx, p, c, ship_node):
    deps = {f"ship:ping={p}:cell={c}": _content_hash(ship_node["payload"])}
    bt = ctx.source_dep("bottom_track", ping=p)
    deps.update(_dep_source(bt))
    deps.update(_param_dep(ctx, "magnetic_declination", ctx.decl_version))
    deps.update(_param_dep(ctx, "transform_order", "cfg"))
    payload = {"ping": p, "cell": c, "basis": "earth", "unit": "m/s"}
    if ship_node["payload"]["value"] is None:
        payload.update(status="unavailable", value=None,
                       diagnostic=ship_node["payload"].get("diagnostic"))
        return payload, deps
    att = ship_node["payload"]["attitude"]
    btp = bt["payload"]
    bt_ship = [btp["bt_x_m_s"], btp["bt_y_m_s"], btp["bt_z_m_s"]]
    meas = transforms.ship_to_earth(ship_node["payload"]["value"], att["heading"],
                                    att["pitch"], att["roll"], ctx.decl_deg, ctx.order)
    ship_motion = transforms.ship_to_earth(bt_ship, att["heading"], att["pitch"],
                                           att["roll"], ctx.decl_deg, ctx.order)
    water = meas - ship_motion
    payload.update(status=ship_node["payload"]["status"],
                   value={"north": float(water[0]), "east": float(water[1]),
                          "down": float(water[2]), "up": float(-water[2])},
                   ship_motion_earth={"north": float(ship_motion[0]),
                                      "east": float(ship_motion[1]),
                                      "down": float(ship_motion[2])},
                   declination_version=ctx.decl_version)
    return payload, deps


def compute_excluded(ctx, p, qc_grid, bt_depth):
    deps = {}
    for c in range(ctx.n_cells):
        for b in range(ctx.n_beams):
            deps[f"qc:ping={p}:cell={c}:beam={b}"] = _content_hash(qc_grid[c][b]["payload"])
    bt = ctx.source_dep("bottom_track", ping=p)
    deps.update(_dep_source(bt))
    deps.update(_param_dep(ctx, "beam_geometry", ctx.geom.version))
    limit = bt_depth * math.cos(math.radians(ctx.geom.beam_angle_deg))
    cells = []
    for c in range(ctx.n_cells):
        reasons = []
        for b in range(ctx.n_beams):
            reasons += [f"beam{b}: {r}" for r in qc_grid[c][b]["payload"]["reasons"]]
        depth = ctx.cell_centers[c]
        if depth > limit:
            reasons.append(f"cell center {depth:.2f}m beyond side-lobe limit "
                           f"{limit:.2f}m (bottom {bt_depth:.2f}m)")
        if reasons:
            cells.append({"cell": c, "depth_m": depth, "reasons": reasons})
    return {"ping": p, "cells": cells}, deps


def compute_profile(ctx, p, earth_nodes, excluded_node):
    deps = {f"excluded:ping={p}": _content_hash(excluded_node["payload"])}
    for c in range(ctx.n_cells):
        deps[f"earth:ping={p}:cell={c}"] = _content_hash(earth_nodes[c]["payload"])
    excluded = {e["cell"] for e in excluded_node["payload"]["cells"]}
    cells, used = [], []
    for c in range(ctx.n_cells):
        e = earth_nodes[c]["payload"]
        entry = {"cell": c, "depth_m": ctx.cell_centers[c],
                 "status": e["status"], "value": e["value"]}
        if c in excluded:
            entry["status"] = "excluded"
            entry["value"] = None
        elif e["value"] is not None:
            used.append(e["value"])
        cells.append(entry)
    mean = None
    if used:
        mean = {k: float(np.mean([u[k] for u in used])) for k in
                ("north", "east", "down", "up")}
    n_ok = len(used)
    status = "ok" if n_ok == ctx.n_cells else ("partial" if n_ok else "empty")
    return {"ping": p, "cells": cells, "mean": mean, "status": status,
            "n_valid": n_ok}, deps


# -- builder -----------------------------------------------------------------

def node_names(ctx):
    for p in range(ctx.n_pings):
        for c in range(ctx.n_cells):
            for b in range(ctx.n_beams):
                yield f"qc:ping={p}:cell={c}:beam={b}"
    for p in range(ctx.n_pings):
        for c in range(ctx.n_cells):
            yield f"instr:ping={p}:cell={c}"
            yield f"ship:ping={p}:cell={c}"
            yield f"earth:ping={p}:cell={c}"
    for p in range(ctx.n_pings):
        yield f"excluded:ping={p}"
        yield f"profile:ping={p}"


def build_version(store, label):
    """Build a new version from current sources/params/decisions.

    Only nodes whose dependency hashes changed are recomputed; the rest are
    carried over with an explicit validity reason.
    """
    parent = store.current_version()
    ctx = Ctx(store)
    fp = pipeline_fingerprint(ctx.order, ctx.geom.version,
                              ctx.decl_version, ctx.mount_version)
    vid = store.begin_version(parent, label, fp)
    parent_rows = {r["node"]: r for r in store.derived(parent)} if parent else {}
    report = {"version": vid, "parent": parent, "recomputed": [], "reused": []}

    def emit(node, payload, deps):
        payload["_deps"] = deps
        ih = short_hash(deps)
        old = parent_rows.get(node)
        if old and old["inputs_hash"] == ih:
            store.put_derived(vid, node, old["payload"], ih,
                              "valid: inputs unchanged")
            report["reused"].append(node)
        else:
            changed = sorted(set(deps) - set(old["payload"].get("_deps", {}))
                             if old else deps)
            changed += [k for k in (old["payload"].get("_deps", {}) if old else {})
                        if k in deps and old["payload"]["_deps"][k] != deps[k]]
            reason = ("recomputed: changed=" + ",".join(sorted(set(changed)))
                      if old else "computed: new node")
            store.put_derived(vid, node, payload, ih, reason)
            report["recomputed"].append(node)
        return payload

    qc = {}
    for p in range(ctx.n_pings):
        for c in range(ctx.n_cells):
            for b in range(ctx.n_beams):
                payload, deps = compute_qc(ctx, p, c, b)
                qc[(p, c, b)] = {"payload": emit(f"qc:ping={p}:cell={c}:beam={b}",
                                                 payload, deps)}
    for p in range(ctx.n_pings):
        bt = ctx.bt(p)
        bt_depth = bt["depth_m"] if bt else float("inf")
        qc_grid = [[qc[(p, c, b)] for b in range(ctx.n_beams)]
                   for c in range(ctx.n_cells)]
        excluded = compute_excluded(ctx, p, qc_grid, bt_depth)
        ep = emit(f"excluded:ping={p}", *excluded)
        earth_nodes = []
        for c in range(ctx.n_cells):
            instr = compute_instr(ctx, p, c, qc_grid[c])
            ip = emit(f"instr:ping={p}:cell={c}", *instr)
            ship = compute_ship(ctx, p, c, {"payload": ip})
            sp = emit(f"ship:ping={p}:cell={c}", *ship)
            earth = compute_earth(ctx, p, c, {"payload": sp})
            earth_nodes.append({"payload": emit(f"earth:ping={p}:cell={c}", *earth)})
        profile = compute_profile(ctx, p, earth_nodes, {"payload": ep})
        emit(f"profile:ping={p}", *profile)
    store.commit_version(vid)
    return vid, report
