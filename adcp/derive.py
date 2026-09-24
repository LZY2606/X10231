"""Dependency-graph derivation engine for ADCP velocity evidence.

Pipeline per ping/cell:
  beam quality -> beam/instrument solve -> instrument -> ship -> earth
  -> vertical profile, plus per-ping bottom track.

Propagation is graph based. Each derived node records dependency edges to the
facts and earlier nodes it needs. Given a set of changed fact nodes the engine
walks those edges to find every affected derived node, recomputes only them,
and keeps every other node untouched (reporting why it stayed valid).

Ambiguity is never collapsed:
  * four good beams -> exactly one four_beam value WITH error velocity;
  * three good beams -> BOTH legal candidates (three_beam_lsq and
    three_beam_wzero), neither adopted, never confused with four_beam;
  * fewer than three good beams -> the cell is excluded, no velocity.
Attitude interpolation never crosses a flagged time gap. The derived
fingerprint always includes transform order, beam geometry sid and magnetic
declination sid/version.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from . import attitude as att
from . import cases
from .fingerprint import fingerprint
from .geometry import look_matrix, solve
from .quality import judge_cell
from .units import ConversionError, to_mps


def osid(otype: str, key: str, token: object) -> str:
    h = hashlib.sha256(repr((otype, key, token)).encode()).hexdigest()[:12]
    return f"sid_{otype}_{h}"


def _f(facts, kind, key):
    return facts.get(kind, {}).get(key)


def _config(facts):
    geo = _f(facts, "beam_geometry", "janus4")
    dec = _f(facts, "declination", "magnetic_declination")
    order = _f(facts, "transform_order", "ship_to_earth")
    return {
        "geo": geo["value"], "geo_sid": geo["sid"],
        "geo_ver": geo["version"],
        "dec": dec["value"], "dec_sid": dec["sid"], "dec_ver": dec["version"],
        "order": order["value"]["order"], "order_sid": order["sid"],
        "order_ver": order["version"],
        "mount": {a: _f(facts, "mount", a)["value"]["deg"]
                  for a in ("heading_deg", "pitch_deg", "roll_deg")},
        "mount_sids": {a: _f(facts, "mount", a)["sid"] for a in
                       ("heading_deg", "pitch_deg", "roll_deg")},
        "thr": _f(facts, "thresholds", "quality")["value"],
        "thr_sid": _f(facts, "thresholds", "quality")["sid"],
    }


def _attitude_samples(facts):
    keys = sorted(facts.get("attitude_heading", {}),
                  key=lambda k: float(k[3:]))
    out = []
    for key in keys:
        tk = key[3:]
        h = facts["attitude_heading"][key]["value"]
        out.append(att.AttitudeSample(
            float(tk), h["deg"], facts["attitude_pitch"][key]["value"]["deg"],
            facts["attitude_roll"][key]["value"]["deg"],
            bool(h.get("gap_after"))))
    return out


def _radials(facts, key, prefix="beam_velocity"):
    vals, bad = [], []
    for b in range(4):
        flag = _f(facts, "bad_beam_flag", f"{key}:b{b}")
        if flag and flag["value"].get("bad"):
            bad.append(b)
        vf = _f(facts, prefix, f"{key}:b{b}")
        if vf is None:
            bad.append(b)
            continue
        try:
            vals.append(to_mps(vf["value"].get("cm_s"),
                               vf.get("raw_unit") or "cm/s"))
        except ConversionError:
            bad.append(b)
    if len(vals) != 4:
        return None, sorted(set(bad))
    return np.asarray(vals), sorted(set(bad))


def _quality_inputs(facts, ping, cell):
    key = f"{ping}:c{cell}"
    corr = {b: _f(facts, "correlation", f"{key}:b{b}")["value"]["count"]
            for b in range(4)}
    amp = {b: _f(facts, "amplitude", f"{key}:b{b}")["value"]["db"]
           for b in range(4)}
    bad = [b for b in range(4)
           if _f(facts, "bad_beam_flag", f"{key}:b{b}")["value"]["bad"]]
    depth_f = _f(facts, "cell_depth", key)
    rng_f = _f(facts, "bottom_track", f"{ping}:range")
    return {"key": key, "corr": corr, "amp": amp, "bad": bad,
            "depth_m": depth_f["value"].get("m"),
            "depth_diag": depth_f["value"].get("diagnostic"),
            "bottom_range_m": rng_f["value"]["range_m"]}


def _chain_fp(cfg, base, family, atti):
    return fingerprint({
        "geometry_sid": cfg["geo_sid"], "geometry_version": cfg["geo_ver"],
        "transform_order": cfg["order"], "order_sid": cfg["order_sid"],
        "order_version": cfg["order_ver"],
        "declination_sid": cfg["dec_sid"],
        "declination_version": cfg["dec_ver"],
        "declination_value": cfg["dec"],
        "mount_sids": cfg["mount_sids"], "mount": cfg["mount"],
        "attitude": atti if atti is not None else None,
        "family": family, "radials": base,
    })


def _velocity_chain(ctx, key, radials, bad, scope, profile_rows=None,
                    cell=None, dep_facts=()):
    facts, cfg, g, m_mount, atti, objects, edges = (
        ctx["facts"], ctx["cfg"], ctx["g"], ctx["m_mount"], ctx["atti"],
        ctx["objects"], ctx["edges"])
    attitude_blocked = atti is None

    def add(otype, k, payload, family=None, status="ok", adopted=False,
            fp=None, deps=()):
        o = {"sid": osid(otype, k, (payload, family, fp)),
             "family": family, "status": status, "payload": payload,
             "fingerprint": fp, "adopted": adopted}
        objects.setdefault(otype, {})[k] = o
        for d in deps:
            edges.append((d, f"{otype}:{k}"))
        return o

    sols = solve(g, radials, bad) if not attitude_blocked else []
    is_three = len(bad) == 1 and radials is not None and not attitude_blocked

    base_token = [round(float(x), 9) for x in (radials if radials is not None
                                               else np.zeros(4))]
    for sol in sols:
        fam = sol.family
        v_inst = [float(x) for x in sol.velocity]
        inst_payload = {
            "key": key, "scope": scope, "family": fam,
            "velocity": v_inst, "frame": "instrument",
            "units": {"velocity": "m/s", "coordinate": "instrument"},
            "used_beams": list(sol.used_beams),
            "error_velocity_m_s": sol.error_velocity,
            "assumption": sol.assumption, "residual": sol.residual,
        }
        fp_inst = _chain_fp(cfg, base_token, fam, atti)
        deps = list(dep_facts) + [f"beam_geometry:janus4"]
        add("inst_velocity", f"{key}:{fam}", inst_payload, family=fam,
            fp=fp_inst, deps=deps)

        v_ship = (m_mount @ sol.velocity).tolist()
        ship_payload = {
            "key": key, "scope": scope, "family": fam,
            "velocity": [float(x) for x in v_ship], "frame": "ship",
            "units": {"velocity": "m/s", "coordinate": "ship"},
            "used_beams": list(sol.used_beams), "assumption": sol.assumption,
        }
        fp_ship = fingerprint({"inst": fp_inst, "mount": cfg["mount_sids"]})
        add("ship_velocity", f"{key}:{fam}", ship_payload, family=fam,
            fp=fp_ship,
            deps=[f"inst_velocity:{key}:{fam}"] + list(cfg["mount_sids"]
                                                       and [
                    f"mount:heading_deg", f"mount:pitch_deg",
                    f"mount:roll_deg"]))

        v_earth = (ctx["m_earth"] @ sol.velocity).tolist()
        earth_payload = {
            "key": key, "scope": scope, "family": fam,
            "velocity": [float(x) for x in v_earth], "frame": "earth",
            "units": {"velocity": "m/s", "coordinate": "earth (ENU, "
                      "declination applied)"},
            "heading_deg": atti["heading_deg"],
            "pitch_deg": atti["pitch_deg"], "roll_deg": atti["roll_deg"],
            "declination_deg": cfg["dec"]["deg"], "order": cfg["order"],
            "used_beams": list(sol.used_beams), "assumption": sol.assumption,
            "ambiguous": fam.startswith("three_beam"),
        }
        fp_earth = fingerprint({
            "ship": fp_ship, "attitude": atti, "dec_sid": cfg["dec_sid"],
            "dec_ver": cfg["dec_ver"], "order_sid": cfg["order_sid"],
            "order": cfg["order"], "family": fam})
        add("earth_velocity", f"{key}:{fam}", earth_payload, family=fam,
            status="ambiguous" if fam.startswith("three_beam") else "ok",
            adopted=False if fam.startswith("three_beam") else True,
            fp=fp_earth,
            deps=[f"ship_velocity:{key}:{fam}", "declination:"
                  "magnetic_declination", "transform_order:ship_to_earth"]
                 + [f"attitude_heading:att{t:g}" for t in
                    {round(atti["bracket"][0], 9),
                     round(atti["bracket"][1], 9)}])
        if profile_rows is not None and fam == "four_beam":
            profile_rows.append({"cell": cell, "status": "ok",
                                 "east_m_s": v_earth[0],
                                 "north_m_s": v_earth[1],
                                 "up_m_s": v_earth[2],
                                 "earth_sid": osid(
                                     "earth_velocity", f"{key}:{fam}",
                                     (earth_payload, fam, fp_earth)),
                                 "family": fam})
        elif profile_rows is not None and fam == "three_beam_lsq":
            profile_rows.append({
                "cell": cell, "status": "ambiguous",
                "candidates": [
                    {"family": s.family,
                     "earth_velocity": (ctx["m_earth"] @ s.velocity).tolist(),
                     "assumption": s.assumption}
                    for s in sols if s.family.startswith("three_beam")],
                "note": "two legal 3-beam candidates; neither adopted"})

    return {"ambiguous": is_three}


def build_objects(facts):
    """Full derivation; returns (objects, edges, diagnostics)."""
    cfg = _config(facts)
    g = look_matrix(cfg["geo"]["beam_angle_deg"], cfg["geo"]["azimuths_deg"])
    samples = _attitude_samples(facts)
    objects: dict[str, dict[str, dict]] = {}
    edges: list[tuple[str, str]] = []
    diagnostics = {"excluded": [], "ambiguous": [], "attitude_gap": []}

    for pi, ping in enumerate(cases.PINGS):
        try:
            atti = att.interpolate_attitude(samples, cases.PING_TIMES[pi])
        except att.AttitudeGap:
            atti = None
            diagnostics["attitude_gap"].append(ping)
        m_mount = att.mount_matrix(cfg["mount"]["roll_deg"],
                                   cfg["mount"]["pitch_deg"],
                                   cfg["mount"]["heading_deg"])
        m_earth = (att.earth_matrix(atti["heading_deg"], atti["pitch_deg"],
                                    atti["roll_deg"], cfg["dec"]["deg"],
                                    cfg["order"])
                   if atti else None)
        ctx = {"facts": facts, "cfg": cfg, "g": g, "m_mount": m_mount,
               "atti": atti, "m_earth": m_earth, "objects": objects,
               "edges": edges}

        bt_key = ping
        bt_radials, bt_bad = _radials(facts, bt_key, prefix="bt_velocity")
        if atti is not None and bt_radials is not None and not bt_bad:
            _velocity_chain(
                ctx, ping, bt_radials, [], "bottom_track",
                dep_facts=[f"bt_velocity:{ping}:b{b}" for b in range(4)]
                + [f"bottom_track:{ping}:range"])

        profile_rows = []
        for cell in cases.CELLS:
            qi = _quality_inputs(facts, ping, cell)
            q = judge_cell(
                ping, cell, qi["depth_m"], qi["bottom_range_m"], qi["bad"],
                qi["corr"], qi["amp"], cfg["thr"]["correlation_count"],
                cfg["thr"]["amplitude_db"],
                attitude_blocked=atti is None)
            qd = q.to_dict()
            if qi["depth_diag"]:
                qd["depth_diagnostic"] = qi["depth_diag"]
            deps = [f"cell_depth:{qi['key']}",
                    f"bottom_track:{ping}:range"]
            for b in range(4):
                deps += [f"beam_velocity:{qi['key']}:b{b}",
                         f"correlation:{qi['key']}:b{b}",
                         f"amplitude:{qi['key']}:b{b}",
                         f"bad_beam_flag:{qi['key']}:b{b}"]
            qobj = {"sid": osid("cell_quality", qi["key"], qd),
                    "family": None,
                    "status": "excluded" if q.excluded else "ok",
                    "payload": qd,
                    "fingerprint": fingerprint({"q": qd,
                                                "thr_sid": cfg["thr_sid"]}),
                    "adopted": False}
            objects.setdefault("cell_quality", {})[qi["key"]] = qobj
            for d in deps:
                edges.append((d, f"cell_quality:{qi['key']}"))
            if q.excluded:
                diagnostics["excluded"].append(
                    {"ping": ping, "cell": cell, "reason": q.reason,
                     "detail": q.detail})
                profile_rows.append({"cell": cell, "status": "excluded",
                                     "reason": q.reason})
                continue
            radials, bad = _radials(facts, qi["key"])
            res = _velocity_chain(
                ctx, qi["key"], radials, bad, "water",
                profile_rows=profile_rows, cell=cell,
                dep_facts=[f"cell_quality:{qi['key']}"])
            if res["ambiguous"]:
                diagnostics["ambiguous"].append({"ping": ping, "cell": cell})

        prof = {"ping": ping, "rows": profile_rows,
                "depth_unit": "m", "velocity_frame": "earth",
                "attitude_status": ("gap_blocked" if atti is None
                                    else "interpolated")}
        fpp = fingerprint({"ping": ping, "rows": profile_rows,
                           "geo": cfg["geo_sid"], "order": cfg["order_sid"],
                           "dec": cfg["dec_sid"]})
        objects.setdefault("profile", {})[ping] = {
            "sid": osid("profile", ping, (prof, fpp)), "family": None,
            "status": "ok", "payload": prof, "fingerprint": fpp,
            "adopted": False}
        for c in cases.CELLS:
            edges.append((f"cell_quality:{ping}:c{c}", f"profile:{ping}"))
            edges.append((f"earth_velocity:{ping}:c{c}:four_beam",
                          f"profile:{ping}"))
    return objects, edges, diagnostics


def derive_all(facts):
    return build_objects(facts)


def affected_nodes(edges, changed_fact_nodes):
    """Forward graph closure: derived nodes transitively dependent on any
    changed source node. Pings not in the closure remain valid."""
    adj: dict[str, set[str]] = {}
    for src, dst in edges:
        adj.setdefault(src, set()).add(dst)
    seen = set()
    queue = deque(changed_fact_nodes)
    while queue:
        node = queue.popleft()
        for dst in adj.get(node, ()):  # noqa: PERF400 - clarity
            if dst not in seen:
                seen.add(dst)
                queue.append(dst)
    return seen


def affected_pings(changed_fact_nodes):
    pings = set()
    for node in changed_fact_nodes:
        kind, _, rest = node.partition(":")
        pid = rest.split(":", 1)[0] if ":" in rest else rest
        if pid in cases.PINGS:
            pings.add(pid)
    return pings
