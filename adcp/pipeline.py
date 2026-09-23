"""Dependency-graph derivation pipeline.

Node graph (topological order)
------------------------------
    attitude/p{n}
    beam_qc/p{n}/c{m}
    instr_vel/p{n}/c{m}      <- beam_qc, beam geometry
    ship_vel/p{n}/c{m}       <- instr_vel, mounting-angle version
    earth_vel/p{n}/c{m}      <- ship_vel, attitude, declination version,
                                transform order, bottom track
    cell_validity/p{n}/c{m}  <- beam_qc, depth/Qc thresholds
    profile/p{n}             <- earth_vel + cell_validity of every cell

Each node stores a fingerprint over its rule version, input source rows,
upstream-node fingerprints and the parameter versions it consumes.
Propagation recomputes only nodes whose fingerprint changed; unchanged
nodes are copied with an explicit "still-valid" reason.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import transforms as tf

RULE_VERSION = "adcp-rules/1.0.0"

ORDER_DEFAULT = tf.ORDER_MOUNTING_FIRST

DEFAULT_PARAMS = [
    ("beam_angle_deg", 20.0, "deg", {"geometry": "janus-convex",
                                      "geometry_version": "geom/1"}),
    ("mounting_angle_deg", 0.5, "deg", {}),
    ("magnetic_declination_deg", -7.2, "deg", {"model": "WMM2020"}),
    ("transform_order", ORDER_DEFAULT, "text", {}),
    ("corr_min", 64.0, "count", {}),
    ("amp_min", 20.0, "count", {}),
    ("amp_max", 220.0, "count", {}),
    ("max_gap_s", 1.5, "s", {}),
    ("blanking_m", 2.0, "m", {}),
    ("cell_size_m", 1.0, "m", {}),
    ("max_depth_m", 7.0, "m", {}),
]


def fp(parts):
    h = hashlib.sha1()
    h.update(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def node_key(*parts):
    return "/".join(str(x) for x in parts)


class Pipeline:
    def __init__(self, store):
        self.store = store

    def ensure_params(self):
        existing = self.store.params()
        for name, value, unit, meta in DEFAULT_PARAMS:
            if name not in existing:
                self.store.set_param(name, value, unit, meta)

    # ----------------------------------------------------------- snapshot
    def _snapshot(self):
        src = self.store.latest_sources()
        idx = {}
        for r in src.values():
            if r["kind"] in ("beam_velocity", "correlation", "amplitude"):
                idx[(r["kind"], r["ping"], r["beam"], r["cell"])] = r
            elif r["kind"] == "bad_beam_flag":
                idx[(r["kind"], r["ping"], r["beam"])] = r
            elif r["kind"] == "bottom_track":
                idx[(r["kind"], r["ping"])] = r
        params = self.store.params()
        p = {k: v["value"] for k, v in params.items()}
        pv = {k: v["version"] for k, v in params.items()}
        vel_rows = [r for r in src.values() if r["kind"] == "beam_velocity"]
        pings = sorted({r["ping"] for r in vel_rows})
        cells = sorted({r["cell"] for r in vel_rows})
        att_rows = sorted((r for r in src.values()
                           if r["kind"] == "attitude"), key=lambda r: r["t"])
        samples = []
        for r in att_rows:
            d = json.loads(r["raw"])
            samples.append((r["t"], float(d["heading"]),
                            float(d["pitch"]), float(d["roll"])))
        ping_t = {}
        for r in vel_rows:
            ping_t.setdefault(r["ping"], r["t"])
        return src, idx, p, pv, pings, cells, samples, ping_t

    def _sig_src(self, row):
        return ["src", row["id"], row["kind"], row["raw"], row["batch_id"]]

    def _sig_param(self, p, pv, name):
        return ["param", name, pv[name], p[name]]

    def _write(self, vid, key, status, payload, sig, reason, prev):
        fingerprint = fp(sig)
        if prev and prev["fingerprint"] == fingerprint \
                and prev["payload"] == payload:
            self.store.put_derived(vid, key, prev["status"], prev["payload"],
                                   fingerprint, sig,
                                   f"still-valid: inputs identical "
                                   f"(copied from v{prev['version_id']})")
        else:
            why = "recomputed"
            if sig and isinstance(sig[0], str):
                pass
            self.store.put_derived(vid, key, status, payload, fingerprint,
                                   sig, reason)
        return fingerprint

    # ---------------------------------------------------------- attitude
    def _attitude_node(self, vid, n, t, samples, src, p, pv, prev, events):
        att_rows = sorted((r for r in src.values()
                           if r["kind"] == "attitude"), key=lambda r: r["t"])
        sig = [["rule", RULE_VERSION], self._sig_param(p, pv, "max_gap_s")]
        crossing = False
        for i in range(1, len(samples)):
            gap_t = samples[i][0] - samples[i - 1][0]
            near = abs(samples[i][0] - t) <= p["max_gap_s"] or \
                abs(samples[i - 1][0] - t) <= p["max_gap_s"]
            if near and abs(samples[i][1] - samples[i - 1][1]) > 180.0:
                crossing = True
        att, diag = tf.interpolate_attitude(samples, t, p["max_gap_s"])
        for r in att_rows:
            if abs(r["t"] - t) <= (p["max_gap_s"] + 0.5):
                sig.append(self._sig_src(r))
        payload = {"ping": n, "t": t, "attitude": att,
                   "diagnostic": diag, "frame": "attitude",
                   "unit": "deg", "heading_unwrapped": crossing}
        reason = "recomputed: attitude interpolation / 360 unwrap"
        fingerprint = self._write(vid, node_key("attitude", f"p{n}"),
                                  "ok" if att else "diagnostic", payload,
                                  sig, reason, prev)
        if diag:
            events.append(("attitude_gap", {
                "ping": n, "t": t, "diagnostic": diag,
                "node": node_key("attitude", f"p{n}"),
                "invariant": "attitude is never interpolated across a time "
                             "gap; values stay in instrument frame with a "
                             "diagnostic"}))
        if crossing:
            events.append(("heading_unwrap", {
                "ping": n, "t": t,
                "node": node_key("attitude", f"p{n}"),
                "rule": "heading difference >180 deg between bracketing "
                        "samples unwrapped to the short arc"}))
        return fingerprint

    # ----------------------------------------------------------- beam qc
    def _beam_qc_node(self, vid, n, m, src, p, pv, prev):
        sig = [["rule", RULE_VERSION]]
        beams = []
        for b in (1, 2, 3, 4):
            vel = src.get(("beam_velocity", n, b, m))
            cor = src.get(("correlation", n, b, m))
            amp = src.get(("amplitude", n, b, m))
            flg = src.get(("bad_beam_flag", n, b))
            for r in (vel, cor, amp, flg):
                if r:
                    sig.append(self._sig_src(r))
            reasons = []
            if flg is not None and flg["value"] == 1:
                reasons.append("bad-beam flag set")
            if cor is not None and cor["value"] < p["corr_min"]:
                reasons.append(
                    f"correlation {cor['value']:.0f} < {p['corr_min']:.0f}")
            if amp is not None and (
                    amp["value"] < p["amp_min"]
                    or amp["value"] > p["amp_max"]):
                reasons.append(
                    f"amplitude {amp['value']:.0f} outside "
                    f"[{p['amp_min']:.0f},{p['amp_max']:.0f}]")
            if vel is None or vel["value"] is None:
                reasons.append("missing beam measurement")
            beams.append({"beam": b,
                          "velocity": None if vel is None else vel["value"],
                          "correlation": None if cor is None else cor["value"],
                          "amplitude": None if amp is None else amp["value"],
                          "bad_flag": None if flg is None else flg["value"],
                          "source_ids": [r["id"] for r in (vel, cor, amp, flg)
                                         if r],
                          "reasons": reasons,
                          "flagged_bad": flg is not None
                          and flg["value"] == 1})
        for name in ("corr_min", "amp_min", "amp_max"):
            sig.append(self._sig_param(p, pv, name))
        payload = {"ping": n, "cell": m, "beams": beams,
                   "frame": "beam", "unit": {"velocity": "m/s",
                                             "correlation": "count",
                                             "amplitude": "count"}}
        return self._write(vid, node_key("beam_qc", f"p{n}", f"c{m}"),
                           "ok", payload, sig,
                           "recomputed: beam quality assessment", prev), beams

    # ------------------------------------------------------- instr / ship
    def _instr_node(self, vid, n, m, beams, src, p, pv, prev, decision,
                    events):
        sig = [["rule", RULE_VERSION],
               ["geom", p["beam_angle_deg"], pv["beam_angle_deg"],
                p.get("geometry", "janus-convex")]]
        sig += [["upstream", node_key("beam_qc", f"p{n}", f"c{m}"),
                 self._beam_qc_fp[(n, m)]]]
        failed = [b["beam"] for b in beams if b["flagged_bad"]]
        usable = [b for b in beams if not b["flagged_bad"]
                  and b["velocity"] is not None]
        candidates = []
        status = "ok"
        if len(usable) >= 4:
            vec, err = tf.instrument_from_beams(
                [b["velocity"] for b in usable], p["beam_angle_deg"])
            candidates.append({
                "id": "cand-beam4", "kind": "beam4", "beams_used": [1, 2, 3, 4],
                "value": [round(x, 6) for x in vec.tolist()],
                "error_velocity": round(err, 6), "adopted": True,
                "note": "four-beam least-squares solution"})
        elif len(usable) == 3:
            vec3, err3 = tf.instrument_from_beams(
                [b["velocity"] for b in usable], p["beam_angle_deg"])
            used = [b["beam"] for b in usable]
            candidates.append({
                "id": "cand-beam3", "kind": "beam3", "beams_used": used,
                "value": [round(x, 6) for x in vec3.tolist()],
                "error_velocity": round(err3, 6), "adopted": None,
                "note": "three-beam solution honouring the bad-beam flag"})
            contradicted = False
            for b in beams:
                if b["flagged_bad"] and not any(
                        ("correlation" in r or "amplitude" in r)
                        for r in b["reasons"]):
                    contradicted = True
            if contradicted:
                vec4, err4 = tf.instrument_from_beams(
                    [b["velocity"] for b in beams], p["beam_angle_deg"])
                candidates.append({
                    "id": "cand-beam4", "kind": "beam4", "beams_used": [1, 2, 3, 4],
                    "value": [round(x, 6) for x in vec4.tolist()],
                    "error_velocity": round(err4, 6), "adopted": None,
                    "note": "four-beam solution: bad-beam flag contradicted "
                            "by good correlation/amplitude"})
                status = "ambiguous"
                events.append(("three_beam_ambiguity", {
                    "ping": n, "cell": m, "failed_beam": failed[0],
                    "node": node_key("instr_vel", f"p{n}", f"c{m}"),
                    "candidate_ids": ["cand-beam3", "cand-beam4"],
                    "invariant": "3-beam and 4-beam solutions are stored as "
                                 "separate candidates, never merged or "
                                 "averaged into one value",
                    "rule": "when uniqueness cannot be established, every "
                            "legal candidate is retained pending a decision"}))
            else:
                candidates[0]["adopted"] = True
                events.append(("single_beam_failure", {
                    "ping": n, "cell": m, "failed_beam": failed[0],
                    "node": node_key("instr_vel", f"p{n}", f"c{m}"),
                    "candidate_id": "cand-beam3",
                    "rule": "flag consistent with poor QC, single legal "
                            "3-beam solution; no 4-beam value fabricated"}))
        else:
            status = "failed"
            payload = {"ping": n, "cell": m, "frame": "instrument",
                       "unit": "m/s", "candidates": [],
                       "diagnostic": "fewer than 3 usable beams",
                       "geometry_version": pv["beam_angle_deg"]}
            sig.append(["decision", None])
            return self._write(vid, node_key("instr_vel", f"p{n}", f"c{m}"),
                               status, payload, sig,
                               "recomputed: beam-to-instrument conversion",
                               prev)
        if decision:
            chosen = {c["id"]: c for c in candidates}
            for c in candidates:
                c["adopted"] = c["id"] == decision
            sig.append(["decision", decision])
            events.append(("candidate_adoption", {
                "ping": n, "cell": m, "node":
                node_key("instr_vel", f"p{n}", f"c{m}"),
                "adopted": decision,
                "valid": decision in chosen}))
        elif status != "ambiguous":
            sig.append(["decision", None])
        else:
            sig.append(["decision", "pending"])
        payload = {"ping": n, "cell": m, "frame": "instrument",
                   "unit": "m/s", "status": status, "candidates": candidates,
                   "geometry": {"beam_angle_deg": p["beam_angle_deg"],
                                "geometry": "janus-convex",
                                "geometry_version": pv["beam_angle_deg"]}}
        return self._write(vid, node_key("instr_vel", f"p{n}", f"c{m}"),
                           status, payload, sig,
                           "recomputed: beam-to-instrument conversion", prev)

    def _ship_node(self, vid, n, m, instr_payload, p, pv, prev):
        sig = [["rule", RULE_VERSION],
               self._sig_param_v(p, pv, "mounting_angle_deg"),
               ["upstream", node_key("instr_vel", f"p{n}", f"c{m}"),
                self._instr_fp[(n, m)]]]
        out = []
        for c in instr_payload.get("candidates", []):
            ship = tf.instrument_to_ship(c["value"], p["mounting_angle_deg"])
            out.append({"id": c["id"],
                        "value": [round(x, 6) for x in ship.tolist()]})
        payload = {"ping": n, "cell": m, "frame": "ship", "unit": "m/s",
                   "mounting_angle_deg": p["mounting_angle_deg"],
                   "mounting_version": pv["mounting_angle_deg"],
                   "candidates": out}
        return self._write(vid, node_key("ship_vel", f"p{n}", f"c{m}"),
                           instr_payload.get("status", "ok"), payload, sig,
                           "recomputed: instrument->ship (mounting angle)",
                           prev), out

    def _sig_param_v(self, p, pv, name):
        return ["param", name, pv[name], p[name]]

    def _earth_node(self, vid, n, m, attitude, ship_cands, src, p, pv, prev):
        bt = src.get(("bottom_track", n))
        bt_uv = None
        if bt is not None:
            d = json.loads(bt["raw"])
            bt_uv = [d["u"], d["v"]]
        sig = [["rule", RULE_VERSION],
               self._sig_param_v(p, pv, "magnetic_declination_deg"),
               self._sig_param_v(p, pv, "transform_order"),
               ["upstream", node_key("ship_vel", f"p{n}", f"c{m}"),
                self._ship_fp[(n, m)]],
               ["upstream", node_key("attitude", f"p{n}"), self._att_fp[n]]]
        if bt is not None:
            sig.append(self._sig_src(bt))
        cands = []
        status = "ok"
        diagnostic = None
        if attitude is None:
            status = "no_attitude"
            diagnostic = "earth transform blocked; velocity retained in " \
                "instrument/ship frame (no interpolation across gap)"
        for c in ship_cands:
            entry = {"id": c["id"]}
            if attitude is None:
                entry["earth"] = None
                entry["water"] = None
            else:
                earth, _, order = tf.instr_to_earth(
                    c["value"], attitude, p["mounting_angle_deg"],
                    p["magnetic_declination_deg"], order=p["transform_order"])
                earth = earth.tolist()
                water = list(earth)
                if bt_uv is not None:
                    water[0] -= bt_uv[0]
                    water[1] -= bt_uv[1]
                entry["earth"] = [round(x, 6) for x in earth]
                entry["water"] = [round(x, 6) for x in water]
            cands.append(entry)
        payload = {"ping": n, "cell": m, "frame": "earth(ENU)",
                   "unit": "m/s", "order": p["transform_order"],
                   "order_version": pv["transform_order"],
                   "declination_deg": p["magnetic_declination_deg"],
                   "declination_version": pv["magnetic_declination_deg"],
                   "bottom_track": bt_uv, "attitude": attitude,
                   "status": status, "diagnostic": diagnostic,
                   "candidates": cands}
        return self._write(vid, node_key("earth_vel", f"p{n}", f"c{m}"),
                           status, payload, sig,
                           "recomputed: ship->earth (attitude/declination/"
                           "order)", prev), payload

    def _validity_node(self, vid, n, m, beams, p, pv, prev):
        depth = p["blanking_m"] + (m - 0.5) * p["cell_size_m"]
        reasons = []
        if depth > p["max_depth_m"]:
            reasons.append(
                f"cell depth {depth:.1f}m beyond max {p['max_depth_m']:.1f}m")
        if all(any("correlation" in r for r in b["reasons"])
               for b in beams if b["velocity"] is not None):
            reasons.append("all beams below correlation threshold")
        sig = [["rule", RULE_VERSION],
               ["upstream", node_key("beam_qc", f"p{n}", f"c{m}"),
                self._beam_qc_fp[(n, m)]],
               self._sig_param_v(p, pv, "blanking_m"),
               self._sig_param_v(p, pv, "cell_size_m"),
               self._sig_param_v(p, pv, "max_depth_m")]
        payload = {"ping": n, "cell": m, "depth_m": round(depth, 3),
                   "included": len(reasons) == 0, "reasons": reasons}
        fpk = self._write(vid, node_key("cell_validity", f"p{n}", f"c{m}"),
                          "ok" if not reasons else "excluded", payload, sig,
                          "recomputed: cell validity / depth", prev)
        if reasons and any("beyond max" in r for r in reasons):
            self._pending_excluded.append(
                ("invalid_depth_cell", n, m, reasons))
        return fpk, payload

    # ------------------------------------------------------------- derive
    def derive(self, label, parent=None, decisions=None, move_pointer=True,
               param_updates=None):
        store = self.store
        self.ensure_params()
        revisions = []
        if param_updates:
            for name, value, unit, meta in param_updates:
                ver = store.set_param(name, value, unit, meta)
                revisions.append((name, value, unit, meta, ver))
        if parent is None:
            parent = store.current_version()
        src, idx, p, pv, pings, cells, samples, ping_t = self._snapshot()
        prev_rows = {r["node_key"]: r for r in store.derived(parent)} \
            if parent else {}
        vid = store.begin_version(parent, label)
        events = []
        for name, value, unit, meta, ver in revisions:
            events.append(("param_revision", {
                "param": name, "version": ver, "value": value,
                "unit": unit, "meta": meta,
                "rule": "parameter revisions create new parameter versions; "
                        "old versions remain addressable in history"}))
        self._beam_qc_fp = {}
        self._instr_fp = {}
        self._ship_fp = {}
        self._att_fp = {}
        self._earth_fp = {}
        self._pending_excluded = []
        decisions = decisions or {}

        events.append(("fingerprint_components", {
            "rule_version": RULE_VERSION,
            "transform_order": p["transform_order"],
            "transform_order_version": pv["transform_order"],
            "beam_angle_deg": p["beam_angle_deg"],
            "beam_angle_version": pv["beam_angle_deg"],
            "beam_geometry": "janus-convex",
            "magnetic_declination_deg": p["magnetic_declination_deg"],
            "declination_version": pv["magnetic_declination_deg"],
            "mounting_angle_deg": p["mounting_angle_deg"],
            "mounting_version": pv["mounting_angle_deg"],
            "invariant": "transform order, beam geometry and magnetic "
                         "declination version all enter the derived "
                         "fingerprint"}))

        for n in pings:
            t = ping_t[n]
            self._att_fp[n] = self._attitude_node(
                vid, n, t, samples, src, p, pv,
                prev_rows.get(node_key("attitude", f"p{n}")), events)
            att_row = store.derived(vid, node_key("attitude", f"p{n}"))[0]
            attitude = att_row["payload"]["attitude"]
            prof_cells = []
            for m in cells:
                qc_fp, beams = self._beam_qc_node(
                    vid, n, m, idx, p, pv,
                    prev_rows.get(node_key("beam_qc", f"p{n}", f"c{m}")))
                self._beam_qc_fp[(n, m)] = qc_fp
                instr_fp = self._instr_node(
                    vid, n, m, beams, src, p, pv,
                    prev_rows.get(node_key("instr_vel", f"p{n}", f"c{m}")),
                    decisions.get((n, m)), events)
                self._instr_fp[(n, m)] = instr_fp
                instr_row = store.derived(
                    vid, node_key("instr_vel", f"p{n}", f"c{m}"))[0]
                ship_fp, ship_cands = self._ship_node(
                    vid, n, m, instr_row["payload"], p, pv,
                    prev_rows.get(node_key("ship_vel", f"p{n}", f"c{m}")))
                self._ship_fp[(n, m)] = ship_fp
                earth_fp, earth_payload = self._earth_node(
                    vid, n, m, attitude, ship_cands, idx, p, pv,
                    prev_rows.get(node_key("earth_vel", f"p{n}", f"c{m}")))
                self._earth_fp[(n, m)] = earth_fp
                val_fp, val_payload = self._validity_node(
                    vid, n, m, beams, p, pv,
                    prev_rows.get(node_key("cell_validity", f"p{n}", f"c{m}")))
                adopted = None
                for c in instr_row["payload"].get("candidates", []):
                    if c.get("adopted"):
                        adopted = c["id"]
                earth_map = {c["id"]: c
                             for c in earth_payload["candidates"]}
                chosen = earth_map.get(adopted, {})
                prof_cells.append({
                    "cell": m, "depth_m": val_payload["depth_m"],
                    "included": val_payload["included"],
                    "reasons": val_payload["reasons"],
                    "adopted_candidate": adopted,
                    "water_velocity": chosen.get("water"),
                    "status": earth_payload["status"],
                    "node": node_key("earth_vel", f"p{n}", f"c{m}")})
            sig = [["rule", RULE_VERSION]]
            for m in cells:
                sig.append(["upstream", f"earth_vel/p{n}/c{m}",
                            self._ship_fp[(n, m)]])
            payload = {"ping": n, "t": t, "frame": "earth(ENU)",
                       "unit": "m/s", "cells": prof_cells}
            self._write(vid, node_key("profile", f"p{n}"), "ok", payload,
                        sig, "recomputed: vertical profile assembly",
                        prev_rows.get(node_key("profile", f"p{n}")))

        for kind, n, m, reasons in self._pending_excluded:
            events.append((kind, {
                "ping": n, "cell": m, "reasons": reasons,
                "node": node_key("cell_validity", f"p{n}", f"c{m}"),
                "rule": "excluded cells stay in place with diagnostics; "
                        "they are never dropped to make a plot cleaner"}))

        version_fp = fp([[str(k), v] for k, v in
                         sorted(self._earth_fp.items())] +
                        [[n, self._att_fp[n]] for n in pings])
        store.commit_version(vid, version_fp, move_pointer=move_pointer)
        for kind, payload in events:
            store.add_event(vid, kind, payload)
        store.add_event(vid, "derive_completed", {
            "version": vid, "label": label, "parent": parent,
            "fingerprint": version_fp,
            "nodes": len(store.derived(vid))})
        return vid

    # ----------------------------------------------------------- decision
    def decide(self, base_version, node, candidate_id, note=""):
        """Record a human decision as a new version (conflict-safe)."""
        store = self.store
        row = store.derived(base_version, node)
        if not row:
            raise ValueError(f"node {node} not found in v{base_version}")
        payload = row[0]["payload"]
        if not any(c["id"] == candidate_id
                   for c in payload.get("candidates", [])):
            raise ValueError(f"candidate {candidate_id} not in {node}")
        n, m = payload["ping"], payload["cell"]

        def build(vid):
            # re-derive from the base with this decision applied
            saved_current = store.current_version()
            src, idx, p, pv, pings, cells, samples, ping_t = self._snapshot()
            prev_rows = {r["node_key"]: r
                         for r in store.derived(base_version)}
            self._beam_qc_fp, self._instr_fp = {}, {}
            self._ship_fp, self._att_fp = {}, {}
            self._earth_fp = {}
            self._pending_excluded = []
            events = []
            for nn in pings:
                t = ping_t[nn]
                self._att_fp[nn] = self._attitude_node(
                    vid, nn, t, samples, src, p, pv,
                    prev_rows.get(node_key("attitude", f"p{nn}")), events)
                att_row = store.derived(
                    vid, node_key("attitude", f"p{nn}"))[0]
                attitude = att_row["payload"]["attitude"]
                prof_cells = []
                for mm in cells:
                    qc_fp, beams = self._beam_qc_node(
                        vid, nn, mm, idx, p, pv,
                        prev_rows.get(node_key("beam_qc", f"p{nn}",
                                               f"c{mm}")))
                    self._beam_qc_fp[(nn, mm)] = qc_fp
                    dec = candidate_id if (nn, mm) == (n, m) else None
                    instr_fp = self._instr_node(
                        vid, nn, mm, beams, src, p, pv,
                        prev_rows.get(node_key("instr_vel", f"p{nn}",
                                               f"c{mm}")), dec, events)
                    self._instr_fp[(nn, mm)] = instr_fp
                    instr_row = store.derived(
                        vid, node_key("instr_vel", f"p{nn}", f"c{mm}"))[0]
                    ship_fp, ship_cands = self._ship_node(
                        vid, nn, mm, instr_row["payload"], p, pv,
                        prev_rows.get(node_key("ship_vel", f"p{nn}",
                                               f"c{mm}")))
                    self._ship_fp[(nn, mm)] = ship_fp
                    earth_fp, earth_payload = self._earth_node(
                        vid, nn, mm, attitude, ship_cands, idx, p, pv,
                        prev_rows.get(node_key("earth_vel", f"p{nn}",
                                                f"c{mm}")))
                    val_fp, val_payload = self._validity_node(
                        vid, nn, mm, beams, p, pv,
                        prev_rows.get(node_key("cell_validity", f"p{nn}",
                                                f"c{mm}")))
                    adopted = None
                    for c in instr_row["payload"].get("candidates", []):
                        if c.get("adopted"):
                            adopted = c["id"]
                    earth_map = {c["id"]: c
                                 for c in earth_payload["candidates"]}
                    chosen = earth_map.get(adopted, {})
                    prof_cells.append({
                        "cell": mm, "depth_m": val_payload["depth_m"],
                        "included": val_payload["included"],
                        "reasons": val_payload["reasons"],
                        "adopted_candidate": adopted,
                        "water_velocity": chosen.get("water"),
                        "status": earth_payload["status"],
                        "node": node_key("earth_vel", f"p{nn}", f"c{mm}")})
                sig = [["rule", RULE_VERSION]]
                for mm in cells:
                    sig.append(["upstream", f"earth_vel/p{nn}/c{mm}",
                                self._ship_fp[(nn, mm)]])
                payload2 = {"ping": nn, "t": t, "frame": "earth(ENU)",
                            "unit": "m/s", "cells": prof_cells}
                self._write(vid, node_key("profile", f"p{nn}"), "ok",
                            payload2, sig,
                            "recomputed: vertical profile assembly",
                            prev_rows.get(node_key("profile", f"p{nn}")))
            version_fp = fp([[str(k), v] for k, v in
                             sorted(self._earth_fp.items())])
            for kind, pl in events:
                store.add_event(vid, kind, pl)
            return version_fp

        vid, conflict_id = store.save_edit(
            base_version, f"decide {node} -> {candidate_id}", build)
        store.add_event(vid, "decision", {
            "node": node, "candidate": candidate_id, "note": note,
            "base_version": base_version,
            "rule": "decisions create new versions; unselected candidates "
                    "are retained, never overwritten"})
        return vid, conflict_id
