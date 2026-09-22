"""依赖图推导：beam -> instrument -> ship -> earth -> profile。

语义红线
--------
- 三波束解与四波束解不混为一值：候选各自带稳定身份
  （<node>#beam4 / <node>#beam3_missing=<b>），指纹不同。
- 不确定/多解不被吞：4 波束误差速度超阈或临界波束时，节点 status=
  indeterminate，全部合法候选保留，没有 canonical。
- 姿态插值不跨时间缺口：缺姿态 ping 仅在同一连续段内插值。
- 坐标转换顺序、波束几何版本、磁偏角版本、安装角版本全部进入指纹。
- 重算按依赖图传播：仅受影响节点产生新版本；指纹相同的结果保持有效。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .fingerprint import digest, derivation_fingerprint, stable_id
from .geometry import BeamGeometry
from .transforms import (Attitude, Declination, MountAngles, interpolate_in_segment,
                         segments_without_gaps, transform_all, unwrap_headings)
from .units import parse_quantity

OP_VERSIONS = {
    "attitude": "op.attitude:v1",
    "instrument": "op.instrument:v1",
    "ship": "op.ship:v1",
    "earth": "op.earth:v1",
    "profile": "op.profile:v1",
    "bt_instrument": "op.instrument:v1",
    "bt_ship": "op.ship:v1",
    "bt_earth": "op.earth:v1",
}


class DerivationEngine:
    def __init__(self, store, branch_id="main"):
        self.store = store
        self.branch_id = branch_id
        self.cfg = {}

    # ---------- 规则 ----------------------------------------------------
    def load_rules(self):
        geo = self.store.get_current_rule("rule:beam_geometry")
        decl = self.store.get_current_rule("rule:declination")
        mount = self.store.get_current_rule("rule:mount")
        qual = self.store.get_current_rule("rule:quality")
        order = self.store.get_current_rule("rule:order")
        g = geo["data"]
        self.geometry = BeamGeometry(
            beam_angle_deg=g["beam_angle_deg"], convex=g["convex"],
            orientation=g["orientation"], version=geo["version"])
        self.declination = Declination(decl["data"]["east_deg"], decl["version"])
        m = mount["data"]
        self.mount = MountAngles(m["heading_deg"], m["pitch_deg"], m["roll_deg"],
                                 mount["version"])
        self.quality = qual["data"]
        self.quality_version = qual["version"]
        self.order = order["data"]["rotation_order"]
        self.order_version = order["version"]

    # ---------- 源读取 --------------------------------------------------
    def _src(self, kind, *scope):
        return self.store.get_current_source(stable_id("src", kind, *scope))

    def _beam_bundle(self, ping, cell, prefix="cell"):
        radials, usable, evidence, corr_hold = [], [], [], True
        for b in range(4):
            sv = self._src("beam_velocity", ping, cell, b)
            co = self._src("correlation", ping, cell, b)
            am = self._src("amplitude", ping, cell, b)
            bd = self._src("bad_beam", ping, cell, b)
            raw_bad = bd["data"]["raw"] if bd else None
            flagged = raw_bad in ("1", "true", "TRUE", "True")
            qv = None
            if sv and sv["data"]["value"] is not None:
                qv = float(sv["data"]["value"])
                radials.append(qv)
            corr_v = co["data"]["value"] if co else None
            ok = (not flagged) and qv is not None
            marginal = False
            reason = None
            hard_gate = self.quality.get("correlation_hard_min", 60)
            soft_gate = self.quality["correlation_min_count"]
            if flagged:
                reason = "bad_beam_flag"
            elif qv is None:
                reason = sv["data"].get("diagnostic") or "missing_velocity"
            elif corr_v is not None and corr_v < hard_gate:
                reason = (f"correlation_below_hard_gate({int(corr_v)}<"
                          f"{hard_gate})")
                ok = False
            elif corr_v is not None and corr_v < soft_gate:
                # 临界带：几何上仍可用，但触发多解/不确定，全部候选保留
                marginal = True
                corr_hold = True
                reason = (f"correlation_marginal({hard_gate}<={int(corr_v)}<"
                          f"{soft_gate})")
            if ok:
                usable.append(b)
            evidence.append({
                "beam": b, "raw": sv["data"]["raw"] if sv else None,
                "unit": sv["data"]["unit"] if sv else None,
                "value": qv, "source_id": sv["sid"] if sv else None,
                "correlation": corr_v, "correlation_source": co["sid"] if co else None,
                "amplitude": am["data"]["value"] if am else None,
                "bad_flag": raw_bad, "usable": ok, "marginal": marginal,
                "reason": reason,
            })
        return radials, usable, evidence, corr_hold

    # ---------- 姿态段 --------------------------------------------------
    def attitude_segments(self):
        times, heads, att_records = [], [], []
        for p in sorted(self._ping_index()):
            a = self._src("attitude", p)
            if a and a["data"]["heading"]["value"] is not None:
                times.append(a["data"]["time"])
                heads.append(a["data"]["heading"]["value"])
                att_records.append(p)
        unwrapped = unwrap_headings(heads)
        max_gap = self.quality["max_attitude_gap_s"]
        seg_bounds = segments_without_gaps(times, max_gap)
        segs = []
        for i0, i1 in seg_bounds:
            segs.append({
                "times": times[i0:i1 + 1],
                "pings": att_records[i0:i1 + 1],
                "heading": unwrapped[i0:i1 + 1],
                "pitch": [self._src("attitude", p)["data"]["pitch"]["value"]
                          for p in att_records[i0:i1 + 1]],
                "roll": [self._src("attitude", p)["data"]["roll"]["value"]
                         for p in att_records[i0:i1 + 1]],
            })
        return segs

    def _ping_index(self):
        pings = set()
        for r in self.store.scan_current_sources("beam_velocity"):
            pings.add(r["ping"])
        return pings

    def ping_time(self, ping):
        a = self._src("attitude", ping)
        if a and a["data"]["time"] is not None:
            return a["data"]["time"]
        d = self._src("beam_velocity", ping, 0, 0)
        return d["data"].get("ping_time") if d else None

    def resolve_attitude(self, ping, segs=None):
        """返回 (attitude|None, method, diagnostics)。绝不跨缺口插值。"""
        segs = segs or self.attitude_segments()
        a = self._src("attitude", ping)
        t = self.ping_time(ping)
        marker = self._src("attitude_gap_marker", ping)
        if marker is not None:
            return None, "gap", [{"code": "attitude_invalidated_record",
                                  "ping": ping, "reason": marker["data"].get("raw"),
                                  "note": "no interpolation across invalidated attitude"}]
        if a is not None and a["data"]["heading"]["value"] is not None:
            d = a["data"]
            return Attitude(d["heading"]["value"], d["pitch"]["value"],
                            d["roll"]["value"]), "measured", []
        diags = []
        for seg in segs:
            if seg["times"][0] <= t <= seg["times"][-1]:
                h = interpolate_in_segment(t, seg["times"], seg["heading"])
                p = interpolate_in_segment(t, seg["times"], seg["pitch"])
                r = interpolate_in_segment(t, seg["times"], seg["roll"])
                if None in (h, p, r):
                    diags.append("attitude_segment_undersampled")
                    return None, "gap", diags
                h360 = h % 360.0
                diags.append({
                    "code": "attitude_interpolated",
                    "segment_pings": seg["pings"],
                    "method": "linear_within_contiguous_segment",
                })
                return Attitude(h360, p, r, h), "interpolated", diags
        return None, "gap", [{"code": "attitude_gap_no_cross_gap_interpolation",
                              "ping": ping, "time": t,
                              "max_gap_s": self.quality["max_attitude_gap_s"]}]

    # ---------- instrument 解（三/四波束候选）--------------------------
    def instrument_candidates(self, ping, cell):
        _, usable, evidence, corr_hold = self._beam_bundle(ping, cell)
        candidates, status, diags = [], "ok", []
        rad = np.array([e["value"] if e["value"] is not None else np.nan
                        for e in evidence], float)

        if len(usable) < 3:
            return {"candidates": [], "status": "excluded", "beam_evidence": evidence,
                    "diagnostics": [{"code": "fewer_than_three_good_beams",
                                     "usable": usable}]}
        if len(usable) == 3:
            xyz, _ = self.geometry.radial_to_xyz(rad[usable], usable)
            candidates.append({
                "candidate_id": f"inst:p{ping}c{cell}#beam3_missing={sorted(set(range(4))-set(usable))[0]}",
                "kind": "beam3", "good_beams": usable,
                "missing_beam": sorted(set(range(4)) - set(usable))[0],
                "vector_ENDframe": np.round(xyz, 6).tolist(),
                "axes": ["fore", "starboard", "down"],
                "error_velocity": None,
            })
            status = "ok"
            diags.append({"code": "determinate_three_beam_solution",
                          "note": "no single error velocity; beam3 != beam4 identity"})
        else:
            xyz4, res4 = self.geometry.radial_to_xyz(rad, [0, 1, 2, 3])
            err = self.geometry.error_velocity(rad)
            cand4 = {
                "candidate_id": f"inst:p{ping}c{cell}#beam4",
                "kind": "beam4", "good_beams": usable, "missing_beam": None,
                "vector_ENDframe": np.round(xyz4, 6).tolist(),
                "axes": ["fore", "starboard", "down"],
                "error_velocity": round(err, 6),
                "residuals": np.round(res4, 6).tolist(),
            }
            candidates.append(cand4)
            # 全部合法三波束候选都保留（多解证据），不与四波束混值
            for drop in range(4):
                beams3 = [b for b in range(4) if b != drop]
                if any(not evidence[b]["usable"] for b in beams3):
                    continue
                xyz3, _ = self.geometry.radial_to_xyz(rad[beams3], beams3)
                candidates.append({
                    "candidate_id": f"inst:p{ping}c{cell}#beam3_missing={drop}",
                    "kind": "beam3", "good_beams": beams3, "missing_beam": drop,
                    "vector_ENDframe": np.round(xyz3, 6).tolist(),
                    "axes": ["fore", "starboard", "down"],
                    "error_velocity": None,
                })
            spread = max(np.linalg.norm(np.array(c["vector_ENDframe"]) - xyz4)
                         for c in candidates[1:])
            suspicious = corr_hold or abs(err) > self.quality["error_velocity_max_mps"]
            if suspicious:
                status = "indeterminate"
                diags.append({
                    "code": "ambiguous_beam4_vs_beam3_candidates",
                    "error_velocity_mps": round(err, 6),
                    "candidate_spread_mps": round(float(spread), 6),
                    "correlation_hold": corr_hold,
                    "note": "all legal candidates retained; awaiting human decision",
                })
            else:
                diags.append({"code": "beam4_adopted",
                              "error_velocity_mps": round(err, 6),
                              "note": "beam4 canonical; beam3 kept as provenance"})
        return {"candidates": candidates, "status": status,
                "beam_evidence": evidence, "diagnostics": diags}

    # ---------- 逐级变换 ------------------------------------------------
    def _chain(self, ping, cell, inst_payload, attitude_result):
        att, att_method, att_diags = attitude_result
        chain = []
        if att is None:
            return None, att_diags
        adopted = next((c for c in inst_payload["candidates"]
                        if c.get("adopted") or c["kind"] == "beam4"), None)
        if adopted is None:
            adopted = inst_payload["candidates"][0]
        out = transform_all(adopted["vector_ENDframe"], att, self.mount,
                            self.declination, self.order)
        return {
            "instrument": {"vector": np.round(out["instrument"], 6).tolist(),
                           "axes": ["fore", "starboard", "down"],
                           "candidate_id": adopted["candidate_id"],
                           "solution_kind": adopted["kind"]},
            "ship": {"vector": np.round(out["ship"], 6).tolist(),
                     "axes": ["fore", "starboard", "down"]},
            "earth": {"vector_END": np.round(out["earth_END"], 6).tolist(),
                      "vector_ENU": np.round(out["earth_ENU"], 6).tolist(),
                      "axes_END": ["east", "north", "down"]},
            "effective_attitude": out["effective"],
            "attitude_method": att_method,
            "attitude_diagnostics": att_diags,
            "order": self.order,
            "geometry_version": self.geometry.version,
            "declination_version": self.declination.version,
            "mount_version": self.mount.version,
        }, att_diags

    # ---------- bottom track 逐级解 -------------------------------------
    def build_bottom_track_records(self, ping, attitude_result):
        att, method_name, adiags = attitude_result
        radials, evidence = [], []
        for b in range(4):
            sv = self._src("bt_velocity", ping, b)
            co = self._src("bt_correlation", ping, b)
            bd = self._src("bt_bad", ping, b)
            flagged = bd and bd["data"]["raw"] in ("1", "true", "TRUE", "True")
            qv = float(sv["data"]["value"]) if sv and sv["data"]["value"] is not None else None
            if (not flagged) and qv is not None:
                radials.append((b, qv))
            evidence.append({"beam": b, "value": qv, "raw": sv["data"]["raw"] if sv else None,
                             "correlation": co["data"]["value"] if co else None,
                             "bad_flag": bd["data"]["raw"] if bd else None,
                             "usable": (not flagged) and qv is not None})
        records = []
        if len(radials) < 3:
            return records
        beams = [b for b, _ in radials]
        vals = [v for _, v in radials]
        if len(radials) == 4:
            xyz, _ = self.geometry.radial_to_xyz(vals, beams)
            kind, cid, missing = "beam4", f"bt:p{ping}#beam4", None
            err = self.geometry.error_velocity(vals)
        else:
            xyz, _ = self.geometry.radial_to_xyz(vals, beams)
            missing = sorted(set(range(4)) - set(beams))[0]
            kind, cid, err = "beam3", f"bt:p{ping}#beam3_missing={missing}", None
        chain, _ = self._chain(ping, ping,
                               {"candidates": [{"candidate_id": cid, "kind": kind,
                                                "vector_ENDframe": np.round(xyz, 6).tolist(),
                                                "adopted": True}]},
                               attitude_result)
        if chain is None:
            return records
        srcs = [self._src_ref(self._src(k, ping, b))
                for k in ("bt_velocity", "bt_correlation", "bt_bad", "bt_range")
                for b in range(4) if self._src(k, ping, b)]
        payload = {"instrument": {"vector": chain["instrument"]["vector"],
                                  "solution_kind": kind, "missing_beam": missing,
                                  "error_velocity": round(err, 6) if err is not None else None},
                   "ship_vector": chain["ship"]["vector"],
                   "earth_END": chain["earth"]["vector_END"],
                   "earth_ENU": chain["earth"]["vector_ENU"],
                   "beam_evidence": evidence,
                   "heading_true_deg": chain["effective_attitude"]["heading_true_deg"],
                   "attitude_method": chain["attitude_method"],
                   "geometry_version": self.geometry.version,
                   "declination_version": self.declination.version}
        records.append({"node_id": f"bt:p{ping}", "op": "bt_earth",
                        "inputs": {"bt_sources": srcs,
                                   "attitude_node": f"att:p{ping}"},
                        "rules": {**self._rule_rules(), "geometry": self.geometry.fingerprint_material(),
                                  "order": self.order,
                                  "mount_version": self.mount.version,
                                  "declination_version": self.declination.version},
                        "payload": payload, "status": "ok"})
        return records

    # ---------- 依赖图传播 ---------------------------------------------
    def _cells_of(self, ping):
        return sorted({r["cell"] for r in self.store.scan_current_sources("beam_velocity")
                       if r["ping"] == ping})

    def _src_ref(self, rec):
        return f"{rec['sid']}@r{rec['revision']}" if rec else None

    def _rule_rules(self):
        return {"rules_version": self.quality.get("rules_version", "rules:v1"),
                "quality_rule": self.quality_version}

    def build_layer_records(self, ping, segs, adoptions):
        """构造一个 ping 下全部派生节点（自底向上）。"""
        records = []
        cells = self._cells_of(ping)

        # attitude 节点
        att, method, adiags = self.resolve_attitude(ping, segs)
        att_srcs = []
        if method == "measured":
            att_srcs.append(self._src_ref(self._src("attitude", ping)))
        elif method == "interpolated":
            for d in adiags:
                if isinstance(d, dict) and d["code"] == "attitude_interpolated":
                    for p in d["segment_pings"]:
                        att_srcs.append(self._src_ref(self._src("attitude", p)))
        att_payload = {"method": method, "diagnostics": adiags}
        if att is not None:
            att_payload["heading_deg"] = round(att.heading_deg, 6)
            att_payload["pitch_deg"] = round(att.pitch_deg, 6)
            att_payload["roll_deg"] = round(att.roll_deg, 6)
            if att.heading_unwrapped is not None:
                att_payload["heading_unwrapped_deg"] = round(att.heading_unwrapped, 6)
        records.append({
            "node_id": f"att:p{ping}", "op": "attitude",
            "inputs": {"attitude_sources": att_srcs,
                       "ping_time": self.ping_time(ping)},
            "rules": dict(self._rule_rules()),
            "payload": att_payload,
            "status": "ok" if att is not None else "excluded",
        })

        inst_by_cell = {}
        for cell in cells:
            sol = self.instrument_candidates(ping, cell)
            node = f"inst:p{ping}c{cell}"
            adopt = adoptions.get(node)
            if adopt:
                hit = next((c for c in sol["candidates"]
                            if c["candidate_id"] == adopt), None)
                if hit:
                    for c in sol["candidates"]:
                        c["adopted"] = c["candidate_id"] == adopt
                    sol["status"] = "ok"
                    sol["decision"] = {"adopted": adopt}
            inst_by_cell[node] = sol
            srcs = []
            for b in range(4):
                for kind in ("beam_velocity", "correlation", "amplitude", "bad_beam"):
                    rec = self._src(kind, ping, cell, b)
                    if rec:
                        srcs.append(self._src_ref(rec))
            records.append({
                "node_id": node, "op": "instrument",
                "inputs": {"beam_sources": srcs},
                "rules": {**self._rule_rules(),
                          "geometry": self.geometry.fingerprint_material(),
                          "geometry_version": self.geometry.version,
                          "adopted_candidate": adopt},
                "payload": sol, "status": sol["status"],
            })

        for cell in cells:
            node = f"inst:p{ping}c{cell}"
            sol = inst_by_cell[node]
            chain, cdiags = self._chain(ping, cell, sol, (att, method, adiags))
            common_in = {
                "instrument_node": node,
                "attitude_node": f"att:p{ping}",
                "geometry_version": self.geometry.version,
            }
            if chain is None:
                for layer, op in (("ship", "ship"), ("earth", "earth")):
                    records.append({
                        "node_id": f"{layer}:p{ping}c{cell}", "op": op,
                        "inputs": dict(common_in),
                        "rules": {**self._rule_rules(), "order": self.order,
                                  "mount_version": self.mount.version,
                                  "declination_version": self.declination.version},
                        "payload": {"available": False, "diagnostics": cdiags},
                        "status": "excluded",
                    })
                continue
            status = "indeterminate" if sol["status"] == "indeterminate" else "ok"
            records.append({
                "node_id": f"ship:p{ping}c{cell}", "op": "ship",
                "inputs": dict(common_in),
                "rules": {**self._rule_rules(), "order": self.order,
                          "mount_version": self.mount.version},
                "payload": {"vector": chain["ship"]["vector"],
                            "axes": chain["ship"]["axes"],
                            "instrument_candidate": chain["instrument"]["candidate_id"],
                            "solution_kind": chain["instrument"]["solution_kind"],
                            "effective_pitch_deg": chain["effective_attitude"]["pitch_deg"],
                            "effective_roll_deg": chain["effective_attitude"]["roll_deg"]},
                "status": status,
            })
            records.append({
                "node_id": f"earth:p{ping}c{cell}", "op": "earth",
                "inputs": {**common_in, "ship_node": f"ship:p{ping}c{cell}"},
                "rules": {**self._rule_rules(), "order": self.order,
                          "mount_version": self.mount.version,
                          "declination_version": self.declination.version},
                "payload": {"vector_END": chain["earth"]["vector_END"],
                            "vector_ENU": chain["earth"]["vector_ENU"],
                            "instrument_candidate": chain["instrument"]["candidate_id"],
                            "solution_kind": chain["instrument"]["solution_kind"],
                            "heading_true_deg": chain["effective_attitude"]["heading_true_deg"],
                            "attitude_method": chain["attitude_method"],
                            "attitude_diagnostics": chain["attitude_diagnostics"]},
                "status": status,
            })

        session_nodes = {r["node_id"]: r for r in records}
        records.append(self._profile_record(ping, cells, session_nodes))
        return records

    def _profile_record(self, ping, cells, session_nodes=None):
        session_nodes = session_nodes or {}
        def _earth(node):
            if node in session_nodes:
                return session_nodes[node]
            return self.store.get_derivation(node, self.branch_id)
        def _inst(node):
            if node in session_nodes:
                return session_nodes[node]
            return self.store.get_derivation(node, self.branch_id)
        bt_range = []
        for b in range(4):
            rec = self._src("bt_range", ping, b)
            if rec and rec["data"]["value"] is not None:
                bt_range.append(rec["data"]["value"])
        bt_valid = self._src("bt_valid", ping)
        bt_ok = bool(bt_valid and bt_valid["data"]["raw"] == "1" and len(bt_range) >= 3)
        bottom_m = min(bt_range) if bt_range else None
        blank = self.quality["blanking_min_m"]
        layers, series, excluded = [], [], []
        for cell in cells:
            depth_rec = self._src("cell_depth", ping, cell)
            depth = depth_rec["data"]["value"] if depth_rec else None
            earth = _earth(f"earth:p{ping}c{cell}")
            inst = _inst(f"inst:p{ping}c{cell}")
            reasons = []
            if depth is None:
                reasons.append("depth_unconvertible")
            if depth is not None and depth < blank:
                reasons.append("in_blind_zone")
            if not bt_ok:
                reasons.append("bottom_track_invalid")
            elif depth is not None and bottom_m is not None and depth > bottom_m:
                reasons.append("below_bottom_track")
            estatus = earth["status"] if earth else "missing"
            if estatus == "indeterminate":
                reasons.append("indeterminate_solution_pending_decision")
            if estatus == "excluded" or inst is None:
                reasons.append("attitude_or_beam_unavailable")
            accepted = not reasons and estatus == "ok"
            entry = {"cell": cell, "depth_m": depth, "status": "ok" if accepted else
                     ("indeterminate" if estatus == "indeterminate" else "excluded"),
                     "reasons": reasons,
                     "earth_node": f"earth:p{ping}c{cell}",
                     "inst_node": f"inst:p{ping}c{cell}"}
            if accepted:
                entry["velocity_END_mps"] = earth["payload"]["vector_END"]
                entry["velocity_ENU_mps"] = earth["payload"]["vector_ENU"]
                series.append(entry)
            elif entry["status"] == "indeterminate":
                excluded.append(entry)
            else:
                excluded.append(entry)
            layers.append(entry)
        return {
            "node_id": f"profile:p{ping}", "op": "profile",
            "inputs": {"earth_nodes": [f"earth:p{ping}c{c}" for c in cells],
                       "bt_valid": self._src_ref(bt_valid),
                       "bt_range": [self._src_ref(self._src("bt_range", ping, b))
                                    for b in range(4)
                                    if self._src("bt_range", ping, b)]},
            "rules": {**self._rule_rules(),
                      "blanking_min_m": blank},
            "payload": {"bottom_m": bottom_m, "bottom_track_ok": bt_ok,
                        "cells": layers, "accepted_series": series,
                        "excluded": excluded},
            "status": "indeterminate" if any(c["status"] == "indeterminate"
                                             for c in layers) else
                     ("ok" if all(c["status"] == "ok" for c in layers) else "excluded"),
        }

    def recompute(self, event_id, adoptions=None, only_pings=None):
        """依赖图传播：拓扑序重建节点；指纹不变的节点复用，不产生新版本。"""
        adoptions = adoptions or {}
        self.load_rules()
        segs = self.attitude_segments()
        pings = only_pings or sorted(self._ping_index())
        report = {"created": [], "unchanged": [], "pings": pings}
        with self.store.tx(immediate=True) as conn:
            # 先 attitude/instrument/ship/earth，最后 profile（拓扑序）
            ordered_ops = {"attitude": 0, "instrument": 1, "ship": 2,
                           "earth": 3, "profile": 4}
            all_records = []
            for ping in pings:
                all_records += self.build_layer_records(ping, segs, adoptions)
                all_records += self.build_bottom_track_records(ping, self.resolve_attitude(ping, segs))
            all_records.sort(key=lambda r: (ordered_ops[r["op"]], r["node_id"]))
            for rec in all_records:
                fp = derivation_fingerprint(rec["op"], OP_VERSIONS[rec["op"]],
                                            rec["inputs"], rec["rules"])
                existing = self.store.get_derivation(rec["node_id"], self.branch_id)
                if existing and existing["fingerprint"] == fp and \
                        existing["branch_id"] == self.branch_id:
                    report["unchanged"].append(rec["node_id"])
                    continue
                self.store.put_derivation(
                    conn, self.branch_id, rec["node_id"], rec["op"],
                    OP_VERSIONS[rec["op"]], rec["inputs"], fp, rec["payload"],
                    rec["status"], event_id)
                report["created"].append({"node_id": rec["node_id"],
                                          "status": rec["status"],
                                          "fingerprint": fp})
        return report

    def explain(self, node_id):
        """说明某结果为何仍有效/为何需重算（依赖与指纹组成）。"""
        d = self.store.get_derivation(node_id, self.branch_id)
        if not d:
            return None
        return {"node_id": node_id, "fingerprint": d["fingerprint"],
                "op": d["op"], "status": d["status"],
                "inputs": d["inputs"], "rules": d.get("inputs") and d["inputs"],
                "created_event": d["created_event"],
                "fingerprint_parts": {
                    "op_version": d["op_version"],
                    "input_snapshot": d["inputs"],
                    "rule_snapshot_in_fp": True}}
