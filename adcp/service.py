"""应用服务：把案例与操作落到存储模型，并触发依赖图传播。

每次操作：
  1. 在一个事务内追加不可变事件 + 不可变源/规则版本（指针随之原子切换）；
  2. 依赖图只重算受影响节点，指纹不变的节点保持旧版本；
  3. 操作返回机器可读事件与受影响对象。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from .derivation import DerivationEngine
from .storage import Store
from .units import parse_quantity


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def eid():
    return "ev:" + uuid.uuid4().hex[:14]


class EvidenceService:
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.store = Store(db_path)
        self.case = None

    # ---------- 初始导入 ------------------------------------------------
    def import_case(self, case, actor="reviewer", branch_id="main"):
        self.case = case
        cfg = case["config"]
        event = eid()
        pings = [p["ping"] for p in case["pings"]]
        with self.store.tx(immediate=True) as conn:
            self.store.add_event(conn, event, "case_imported", actor, {
                "case_id": case["case_id"], "title": case["title"],
                "pings": pings, "separate_records": [
                    "beam_velocity", "correlation", "amplitude", "bad_beam",
                    "heading_pitch_roll", "declination", "mount_angles"],
                "schema_version": case.get("schema_version", 1),
            }, scope={"case_id": case["case_id"]}, affected={"pings": pings},
                branch_id=branch_id)
            # 规则分开记录
            self.store.put_rule(conn, "beam_geometry", "rule:beam_geometry",
                                case["beam_geometry"],
                                case["beam_geometry"]["version"], event)
            decl = case["declination"]
            self.store.put_rule(conn, "declination", "rule:declination",
                                {"east_deg": float(decl["value"]),
                                 "unit": decl["unit"], "raw": decl["value"],
                                 "note": decl.get("note")},
                                decl["version"], event)
            mo = case["mount"]
            self.store.put_rule(conn, "mount", "rule:mount",
                                {"heading_deg": float(mo["heading"]),
                                 "pitch_deg": float(mo["pitch"]),
                                 "roll_deg": float(mo["roll"]),
                                 "unit": mo["unit"], "raw": [mo["heading"], mo["pitch"], mo["roll"]]},
                                mo["version"], event)
            self.store.put_rule(conn, "quality", "rule:quality", cfg,
                                cfg["rules_version"], event)
            self.store.put_rule(conn, "order", "rule:order",
                                {"rotation_order": cfg["rotation_order"]},
                                "order:v1", event)
            # 每个 ping 的源数据：七类测量分开成行
            for p in case["pings"]:
                ping = p["ping"]
                pt = parse_quantity(p["time"], kind="time")
                if p.get("attitude"):
                    a = p["attitude"]
                    self.store.put_source(conn, "attitude", [ping], {
                        "time": pt.value, "time_raw": p["time"],
                        "heading": vars(parse_quantity(a["heading"], a["unit"],
                                                       datum="magnetic", kind="angle")),
                        "pitch": vars(parse_quantity(a["pitch"], a["unit"], kind="angle")),
                        "roll": vars(parse_quantity(a["roll"], a["unit"], kind="angle")),
                    }, event)
                bt = p["bottom_track"]
                self.store.put_source(conn, "bt_valid", [ping],
                                      vars(parse_quantity(bt["valid"], kind="flag")), event)
                for b in range(4):
                    self.store.put_source(conn, "bt_range", [ping, None, b],
                        vars(parse_quantity(bt["range"][b], bt["range_unit"], kind="length")), event)
                    self.store.put_source(conn, "bt_velocity", [ping, None, b],
                        vars(parse_quantity(bt["beam_velocity"][b], bt["velocity_unit"],
                                            kind="velocity")), event)
                    self.store.put_source(conn, "bt_correlation", [ping, None, b],
                        vars(parse_quantity(bt["correlation"][b], "count",
                                            kind="dimensionless")), event)
                    self.store.put_source(conn, "bt_bad", [ping, None, b],
                        vars(parse_quantity(bt["bad_beam"][b], kind="flag")), event)
                for cell in [c0["cell"] for c0 in p["cells"]]:
                    cd = next(x for x in p["cells"] if x["cell"] == cell)
                    qdepth = parse_quantity(cd["depth"], cd["depth_unit"], kind="length")
                    self.store.put_source(conn, "cell_depth", [ping, cell],
                        {**vars(qdepth), "ping_time": pt.value}, event)
                    for b in range(4):
                        self.store.put_source(conn, "beam_velocity", [ping, cell, b],
                            {**vars(parse_quantity(cd["beam_velocity"][b],
                                                   cd["velocity_unit"], kind="velocity")),
                             "ping_time": pt.value}, event)
                        self.store.put_source(conn, "correlation", [ping, cell, b],
                            vars(parse_quantity(cd["correlation"][b], "count",
                                                kind="dimensionless")), event)
                        self.store.put_source(conn, "amplitude", [ping, cell, b],
                            vars(parse_quantity(cd["amplitude"][b], "count",
                                                kind="dimensionless")), event)
                        self.store.put_source(conn, "bad_beam", [ping, cell, b],
                            vars(parse_quantity(cd["bad_beam"][b], kind="flag")), event)
        report = self.recompute(event, branch_id=branch_id)
        return {"event_id": event, "report": report}

    # ---------- 派生 ----------------------------------------------------
    def engine(self, branch_id="main"):
        eng = DerivationEngine(self.store, branch_id)
        eng.load_rules()
        return eng

    def recompute(self, event_id, branch_id="main", only_pings=None, adoptions=None):
        eng = DerivationEngine(self.store, branch_id)
        return eng.recompute(event_id, adoptions=adoptions, only_pings=only_pings)

    # ---------- 源数据修订（只改动一个直接来源 -> 局部重算）-------------
    def revise_beam_velocity(self, ping, cell, beam, raw_value, unit,
                             actor="hydrologist", reason="source_revision"):
        event = eid()
        kind = "beam_velocity"
        with self.store.tx(immediate=True) as conn:
            old = self.store.get_current_source(self._sid(kind, ping, cell, beam))
            parsed = parse_quantity(raw_value, unit, kind="velocity")
            ref = self.store.put_source(conn, kind, [ping, cell, beam],
                                        {**vars(parsed),
                                         "ping_time": old["data"].get("ping_time")}, event)
            self.store.add_event(conn, event, "source_revised", actor, {
                "kind": kind, "ping": ping, "cell": cell, "beam": beam,
                "old": {"raw": old["data"].get("raw"), "unit": old["data"].get("unit"),
                        "revision": old["revision"]},
                "new": {"raw": raw_value, "unit": unit,
                        "canonical_value": parsed.value,
                        "diagnostic": parsed.diagnostic, "revision": ref["revision"]},
                "reason": reason,
            }, scope={"ping": ping, "cell": cell, "beam": beam},
                affected={"pings": [ping],
                          "nodes": [f"inst:p{ping}c{cell}", f"ship:p{ping}c{cell}",
                                    f"earth:p{ping}c{cell}", f"profile:p{ping}"]},
                branch_id="main")
        report = self.recompute(event, only_pings=[ping])
        return {"event_id": event, "report": report}

    def set_bad_beam(self, ping, cell, beam, flagged, actor="reviewer",
                     reason="bad_beam_flag_set"):
        event = eid()
        kind = "bad_beam"
        with self.store.tx(immediate=True) as conn:
            old = self.store.get_current_source(self._sid(kind, ping, cell, beam))
            ref = self.store.put_source(
                conn, kind, [ping, cell, beam],
                {"value": None, "unit": "", "raw": "1" if flagged else "0",
                 "kind": "flag"}, event)
            self.store.add_event(conn, event, "source_revised", actor, {
                "kind": kind, "ping": ping, "cell": cell, "beam": beam,
                "old": old["data"].get("raw"), "new": "1" if flagged else "0",
                "reason": reason,
            }, scope={"ping": ping, "cell": cell, "beam": beam},
                affected={"pings": [ping], "nodes": [f"inst:p{ping}c{cell}",
                          f"ship:p{ping}c{cell}", f"earth:p{ping}c{cell}",
                          f"profile:p{ping}"]})
        report = self.recompute(event, only_pings=[ping])
        return {"event_id": event, "report": report}

    def set_bottom_track_valid(self, ping, valid, actor="reviewer",
                               reason="bottom_track_review"):
        event = eid()
        with self.store.tx(immediate=True) as conn:
            old = self.store.get_current_source(self._sid("bt_valid", ping))
            self.store.put_source(conn, "bt_valid", [ping],
                                  {"value": None, "unit": "", "kind": "flag",
                                   "raw": "1" if valid else "0"}, event)
            self.store.add_event(conn, event, "source_revised", actor, {
                "kind": "bt_valid", "ping": ping, "old": old["data"].get("raw"),
                "new": "1" if valid else "0", "reason": reason,
            }, scope={"ping": ping},
                affected={"pings": [ping], "nodes": [f"profile:p{ping}", f"bt:p{ping}"]})
        report = self.recompute(event, only_pings=[ping])
        return {"event_id": event, "report": report}

    def invalidate_attitude(self, ping, actor="reviewer", reason="attitude_record_gap"):
        """删除某 ping 的姿态来源（逻辑删除：写一条标记，不再提供姿态）。

        实现为追加一条 attitude_gap 规则化源记录，引擎读不到可用姿态，
        仅允许在连续段内插值，绝不跨缺口。
        """
        event = eid()
        with self.store.tx(immediate=True) as conn:
            ref = self.store.put_source(conn, "attitude_gap_marker", [ping],
                                        {"value": None, "unit": "", "raw": reason}, event)
            self.store.add_event(conn, event, "source_revised", actor, {
                "kind": "attitude", "ping": ping,
                "new": "unavailable", "reason": reason,
            }, scope={"ping": ping},
                affected={"pings": [ping], "nodes": [f"att:p{ping}",
                          f"earth:p{ping}c*", f"profile:p{ping}"]})
        return {"event_id": event,
                "report": {"created": [], "unchanged": [], "note": "attitude marker; recompute required"}}

    # ---------- 规则修订（进入指纹，沿图传播）---------------------------
    def revise_rule(self, kind, new_version, new_data, actor, reason,
                    extra=None, branch_id="main"):
        rid = {"beam_geometry": "rule:beam_geometry", "declination": "rule:declination",
               "mount": "rule:mount", "quality": "rule:quality",
               "order": "rule:order"}[kind]
        event = eid()
        with self.store.tx(immediate=True) as conn:
            ref = self.store.put_rule(conn, kind, rid, new_data, new_version, event)
            affected_nodes = self._nodes_for_rule(kind)
            self.store.add_event(conn, event, "rule_revised", actor, {
                "rule": kind, "rule_id": rid, "new_version": new_version,
                "new_data": new_data, "revision": ref["revision"], "reason": reason,
                "fingerprint_impact": ["coordinate_transform_order"] if kind == "order" else
                    (["beam_geometry"] if kind == "beam_geometry" else
                     (["magnetic_declination"] if kind == "declination" else
                      (["mount_angles"] if kind == "mount" else ["quality_gate"]))),
                "extra": extra or {},
            }, scope={"rule": kind}, affected={"nodes": affected_nodes},
                branch_id=branch_id)
        report = self.recompute(event, branch_id=branch_id)
        return {"event_id": event, "report": report}

    def revise_order(self, order, actor="hydrologist", reason="transform_order_revision",
                     branch_id="main"):
        return self.revise_rule("order", f"order:{order.replace(',', '')}:v2",
                                {"rotation_order": order}, actor, reason,
                                branch_id=branch_id)

    def revise_mount(self, heading=None, pitch=None, roll=None,
                     new_version="mount:revised:v2", actor="surveyor",
                     reason="mount_angle_revision", branch_id="main"):
        cur = self.store.get_current_rule("rule:mount")["data"]
        data = {"heading_deg": cur["heading_deg"] if heading is None else heading,
                "pitch_deg": cur["pitch_deg"] if pitch is None else pitch,
                "roll_deg": cur["roll_deg"] if roll is None else roll,
                "unit": "deg", "raw": "mount revision"}
        return self.revise_rule("mount", new_version, data, actor, reason,
                                branch_id=branch_id)

    def _nodes_for_rule(self, kind):
        pings = self.engine()._ping_index()
        cells_by_ping = {p: self.engine()._cells_of(p) for p in pings}
        nodes = []
        for p in pings:
            if kind in ("mount", "declination", "order"):
                nodes += [f"earth:p{p}c{c}" for c in cells_by_ping[p]]
                nodes.append(f"bt:p{p}")
            elif kind == "beam_geometry":
                nodes += [f"inst:p{p}c{c}" for c in cells_by_ping[p]]
                nodes += [f"ship:p{p}c{c}" for c in cells_by_ping[p]]
                nodes += [f"earth:p{p}c{c}" for c in cells_by_ping[p]]
                nodes += [f"profile:p{p}", f"bt:p{p}"]
            elif kind == "quality":
                nodes += [f"inst:p{p}c{c}" for c in cells_by_ping[p]]
                nodes.append(f"profile:p{p}")
        return nodes

    # ---------- 裁定与冲突（保留未选候选，不覆盖历史）-------------------
    def decide_candidate(self, node_id, candidate_id, actor="reviewer",
                         expected_base_event=None):
        """对不确定节点选择继续哪个候选；其它候选原样保留。"""
        current = self.store.get_derivation(node_id)
        if current is None:
            raise ValueError(f"node {node_id} not found")
        valid = {c["candidate_id"] for c in current["payload"]["candidates"]}
        if candidate_id not in valid:
            raise ValueError(f"candidate {candidate_id} not legal for {node_id}: {valid}")
        base = self.store.latest_event_id()
        if expected_base_event and expected_base_event != base:
            return {"conflict": True, "reason": "stale_base",
                    "expected": expected_base_event, "actual": base}
        event = eid()
        with self.store.tx(immediate=True) as conn:
            self.store.add_event(conn, event, "candidate_decided", actor, {
                "node_id": node_id, "selected": candidate_id,
                "retained_unselected": sorted(valid - {candidate_id}),
                "base_event": base,
                "note": "decision creates new version; history never overwritten",
            }, scope={"node": node_id}, affected={"nodes": [node_id]})
        ping = int(node_id.split(":p")[1].split("c")[0])
        report = self.recompute(event, only_pings=[ping],
                                adoptions={node_id: candidate_id})
        return {"event_id": event, "report": report, "base_event": base}

    def conflicting_decisions(self, node_id, candidate_a, candidate_b,
                              actor_a="reviewer-A", actor_b="reviewer-B"):
        """两位使用者从同一版本编辑 -> 可解决的冲突。"""
        first = self.decide_candidate(node_id, candidate_a, actor_a)
        if first.get("conflict"):
            return first
        event = eid()
        with self.store.tx(immediate=True) as conn:
            self.store.add_event(conn, event, "decision_conflicted", actor_b, {
                "node_id": node_id, "selected": candidate_b,
                "concurrent_with": first["event_id"],
                "base_event": first.get("base_event"),
                "resolution": "required",
            }, scope={"node": node_id}, affected={"nodes": [node_id]})
            cid = "conflict:" + node_id.replace(":", "_")
            self.store.record_conflict(conn, cid, node_id, first["event_id"], event)
        return {"conflict_recorded": True, "conflict_id": cid,
                "events": [first["event_id"], event]}

    def resolve_conflict(self, conflict_id, candidate_id, actor="lead"):
        conf = next((c for c in self.store.list_conflicts(True)
                     if c["conflict_id"] == conflict_id), None)
        if not conf:
            raise ValueError("no open conflict")
        out = self.decide_candidate(conf["node_id"], candidate_id, actor)
        with self.store.tx(immediate=True) as conn:
            self.store.resolve_conflict(conn, conflict_id, out["event_id"])
        out["conflict_resolved"] = conflict_id
        return out

    # ---------- 分支 ----------------------------------------------------
    def fork_branch(self, branch_id, name, from_branch="main", actor="reviewer"):
        event = eid()
        with self.store.tx(immediate=True) as conn:
            base_seq = self.store.create_branch(conn, branch_id, name, from_branch, event)
            self.store.add_event(conn, event, "branch_forked", actor, {
                "branch": branch_id, "from": from_branch, "base_seq": base_seq,
            }, branch_id=branch_id)
        return {"event_id": event, "base_seq": base_seq}

    # ---------- 查询 ----------------------------------------------------
    def _sid(self, kind, *scope):
        from .fingerprint import stable_id
        return stable_id("src", kind, *scope)

    def ping_view(self, ping, upto_event_seq=None, branch_id="main"):
        if upto_event_seq is None:
            derivs = {n: self.store.get_derivation(n, branch_id)
                      for n in self._all_nodes(ping, branch_id)}
            derivs = {k: v for k, v in derivs.items() if v is not None}
        else:
            derivs = self.store.derivations_at(branch_id, upto_event_seq)
        return {
            "ping": ping,
            "time": self.engine(branch_id).ping_time(ping),
            "attitude_node": derivs.get(f"att:p{ping}"),
            "cells": [derivs.get(f"inst:p{ping}c{c}") and {
                "cell": c,
                "instrument": derivs.get(f"inst:p{ping}c{c}"),
                "ship": derivs.get(f"ship:p{ping}c{c}"),
                "earth": derivs.get(f"earth:p{ping}c{c}"),
            } for c in self.engine(branch_id)._cells_of(ping)
                if derivs.get(f"inst:p{ping}c{c}")],
            "bottom_track": derivs.get(f"bt:p{ping}"),
            "profile": derivs.get(f"profile:p{ping}"),
        }

    def _all_nodes(self, ping, branch_id):
        eng = self.engine(branch_id)
        nodes = [f"att:p{ping}", f"profile:p{ping}", f"bt:p{ping}"]
        for c in eng._cells_of(ping):
            nodes += [f"inst:p{ping}c{c}", f"ship:p{ping}c{c}", f"earth:p{ping}c{c}"]
        return nodes

    def overview(self, branch_id="main"):
        eng = self.engine(branch_id)
        pings = sorted(eng._ping_index())
        rows = []
        for p in pings:
            prof = self.store.get_derivation(f"profile:p{p}", branch_id)
            att = self.store.get_derivation(f"att:p{p}", branch_id)
            rows.append({"ping": p,
                         "attitude_status": att["status"] if att else None,
                         "attitude_method": (att["payload"].get("method") if att else None),
                         "profile_status": prof["status"] if prof else None,
                         "accepted_cells": (len(prof["payload"]["accepted_series"])
                                            if prof else 0),
                         "excluded_cells": len(prof["payload"]["excluded"]) if prof else 0})
        return {"case_id": self.case and self.case["case_id"], "pings": rows,
                "branches": self.store.list_branches(),
                "conflicts": self.store.list_conflicts()}

    def events(self, branch_id="main"):
        return self.store.events(branch_id)

    def explain_node(self, node_id, branch_id="main"):
        return DerivationEngine(self.store, branch_id).explain(node_id)

    # ---------- 导出/导入 -----------------------------------------------
    def export_bundle(self):
        return self.store.export_bundle()
