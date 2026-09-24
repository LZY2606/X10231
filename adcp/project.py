"""Application service: seeded narrative, branches, decisions, replay.

Timeline (branch ``main``)
--------------------------
e0001 baseline imported (separable source records)
e0002 installation-angle revision (declination unchanged) -> local recompute
e0003 one direct source changed (p02 c3 beam 2 velocity) -> local recompute
       and one cell becomes ambiguous
e0004 reviewer A accepts one 3-beam candidate
e0005 reviewer B attempts a conflicting decision on the moved head -> conflict
       recorded, resolvable, never overwrites e0004

Seven counterexample branches fork from baseline and change exactly one
direct source or one rule version.
"""
from __future__ import annotations

import copy
import json

from . import graph
from .casepack import (build_baseline, GEOMETRY_V1, DECL_V1, DECL_V2,
                       INSTALL_V1, INSTALL_V2, ORDER_V1, ORDER_V2,
                       POLICY_V1, CELLS_V1, N_BEAMS)
from .graph import build_snapshot, recompute, project_state, accepted_decisions
from .storage import Store, ConflictError

BASELINE_EVENT = "e0001"


def _baseline_records(baseline: dict):
    sources, rules = [], []
    for kind, recs in baseline["sources"].items():
        for r in recs:
            if "payload" in r:
                key, version, payload = r["record_key"], r["version"], r["payload"]
            else:  # attitude samples are stored as raw payloads
                key, version, payload = r["sample_id"], 1, dict(r)
            sources.append({"kind": kind, "record_key": key,
                            "version": version, "payload": payload})
    for rule, r in baseline["rules"].items():
        rules.append({"rule": rule, "version": 1, "payload": {
            **r["payload"], "version_id": r["version_id"]}})
    return sources, rules


COUNTEREXAMPLES = [
    ("x_single_beam", "单 beam 失效", "bad_flag"),
    ("x_heading_wrap", "姿态跨越 360 度", "attitude"),
    ("x_time_gap", "时间缺口：姿态插值不跨缺口", "attitude_gap"),
    ("x_three_beam_ambiguous", "三波束多解：不与四波束混为一值", "source_velocity"),
    ("x_install_revision", "安装角修订", "rule_install"),
    ("x_transform_order", "坐标变换顺序进入指纹", "rule_order"),
    ("x_invalid_cell", "无效深度 cell", "cells"),
]


class Project:
    def __init__(self, store: Store):
        self.store = store

    # ----- seeding -----------------------------------------------------
    def seed(self) -> None:
        if self.store.list_events("main"):
            return
        baseline = build_baseline()
        sources, rules = _baseline_records(baseline)
        with self.store.run("baseline import", run_id="run-baseline") as batch:
            for s in sources:
                self.store.put_source(batch.run_id, s["kind"], s["record_key"],
                                      s["version"], s["payload"])
            for r in rules:
                self.store.put_rule(batch.run_id, r["rule"], 1, r["payload"])
            self.store.put_branch("main", "主干评审", None, batch.run_id)
            snap = build_snapshot(self.store.committed_sources(batch.run_id),
                                  self.store.committed_rules(batch.run_id))
            nodes = graph.materialize(snap)
            self.store.put_derived(batch.run_id, BASELINE_EVENT, nodes)
            self.store.append_event(
                "main", "baseline",
                {"summary": "基线导入：波束速度/correlation/amplitude/坏 beam 标志/"
                            "姿态/磁偏角/安装角分开记录",
                 "rule_versions": {k: v["version_id"] for k, v in baseline["rules"].items()},
                 "node_count": len(nodes)},
                actor="importer", event_id=BASELINE_EVENT,
                parent_event_id=None, run_id=batch.run_id)

    def seed_narrative(self) -> None:
        """Install-angle revision, one local source fix, conflicting decision."""
        existing = {e["event_id"] for e in self.store.list_events("main")}
        if "e0002" not in existing:
            self._revise_install()
        if "e0003" not in existing:
            self._local_source_fix()
        if "e0004" not in existing:
            self._first_decision()
        if "e0005" not in existing:
            self._conflicting_decision()
        self.seed_counterexamples()

    def _baseline_snapshot(self):
        return build_snapshot(self.store.committed_sources(),
                              self.store.committed_rules())

    def _revise_install(self) -> None:
        record = {"heading_offset_deg": 1.5, "pitch_offset_deg": 0.0,
                  "roll_offset_deg": 0.0, "unit": "deg",
                  "version_id": INSTALL_V2, "reason": "安装角实测修订 +1.5°"}
        with self.store.run("install revision", run_id="run-install-v2") as batch:
            self.store.put_rule(batch.run_id, "install", 2, record)
            self.store.append_event(
                "main", "rule_revision",
                {"rule": "install",
                 "summary": "安装角修订：heading_offset 0° -> +1.5°；波束层不变，"
                            "ship/earth 层重算",
                 "record": {"version": 2, "payload": record}},
                actor="surveyor", event_id="e0002",
                parent_event_id="e0001", run_id=batch.run_id)
            snap = build_snapshot(self.store.committed_sources(batch.run_id),
                                  self.store.committed_rules(batch.run_id))
            self.store.put_derived(batch.run_id, "e0002", graph.materialize(snap))

    def _local_source_fix(self) -> None:
        # Perturb ONE beam velocity at p02 cell 3 so 4-beam error velocity
        # exceeds policy: one direct source changes, only p02:c3 recomputes.
        rk = "p02:c03:b2"
        old = self._payload("beam_velocity", rk, 1)
        injected = old["q"]["value"] + 20.0  # cm/s, along-beam
        payload = copy.deepcopy(old)
        payload["q"] = {"value": injected, "unit": "cm/s",
                        "raw": f"{injected:.4f} cm/s (corrected)"}
        payload["edit_note"] = "源数据修订：单 beam 读数校正"
        with self.store.run("local source fix", run_id="run-fix-p02") as batch:
            self.store.put_source(batch.run_id, "beam_velocity", rk, 2, payload)
            self.store.append_event(
                "main", "source_fix",
                {"summary": "仅改动直接来源 p02:c03:b2；依赖图只重算该 cell，"
                            "四波束出现高 error velocity -> 三波束多解",
                 "sources": [{"kind": "beam_velocity", "record_key": rk,
                              "version": 2, "payload": payload}]},
                actor="surveyor", event_id="e0003",
                parent_event_id="e0002", run_id=batch.run_id)
            snap = build_snapshot(self.store.committed_sources(batch.run_id),
                                  self.store.committed_rules(batch.run_id))
            self.store.put_derived(batch.run_id, "e0003", graph.materialize(snap))

    def _payload(self, kind: str, key: str, version: int) -> dict:
        for r in self.store.committed_sources():
            if r["kind"] == kind and r["record_key"] == key and r["version"] == version:
                return copy.deepcopy(r["payload"])
        raise KeyError((kind, key, version))

    def _first_decision(self) -> None:
        state = self.view_branch_head("main")
        node = self._candidate(state, "p02", 2, "c3_134")
        with self.store.run("decision A", run_id="run-dec-a") as batch:
            self.store.append_event(
                "main", "decision",
                {"node_id": node["node_id"], "ping_id": "p02", "cell_idx": 2,
                 "candidate_id": "c3_134", "fingerprint": node["fingerprint"],
                 "verdict": "accepted",
                 "summary": "审阅者 A：采纳剔除 beam 2 的三波束解；c4 与其余"
                            "三波束候选保留不覆盖"},
                actor="reviewer-A", event_id="e0004",
                parent_event_id="e0003", expected_head="e0003",
                run_id=batch.run_id)

    def _conflicting_decision(self) -> None:
        # Reviewer B still assumes head e0003 and accepts a different candidate.
        state = self.state_at("main", "e0003")
        node = self._candidate(state, "p02", 2, "c3_123")
        try:
            self.store.append_event(
                "main", "decision",
                {"node_id": node["node_id"], "ping_id": "p02", "cell_idx": 2,
                 "candidate_id": "c3_123", "fingerprint": node["fingerprint"],
                 "verdict": "accepted",
                 "summary": "审阅者 B：基于旧版本 e0003 尝试采纳 c3_123"},
                actor="reviewer-B", event_id="e0005",
                parent_event_id="e0003", expected_head="e0003")
        except ConflictError as exc:
            self.store.append_event(
                "main", "decision_conflict",
                {"attempted": {"node_id": node["node_id"],
                               "candidate_id": "c3_123",
                               "fingerprint": node["fingerprint"],
                               "actor": "reviewer-B"},
                 "winner_event": "e0004", "resolution": "open",
                 "summary": f"两位使用者从同一版本编辑：{exc}；冲突可解决，"
                            "A 的裁定保留，B 可在新版本重新提交"},
                actor="reviewer-B", event_id="e0005",
                parent_event_id="e0004")

    # ----- counterexample branches ------------------------------------
    def seed_counterexamples(self) -> None:
        existing = {b["branch_id"] for b in self.store.list_branches()}
        builders = {
            "x_single_beam": self._x_single_beam,
            "x_heading_wrap": self._x_heading_wrap,
            "x_time_gap": self._x_time_gap,
            "x_three_beam_ambiguous": self._x_three_beam,
            "x_install_revision": self._x_install,
            "x_transform_order": self._x_order,
            "x_invalid_cell": self._x_invalid_cell,
        }
        for bid, _, _ in COUNTEREXAMPLES:
            if bid not in existing:
                builders[bid]()

    def _new_branch(self, bid: str, name: str, etype: str, payload: dict,
                    nodes) -> None:
        rid = f"run-{bid}"
        with self.store.run(name, run_id=rid) as batch:
            self.store.put_branch(bid, name, BASELINE_EVENT, batch.run_id)
            self.store.append_event(
                bid, etype, payload, actor="casepack",
                event_id=f"{bid}:e1", parent_event_id=BASELINE_EVENT,
                expected_head=None, run_id=batch.run_id)
            self.store.put_derived(batch.run_id, f"{bid}:e1", list(nodes))

    def _branch_events(self, bid: str):
        return [self.store.get_event(BASELINE_EVENT)] + self.store.list_events(bid)

    def _x_single_beam(self) -> None:
        rec = {"ping_id": "p03", "beam": 2, "bad": True,
               "source": "quality_flag", "injected": "single_beam_failure"}
        events = [{"type": "source_fix",
                   "payload": {"sources": [
                       {"kind": "bad_flag", "record_key": "p03:b2",
                        "version": 2, "payload": rec}]}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_single_beam", "反例：单 beam 失效", "source_fix",
            {"summary": "只把 p03 beam 2 坏标志置真；仅保留剔除 beam 2 的"
                        "三波束解 c3_134，四波束解被排除且不混入",
             "sources": [{"kind": "bad_flag", "record_key": "p03:b2",
                          "version": 2, "payload": rec}]},
            state.nodes.values())

    def _x_heading_wrap(self) -> None:
        old = self._payload("attitude", "a06", 1)
        rec = copy.deepcopy(old)
        rec["heading"] = 271.0  # equivalent to -89 = 91 - 180 ... forces wrap
        rec["injected"] = "heading_crosses_360"
        # a06 true heading is 90; 270 differs by -180 (use 272 vs 92 -> exact -180)
        rec["heading"] = 92.0 - 180.0
        events = [{"type": "source_fix", "payload": {"sources": [
            {"kind": "attitude", "record_key": "a06", "version": 2, "payload": rec}]}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_heading_wrap", "反例：姿态跨越 360 度", "source_fix",
            {"summary": "只改 a06 一条姿态记录为 -88°（表示为 272°）；解卷绕后"
                        "p05/p06 插值保持连续，不产生 180° 错误跳变",
             "sources": [{"kind": "attitude", "record_key": "a06",
                          "version": 2, "payload": rec}]},
            state.nodes.values())

    def _x_time_gap(self) -> None:
        old = self._payload("attitude", "a08", 1)
        rec = copy.deepcopy(old)
        rec["t"] = old["t"] + 60.0  # 20s cadence -> 80s gap between a07 and a08
        rec["injected"] = "time_gap"
        events = [{"type": "source_fix", "payload": {"sources": [
            {"kind": "attitude", "record_key": "a08", "version": 2, "payload": rec}]}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_time_gap", "反例：时间缺口", "source_fix",
            {"summary": "只移动 a08 时间戳制造 80s 缺口；p07 落入缺口，姿态"
                        "状态为 gap，插值不跨缺口，earth 速度保持排除",
             "sources": [{"kind": "attitude", "record_key": "a08",
                          "version": 2, "payload": rec}]},
            state.nodes.values())

    def _x_three_beam(self) -> None:
        rk = "p04:c05:b3"
        old = self._payload("beam_velocity", rk, 1)
        injected = old["q"]["value"] + 25.0
        rec = copy.deepcopy(old)
        rec["q"] = {"value": injected, "unit": "cm/s", "raw": f"{injected:.4f} cm/s"}
        rec["injected"] = "three_beam_ambiguity"
        events = [{"type": "source_fix", "payload": {"sources": [
            {"kind": "beam_velocity", "record_key": rk,
             "version": 2, "payload": rec}]}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_three_beam_ambiguous", "反例：三波束多解", "source_fix",
            {"summary": "只改 p04:c05:b3 使 error velocity 超阈；c4 标记 suspect，"
                        "四个合法三波束候选全部保留，无唯一主候选",
             "sources": [{"kind": "beam_velocity", "record_key": rk,
                          "version": 2, "payload": rec}]},
            state.nodes.values())

    def _x_install(self) -> None:
        rec = {"heading_offset_deg": -2.0, "pitch_offset_deg": 0.0,
               "roll_offset_deg": 0.0, "unit": "deg",
               "version_id": INSTALL_V2, "injected": "install_revision"}
        events = [{"type": "rule_revision",
                   "payload": {"rule": "install",
                               "record": {"version": 2, "payload": rec}}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_install_revision", "反例：安装角修订", "rule_revision",
            {"summary": "安装角 install:v1 -> install:v2；instrument 层指纹不变，"
                        "ship/earth 层指纹变化",
             "rule": "install", "record": {"version": 2, "payload": rec}},
            state.nodes.values())

    def _x_order(self) -> None:
        rec = {"order": ORDER_V2, "version_id": ORDER_V2}
        events = [{"type": "rule_revision",
                   "payload": {"rule": "transform_order",
                               "record": {"version": 2, "payload": rec}}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_transform_order", "反例：坐标变换顺序", "rule_revision",
            {"summary": "变换顺序 RDI_PR -> ALT_RP；beam/instrument 层不变，"
                        "ship 层改变，顺序版本进入派生指纹",
             "rule": "transform_order", "record": {"version": 2, "payload": rec}},
            state.nodes.values())

    def _x_invalid_cell(self) -> None:
        old = self._payload("depth_cell_bounds", "profile", 1)
        rec = copy.deepcopy(old)
        rec["cell_depths_m"] = list(rec["cell_depths_m"])
        rec["cell_depths_m"][11] = 17.0  # c12 beyond bottom range 16.5
        rec["injected"] = "invalid_depth_cell"
        events = [{"type": "source_fix", "payload": {"sources": [
            {"kind": "depth_cell_bounds", "record_key": "profile",
             "version": 2, "payload": rec}]}}]
        state = project_state(self._baseline_snapshot(), events)
        self._new_branch(
            "x_invalid_cell", "反例：无效深度 cell", "source_fix",
            {"summary": "只把 c12 深度改到 17m（越过 bottom track 16.5m）；"
                        "该 cell 所有候选保留但标记 excluded_depth，不进入剖面",
             "sources": [{"kind": "depth_cell_bounds", "record_key": "profile",
                          "version": 2, "payload": rec}]},
            state.nodes.values())

    # ----- read models -------------------------------------------------
    def _candidate(self, state, ping_id: str, cell_idx: int, candidate_id: str):
        nid = f"n:{ping_id}:c{cell_idx + 1:02d}:{candidate_id}"
        node = state["nodes"][nid]
        stored = self.store.get_derived(nid, node["fingerprint"])
        out = copy.deepcopy(node)
        out["stored"] = bool(stored)
        return out

    def state_at(self, branch_id: str, event_id: str | None) -> dict:
        """Replay branch history up to and including ``event_id``.

        Returns a projected snapshot without mutating the current pointer.
        """
        all_events = self.store.list_events(branch_id)
        up_to = []
        for e in all_events:
            up_to.append(e)
            if event_id and e["event_id"] == event_id:
                break
        base = self._baseline_snapshot()
        # Re-fork logic: branch events apply on the committed baseline snapshot.
        snap = project_state(base, up_to)
        return {"event_id": event_id or (up_to[-1]["event_id"] if up_to else None),
                "nodes": snap.nodes, "rules": snap.rules,
                "decisions": accepted_decisions(up_to),
                "events": up_to}

    def view_branch_head(self, branch_id: str) -> dict:
        head = self.store.branch_head(branch_id)
        return self.state_at(branch_id, head)

    def view_ping(self, branch_id: str, ping_id: str,
                  event_id: str | None = None) -> dict:
        state = self.state_at(branch_id, event_id)
        cells: dict[int, dict] = {}
        for nid, node in state["nodes"].items():
            if node.get("kind") != "earth_velocity_cell":
                continue
            if node.get("ping_id") != ping_id:
                continue
            p = node["payload"]
            cell = cells.setdefault(p["cell_idx"], {
                "cell_idx": p["cell_idx"], "depth_m": p["depth_m"],
                "depth_unit": p["depth_unit"], "candidates": {},
                "depth_check": p["depth_check"]})
            accepted = state["decisions"].get(nid)
            cand = {"candidate_id": p["candidate_id"],
                    "candidate_kind": p["candidate_kind"], "beams": p["beams"],
                    "node_id": nid, "fingerprint": node["fingerprint"],
                    "v_instrument": p["v_instrument"], "v_ship": p["v_ship"],
                    "v_earth": p["v_earth"],
                    "error_velocity_mps": p["error_velocity_mps"],
                    "candidate_status": p["candidate_status"], "status": p["status"],
                    "attitude": p["attitude"], "rule_versions": p["rule_versions"],
                    "source_versions": p["source_versions"],
                    "decision": accepted}
            cell["candidates"][p["candidate_id"]] = cand
        bt = state["nodes"].get(f"n:{ping_id}:bottomtrack")
        return {"branch_id": branch_id, "ping_id": ping_id,
                "event_id": state["event_id"],
                "cells": [cells[k] for k in sorted(cells)],
                "bottom_track": copy.deepcopy(bt["payload"]) if bt else None,
                "events": [{"event_id": e["event_id"], "type": e["type"],
                            "actor": e["actor"],
                            "summary": e["payload"].get("summary", "")}
                           for e in state["events"]]}

    def list_pings(self) -> list[dict]:
        return [{"ping_id": r["payload"]["ping_id"], "t": r["payload"]["t"]}
                for r in sorted(self.store.committed_sources(),
                                key=lambda r: r["payload"].get("t", 0))
                if r["kind"] == "ping"]

    def source_record(self, kind: str, key: str) -> dict:
        rows = [r for r in self.store.committed_sources()
                if r["kind"] == kind and r["record_key"] == key]
        rows.sort(key=lambda r: r["version"])
        return {"kind": kind, "record_key": key,
                "versions": [{"version": r["version"], "payload": r["payload"],
                              "run_id": r["run_id"]} for r in rows]}

    def profile(self, branch_id: str, event_id: str | None = None) -> dict:
        """Vertical profile using adopted primary candidates per cell."""
        state = self.state_at(branch_id, event_id)
        pings = sorted({n["ping_id"] for n in state["nodes"].values()
                        if n.get("kind") == "earth_velocity_cell"})
        out_pings = []
        for ping_id in pings:
            points, excluded = [], []
            for nid, node in sorted(state["nodes"].items()):
                if node.get("kind") != "earth_velocity_cell" or node.get("ping_id") != ping_id:
                    continue
                p = node["payload"]
                if p["candidate_id"] != "c4":
                    continue
                decision = state["decisions"].get(nid)
                record = {"cell_idx": p["cell_idx"], "depth_m": p["depth_m"],
                          "status": p["status"], "node_id": nid,
                          "fingerprint": node["fingerprint"]}
                if p["status"] == "adopted" and p["v_earth"]:
                    record["v_earth"] = p["v_earth"]
                    points.append(record)
                else:
                    excluded.append(record)
            # ambiguous cells: primary chosen by decision if any
            for nid, decision in state["decisions"].items():
                node = state["nodes"].get(nid)
                if not node or node.get("ping_id") != ping_id:
                    continue
                p = node["payload"]
                if decision["verdict"] == "accepted" and p["v_earth"]:
                    points.append({"cell_idx": p["cell_idx"], "depth_m": p["depth_m"],
                                   "status": "adopted_by_decision", "node_id": nid,
                                   "fingerprint": node["fingerprint"],
                                   "v_earth": p["v_earth"],
                                   "decision_event": decision["event_id"]})
            out_pings.append({"ping_id": ping_id, "points": points,
                              "excluded": excluded})
        return {"branch_id": branch_id, "event_id": state["event_id"],
                "pings": out_pings}

    # ----- write actions from review UI --------------------------------
    def submit_decision(self, branch_id: str, ping_id: str, cell_idx: int,
                        candidate_id: str, verdict: str, actor: str,
                        expected_head: str | None = None,
                        note: str = "") -> dict:
        state = self.view_branch_head(branch_id)
        nid = f"n:{ping_id}:c{cell_idx + 1:02d}:{candidate_id}"
        node = state["nodes"].get(nid)
        if node is None:
            raise KeyError(nid)
        payload = {"node_id": nid, "ping_id": ping_id, "cell_idx": cell_idx,
                   "candidate_id": candidate_id, "fingerprint": node["fingerprint"],
                   "verdict": verdict, "note": note}
        try:
            event = self.store.append_event(
                branch_id, "decision", payload, actor=actor,
                parent_event_id=None, expected_head=expected_head)
        except ConflictError:
            raise
        return event

    def adjust_parameter(self, branch_id: str, rule: str, payload: dict,
                         actor: str, note: str = "",
                         expected_head: str | None = None) -> dict:
        versions = [r["version"] for r in self.store.committed_rules()
                    if r["rule"] == rule]
        version = (max(versions) if versions else 0) + 1
        rid = f"run-param-{rule}-{version}"
        with self.store.run(f"parameter adjustment {rule}", run_id=rid) as batch:
            self.store.put_rule(batch.run_id, rule, version, payload)
            event = self.store.append_event(
                branch_id, "rule_revision",
                {"rule": rule, "note": note,
                 "record": {"version": version, "payload": payload}},
                actor=actor, expected_head=expected_head, run_id=batch.run_id)
            snap = build_snapshot(self.store.committed_sources(batch.run_id),
                                  self.store.committed_rules(batch.run_id))
            self.store.put_derived(batch.run_id, event["event_id"],
                                   graph.materialize(snap))
        return event

    def compare_branches(self, branch_a: str, node_id: str,
                         branch_b: str = "main") -> dict:
        a = self.view_branch_head(branch_a)["nodes"].get(node_id)
        b = self.view_branch_head(branch_b)["nodes"].get(node_id)
        return {
            "node_id": node_id,
            "same_fingerprint": bool(a and b and a["fingerprint"] == b["fingerprint"]),
            "a": {"branch": branch_a,
                  "fingerprint": a["fingerprint"] if a else None,
                  "payload": a["payload"] if a else None},
            "b": {"branch": branch_b,
                  "fingerprint": b["fingerprint"] if b else None,
                  "payload": b["payload"] if b else None}}

    def propagation_report(self, event_id: str) -> dict:
        """Diff snapshots either side of an event (why nodes stayed valid)."""
        # Locate event branch.
        target = None
        for branch in self.store.list_branches():
            for e in self.store.list_events(branch["branch_id"]):
                if e["event_id"] == event_id:
                    target = (branch["branch_id"], e)
                    break
        if target is None:
            raise KeyError(event_id)
        branch_id, event = target
        before_parent = event["parent_event_id"]
        before = self.state_at(branch_id, before_parent)
        after = self.state_at(branch_id, event_id)
        before_ids = {nid: n["fingerprint"] for nid, n in before["nodes"].items()}
        changed, unchanged, added = [], [], []
        for nid, n in after["nodes"].items():
            if nid not in before_ids:
                added.append(nid)
            elif before_ids[nid] == n["fingerprint"]:
                unchanged.append(nid)
            else:
                changed.append(nid)
        return {"event_id": event_id, "branch_id": branch_id,
                "unchanged_count": len(unchanged),
                "recomputed": sorted(changed),
                "added": sorted(added),
                "unchanged_sample": sorted(unchanged)[:10]}

    # ----- export / import ---------------------------------------------
    def export_bundle(self) -> dict:
        return self.store.export_json()

    @classmethod
    def import_bundle(cls, store: Store, bundle: dict) -> "Project":
        store.import_json(bundle)
        return cls(store)

    # ----- guided tour --------------------------------------------------
    def tour(self) -> list[dict]:
        return [
            {"step": 1, "title": "基线分开记录",
             "target": "/ui#ping=p01",
             "text": "p01 展示波束质量、四级保留候选、逐级向量与 ENU 剖面。"},
            {"step": 2, "title": "单 beam 失效",
             "target": "/ui?branch=x_single_beam#ping=p03",
             "text": "beam 2 置坏，仅 c3_134 合法，四波束解被排除不混入。"},
            {"step": 3, "title": "姿态跨越 360 度",
             "target": "/ui?branch=x_heading_wrap#ping=p05",
             "text": "一条姿态记录改写为 272°，解卷绕后插值仍连续。"},
            {"step": 4, "title": "时间缺口",
             "target": "/ui?branch=x_time_gap#ping=p07",
             "text": "80s 缺口内不插值，earth 速度保持排除。"},
            {"step": 5, "title": "三波束多解",
             "target": "/ui?branch=x_three_beam_ambiguous#ping=p04",
             "text": "c4 suspect，四个三波束候选全部保留待裁定。"},
            {"step": 6, "title": "安装角修订与变换顺序",
             "target": "/ui?branch=x_install_revision#ping=p02",
             "text": "instrument 指纹不变、ship/earth 变化；顺序分支同理。"},
            {"step": 7, "title": "无效深度 cell",
             "target": "/ui?branch=x_invalid_cell#ping=p01",
             "text": "c12 越过 bottom track 被排除且证据保留。"},
            {"step": 8, "title": "冲突裁定与历史回放",
             "target": "/ui?branch=main&at=e0004#ping=p02",
             "text": "A/B 冲突可解决；回到 e0002 恢复当时依赖与结论。"},
        ]
