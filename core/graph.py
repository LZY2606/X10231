"""依赖图传播：源 -> 质量 -> instrument -> ship -> earth -> 剖面/底跟踪。

节点身份稳定：node_id 只依赖语义位置（kind/ping/cell/beam/candidate），
跨运行不变；节点内容（value_id）由“上游指纹 + 规则版本”的派生指纹决定，
因此可以说明某结果为何仍有效（指纹命中缓存）或为何必须重算（输入指纹变了）。

语义硬约束：
- 三波束解与四波束解是不同 kind/身份，绝不混为一个值；欠定时保留全部合法候选。
- 姿态插值不跨缺口；缺口内 ping 的 earth 节点为 blocked（不是丢弃）。
- 坐标转换顺序、波束几何、磁偏角版本、安装角版本都进入派生指纹。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .attitude import AttitudeSeries, interpolate
from .fingerprint import fingerprint, short
from .geometry import radial_to_instrument
from .model import ProfileSources, beam_quantities, depth_quantity
from .quality import CellQuality, judge_beam, judge_cell
from .rotation import instrument_to_ship, ship_to_earth
from .units import require

NODE_KINDS = (
    "src", "attitude", "quality", "instrument", "ship",
    "earth", "profile", "bottomtrack",
)


@dataclass
class Node:
    node_id: str
    kind: str
    ping: int | None
    cell: int | None
    level: str
    status: str                     # ok | blocked | multi | excluded | unresolved
    value_id: str
    fp: str
    deps: list[str]
    payload: dict[str, Any]
    reused: bool = False
    reused_from_run: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "kind": self.kind, "ping": self.ping,
            "cell": self.cell, "level": self.level, "status": self.status,
            "value_id": self.value_id, "fp": self.fp, "fp_short": short(self.fp),
            "deps": self.deps, "payload": self.payload,
            "reused": self.reused, "reused_from_run": self.reused_from_run,
        }


def _source_basis(snap: ProfileSources) -> dict[str, Any]:
    """所有派生共享的“规则与配置依据”，进入每个派生指纹。"""
    return {
        "geometry": {"version": snap.geometry_version, "config": snap.geometry.to_dict()},
        "mounting": {"version": snap.mounting_version, "angles": snap.mounting.to_dict()},
        "declination": {"version": snap.source_versions.get("declination", ""),
                        "value": snap.declination},
        "rules": {"version": snap.rules_version, "rules": snap.rules.to_dict()},
        "transform": {"order": snap.transform_order, "version": snap.transform_version},
        "attitude_max_gap_s": snap.attitude_max_gap_s,
    }


def _src_node(kind: str, ping: int, cell: int, beam: int | None,
              raw: Any, unit: str, fp_input: str) -> Node:
    ident = {"kind": "src", "source": kind, "ping": ping, "cell": cell, "beam": beam}
    nid = f"src:{kind}:p{ping}:c{cell}" + (f":b{beam}" if beam is not None else "")
    fp = fingerprint({"ident": ident, "raw": raw, "unit": unit, "input": fp_input})
    return Node(nid, "src", ping, cell, "source", "ok", f"val:{short(fp)}", fp, [],
                {"kind": kind, "raw": raw, "unit": unit})


class GraphRun:
    """一次依赖图计算。node_cache 为跨运行内容寻址缓存 {fp: value_id}。"""

    def __init__(self, snap: ProfileSources, run_id: str,
                 node_cache: dict[str, str] | None = None,
                 previous_nodes: dict[str, Node] | None = None):
        self.snap = snap
        self.run_id = run_id
        self.nodes: dict[str, Node] = {}
        self.node_cache = node_cache or {}
        self.previous = previous_nodes or {}
        self.changed_reasons: dict[str, list[str]] = {}
        self.basis = _source_basis(snap)
        self._attitude_series = AttitudeSeries(
            times=np.asarray(snap.attitude["times"], dtype=float),
            heading_deg=np.asarray(snap.attitude["heading_deg"], dtype=float),
            pitch_deg=np.asarray(snap.attitude["pitch_deg"], dtype=float),
            roll_deg=np.asarray(snap.attitude["roll_deg"], dtype=float),
            max_gap_s=snap.attitude_max_gap_s,
        )

    def _store(self, node: Node) -> Node:
        prev = self.previous.get(node.node_id)
        if prev is not None and prev.fp == node.fp:
            node.reused = True
            node.reused_from_run = self.run_id
            node.value_id = prev.value_id
            self.changed_reasons[node.node_id] = ["unchanged-fingerprint"]
        else:
            if prev is not None:
                self.changed_reasons[node.node_id] = self._why_changed(prev, node)
            else:
                self.changed_reasons[node.node_id] = ["new-node"]
        self.nodes[node.node_id] = node
        return node

    @staticmethod
    def _why_changed(prev: Node, new: Node) -> list[str]:
        old_deps = set(prev.deps)
        new_deps = set(new.deps)
        reasons = []
        if old_deps != new_deps:
            reasons.append("dependency-set-changed")
        # 指纹比较由上层 detail 展开；这里给出规则版本差异类别
        return reasons or ["input-or-rule-fingerprint-changed"]

    def run(self) -> dict[str, Node]:
        self._build_attitude_nodes()
        fourbeam_w_by_ping: dict[int, list[float]] = {}
        # 第一遍：源节点、质量、instrument（先求四波束 w 参考）
        for ping_raw in self.snap.pings:
            p = ping_raw.ping
            ws: list[float] = []
            for c in range(ping_raw.cell_count()):
                self._build_cell_up_to_instrument(p, c)
                inst = self.nodes[f"instrument:p{p}:c{c}"]
                if inst.status == "ok":
                    ws.append(inst.payload["uvw"][2])
            fourbeam_w_by_ping[p] = ws
        # 第二遍：三波束用同 ping 四波束 w 中位数做候选；再 ship/earth/profile/bt
        for ping_raw in self.snap.pings:
            p = ping_raw.ping
            ref_w = None
            if fourbeam_w_by_ping[p]:
                ref_w = float(np.median(fourbeam_w_by_ping[p]))
            for c in range(ping_raw.cell_count()):
                self._build_ship_earth(p, c, ref_w)
            self._build_profile(p)
            self._build_bottomtrack(p, ref_w)
        return self.nodes

    # ---- 姿态 ----
    def _build_attitude_nodes(self) -> None:
        snap = self.snap
        att_fp_input = fingerprint({"times": snap.attitude["times"],
                                    "heading": snap.attitude["heading_deg"],
                                    "pitch": snap.attitude["pitch_deg"],
                                    "roll": snap.attitude["roll_deg"],
                                    "version": snap.source_versions.get("attitude", "")})
        src_nid = "src:attitude:series"
        src_fp = fingerprint({"ident": {"kind": "src", "source": "attitude"},
                              "input": att_fp_input, **self.basis})
        self._store(Node(src_nid, "src", None, None, "source", "ok",
                         f"val:{short(src_fp)}", src_fp, [],
                         {"kind": "attitude", "samples": len(snap.attitude["times"])}))
        for ping_raw in snap.pings:
            p = ping_raw.ping
            point = interpolate(self._attitude_series, ping_raw.time_s)
            deps = [src_nid]
            parts = {"ident": {"kind": "attitude", "ping": p},
                     "deps": deps, "point_time_s": point.time_s,
                     "status": point.status, "basis": self.basis,
                     "evidence": point.evidence, "segment": point.segment}
            if point.status == "ok":
                parts.update({"heading_deg": point.heading_deg,
                              "pitch_deg": point.pitch_deg, "roll_deg": point.roll_deg})
            fp = fingerprint(parts)
            self._store(Node(
                f"attitude:p{p}", "attitude", p, None, "attitude",
                "ok" if point.status == "ok" else "unresolved",
                f"val:{short(fp)}", fp, deps,
                {"time_s": point.time_s, "status": point.status, "segment": point.segment,
                 "heading_deg": point.heading_deg, "pitch_deg": point.pitch_deg,
                 "roll_deg": point.roll_deg, "evidence": point.evidence},
            ))

    # ---- 单 cell：源 -> 质量 -> instrument ----
    def _build_cell_up_to_instrument(self, p: int, c: int) -> None:
        pr = self.snap.ping(p)
        unit = pr.velocity_unit
        raw_row = pr.beam_velocity_raw[c]
        corr_row = pr.correlation[c]
        amp_row = pr.amplitude[c]
        bad_row = pr.bad_beam[c]
        depth = pr.cells[c]

        dep_ids: list[str] = []
        beam_quants = beam_quantities(raw_row, unit)
        for b in range(4):
            fp_in = fingerprint({"source_version": self.snap.source_versions.get("beam_velocity", ""),
                                 "raw": raw_row[b]})
            n = _src_node("beam_velocity", p, c, b, raw_row[b], unit, fp_in)
            self._store(n)
            dep_ids.append(n.node_id)
            cf = fingerprint({"source_version": self.snap.source_versions.get("correlation", ""),
                              "raw": corr_row[b]})
            self._store(_src_node("correlation", p, c, b, corr_row[b], "counts", cf))
            af = fingerprint({"source_version": self.snap.source_versions.get("amplitude", ""),
                              "raw": amp_row[b]})
            self._store(_src_node("amplitude", p, c, b, amp_row[b], "counts", af))
            bf = fingerprint({"source_version": self.snap.source_versions.get("bad_beam", ""),
                              "raw": bad_row[b]})
            self._store(_src_node("bad_beam", p, c, b, bad_row[b], "flag", bf))

        depth_q = depth_quantity(depth)
        depth_fp = fingerprint({"source_version": self.snap.source_versions.get("depth_cell", ""),
                                "raw": depth.depth_raw, "unit": depth.depth_unit,
                                "invalid": depth.invalid})
        self._store(_src_node("depth_cell", p, c, None, depth.depth_raw,
                              depth.depth_unit, depth_fp))

        bq = [judge_beam(b, beam_quants[b], corr_row[b], amp_row[b], bad_row[b],
                         self.snap.rules) for b in range(4)]
        good_mask = [b.good for b in bq]
        q_deps = dep_ids + [f"src:correlation:p{p}:c{c}:b{b}" for b in range(4)] + \
                 [f"src:amplitude:p{p}:c{c}:b{b}" for b in range(4)] + \
                 [f"src:bad_beam:p{p}:c{c}:b{b}" for b in range(4)] + \
                 [f"src:depth_cell:p{p}:c{c}"]
        sol = radial_to_instrument(beam_quants, self.snap.geometry, good_mask=good_mask)
        err = sol.candidates[0]["error_velocity"] if sol.kind == "four-beam" else None
        cellq = judge_cell(p, c, depth_q, bq, self.snap.rules,
                           error_velocity_mps=err, explicit_invalid=depth.invalid)
        q_fp = fingerprint({"ident": {"kind": "quality", "ping": p, "cell": c},
                            "deps": q_deps, "cell": cellq.as_dict(),
                            "basis": self.basis})
        self._store(Node(
            f"quality:p{p}:c{c}", "quality", p, c, "quality",
            "excluded" if not cellq.accepted and sol.kind != "three-beam" else
            ("multi" if sol.kind == "three-beam" else "ok"),
            f"val:{short(q_fp)}", q_fp, q_deps, cellq.as_dict(),
        ))

        status = {"four-beam": "ok", "three-beam": "multi",
                  "unsolvable": "blocked"}[sol.kind]
        inst_payload = {
            "kind": sol.kind, "good_beams": list(sol.good_beams),
            "note": sol.note,
            "candidates": sol.candidates,  # 三波束时全部候选在此；四波束仅一个精确解
            "unit": "m/s", "frame": "instrument",
        }
        if sol.kind == "four-beam":
            inst_payload["uvw"] = sol.candidates[0]["uvw"]
            inst_payload["error_velocity"] = sol.candidates[0]["error_velocity"]
        i_deps = [f"quality:p{p}:c{c}"] + [f"src:beam_velocity:p{p}:c{c}:b{b}" for b in range(4)]
        i_fp = fingerprint({"ident": {"kind": "instrument", "ping": p, "cell": c},
                            "deps": i_deps, "solution": inst_payload,
                            "basis": self.basis})
        self._store(Node(
            f"instrument:p{p}:c{c}", "instrument", p, c, "derived",
            status, f"val:{short(i_fp)}", i_fp, i_deps, inst_payload,
        ))

    # ---- ship / earth：每个 instrument 候选独立成节点（多解保留） ----
    def _build_ship_earth(self, p: int, c: int, ref_w: float | None) -> None:
        inst = self.nodes[f"instrument:p{p}:c{c}"]
        att = self.nodes[f"attitude:p{p}"]
        mount = self.snap.mounting
        order = self.snap.transform_order
        decl_deg = float(self.snap.declination["value_deg"])

        for cand in inst.payload["candidates"]:
            assumption = cand["assumption"]
            uvw = np.asarray(cand["uvw"], dtype=float)
            v_ship = instrument_to_ship(uvw, mount, order=order)
            s_nid = f"ship:p{p}:c{c}:{assumption}"
            s_deps = [inst.node_id]
            s_payload = {"assumption": assumption, "ship": [float(x) for x in v_ship],
                         "unit": "m/s", "frame": "ship", "parent_kind": inst.payload["kind"]}
            s_fp = fingerprint({"ident": {"kind": "ship", "ping": p, "cell": c,
                                          "assumption": assumption},
                                "deps": s_deps, "payload": s_payload, "basis": self.basis})
            self._store(Node(s_nid, "ship", p, c, "derived",
                             "multi" if inst.status == "multi" else "ok",
                             f"val:{short(s_fp)}", s_fp, s_deps, s_payload))

            e_nid = f"earth:p{p}:c{c}:{assumption}"
            e_deps = [s_nid, att.node_id]
            if att.status != "ok":
                e_payload = {"assumption": assumption, "earth": None, "unit": "m/s",
                             "frame": "earth", "datum": "ENU",
                             "blocked_reason": att.payload["evidence"].get(
                                 "reason", "attitude-unresolved"),
                             "heading_true_deg": None}
                e_fp = fingerprint({"ident": {"kind": "earth", "ping": p, "cell": c,
                                              "assumption": assumption},
                                    "deps": e_deps, "payload": e_payload,
                                    "basis": self.basis})
                self._store(Node(e_nid, "earth", p, c, "derived", "blocked",
                                 f"val:{short(e_fp)}", e_fp, e_deps, e_payload))
                continue
            v_earth = ship_to_earth(
                v_ship, att.payload["heading_deg"], att.payload["pitch_deg"],
                att.payload["roll_deg"], decl_deg, mount, order=order)
            e_payload = {
                "assumption": assumption,
                "earth": [float(x) for x in v_earth],
                "unit": "m/s", "frame": "earth", "datum": "ENU",
                "heading_mag_deg": att.payload["heading_deg"],
                "pitch_deg": att.payload["pitch_deg"],
                "roll_deg": att.payload["roll_deg"],
                "declination_deg": decl_deg,
                "heading_true_deg": (att.payload["heading_deg"] + decl_deg
                                     + mount.heading_alignment_deg) % 360.0,
                "attitude_evidence": att.payload["evidence"],
                "transform_order": order,
            }
            e_fp = fingerprint({"ident": {"kind": "earth", "ping": p, "cell": c,
                                          "assumption": assumption},
                                "deps": e_deps, "payload": e_payload, "basis": self.basis})
            self._store(Node(e_nid, "earth", p, c, "derived",
                             "multi" if inst.status == "multi" else "ok",
                             f"val:{short(e_fp)}", e_fp, e_deps, e_payload))

    # ---- 垂向剖面：accepted cell 入图；被排除 cell 保留证据 ----
    def _build_profile(self, p: int) -> None:
        entries: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        pr = self.snap.ping(p)
        for c in range(pr.cell_count()):
            qn = self.nodes[f"quality:p{p}:c{c}"]
            inst = self.nodes[f"instrument:p{p}:c{c}"]
            row = {"cell": c, "depth_m": qn.payload["depth_m"],
                   "reason_codes": qn.payload["reason_codes"]}
            if qn.payload["accepted"] and inst.status == "ok":
                earth = self.nodes[f"earth:p{p}:c{c}:four-beam-exact"].payload
                row["earth"] = earth["earth"]
                row["earth_node"] = f"earth:p{p}:c{c}:four-beam-exact"
                entries.append(row)
            else:
                row["instrument_kind"] = inst.payload["kind"]
                row["candidate_assumptions"] = [x["assumption"]
                                                for x in inst.payload["candidates"]]
                excluded.append(row)
        payload = {"ping": p, "accepted": entries, "excluded": excluded,
                  "earth_datum": "ENU", "depth_unit": "m", "velocity_unit": "m/s"}
        deps = [f"quality:p{p}:c{c}" for c in range(pr.cell_count())] + \
               [f"instrument:p{p}:c{c}" for c in range(pr.cell_count())]
        fp = fingerprint({"ident": {"kind": "profile", "ping": p},
                          "deps": deps, "payload": payload, "basis": self.basis})
        self._store(Node(f"profile:p{p}", "profile", p, None, "derived", "ok",
                         f"val:{short(fp)}", fp, deps, payload))

    # ---- bottom track：独立重算 instrument/ship/earth ----
    def _build_bottomtrack(self, p: int, ref_w: float | None) -> None:
        pr = self.snap.ping(p)
        att = self.nodes[f"attitude:p{p}"]
        deps = [att.node_id]
        if pr.bottom_track_raw is None:
            payload = {"ping": p, "available": False, "reason": "no-bottom-track-source"}
            fp = fingerprint({"ident": {"kind": "bottomtrack", "ping": p},
                              "payload": payload, "basis": self.basis})
            self._store(Node(f"bottomtrack:p{p}", "bottomtrack", p, None,
                             "derived", "blocked", f"val:{short(fp)}", fp, deps, payload))
            return
        sol = radial_to_instrument(
            beam_quantities(pr.bottom_track_raw, pr.bottom_track_unit),
            self.snap.geometry)
        payload: dict[str, Any] = {"ping": p, "available": True,
                                   "solution_kind": sol.kind,
                                   "candidates": sol.candidates}
        if sol.kind == "four-beam" and att.status == "ok":
            uvw = np.asarray(sol.candidates[0]["uvw"], dtype=float)
            mount = self.snap.mounting
            v_ship = instrument_to_ship(uvw, mount, order=self.snap.transform_order)
            v_earth = ship_to_earth(
                v_ship, att.payload["heading_deg"], att.payload["pitch_deg"],
                att.payload["roll_deg"], float(self.snap.declination["value_deg"]),
                mount, order=self.snap.transform_order)
            payload.update({"ship": [float(x) for x in v_ship],
                            "earth": [float(x) for x in v_earth],
                            "heading_true_deg": self.nodes[
                                f"earth:p{p}:c0:four-beam-exact"].payload.get(
                                "heading_true_deg") if any(
                                k == f"earth:p{p}:c0:four-beam-exact"
                                for k in self.nodes) else None})
        elif att.status != "ok":
            payload["blocked_reason"] = att.payload["evidence"].get(
                "reason", "attitude-unresolved")
        fp = fingerprint({"ident": {"kind": "bottomtrack", "ping": p}, "deps": deps,
                          "payload": payload, "basis": self.basis})
        status = "ok" if payload.get("earth") is not None else (
            "multi" if sol.kind == "three-beam" else "blocked")
        self._store(Node(f"bottomtrack:p{p}", "bottomtrack", p, None,
                         "derived", status, f"val:{short(fp)}", fp, deps, payload))


def build_graph(snap: ProfileSources, run_id: str,
                previous: dict[str, Node] | None = None) -> tuple[dict[str, Node], dict[str, list[str]]]:
    gr = GraphRun(snap, run_id, previous_nodes=previous)
    nodes = gr.run()
    return nodes, gr.changed_reasons
