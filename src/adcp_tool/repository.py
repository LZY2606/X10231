"""仓库：原始材料导入、配置版本、内容寻址派生图、原子版本提交、
只追加事件、乐观锁冲突、历史视图与来源解释。
"""
from __future__ import annotations

import json
import sqlite3
import time
import typing
from typing import Any, Dict, List, Optional, Tuple

from .config import (DeclinationConfig, GeometryConfig, InstallConfig,
                     TransformConfig)
from .derive import (depth_cell_validity, derive_cell, solve_bottom_track)
from .attitude import AttitudeInterpConfig, interpolate_at
from .geometry import BeamGeometry
from .quality import QCThresholds
from .fingerprint import canonical, fingerprint, node_id
from . import db as dbmod


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime(
        "%Y-%m-%dT%H:%M:%S")


def _geometry_from_payload(p: dict) -> GeometryConfig:
    g = p["geometry"]
    return GeometryConfig(
        version=p["version"],
        geometry=BeamGeometry(
            azimuth_deg=tuple(g["azimuth_deg"]),
            zenith_deg=tuple(g["zenith_deg"]),
            convention=g["convention"],
        ),
    )


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------- 原始材料 ----------
    def import_case(self, case: dict, batch_id: Optional[str] = None) -> str:
        """导入一个离线案例包（原始四类证据分开记录，保留原始表示）。"""
        conn = self.conn
        batch_id = batch_id or case.get("batch_id") or f"batch_{_now_id()}"
        conn.execute(
            "INSERT OR REPLACE INTO source_batch(batch_id,kind,imported_at,"
            "original_encoding) VALUES(?,?,?,?)",
            (batch_id, case.get("kind", "adcp_case"), _now(),
               case.get("encoding")),
        )
        for ping in case["pings"]:
            pid = ping["ping_id"]
            conn.execute(
                "INSERT OR REPLACE INTO ping(ping_id,batch_id,seq,t_s,"
                "raw_time_text,time_unit) VALUES(?,?,?,?,?,?)",
                (pid, batch_id, ping["seq"], ping["t_s"],
                 ping.get("raw_time"), ping.get("time_unit", "s")),
            )
            for cell in ping["cells"]:
                cid = cell["cell_id"]
                conn.execute(
                    "INSERT OR REPLACE INTO cell(cell_id,ping_id,bin_index,"
                    "depth_raw,depth_unit) VALUES(?,?,?,?,?)",
                    (cid, pid, cell["bin_index"], cell.get("depth_raw"),
                     cell.get("depth_unit", "m")),
                )
                for b, ev in enumerate(cell["beams"]):
                    self._upsert_beam(cid, b, ev)
            if "bottom_track" in ping:
                for b, ev in enumerate(ping["bottom_track"]):
                    conn.execute(
                        "INSERT OR REPLACE INTO bottom_track(id,ping_id,beam,"
                        "velocity_raw,velocity_unit,correlation_raw,"
                        "correlation_unit,amplitude_raw,amplitude_unit,"
                        "bad_flag,range_raw,range_unit) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"{pid}/bt/{b}", pid, b,
                         ev.get("velocity"), ev.get("velocity_unit"),
                         ev.get("correlation"), ev.get("correlation_unit"),
                         ev.get("amplitude"), ev.get("amplitude_unit"),
                         None if ev.get("bad_flag") is None else int(ev["bad_flag"]),
                         ev.get("range"), ev.get("range_unit", "m")),
                    )
        for i, s in enumerate(case.get("attitude", [])):
            conn.execute(
                "INSERT OR REPLACE INTO attitude_sample(id,batch_id,seq,t_s,"
                "heading_raw,heading_unit,heading_datum,pitch_raw,pitch_unit,"
                "roll_raw,roll_unit) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (f"{batch_id}/att/{s.get('seq', i)}", batch_id,
                 s.get("seq", i), s["t_s"],
                 s.get("heading_deg"), s.get("heading_unit", "deg"),
                 s.get("heading_datum", "MAGNETIC"),
                 s.get("pitch_deg"), s.get("pitch_unit", "deg"),
                 s.get("roll_deg"), s.get("roll_unit", "deg")),
            )
        return batch_id

    def _upsert_beam(self, cid: str, b: int, ev: dict) -> None:
        conn = self.conn
        conn.execute(
            "INSERT OR REPLACE INTO beam_velocity(id,cell_id,beam,raw_value,"
            "raw_unit) VALUES(?,?,?,?,?)",
            (f"{cid}/v/{b}", cid, b, ev.get("velocity"), ev.get("velocity_unit")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO beam_correlation(id,cell_id,beam,raw_value,"
            "raw_unit) VALUES(?,?,?,?,?)",
            (f"{cid}/corr/{b}", cid, b, ev.get("correlation"),
             ev.get("correlation_unit", "count")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO beam_amplitude(id,cell_id,beam,raw_value,"
            "raw_unit) VALUES(?,?,?,?,?)",
            (f"{cid}/amp/{b}", cid, b, ev.get("amplitude"),
             ev.get("amplitude_unit", "dB")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO beam_bad_flag(id,cell_id,beam,flag) "
            "VALUES(?,?,?,?)",
            (f"{cid}/bad/{b}", cid, b,
             0 if ev.get("bad_flag") is None else int(ev["bad_flag"])),
        )

    def patch_beam_evidence(self, cell_id: str, beam: int, field: str,
                            value: Any) -> None:
        """只改动一个直接来源（局部源数据改动）。"""
        table_map = {
            "velocity": ("beam_velocity", "raw_value"),
            "correlation": ("beam_correlation", "raw_value"),
            "amplitude": ("beam_amplitude", "raw_value"),
            "bad_flag": ("beam_bad_flag", "flag"),
        }
        table, col = table_map[field]
        conn = self.conn
        if field == "bad_flag":
            conn.execute(
                f"UPDATE {table} SET {col}=? WHERE cell_id=? AND beam=?",
                (None if value is None else int(value), cell_id, beam))
        else:
            conn.execute(
                f"UPDATE {table} SET {col}=? WHERE cell_id=? AND beam=?",
                (value, cell_id, beam))

    # ---------- 读取原始材料 ----------
    def list_pings(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM ping ORDER BY seq").fetchall()]

    def list_cells(self, ping_id: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM cell WHERE ping_id=? ORDER BY bin_index",
            (ping_id,)).fetchall()]

    def cell_evidence(self, cell_id: str) -> Dict[int, dict]:
        conn = self.conn
        out: Dict[int, dict] = {}
        rows = conn.execute(
            "SELECT bv.beam, bv.raw_value velocity, bv.raw_unit velocity_unit,"
            " bc.raw_value correlation, bc.raw_unit correlation_unit,"
            " ba.raw_value amplitude, ba.raw_unit amplitude_unit,"
            " bf.flag bad_flag FROM beam_velocity bv "
            "JOIN beam_correlation bc ON bc.cell_id=bv.cell_id AND bc.beam=bv.beam "
            "JOIN beam_amplitude ba ON ba.cell_id=bv.cell_id AND ba.beam=bv.beam "
            "JOIN beam_bad_flag bf ON bf.cell_id=bv.cell_id AND bf.beam=bv.beam "
            "WHERE bv.cell_id=?", (cell_id,)).fetchall()
        for r in rows:
            d = dict(r)
            out[d.pop("beam")] = d
        return out

    def bottom_track_evidence(self, ping_id: str) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for r in self.conn.execute(
                "SELECT * FROM bottom_track WHERE ping_id=? ORDER BY beam",
                (ping_id,)).fetchall():
            d = dict(r)
            b = d.pop("beam")
            out[b] = {
                "velocity": d["velocity_raw"], "velocity_unit": d["velocity_unit"],
                "correlation": d["correlation_raw"],
                "correlation_unit": d["correlation_unit"],
                "amplitude": d["amplitude_raw"],
                "amplitude_unit": d["amplitude_unit"],
                "bad_flag": d["bad_flag"], "range_m": d["range_raw"],
            }
        return out

    def attitude_series(self, batch_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attitude_sample WHERE batch_id=? ORDER BY seq",
            (batch_id,)).fetchall()
        return [{
            "t": r["t_s"], "seq": r["seq"],
            "heading_deg": r["heading_raw"], "pitch_deg": r["pitch_raw"],
            "roll_deg": r["roll_raw"],
        } for r in rows]

    # ---------- 配置 ----------
    def put_config(self, ctype: str, version: str, payload: dict,
                   event_seq: Optional[int] = None) -> str:
        cid = f"cfg/{ctype}/{version}"
        self.conn.execute(
            "INSERT OR REPLACE INTO config_blob(config_id,type,version,"
            "payload_json,created_event_seq) VALUES(?,?,?,?,?)",
            (cid, ctype, version, json.dumps(payload, ensure_ascii=False),
             event_seq))
        return cid

    def get_config(self, ctype: str, version: str) -> dict:
        r = self.conn.execute(
            "SELECT payload_json FROM config_blob WHERE type=? AND version=?",
            (ctype, version)).fetchone()
        if r is None:
            raise KeyError(f"缺少配置 {ctype}@{version}")
        return json.loads(r["payload_json"])

    def get_configs(self, versions: Dict[str, str]) -> Dict[str, Any]:
        p = self.get_config("geometry", versions["geometry"])
        geo = _geometry_from_payload(p)
        inst = InstallConfig(**self.get_config("install", versions["install"]))
        decl = DeclinationConfig(**self.get_config("declination",
                                                   versions["declination"]))
        tform = TransformConfig(**self.get_config("transform",
                                                  versions["transform"]))
        thr = QCThresholds(**self.get_config("qc", versions["qc"]))
        acfg = AttitudeInterpConfig(**self.get_config("attitude",
                                                      versions["attitude"]))
        depth = self.get_config("depth", versions["depth"])
        return {"geometry": geo, "install": inst, "decl": decl,
                "tform": tform, "qc": thr, "attitude": acfg, "depth": depth,
                "versions": versions}

    # ---------- 派生图 ----------
    def get_or_create_node(self, kind: str, fp_input: Any,
                           deps: Dict[str, str], event_seq: int,
                           explain: Optional[dict] = None
                           ) -> Tuple[str, bool, dict]:
        """内容寻址取/建节点。返回 (node_id, reused, payload)。

        deps: slot -> dep_node_id。节点的派生输入（含几何、变换顺序、磁偏角
        版本、上游节点）全部进入 fp_input，因此这些任一变化都会产生新指纹、
        新节点身份。
        """
        conn = self.conn
        material = {"input": fp_input,
                    "deps": {k: deps[k] for k in sorted(deps)}}
        fp = fingerprint(material)
        row = conn.execute(
            "SELECT node_id,payload_json FROM node WHERE fingerprint=?",
            (fp,)).fetchone()
        if row is not None:
            return row["node_id"], True, json.loads(row["payload_json"])
        nid = node_id(kind, fp_input)
        payload = {"input": fp_input, "result": None}
        conn.execute(
            "INSERT INTO node(node_id,kind,fingerprint,payload_json,"
            "created_event_seq) VALUES(?,?,?,?,?)",
            (nid, kind, fp, json.dumps(payload, ensure_ascii=False), event_seq))
        for slot, dep in deps.items():
            conn.execute(
                "INSERT OR IGNORE INTO node_dep(node_id,dep_node_id,dep_slot)"
                " VALUES(?,?,?)", (nid, dep, slot))
        return nid, False, payload

    def set_node_result(self, node_id: str, result: Any) -> None:
        row = self.conn.execute(
            "SELECT payload_json FROM node WHERE node_id=?", (node_id,)).fetchone()
        payload = json.loads(row["payload_json"])
        payload["result"] = result
        if "explain" not in payload:
            payload["explain"] = {}
        self.conn.execute("UPDATE node SET payload_json=? WHERE node_id=?",
                          (json.dumps(payload, ensure_ascii=False), node_id))

    def get_node(self, node_id: str) -> dict:
        r = self.conn.execute(
            "SELECT * FROM node WHERE node_id=?", (node_id,)).fetchone()
        if r is None:
            raise KeyError(node_id)
        return {"node_id": node_id, "kind": r["kind"],
                "fingerprint": r["fingerprint"],
                "payload": json.loads(r["payload_json"])}

    def node_deps(self, node_id: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT dep_node_id,dep_slot FROM node_dep WHERE node_id=? "
            "ORDER BY dep_slot", (node_id,)).fetchall()]

    def explain_node(self, node_id: str) -> dict:
        """某项结果为何有效/依赖什么：返回完整直接来源链（一层）。"""
        node = self.get_node(node_id)
        return {"node": node, "deps": self.node_deps(node_id)}

    def ancestors(self, node_id: str) -> List[str]:
        seen, stack = set(), [node_id]
        while stack:
            cur = stack.pop()
            for d in self.node_deps(cur):
                if d["dep_node_id"] not in seen:
                    seen.add(d["dep_node_id"])
                    stack.append(d["dep_node_id"])
        return sorted(seen)

    # ---------- 事件 ----------
    def append_event(self, event_type: str, basis: dict, actor: str = "reviewer",
                     version_id: Optional[str] = None,
                     parent_seq: Optional[int] = None) -> int:
        conn = self.conn
        seq = self.next_event_seq()
        conn.execute(
            "INSERT INTO event(seq,event_type,version_id,basis_json,created_at,"
            "actor,parent_seq) VALUES(?,?,?,?,?,?,?)",
            (seq, event_type, version_id,
             json.dumps(basis, ensure_ascii=False), _now(), actor, parent_seq))
        return seq

    def next_event_seq(self) -> int:
        r = self.conn.execute("SELECT MAX(seq) m FROM event").fetchone()
        return int(r["m"] or 0) + 1

    def list_events(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM event ORDER BY seq").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["basis"] = json.loads(d.pop("basis_json"))
            out.append(d)
        return out

    # ---------- 版本提交（原子） ----------
    def current_version(self) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT version_id,event_seq FROM current_pointer WHERE only_id=1"
        ).fetchone()
        if r is None:
            return None
        return {"version_id": r["version_id"], "event_seq": r["event_seq"]}

    def commit_version(self, version_id: str, event_seq: int,
                       slots: Dict[str, str]) -> None:
        """在一个事务内：staged -> 写槽位 -> committed -> 切 current。

        只有完整写完后才切换 current 指针；与启动恢复配合实现“旧版本或
        完整新版本”，不出现半版本。
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO version_stage(version_id,created_event_seq,state)"
                " VALUES(?,?, 'staged')", (version_id, event_seq))
            for key, nid in slots.items():
                conn.execute(
                    "INSERT OR REPLACE INTO version_slot(version_id,slot_key,"
                    "node_id) VALUES(?,?,?)", (version_id, key, nid))
            conn.execute(
                "UPDATE version_stage SET state='committed' WHERE version_id=?",
                (version_id,))
            conn.execute(
                "INSERT INTO current_pointer(only_id,version_id,event_seq) "
                "VALUES(1,?,?) ON CONFLICT(only_id) DO UPDATE SET "
                "version_id=excluded.version_id,event_seq=excluded.event_seq",
                (version_id, event_seq))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def version_slots(self, version_id: str) -> Dict[str, str]:
        return {r["slot_key"]: r["node_id"] for r in self.conn.execute(
            "SELECT slot_key,node_id FROM version_slot WHERE version_id=?",
            (version_id,)).fetchall()}

    # ---------- 冲突 ----------
    def register_conflict(self, scope_key: str, base_version_id: str,
                          winner: Optional[str], loser: Optional[str],
                          detail: dict) -> str:
        cid = f"conflict/{scope_key}/{_now_id()}"
        self.conn.execute(
            "INSERT INTO edit_conflict(conflict_id,scope_key,base_version_id,"
            "winner_version_id,loser_version_id,status,detail_json) "
            "VALUES(?,?,?,?,?, 'open',?)",
            (cid, scope_key, base_version_id, winner, loser,
             json.dumps(detail, ensure_ascii=False)))
        return cid

    def resolve_conflict(self, conflict_id: str, winner_version_id: str) -> None:
        self.conn.execute(
            "UPDATE edit_conflict SET status='resolved',"
            "winner_version_id=? WHERE conflict_id=?",
            (winner_version_id, conflict_id))

    def list_conflicts(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edit_conflict ORDER BY conflict_id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json"))
            out.append(d)
        return out

    # ---------- 候选记录 ----------
    def record_candidates(self, event_seq: int, scope_key: str,
                          candidate_nodes: List[Tuple[str, str]]) -> None:
        """candidate_nodes: [(node_id,label)]。被采纳与未采纳候选都保留。"""
        for nid, label in candidate_nodes:
            self.conn.execute(
                "INSERT OR IGNORE INTO candidate(candidate_id,event_seq,"
                "scope_key,node_id,label,adopted) VALUES(?,?,?,?,?,0)",
                (f"cand/{scope_key}/{nid}", event_seq, scope_key, nid, label))

    def mark_candidate(self, scope_key: str, node_id: str, adopted: bool) -> None:
        self.conn.execute(
            "UPDATE candidate SET adopted=? WHERE scope_key=? AND node_id=?",
            (1 if adopted else 0, scope_key, node_id))

    def candidates_for(self, scope_key: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM candidate WHERE scope_key=? ORDER BY candidate_id",
            (scope_key,)).fetchall()]


def _now_id() -> str:
    return str(int(time.time() * 1000) % 10_000_000_000)
