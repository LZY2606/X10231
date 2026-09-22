"""SQLite storage: immutable sources, atomic commit versions, events, export.

Guarantees:
* every derivation commit is a full, independent version; the branch current
  pointer moves only after all rows landed (inside one transaction).
* a crash mid-write leaves the old pointer, and on restart the orphan partial
  version is removed (either old or complete new, never torn state).
* two users editing from the same base_commit get a resolvable
  EditConflict instead of silent overwrite.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from adcp.fingerprint import fingerprint

SCHEMA_VERSION = 2


class EditConflict(Exception):
    def __init__(self, message: str, conflict_id: str):
        super().__init__(message)
        self.conflict_id = conflict_id


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS sources(
            sid TEXT NOT NULL, rev INTEGER NOT NULL, type TEXT NOT NULL,
            ping INTEGER, cell INTEGER, beam INTEGER, time_index INTEGER,
            value_json TEXT NOT NULL, unit TEXT NOT NULL,
            supersedes INTEGER, note TEXT DEFAULT '',
            PRIMARY KEY(sid, rev));
        CREATE TABLE IF NOT EXISTS parameters(key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL, version_tag TEXT NOT NULL,
            updated_event_id TEXT);
        CREATE TABLE IF NOT EXISTS branches(name TEXT PRIMARY KEY,
            current_cid TEXT, base_cid TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS commits(
            cid TEXT PRIMARY KEY, parent_cid TEXT, branch TEXT,
            created_at REAL, label TEXT, kind TEXT, note TEXT,
            manifest_json TEXT NOT NULL, complete INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS derived(
            uid TEXT, cid TEXT, kind TEXT, fp TEXT,
            value_json TEXT NOT NULL, deps_json TEXT NOT NULL,
            refs_json TEXT NOT NULL, reused_from_cid TEXT,
            PRIMARY KEY(uid, cid));
        CREATE TABLE IF NOT EXISTS events(
            eid TEXT PRIMARY KEY, seq INTEGER UNIQUE, at REAL,
            commit_cid TEXT, type TEXT, payload_json TEXT NOT NULL,
            actor TEXT, superseded_by TEXT, branch TEXT);
        CREATE TABLE IF NOT EXISTS conflicts(
            id TEXT PRIMARY KEY, at REAL, branch TEXT, base_cid TEXT,
            committed_cid TEXT, attempted_label TEXT, attempted_payload TEXT,
            resolved_cid TEXT, resolution TEXT);
        """)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
        (str(SCHEMA_VERSION),))
    conn.commit()
    return conn


def recover(conn: sqlite3.Connection) -> List[str]:
    """Remove torn commits (created but never marked complete)."""
    rows = conn.execute(
        "SELECT cid FROM commits WHERE complete=0").fetchall()
    torn = [r["cid"] for r in rows]
    for cid in torn:
        conn.execute("DELETE FROM derived WHERE cid=?", (cid,))
        conn.execute("DELETE FROM commits WHERE cid=?", (cid,))
    conn.commit()
    return torn


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _seq(conn: sqlite3.Connection) -> int:
    r = conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM events").fetchone()
    return int(r["n"])


def _crashpoint(conn: sqlite3.Connection, label: str,
                crash_hook: Optional[Any] = None) -> None:
    if crash_hook is not None:
        crash_hook(label, conn)


class Store:
    def __init__(self, db_path: str = ":memory:",
                 crash_hook: Optional[Any] = None):
        self.db_path = db_path
        self.conn = init_db(db_path)
        self.crash_hook = crash_hook
        recover(self.conn)

    # ---------------- sources ---------------- #
    def seed_pack(self, pack: Dict[str, Any]) -> None:
        with self.conn:
            for key, val in pack["parameters"].items():
                tag = val.get("version", "v1") if isinstance(val, dict)                     else "v1"
                self.conn.execute(
                    "INSERT OR REPLACE INTO parameters(key,value_json,"
                    "version_tag,updated_event_id) VALUES(?,?,?,?)",
                    (key, _j(val), tag, None))
            for s in pack["sources"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO sources(sid,rev,type,ping,cell,"
                    "beam,time_index,value_json,unit,supersedes,note) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (s["sid"], s["rev"], s["type"], s.get("ping"),
                     s.get("cell"), s.get("beam"), s.get("time_index"),
                     _j(s["value"]), s["unit"], None, s.get("note", "")))
            self.conn.execute(
                "INSERT OR IGNORE INTO branches(name,current_cid,base_cid,"
                "created_at) VALUES('main',NULL,NULL,?)", (time.time(),))

    def add_source_revision(self, sid: str, value: Any, unit: str,
                            actor: str = "hydrologist",
                            note: str = "") -> Dict[str, Any]:
        r = self.conn.execute(
            "SELECT * FROM sources WHERE sid=? ORDER BY rev DESC LIMIT 1",
            (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        rev = int(r["rev"]) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO sources(sid,rev,type,ping,cell,beam,time_index,"
                "value_json,unit,supersedes,note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (sid, rev, r["type"], r["ping"], r["cell"], r["beam"],
                 r["time_index"], _j(value), unit, r["rev"], note))
            eid = self._log_event("source_revised",
                                  {"sid": sid, "rev": rev, "from": json.loads(
                                      r["value_json"]), "to": value,
                                   "unit": unit, "note": note}, actor)
        return {"sid": sid, "rev": rev, "event_id": eid}

    def active_sources(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.* FROM sources s JOIN (SELECT sid, MAX(rev) mrev "
            "FROM sources GROUP BY sid) m ON s.sid=m.sid AND s.rev=m.mrev"
        ).fetchall()
        return [dict(r) for r in rows]

    def active_pack(self) -> Dict[str, Any]:
        params = {r["key"]: json.loads(r["value_json"])
                  for r in self.conn.execute("SELECT * FROM parameters")}
        sources = []
        for r in self.active_sources():
            sources.append({"sid": r["sid"], "rev": r["rev"],
                            "type": r["type"], "ping": r["ping"],
                            "cell": r["cell"], "beam": r["beam"],
                            "time_index": r["time_index"],
                            "value": json.loads(r["value_json"]),
                            "unit": r["unit"], "note": r["note"] or ""})
        return {"dataset_id": "liushu-conversion-bench-v1",
                "title": "流束换算台", "parameters": params,
                "meta": {"n_pings": 12, "n_cells": 6}, "sources": sources}

    # ---------------- parameters ---------------- #
    def update_parameter(self, key: str, value: Dict[str, Any],
                         actor: str = "hydrologist", note: str = "") -> str:
        r = self.conn.execute("SELECT * FROM parameters WHERE key=?",
                              (key,)).fetchone()
        old_tag = r["version_tag"] if r else "v0"
        new_tag = value.get("version", old_tag)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO parameters(key,value_json,version_tag,"
                "updated_event_id) VALUES(?,?,?,?)",
                (key, _j(value), new_tag, None))
            eid = self._log_event(
                "parameter_revised",
                {"key": key, "old_version": old_tag, "new_version": new_tag,
                 "value": value, "note": note}, actor)
            self.conn.execute(
                "UPDATE parameters SET updated_event_id=? WHERE key=?",
                (eid, key))
        return eid

    # ---------------- commits ---------------- #
    def branch_head(self, branch: str = "main") -> Optional[str]:
        r = self.conn.execute("SELECT current_cid FROM branches WHERE name=?",
                              (branch,)).fetchone()
        return r["current_cid"] if r else None

    def commit(self, branch: str, nodes: List[dict], label: str, kind: str,
               note: str, base_commit: Optional[str], actor: str,
               event_type: str, event_payload: Dict[str, Any],
               parent_commit: Optional[str] = None) -> Dict[str, str]:
        """Atomic version creation + pointer flip with optimistic conflict."""
        head = self.branch_head(branch)
        if base_commit is not None and head is not None \
                and base_commit != head:
            return self._record_conflict(branch, base_commit, head, label,
                                         event_payload)
        parent_cid = parent_commit if parent_commit is not None else head
        manifest = {"nodes": sorted(n["uid"] for n in nodes),
                    "node_count": len(nodes)}
        cid = "c-" + fingerprint([branch, label, kind, parent_cid, manifest,
                                  time.time_ns()])[:12]
        with self.conn:
            self.conn.execute(
                "INSERT INTO commits(cid,parent_cid,branch,created_at,label,"
                "kind,note,manifest_json,complete) VALUES(?,?,?,?,?,?,?,?,0)",
                (cid, parent_cid, branch, time.time(), label, kind, note,
                 _j(manifest)))
            _crashpoint(self.conn, "after_commit_row", self.crash_hook)
            for n in nodes:
                self.conn.execute(
                    "INSERT INTO derived(uid,cid,kind,fp,value_json,deps_json,"
                    "refs_json,reused_from_cid) VALUES(?,?,?,?,?,?,?,?)",
                    (n["uid"], cid, n["kind"], n["fp"], _j(n["value"]),
                     _j(n["deps"]), _j(n["refs"]), None))
            _crashpoint(self.conn, "after_derived_rows", self.crash_hook)
            eid = self._log_event(event_type, event_payload, actor,
                                  commit_cid=cid, branch=branch)
            self.conn.execute(
                "UPDATE commits SET complete=1 WHERE cid=?", (cid,))
            _crashpoint(self.conn, "before_pointer_flip", self.crash_hook)
            self.conn.execute(
                "UPDATE branches SET current_cid=?, base_cid=COALESCE("
                "base_cid,?) WHERE name=?", (cid, parent_cid, branch))
            _crashpoint(self.conn, "after_pointer_flip", self.crash_hook)
        return {"cid": cid, "event_id": eid}

    # ---------------- events / conflicts ---------------- #
    def _log_event(self, etype: str, payload: Dict[str, Any],
                   actor: str = "system", commit_cid: Optional[str] = None,
                   branch: str = "main") -> str:
        seq = _seq(self.conn)
        eid = "e-" + str(seq).zfill(4) + "-" + fingerprint(
            [etype, payload, seq])[:6]
        self.conn.execute(
            "INSERT INTO events(eid,seq,at,commit_cid,type,payload_json,actor,"
            "branch) VALUES(?,?,?,?,?,?,?,?)",
            (eid, seq, time.time(), commit_cid, etype, _j(payload), actor,
             branch))
        return eid

    def log_decision(self, decision: Dict[str, Any], actor: str,
                     commit_cid: Optional[str] = None) -> str:
        with self.conn:
            return self._log_event("human_decision", decision, actor,
                                   commit_cid=commit_cid)

    def list_events(self, branch: Optional[str] = None) -> List[dict]:
        q = "SELECT * FROM events"
        args: tuple = ()
        if branch:
            q += " WHERE branch=?"
            args = (branch,)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args)]

    def _record_conflict(self, branch: str, base_cid: str, head_cid: str,
                         label: str, payload: Dict[str, Any]) -> Dict[str, str]:
        cid = "conf-" + fingerprint([branch, base_cid, head_cid, label,
                                     time.time_ns()])[:12]
        with self.conn:
            self.conn.execute(
                "INSERT INTO conflicts(id,at,branch,base_cid,committed_cid,"
                "attempted_label,attempted_payload,resolution) "
                "VALUES(?,?,?,?,?,?,?,NULL)",
                (cid, time.time(), branch, base_cid, head_cid, label,
                 _j(payload)))
            self._log_event("edit_conflict",
                            {"conflict_id": cid, "branch": branch,
                             "base_cid": base_cid, "head_cid": head_cid,
                             "attempted": label, "resolvable": True},
                            "system", branch=branch)
        raise EditConflict(
            f"branch {branch} moved from {base_cid} to {head_cid} "
            "before your edit landed", cid)

    def resolve_conflict(self, conflict_id: str, resolution: str,
                         nodes: Optional[List[dict]] = None, label: str = "",
                         actor: str = "reviewer") -> str:
        r = self.conn.execute("SELECT * FROM conflicts WHERE id=?",
                              (conflict_id,)).fetchone()
        if r is None:
            raise KeyError(conflict_id)
        with self.conn:
            self.conn.execute(
                "UPDATE conflicts SET resolution=?, resolved_cid=? WHERE id=?",
                (resolution, self.branch_head(r["branch"]), conflict_id))
            eid = self._log_event(
                "conflict_resolved",
                {"conflict_id": conflict_id, "resolution": resolution},
                actor, branch=r["branch"])
        return eid

    def list_conflicts(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM conflicts ORDER BY at")]

    # ---------------- reads / graph propagation ---------------- #
    def get_commit(self, cid: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM commits WHERE cid=?",
                              (cid,)).fetchone()
        return dict(r) if r else None

    def get_nodes(self, cid: str) -> Dict[str, dict]:
        rows = self.conn.execute("SELECT * FROM derived WHERE cid=?",
                                 (cid,)).fetchall()
        out = {}
        for r in rows:
            out[r["uid"]] = {"uid": r["uid"], "kind": r["kind"],
                             "fp": r["fp"], "value": json.loads(
                                 r["value_json"]),
                             "deps": json.loads(r["deps_json"]),
                             "refs": json.loads(r["refs_json"])}
        return out

    def node_history(self, uid: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT cid,fp,kind FROM derived WHERE uid=? ORDER BY cid",
            (uid,))]

    def propagate(self, parent_cid: Optional[str],
                  new_nodes: List[dict]) -> List[dict]:
        """Reuse unchanged parent nodes; only fingerprints that changed renew.

        Returns the full node set for the new commit; reused nodes are copied
        with reused_from_cid set, and a 'propagation' report explains why each
        uid was kept or recomputed.
        """
        if parent_cid is None:
            return list(new_nodes)
        parent = self.get_nodes(parent_cid)
        merged: Dict[str, dict] = {}
        reused = 0
        recomputed = 0
        for n in new_nodes:
            old = parent.get(n["uid"])
            if old is not None and old["fp"] == n["fp"]:
                merged[n["uid"]] = dict(old)
                merged[n["uid"]]["reused_from_cid"] = parent_cid
                reused += 1
            else:
                merged[n["uid"]] = n
                recomputed += 1
        self._prop_report = {"reused": reused, "recomputed": recomputed,
                             "parent_cid": parent_cid}
        return list(merged.values())

    def propagation_report(self) -> dict:
        return getattr(self, "_prop_report", {})

    # ---------------- branches ---------------- #
    def create_branch(self, name: str, from_cid: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO branches(name,current_cid,base_cid,"
                "created_at) VALUES(?,?,?,?)",
                (name, from_cid, from_cid, time.time()))
            self._log_event("branch_created", {"branch": name,
                                               "from_cid": from_cid},
                            "system", branch=name)

    def list_branches(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY name")]

    def continue_branch(self, branch: str, cid: str,
                        actor: str = "reviewer") -> None:
        """Point main at a chosen branch commit; alternatives stay readable."""
        with self.conn:
            self.conn.execute(
                "UPDATE branches SET current_cid=? WHERE name=?", (cid, branch))
            self._log_event("branch_selected",
                            {"branch": branch, "cid": cid}, actor,
                            commit_cid=cid)

    # ---------------- export / import ---------------- #
    def export_bundle(self) -> Dict[str, Any]:
        def rows(table: str) -> List[dict]:
            return [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} ORDER BY 1")]
        return {"format": "adcp-evidence-bundle",
                "schema_version": SCHEMA_VERSION,
                "meta": {r["key"]: r["value"]
                         for r in self.conn.execute("SELECT * FROM meta")},
                "sources": rows("sources"), "parameters": rows("parameters"),
                "branches": rows("branches"), "commits": rows("commits"),
                "derived": rows("derived"), "events": rows("events"),
                "conflicts": rows("conflicts")}

    def import_bundle(self, bundle: Dict[str, Any],
                      target_path: str) -> "Store":
        if bundle.get("format") != "adcp-evidence-bundle":
            raise ValueError("not an adcp evidence bundle")
        other = Store(target_path)
        other.migrate_from(bundle)
        return other

    def migrate_from(self, bundle: Dict[str, Any]) -> None:
        with self.conn:
            for r in bundle["sources"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO sources VALUES("
                    ":sid,:rev,:type,:ping,:cell,:beam,:time_index,"
                    ":value_json,:unit,:supersedes,:note)", r)
            for r in bundle["parameters"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO parameters VALUES("
                    ":key,:value_json,:version_tag,:updated_event_id)", r)
            for r in bundle["branches"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO branches VALUES("
                    ":name,:current_cid,:base_cid,:created_at)", r)
            for r in bundle["commits"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO commits VALUES("
                    ":cid,:parent_cid,:branch,:created_at,:label,:kind,:note,"
                    ":manifest_json,:complete)", r)
            for r in bundle["derived"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO derived VALUES("
                    ":uid,:cid,:kind,:fp,:value_json,:deps_json,:refs_json,"
                    ":reused_from_cid)", r)
            for r in bundle["events"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO events VALUES("
                    ":eid,:seq,:at,:commit_cid,:type,:payload_json,:actor,"
                    ":superseded_by,:branch)", r)
            for r in bundle["conflicts"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO conflicts VALUES("
                    ":id,:at,:branch,:base_cid,:committed_cid,"
                    ":attempted_label,:attempted_payload,:resolved_cid,"
                    ":resolution)", r)
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                ("schema_version", str(SCHEMA_VERSION)))

    def close(self) -> None:
        self.conn.close()
