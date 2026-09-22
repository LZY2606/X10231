"""Append-only versioned SQLite store for ADCP evidence.

Guarantees:
  * beam velocities, correlation, amplitude, bad-beam flags, heading/pitch/
    roll, magnetic declination and mount angles are stored as *separate*
    sources, each with its own immutable version chain and a current pointer.
  * a derivation is written as a complete set in one transaction; the current
    pointer only flips after full success. A crash inside the write leaves a
    `pending` set which is aborted on recovery, so a restarted process sees
    either the old set or the complete new one.
  * two users editing the same base version must hit a resolvable conflict
    (optimistic version check -> 409-style StoreConflict carrying both sides).
  * export/import preserves content identities, order and provenance.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .fingerprint import content_id

SCHEMA_VERSION = 2
DEFAULT_BRANCH = "main"


class StoreConflict(Exception):
    """Raised on an optimistic-concurrency mismatch; carries both sides."""

    def __init__(self, kind: str, key: str, expected: str, actual: str, incoming: dict,
                 current: dict):
        super().__init__("conflict on %s/%s: expected %s actual %s"
                         % (kind, key, expected, actual))
        self.kind, self.key = kind, key
        self.expected, self.actual = expected, actual
        self.incoming, self.current = incoming, current


def _now() -> str:
    return "%.6f" % time.time()


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
  source_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  seq INTEGER NOT NULL,
  payload TEXT NOT NULL,
  unit TEXT, datum TEXT,
  created_at TEXT NOT NULL,
  author TEXT NOT NULL DEFAULT 'reviewer',
  replaces TEXT,
  PRIMARY KEY(source_id, version_id));
CREATE TABLE IF NOT EXISTS source_current(
  source_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL, key TEXT NOT NULL,
  version_id TEXT NOT NULL, seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sets(
  set_id TEXT PRIMARY KEY,
  parent_set_id TEXT,
  base_source_versions TEXT NOT NULL,
  tag TEXT, status TEXT NOT NULL,
  created_at TEXT NOT NULL, committed_at TEXT);
CREATE TABLE IF NOT EXISTS nodes(
  set_id TEXT NOT NULL,
  node_key TEXT NOT NULL,
  kind TEXT NOT NULL, ping TEXT, bin INTEGER,
  candidate_id TEXT,
  fingerprint TEXT NOT NULL,
  deps TEXT NOT NULL,
  payload TEXT NOT NULL,
  seq INTEGER NOT NULL,
  status TEXT NOT NULL,
  PRIMARY KEY(set_id, node_key));
CREATE TABLE IF NOT EXISTS branches(name TEXT PRIMARY KEY, set_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  at TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  type TEXT NOT NULL,
  set_id TEXT,
  payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY,
  at TEXT NOT NULL, branch TEXT NOT NULL,
  set_id TEXT, conflict_key TEXT,
  chosen TEXT NOT NULL, rejected TEXT,
  payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS locks(id INTEGER PRIMARY KEY CHECK(id=1), holder TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init()

    # ------------------------------------------------------------------ basic
    def _init(self) -> None:
        with self._tx() as cur:
            cur.executescript(SCHEMA)
            cur.execute("INSERT OR IGNORE INTO meta(k,v) VALUES('schema_version','%d')"
                        % SCHEMA_VERSION)
            cur.execute("INSERT OR IGNORE INTO branches(name,set_id) VALUES(?,'')",
                        (DEFAULT_BRANCH,))
        self._migrate()
        self._recover_pending()

    @contextmanager
    def _tx(self):
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            yield cur
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------- migration / recovery
    def _migrate(self) -> None:
        row = self.conn.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()
        ver = int(row["v"]) if row else 1
        if ver < SCHEMA_VERSION:
            with self._tx() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO meta(k,v) VALUES('schema_version','%d')"
                    % SCHEMA_VERSION)
                self._append_event(cur, "schema_migrated",
                                   {"from": ver, "to": SCHEMA_VERSION})

    def _recover_pending(self) -> None:
        rows = self.conn.execute(
            "SELECT set_id FROM sets WHERE status='pending'").fetchall()
        if not rows:
            return
        with self._tx() as cur:
            for r in rows:
                cur.execute("DELETE FROM nodes WHERE set_id=?", (r["set_id"],))
                cur.execute("DELETE FROM sets WHERE set_id=?", (r["set_id"],))
            self._append_event(cur, "pending_sets_aborted_on_recovery",
                               {"sets": [r["set_id"] for r in rows]})

    # ------------------------------------------------------------- event log
    def _append_event(self, cur, etype: str, payload: dict,
                      branch: str = DEFAULT_BRANCH, set_id: Optional[str] = None) -> str:
        seq_row = cur.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS n FROM events").fetchone()
        seq = seq_row["n"]
        event_id = content_id(["event", branch, seq, etype,
                               json.dumps(payload, sort_keys=True)])
        cur.execute(
            "INSERT INTO events(event_id,seq,at,branch,type,set_id,payload)"
            " VALUES(?,?,?,?,?,?,?)",
            (event_id, seq, _now(), branch, etype, set_id,
             json.dumps(payload, ensure_ascii=False, sort_keys=True)))
        return event_id

    def append_event(self, etype: str, payload: dict,
                     branch: str = DEFAULT_BRANCH, set_id: Optional[str] = None) -> str:
        with self._tx() as cur:
            return self._append_event(cur, etype, payload, branch, set_id)

    def list_events(self, upto_seq: Optional[int] = None,
                    branch: Optional[str] = None) -> List[dict]:
        q = "SELECT * FROM events WHERE 1=1"
        args: list = []
        if upto_seq is not None:
            q += " AND seq<=?"
            args.append(upto_seq)
        if branch is not None:
            q += " AND branch=?"
            args.append(branch)
        q += " ORDER BY seq"
        return [self._event_dict(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _event_dict(r: sqlite3.Row) -> dict:
        return {"event_id": r["event_id"], "seq": r["seq"], "at": r["at"],
                "branch": r["branch"], "type": r["type"], "set_id": r["set_id"],
                "payload": json.loads(r["payload"])}

    # -------------------------------------------------------------- sources
    def put_source(self, kind: str, key: str, payload: dict,
                   expected_version: Optional[str] = None,
                   unit: Optional[str] = None, datum: Optional[str] = None,
                   author: str = "reviewer",
                   branch: str = DEFAULT_BRANCH) -> dict:
        """Insert an immutable source version; optimistic conflict detection."""
        cur_row = self.conn.execute(
            "SELECT * FROM source_current WHERE kind=? AND key=?",
            (kind, key)).fetchone()
        if cur_row is None:
            if expected_version is not None and expected_version != "":
                raise StoreConflict("source", "%s:%s" % (kind, key),
                                    expected_version, "", payload, {})
            source_id = "src:%s:%s" % (kind, key)
            seq = 1
            replaces = None
        else:
            if expected_version != cur_row["version_id"]:
                raise StoreConflict(
                    "source", "%s:%s" % (kind, key),
                    "" if expected_version is None else expected_version,
                    cur_row["version_id"], payload,
                    json.loads(cur_row["payload"]) if "payload" in cur_row.keys()
                    else self._source_payload(cur_row))
            source_id = cur_row["source_id"]
            seq = cur_row["seq"] + 1
            replaces = cur_row["version_id"]
        version_id = content_id([
            "source-version", source_id, seq,
            json.dumps(payload, sort_keys=True, ensure_ascii=False),
            unit or "", datum or ""])
        rec = {"source_id": source_id, "version_id": version_id, "kind": kind,
               "key": key, "seq": seq, "payload": payload, "unit": unit,
               "datum": datum, "replaces": replaces}
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO sources(source_id,version_id,kind,key,seq,payload,"
                "unit,datum,created_at,author,replaces) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (source_id, version_id, kind, key, seq,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 unit, datum, _now(), author, replaces))
            cur.execute(
                "INSERT INTO source_current(source_id,kind,key,version_id,seq)"
                " VALUES(?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET"
                " version_id=excluded.version_id, seq=excluded.seq",
                (source_id, kind, key, version_id, seq))
            self._append_event(cur, "source_version",
                               {"source_id": source_id, "version_id": version_id,
                                "kind": kind, "key": key, "seq": seq,
                                "replaces": replaces, "unit": unit, "datum": datum},
                               branch=branch)
        return rec

    def _source_payload(self, cur_row: sqlite3.Row) -> dict:
        r = self.conn.execute("SELECT payload FROM sources WHERE source_id=? AND version_id=?",
                              (cur_row["source_id"], cur_row["version_id"])).fetchone()
        return json.loads(r["payload"]) if r else {}

    def get_source_current(self, kind: str, key: str) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT * FROM source_current WHERE kind=? AND key=?",
            (kind, key)).fetchone()
        if not r:
            return None
        p = self.conn.execute(
            "SELECT payload,unit,datum,created_at,author,replaces FROM sources"
            " WHERE source_id=? AND version_id=?",
            (r["source_id"], r["version_id"])).fetchone()
        return {"source_id": r["source_id"], "version_id": r["version_id"],
                "kind": r["kind"], "key": r["key"], "seq": r["seq"],
                "payload": json.loads(p["payload"]), "unit": p["unit"],
                "datum": p["datum"], "replaces": p["replaces"]}

    def get_source_version(self, version_id: str) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT * FROM sources WHERE version_id=?", (version_id,)).fetchone()
        if not r:
            return None
        return {"source_id": r["source_id"], "version_id": r["version_id"],
                "kind": r["kind"], "key": r["key"], "seq": r["seq"],
                "payload": json.loads(r["payload"]), "unit": r["unit"],
                "datum": r["datum"], "replaces": r["replaces"]}

    def all_source_current(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT source_id,version_id,kind,key,seq FROM source_current"
            " ORDER BY kind,key").fetchall()
        out = []
        for r in rows:
            out.append(self.get_source_current(r["kind"], r["key"]))
        return out

    def current_versions(self) -> Dict[str, str]:
        return {r["source_id"]: r["version_id"]
                for r in self.conn.execute(
                    "SELECT source_id,version_id FROM source_current")}

    # ------------------------------------------------------------ sets/nodes
    def begin_set(self, base_source_versions: Dict[str, str],
                  tag: str = "",
                  branch: str = DEFAULT_BRANCH,
                  parent_set_id: Optional[str] = None,
                  fail_before_commit: bool = False) -> str:
        """Phase 1: insert a pending set. Crash here is safe (aborted later)."""
        set_id = content_id([
            "set", json.dumps(base_source_versions, sort_keys=True), tag,
            parent_set_id or ""])
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO sets(set_id,parent_set_id,"
                "base_source_versions,tag,status,created_at) VALUES(?,?,?,?,?,?)",
                (set_id, parent_set_id,
                 json.dumps(base_source_versions, sort_keys=True), tag,
                 "pending", _now()))
            self._append_event(cur, "set_begin", {"set_id": set_id, "tag": tag},
                               branch=branch, set_id=set_id)
        if fail_before_commit:
            # Simulates a process killed between phase 1 and phase 2.
            raise RuntimeError("simulated crash before set commit")
        return set_id

    def add_node(self, set_id: str, node_key: str, kind: str,
                 ping: Optional[str], bin_idx: Optional[int],
                 candidate_id: Optional[str], fingerprint: str,
                 deps: Dict[str, str], payload: dict, seq: int,
                 status: str = "ok") -> None:
        with self._tx() as cur:
            cur.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (set_id, node_key, kind, ping, bin_idx, candidate_id,
                         fingerprint, json.dumps(deps, sort_keys=True),
                         json.dumps(payload, ensure_ascii=False, sort_keys=True),
                         seq, status))

    def add_nodes_bulk(self, set_id: str, nodes: List[dict]) -> None:
        with self._tx() as cur:
            for n in nodes:
                cur.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (set_id, n["node_key"], n["kind"], n.get("ping"),
                             n.get("bin"), n.get("candidate_id"), n["fingerprint"],
                             json.dumps(n["deps"], sort_keys=True),
                             json.dumps(n["payload"], ensure_ascii=False,
                                        sort_keys=True), n["seq"],
                             n.get("status", "ok")))

    def commit_set(self, set_id: str, branch: str = DEFAULT_BRANCH,
                   event_payload: Optional[dict] = None) -> None:
        """Phase 2: flip set status and branch pointer atomically."""
        with self._tx() as cur:
            r = cur.execute("SELECT status FROM sets WHERE set_id=?",
                            (set_id,)).fetchone()
            if not r:
                raise KeyError("unknown set %s" % set_id)
            if r["status"] == "committed":
                return
            cur.execute("UPDATE sets SET status='committed', committed_at=? "
                        "WHERE set_id=?", (_now(), set_id))
            cur.execute(
                "INSERT INTO branches(name,set_id) VALUES(?,?) ON CONFLICT(name)"
                " DO UPDATE SET set_id=excluded.set_id", (branch, set_id))
            self._append_event(cur, "set_committed",
                               event_payload or {"set_id": set_id},
                               branch=branch, set_id=set_id)

    def current_set_id(self, branch: str = DEFAULT_BRANCH) -> Optional[str]:
        r = self.conn.execute("SELECT set_id FROM branches WHERE name=?",
                              (branch,)).fetchone()
        sid = r["set_id"] if r else ""
        return sid or None

    def set_is_committed(self, set_id: str) -> bool:
        r = self.conn.execute("SELECT status FROM sets WHERE set_id=?",
                              (set_id,)).fetchone()
        return bool(r and r["status"] == "committed")

    def list_nodes(self, set_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE set_id=? ORDER BY seq", (set_id,)).fetchall()
        return [self._node_dict(r) for r in rows]

    def get_node(self, set_id: str, node_key: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM nodes WHERE set_id=? AND node_key=?",
                              (set_id, node_key)).fetchone()
        return self._node_dict(r) if r else None

    @staticmethod
    def _node_dict(r: sqlite3.Row) -> dict:
        return {"set_id": r["set_id"], "node_key": r["node_key"],
                "kind": r["kind"], "ping": r["ping"], "bin": r["bin"],
                "candidate_id": r["candidate_id"], "fingerprint": r["fingerprint"],
                "deps": json.loads(r["deps"]),
                "payload": json.loads(r["payload"]), "seq": r["seq"],
                "status": r["status"]}

    def node_history(self, node_key: str, branch: str = DEFAULT_BRANCH) -> List[dict]:
        rows = self.conn.execute(
            "SELECT n.* FROM nodes n JOIN sets s ON s.set_id=n.set_id "
            "WHERE n.node_key=? AND s.status='committed' ORDER BY n.set_id",
            (node_key,)).fetchall()
        return [self._node_dict(r) for r in rows]

    # ------------------------------------------------------------ decisions
    def add_decision(self, chosen: str, rejected: Optional[str],
                     conflict_key: str, payload: dict,
                     branch: str = DEFAULT_BRANCH,
                     set_id: Optional[str] = None) -> str:
        decision_id = content_id(["decision", conflict_key, chosen,
                                  rejected or "",
                                  json.dumps(payload, sort_keys=True)])
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO decisions(decision_id,at,branch,set_id,"
                "conflict_key,chosen,rejected,payload) VALUES(?,?,?,?,?,?,?,?)",
                (decision_id, _now(), branch, set_id, conflict_key, chosen,
                 rejected, json.dumps(payload, ensure_ascii=False, sort_keys=True)))
            self._append_event(cur, "decision",
                               {"decision_id": decision_id, "chosen": chosen,
                                "rejected": rejected, "conflict_key": conflict_key,
                                **payload}, branch=branch, set_id=set_id)
        return decision_id

    def list_decisions(self, branch: str = DEFAULT_BRANCH) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE branch=? ORDER BY at", (branch,)).fetchall()
        return [dict(r, payload=json.loads(r["payload"])) for r in rows]

    # ------------------------------------------------------------- branches
    def create_branch(self, name: str, set_id: str) -> None:
        with self._tx() as cur:
            cur.execute("INSERT INTO branches(name,set_id) VALUES(?,?) "
                        "ON CONFLICT(name) DO NOTHING", (name, set_id))
            self._append_event(cur, "branch_created",
                               {"branch": name, "set_id": set_id}, branch=name)

    def list_branches(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT name,set_id FROM branches ORDER BY name")]

    # ------------------------------------------------------- export/import
    def export_bundle(self) -> dict:
        """Deterministic JSON bundle; IDs are content-based and stable."""
        sources = [dict(r, payload=json.loads(r["payload"]))
                   for r in self.conn.execute(
                       "SELECT * FROM sources ORDER BY source_id,seq")]
        sets = [dict(r, base_source_versions=json.loads(r["base_source_versions"]))
                for r in self.conn.execute("SELECT * FROM sets ORDER BY set_id")]
        nodes = [self._node_dict(r) for r in self.conn.execute(
            "SELECT * FROM nodes ORDER BY set_id,seq")]
        events = [self._event_dict(r) for r in self.conn.execute(
            "SELECT * FROM events ORDER BY seq")]
        branches = self.list_branches()
        decisions = [dict(r, payload=json.loads(r["payload"]))
                     for r in self.conn.execute("SELECT * FROM decisions")]
        bundle = {"format": "adcp-evidence-bundle", "schema_version": SCHEMA_VERSION,
                  "sources": sources, "sets": sets, "nodes": nodes,
                  "events": events, "branches": branches, "decisions": decisions}
        from .fingerprint import fingerprint
        bundle["bundle_id"] = fingerprint([
            json.dumps(bundle, sort_keys=True, ensure_ascii=False)])
        return bundle

    def import_bundle(self, bundle: dict) -> str:
        """Import a bundle into a fresh/current DB; keeps all IDs and order."""
        with self._tx() as cur:
            for s in bundle["sources"]:
                cur.execute(
                    "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (s["source_id"], s["version_id"], s["kind"], s["key"],
                     s["seq"],
                     json.dumps(s["payload"], ensure_ascii=False, sort_keys=True),
                     s["unit"], s["datum"], s["created_at"],
                     s.get("author", "reviewer"), s.get("replaces")))
                cur.execute(
                    "INSERT INTO source_current(source_id,kind,key,version_id,seq)"
                    " VALUES(?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET"
                    " version_id=excluded.version_id, seq=excluded.seq",
                    (s["source_id"], s["kind"], s["key"], s["version_id"], s["seq"]))
            for st in bundle["sets"]:
                cur.execute("INSERT OR REPLACE INTO sets VALUES(?,?,?,?,?,?)",
                            (st["set_id"], st["parent_set_id"],
                             json.dumps(st["base_source_versions"], sort_keys=True),
                             st["tag"], st["status"], st.get("committed_at")))
            for n in bundle["nodes"]:
                cur.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (n["set_id"], n["node_key"], n["kind"], n["ping"],
                             n["bin"], n["candidate_id"], n["fingerprint"],
                             json.dumps(n["deps"], sort_keys=True),
                             json.dumps(n["payload"], ensure_ascii=False,
                                        sort_keys=True), n["seq"], n["status"]))
            for b in bundle.get("branches", []):
                cur.execute("INSERT OR REPLACE INTO branches VALUES(?,?)",
                            (b["name"], b["set_id"]))
            for e in bundle["events"]:
                cur.execute(
                    "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",
                    (e["event_id"], e["seq"], e["at"], e["branch"], e["type"],
                     e["set_id"],
                     json.dumps(e["payload"], ensure_ascii=False, sort_keys=True)))
            for d in bundle.get("decisions", []):
                cur.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?,?,?,?,?,?)",
                            (d["decision_id"], d["at"], d["branch"], d["set_id"],
                             d["conflict_key"], d["chosen"], d["rejected"],
                             json.dumps(d["payload"], ensure_ascii=False,
                                        sort_keys=True)))
            self._append_event(cur, "bundle_imported",
                               {"bundle_id": bundle.get("bundle_id", "?")})
        return bundle.get("bundle_id", "?")
