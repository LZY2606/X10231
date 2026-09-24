"""SQLite versioned evidence store.

Guarantees:
  * sources (beam velocity / correlation / amplitude / bad flag / attitude
    / declination / mounting) are stored as separate records;
  * every derivation runs inside a batch; the current pointer flips only on
    the completing commit event (single SQLite transaction), so a crash at
    write time leaves either the old version or a complete new version;
  * events are append-only and content-addressed nodes are immutable;
  * two users editing the same base version get a resolve-on-merge conflict.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL,
  head_batch INTEGER);

CREATE TABLE IF NOT EXISTS batches(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  parent_batch INTEGER,
  branch TEXT NOT NULL,
  rule_set_id TEXT,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,                -- pending | committed
  label TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(id));

CREATE TABLE IF NOT EXISTS sources(
  session_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  rev INTEGER NOT NULL,                -- bump per revision; identity stable
  kind TEXT NOT NULL,                  -- beam_velocity|correlation|amplitude|
                                       -- bad_beam_flag|heading|pitch|roll|
                                       -- declination|mounting|geometry|...
  ping_idx INTEGER, cell_idx INTEGER, beam INTEGER,
  payload TEXT NOT NULL,               -- original representation preserved
  batch_id INTEGER NOT NULL,
  PRIMARY KEY(session_id, batch_id, source_id));

CREATE TABLE IF NOT EXISTS nodes(
  id TEXT NOT NULL, session_id TEXT NOT NULL, batch_id INTEGER NOT NULL,
  kind TEXT NOT NULL, payload TEXT NOT NULL,
  PRIMARY KEY(session_id, batch_id, id));

CREATE TABLE IF NOT EXISTS node_order(
  session_id TEXT NOT NULL, batch_id INTEGER NOT NULL,
  ord INTEGER NOT NULL, node_id TEXT NOT NULL,
  scope TEXT NOT NULL,                 -- ping:k | cell:k:l | bottomtrack:k
  PRIMARY KEY(session_id, batch_id, scope, ord));

CREATE TABLE IF NOT EXISTS edges(
  session_id TEXT NOT NULL, batch_id INTEGER NOT NULL,
  upstream_node_id TEXT, upstream_source_key TEXT,
  node_id TEXT NOT NULL, rel TEXT NOT NULL,
  PRIMARY KEY(session_id, batch_id, node_id, rel,
              COALESCE(upstream_node_id,''), COALESCE(upstream_source_key,'')));

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL, batch_id INTEGER NOT NULL,
  seq INTEGER NOT NULL, branch_seq INTEGER NOT NULL, branch TEXT NOT NULL,
  kind TEXT NOT NULL, actor TEXT NOT NULL, payload TEXT NOT NULL,
  created_at REAL NOT NULL);

CREATE TABLE IF NOT EXISTS edit_locks(
  session_id TEXT PRIMARY KEY, base_batch INTEGER NOT NULL,
  actor TEXT NOT NULL, taken_at REAL NOT NULL);

CREATE INDEX IF NOT EXISTS idx_nodes_batch ON nodes(session_id, batch_id);
CREATE INDEX IF NOT EXISTS idx_events_batch ON events(session_id, batch_id);
"""


def _now() -> float:
    return time.time()


class EditConflict(Exception):
    def __init__(self, msg, base_batch, other_batch, changed_sources):
        super().__init__(msg)
        self.base_batch = base_batch
        self.other_batch = other_batch
        self.changed_sources = changed_sources


class Store:
    def __init__(self, path: str):
        self.path = path
        fresh = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if fresh:
            self._init_v2()
        self._migrate()

    # ------------------------------------------------------------------ ddl
    def _init_v2(self):
        c = self.conn
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO meta(key,value) VALUES('schema_version','2')")
        c.commit()

    def _migrate(self):
        c = self.conn
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'"
                        ).fetchone()
        version = int(row["value"]) if row else 1
        if version == 1:
            # legacy v1: one flat table of sources, no batches/events.
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in c.fetchall()}
            legacy = []
            if "legacy_sources" in tables:
                legacy = [dict(r) for r in c.execute(
                    "SELECT * FROM legacy_sources").fetchall()]
            c.executescript(_SCHEMA)
            c.execute("INSERT INTO meta(key,value) VALUES('schema_version','2')")
            if legacy:
                sid = "migrated"
                c.execute("INSERT INTO sessions(id,name,created_at,head_batch)"
                          " VALUES(?,?,?,?)", (sid, "migrated", _now(), None))
                bid = self._create_batch_row(c, sid, None, "main", None,
                                             "legacy v1 migration")
                for i, rec in enumerate(legacy):
                    payload = {k: rec[k] for k in rec.keys()
                               if k not in ("source_id", "kind")}
                    c.execute("INSERT INTO sources(session_id,source_id,rev,"
                              "kind,ping_idx,cell_idx,beam,payload,batch_id)"
                              " VALUES(?,?,?,0,?,?,?,?,?)",
                              (sid, rec.get("source_id", f"src{i}"),
                               rec.get("kind", "unknown"),
                               rec.get("ping_idx"), rec.get("cell_idx"),
                               rec.get("beam"), json.dumps(payload), bid))
                c.execute("UPDATE batches SET status='committed' WHERE id=?",
                          (bid,))
                c.execute("UPDATE sessions SET head_batch=? WHERE id=?",
                          (bid, sid))
                self._insert_event(c, sid, bid, "MIGRATION", "system",
                                   {"from_schema": 1, "to_schema": 2,
                                    "legacy_records": len(legacy)})
            c.commit()
        elif version != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported schema version {version}")

    def _create_batch_row(self, c, session_id, parent, branch, rule_set_id,
                          label):
        cur = c.execute(
            "INSERT INTO batches(session_id,parent_batch,branch,rule_set_id,"
            "created_at,status,label) VALUES(?,?,?,?,?, 'pending', ?)",
            (session_id, parent, branch, rule_set_id, _now(), label))
        return int(cur.lastrowid)

    def _next_seq(self, c, session_id):
        r = c.execute("SELECT COALESCE(MAX(seq),0)+1 AS s FROM events "
                      "WHERE session_id=?", (session_id,)).fetchone()
        return int(r["s"])

    def _next_branch_seq(self, c, session_id, branch):
        r = c.execute("SELECT COALESCE(MAX(branch_seq),0)+1 AS s FROM events "
                      "WHERE session_id=? AND branch=?",
                      (session_id, branch)).fetchone()
        return int(r["s"])

    def _insert_event(self, c, session_id, batch_id, kind, actor, payload,
                      branch="main"):
        seq = self._next_seq(c, session_id)
        bseq = self._next_branch_seq(c, session_id, branch)
        c.execute("INSERT INTO events(session_id,batch_id,seq,branch_seq,"
                  "branch,kind,actor,payload,created_at) VALUES(?,?,?,?,?,"
                  "?,?,?,?)",
                  (session_id, batch_id, seq, bseq, branch, kind, actor,
                   json.dumps(payload, ensure_ascii=False), _now()))
        return seq

    # -------------------------------------------------------------- sessions
    def create_session(self, session_id: str, name: str) -> str:
        self.conn.execute("INSERT INTO sessions(id,name,created_at,head_batch)"
                          " VALUES(?,?,?,NULL)", (session_id, name, _now()))
        self.conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sessions WHERE id=?",
                                 (session_id,)).fetchone() is not None

    def head_batch(self, session_id: str) -> Optional[int]:
        r = self.conn.execute("SELECT head_batch FROM sessions WHERE id=?",
                              (session_id,)).fetchone()
        return None if r is None or r["head_batch"] is None else int(
            r["head_batch"])

    def batch_branch(self, session_id, batch_id) -> str:
        r = self.conn.execute("SELECT branch FROM batches WHERE session_id=? "
                              "AND id=?", (session_id, batch_id)).fetchone()
        return r["branch"] if r else "main"
