"""SQLite storage: separated source nodes, immutable versions, linear commits.

Semantics enforced here
------------------------
* Beam velocity / correlation / amplitude / bad-beam flag / heading-pitch-roll
  / declination / mounting are *separate nodes* - never one blended row.
* Every derivation is an independent version; the current pointer moves only
  after a commit fully succeeds (single SQLite transaction => crash sees
  either the old commit or the complete new one).
* Two users editing from the same commit get a 409-style conflict that must
  be explicitly resolved; no last-writer-wins.
* Import/export round-trips stable identities, order and provenance edges.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    tombstones INTEGER NOT NULL DEFAULT 0,
    created_commit INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS node_versions (
    id          TEXT NOT NULL,
    version     INTEGER NOT NULL,
    commit_id   INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    tombstone   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (id, version)
);
CREATE TABLE IF NOT EXISTS commits (
    id          INTEGER PRIMARY KEY,
    parent_id   INTEGER,
    created_at  REAL NOT NULL,
    author      TEXT NOT NULL,
    message     TEXT NOT NULL,
    batch       TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    commit_id   INTEGER NOT NULL,
    src_id      TEXT NOT NULL,
    src_version INTEGER NOT NULL,
    dst_id      TEXT NOT NULL,
    dst_version INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    PRIMARY KEY (commit_id, src_id, src_version, dst_id, dst_version, kind)
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    commit_id   INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    at          REAL NOT NULL,
    actor       TEXT NOT NULL,
    etype       TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch       TEXT PRIMARY KEY,
    commit_id   INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    kind        TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    detail      TEXT NOT NULL
);
"""


class ConflictError(Exception):
    """Raised when a writer is stale relative to current HEAD."""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.executescript(SCHEMA)
            self._init_meta()

    # ---------- basic helpers ----------
    def _init_meta(self) -> None:
        cur = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('head',?)", ("0",)
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('node_seq',?)", ("0",)
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('event_seq',?)", ("0",)
            )
            self.conn.execute(
                "INSERT INTO commits(id,parent_id,created_at,author,message,batch) "
                "VALUES(0,NULL,?,'system','baseline',NULL)",
                (time.time(),),
            )

    @property
    def head(self) -> int:
        return int(self.conn.execute("SELECT value FROM meta WHERE key='head'")
                   .fetchone()[0])

    def close(self) -> None:
        self.conn.close()

    # ---------- reads ----------
    def _at_clause(self, at_commit: Optional[int]) -> Tuple[str, int]:
        at = self.head if at_commit is None else int(at_commit)
        return ("nv.commit_id <= ?", at)

    def get_node(self, node_id: str, at_commit: Optional[int] = None
                 ) -> Optional[Dict[str, Any]]:
        where, at = self._at_clause(at_commit)
        row = self.conn.execute(
            f"""SELECT id, version, kind, payload, tombstone
                FROM node_versions nv
                WHERE id=? AND {where}
                ORDER BY version DESC LIMIT 1""",
            (node_id, at),
        ).fetchone()
        if row is None:
            return None
        d = {
            "id": row["id"], "version": row["version"], "kind": row["kind"],
            "payload": json.loads(row["payload"]),
            "tombstone": bool(row["tombstone"]),
        }
        return d

    def nodes_by_kind(self, kind: str, at_commit: Optional[int] = None,
                      alive_only: bool = True) -> List[Dict[str, Any]]:
        where, at = self._at_clause(at_commit)
        sql = f"""
        SELECT nv.id, nv.version, nv.kind, nv.payload, nv.tombstone,
               n.seq AS nseq
        FROM node_versions nv
        JOIN nodes n ON n.id = nv.id
        JOIN (SELECT id, MAX(version) AS mv FROM node_versions
              WHERE {where} GROUP BY id) AS cur
          ON cur.id = nv.id AND cur.mv = nv.version
        WHERE nv.{where} AND nv.kind=?
        """
        if alive_only:
            sql += " AND nv.tombstone=0"
        sql += " ORDER BY n.seq"
        rows = self.conn.execute(sql, (at, at, kind)).fetchall()
        return [
            {"id": r["id"], "version": r["version"], "kind": r["kind"],
             "payload": json.loads(r["payload"]),
             "tombstone": bool(r["tombstone"]), "seq": r["nseq"]}
            for r in rows
        ]

    def all_alive(self, at_commit: Optional[int] = None) -> List[Dict[str, Any]]:
        where, at = self._at_clause(at_commit)
        rows = self.conn.execute(
            f"""
            SELECT nv.id, nv.version, nv.kind, nv.payload, n.seq AS nseq
            FROM node_versions nv
            JOIN nodes n ON n.id=nv.id
            JOIN (SELECT id, MAX(version) AS mv FROM node_versions
                  WHERE {where} GROUP BY id) cur
              ON cur.id=nv.id AND cur.mv=nv.version
            WHERE nv.{where} AND nv.tombstone=0
            ORDER BY n.seq""",
            (at, at),
        ).fetchall()
        return [{"id": r["id"], "version": r["version"], "kind": r["kind"],
                 "payload": json.loads(r["payload"]), "seq": r["nseq"]}
                for r in rows]

    def history(self, node_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT version, commit_id, kind, payload, tombstone
               FROM node_versions WHERE id=? ORDER BY version""",
            (node_id,),
        ).fetchall()
        return [{"version": r["version"], "commit_id": r["commit_id"],
                 "kind": r["kind"], "payload": json.loads(r["payload"]),
                 "tombstone": bool(r["tombstone"])} for r in rows]

    def edges_at(self, commit_id: Optional[int] = None
                 ) -> List[Dict[str, Any]]:
        at = self.head if commit_id is None else int(commit_id)
        rows = self.conn.execute(
            """SELECT commit_id, src_id, src_version, dst_id, dst_version, kind
               FROM edges WHERE commit_id<=? ORDER BY rowid""",
            (at,),
        ).fetchall()
        return [dict(r) for r in rows]

    def commits(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM commits ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def commit(self, commit_id: int) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM commits WHERE id=?",
                              (commit_id,)).fetchone()
        return dict(r) if r else None

    def events(self, commit_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if commit_id is None:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE commit_id<=? ORDER BY id",
                (commit_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def batches(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM batches ORDER BY rowid").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["fingerprint"] = json.loads(d["fingerprint"])
            d["detail"] = json.loads(d["detail"])
            out.append(d)
        return out
