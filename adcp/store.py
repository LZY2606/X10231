"""SQLite storage layer.

Semantics:
  * beam measurements, correlation, amplitude, bad-beam flags,
    heading/pitch/roll, magnetic declination and mounting angle are stored
    as *separate* source records, each with a stable identity;
  * every derivation produces an independent version; the current pointer is
    switched only after the whole version committed successfully -- a crash
    mid-write leaves either the old version or a complete new one;
  * two users editing from the same base version produce a resolvable
    conflict: both branches are kept, one is later chosen by an event;
  * export / delete / re-import preserves identity, order and provenance.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,            -- 'source' | 'derived' | 'decision'
    parent TEXT,
    status TEXT NOT NULL,          -- 'building' | 'complete' | 'abandoned'
    created REAL NOT NULL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    kind TEXT NOT NULL,            -- beam_velocity|correlation|amplitude|
                                   -- bad_beam_flag|heading|pitch|roll|
                                   -- declination|mounting_angle|bottom_track
    ping INTEGER,
    cell INTEGER,
    beam INTEGER,
    raw TEXT NOT NULL,             -- original representation, never rewritten
    unit TEXT NOT NULL,
    frame TEXT NOT NULL,
    diagnostic TEXT,
    seq INTEGER NOT NULL           -- stable import order
);
CREATE TABLE IF NOT EXISTS derived_values (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    stage TEXT NOT NULL,           -- cell_quality|instrument|ship|earth|profile
    ping INTEGER,
    cell INTEGER,
    component TEXT,                -- x|y|z|error|...
    candidate INTEGER NOT NULL DEFAULT 0,
    assumption TEXT,
    value REAL,
    unit TEXT NOT NULL,
    frame TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    deps TEXT NOT NULL,            -- JSON list of source/derived ids
    status TEXT NOT NULL DEFAULT 'valid',   -- valid|stale|excluded|candidate-rejected
    detail TEXT                    -- JSON: exclusion reasons, candidates, ...
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    base_version TEXT,
    result_version TEXT,
    payload TEXT NOT NULL          -- machine-readable JSON
);
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    base_version TEXT NOT NULL,
    version_a TEXT NOT NULL,
    version_b TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open|resolved
    chosen_version TEXT,
    resolved_event TEXT
);
CREATE TABLE IF NOT EXISTS current_pointer (
    name TEXT PRIMARY KEY,         -- 'source' | 'derived' | 'decision'
    version_id TEXT NOT NULL
);
"""


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.db = sqlite3.connect(path, isolation_level=None,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # ---------------------------------------------------------- migration
    def migrate(self) -> None:
        row = self.db.execute("PRAGMA user_version").fetchone()
        version = row[0]
        if version == SCHEMA_VERSION:
            return
        if version == 0:
            self.db.executescript(SCHEMA)
        elif version == 1:
            # v1 -> v2: sources gained 'frame' and 'diagnostic' columns
            cols = [r[1] for r in self.db.execute("PRAGMA table_info(sources)")]
            if "frame" not in cols:
                self.db.execute(
                    "ALTER TABLE sources ADD COLUMN frame TEXT NOT NULL DEFAULT 'beam'")
            if "diagnostic" not in cols:
                self.db.execute("ALTER TABLE sources ADD COLUMN diagnostic TEXT")
        else:
            raise RuntimeError(f"unsupported schema version {version}")
        self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.db.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),))

    # ------------------------------------------------------------ versions
    def begin_version(self, kind: str, parent: str | None,
                      message: str = "") -> str:
        vid = _uid("ver")
        self.db.execute(
            "INSERT INTO versions(id,kind,parent,status,created,message)"
            " VALUES(?,?,?,?,?,?)",
            (vid, kind, parent, "building", time.time(), message))
        return vid

    def complete_version_and_switch(self, version_id: str, pointer: str) -> None:
        """Atomically mark a version complete and move the current pointer."""
        cur = self.db.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("UPDATE versions SET status='complete' WHERE id=?",
                        (version_id,))
            cur.execute(
                "INSERT INTO current_pointer(name,version_id) VALUES(?,?)"
                " ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                (pointer, version_id))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def abandon_version(self, version_id: str) -> None:
        self.db.execute("UPDATE versions SET status='abandoned' WHERE id=?",
                        (version_id,))

    def current(self, pointer: str) -> str | None:
        row = self.db.execute(
            "SELECT version_id FROM current_pointer WHERE name=?",
            (pointer,)).fetchone()
        return row["version_id"] if row else None

    # ------------------------------------------------------------- sources
    def add_source(self, version_id: str, kind: str, raw: str, unit: str,
                   frame: str, ping=None, cell=None, beam=None,
                   diagnostic: str | None = None, seq: int | None = None,
                   source_id: str | None = None) -> str:
        sid = source_id or _uid("src")
        if seq is None:
            row = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 AS s FROM sources").fetchone()
            seq = row["s"]
        self.db.execute(
            "INSERT INTO sources(id,version_id,kind,ping,cell,beam,raw,unit,"
            "frame,diagnostic,seq) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (sid, version_id, kind, ping, cell, beam, raw, unit, frame,
             diagnostic, seq))
        return sid

    def sources(self, kind: str | None = None, version_id: str | None = None,
                ping: int | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM sources WHERE 1=1", []
        if kind:
            q += " AND kind=?"; args.append(kind)
        if version_id:
            q += " AND version_id=?"; args.append(version_id)
        if ping is not None:
            q += " AND ping=?"; args.append(ping)
        return self.db.execute(q + " ORDER BY seq", args).fetchall()

    # -------------------------------------------------------- derived values
    def add_derived(self, version_id: str, stage: str, value, unit: str,
                    frame: str, fingerprint: str, deps: list[str],
                    ping=None, cell=None, component=None, candidate: int = 0,
                    assumption: str | None = None, status: str = "valid",
                    detail: dict | None = None,
                    value_id: str | None = None) -> str:
        did = value_id or _uid("dv")
        self.db.execute(
            "INSERT INTO derived_values(id,version_id,stage,ping,cell,"
            "component,candidate,assumption,value,unit,frame,fingerprint,"
            "deps,status,detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, version_id, stage, ping, cell, component, candidate,
             assumption, value, unit, frame, fingerprint,
             json.dumps(deps), status,
             json.dumps(detail) if detail is not None else None))
        return did

    def derived(self, version_id: str | None = None, stage: str | None = None,
                ping: int | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM derived_values WHERE 1=1", []
        if version_id:
            q += " AND version_id=?"; args.append(version_id)
        if stage:
            q += " AND stage=?"; args.append(stage)
        if ping is not None:
            q += " AND ping=?"; args.append(ping)
        return self.db.execute(
            q + " ORDER BY ping, cell, stage, component, candidate", args).fetchall()

    def mark_derived_status(self, value_ids: list[str], status: str) -> None:
        self.db.executemany("UPDATE derived_values SET status=? WHERE id=?",
                            [(status, v) for v in value_ids])

    # -------------------------------------------------------------- events
    def add_event(self, actor: str, kind: str, payload: dict,
                  base_version: str | None = None,
                  result_version: str | None = None,
                  event_id: str | None = None, seq: int | None = None,
                  ts: float | None = None) -> str:
        eid = event_id or _uid("evt")
        if seq is None:
            row = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 AS s FROM events").fetchone()
            seq = row["s"]
        self.db.execute(
            "INSERT INTO events(id,seq,ts,actor,kind,base_version,"
            "result_version,payload) VALUES(?,?,?,?,?,?,?,?)",
            (eid, seq, ts if ts is not None else time.time(), actor, kind,
             base_version, result_version, json.dumps(payload, sort_keys=True)))
        return eid

    def events(self, upto_seq: int | None = None) -> list[sqlite3.Row]:
        if upto_seq is None:
            return self.db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        return self.db.execute(
            "SELECT * FROM events WHERE seq<=? ORDER BY seq", (upto_seq,)).fetchall()

    # ------------------------------------------------------------- branches
    def open_branch(self, base_version: str, version_a: str,
                    version_b: str) -> str:
        bid = _uid("br")
        self.db.execute(
            "INSERT INTO branches(id,base_version,version_a,version_b)"
            " VALUES(?,?,?,?)", (bid, base_version, version_a, version_b))
        return bid

    def resolve_branch(self, branch_id: str, chosen_version: str,
                       event_id: str) -> None:
        self.db.execute(
            "UPDATE branches SET status='resolved', chosen_version=?,"
            " resolved_event=? WHERE id=?",
            (chosen_version, event_id, branch_id))

    def branches(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.db.execute(
                "SELECT * FROM branches WHERE status=? ORDER BY rowid",
                (status,)).fetchall()
        return self.db.execute("SELECT * FROM branches ORDER BY rowid").fetchall()

    # ------------------------------------------------------- export / import
    def export(self) -> dict:
        dump = {"schema_version": SCHEMA_VERSION}
        for table in ("versions", "sources", "derived_values", "events",
                      "branches", "current_pointer", "schema_meta"):
            rows = self.db.execute(f"SELECT * FROM {table}").fetchall()
            dump[table] = [dict(r) for r in rows]
        return dump

    def import_dump(self, dump: dict) -> None:
        """Load an export into this (empty) database, preserving identity."""
        cur = self.db.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            for table in ("versions", "sources", "derived_values", "events",
                          "branches", "current_pointer", "schema_meta"):
                for row in dump.get(table, []):
                    cols = ",".join(row.keys())
                    marks = ",".join("?" for _ in row)
                    cur.execute(f"INSERT OR REPLACE INTO {table}({cols}) VALUES({marks})",
                                tuple(row.values()))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.db.close()
