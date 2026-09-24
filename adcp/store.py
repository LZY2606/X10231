"""SQLite evidence store.

Semantics:
- Raw material (beam velocity, correlation, amplitude, bad-beam flags,
  heading/pitch/roll, magnetic declination, mounting angle) is recorded as
  separate source rows, never merged.
- Every derivation produces a new immutable version. The `current` pointer is
  flipped only after the version is fully written, inside one transaction.
  A crash mid-write leaves either the old version or a complete new one.
- Two users editing from the same base version produce a resolvable conflict;
  the losing candidate is retained, never overwritten.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sources(
    id TEXT PRIMARY KEY, kind TEXT, ping INTEGER, cell INTEGER, beam INTEGER,
    unit TEXT, basis TEXT, payload TEXT, seq INTEGER, superseded INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS params(
    name TEXT, version TEXT, value TEXT, PRIMARY KEY(name, version));
CREATE TABLE IF NOT EXISTS param_current(name TEXT PRIMARY KEY, version TEXT);
CREATE TABLE IF NOT EXISTS versions(
    id TEXT PRIMARY KEY, parent TEXT, label TEXT, status TEXT,
    created REAL, fingerprint TEXT);
CREATE TABLE IF NOT EXISTS derived(
    version TEXT, node TEXT, payload TEXT, inputs_hash TEXT, reason TEXT,
    PRIMARY KEY(version, node));
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, kind TEXT, payload TEXT,
    base_version TEXT, result_version TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS decisions(
    id TEXT PRIMARY KEY, scope TEXT, base_version TEXT, choice TEXT,
    author TEXT, status TEXT, payload TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS current(version TEXT);
"""

# v1 (legacy) schema lacked derived.reason and the decisions table.
MIGRATIONS = {
    1: [
        "ALTER TABLE derived ADD COLUMN reason TEXT DEFAULT ''",
        """CREATE TABLE IF NOT EXISTS decisions(
            id TEXT PRIMARY KEY, scope TEXT, base_version TEXT, choice TEXT,
            author TEXT, status TEXT, payload TEXT, ts REAL)""",
    ],
}


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_or_migrate()
        self.recover()

    def close(self):
        self.conn.close()

    # -- schema management -------------------------------------------------
    def _init_or_migrate(self):
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            self.conn.executescript(SCHEMA)
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.conn.commit()
            return
        version = int(self.get_meta("schema_version", "1"))
        while version < SCHEMA_VERSION:
            for stmt in MIGRATIONS[version]:
                self.conn.execute(stmt)
            version += 1
            self.set_meta("schema_version", str(version))
        self.conn.commit()

    def recover(self):
        """After an interrupted write, incomplete versions are never current."""
        cur = self.conn.execute("SELECT version FROM current").fetchone()
        if cur is not None:
            ok = self.conn.execute(
                "SELECT status FROM versions WHERE id=?", (cur["version"],)
            ).fetchone()
            assert ok is None or ok["status"] == "complete", \
                "current pointer references an incomplete version"
        self.conn.execute("DELETE FROM derived WHERE version IN "
                          "(SELECT id FROM versions WHERE status='building')")
        self.conn.execute("DELETE FROM versions WHERE status='building'")
        self.conn.commit()

    # -- meta / current pointer -------------------------------------------
    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                          (key, value))

    def current_version(self):
        row = self.conn.execute("SELECT version FROM current").fetchone()
        return row["version"] if row else None

    def view_version(self):
        return self.get_meta("view_version") or self.current_version()

    def set_view_version(self, version):
        self.set_meta("view_version", version)
        self.conn.commit()

    # -- sources -----------------------------------------------------------
    def add_source(self, sid, kind, payload, unit, basis, ping=None, cell=None,
                   beam=None, seq=0, commit=True):
        self.conn.execute(
            "INSERT INTO sources(id,kind,ping,cell,beam,unit,basis,payload,seq)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (sid, kind, ping, cell, beam, unit, basis, json.dumps(payload), seq))
        if commit:
            self.conn.commit()

    def supersede_source(self, old_id, new_id, kind, payload, unit, basis,
                         ping=None, cell=None, beam=None, seq=0):
        """Local source change: old row kept in place, new row supersedes it."""
        with self.conn:
            self.conn.execute("UPDATE sources SET superseded=1 WHERE id=?", (old_id,))
            self.add_source(new_id, kind, payload, unit, basis, ping, cell, beam,
                            seq, commit=False)

    def sources(self, kind=None, include_superseded=False):
        q = "SELECT * FROM sources"
        cond, args = [], []
        if kind:
            cond.append("kind=?")
            args.append(kind)
        if not include_superseded:
            cond.append("superseded=0")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq, id"
        rows = self.conn.execute(q, args).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    # -- params ------------------------------------------------------------
    def set_param(self, name, version, value):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO params(name,version,value) VALUES(?,?,?)",
                (name, version, json.dumps(value)))
            self.conn.execute(
                "INSERT OR REPLACE INTO param_current(name,version) VALUES(?,?)",
                (name, version))

    def param(self, name):
        row = self.conn.execute(
            "SELECT p.version, p.value FROM param_current c JOIN params p"
            " ON p.name=c.name AND p.version=c.version WHERE c.name=?",
            (name,)).fetchone()
        if row is None:
            return None, None
        return row["version"], json.loads(row["value"])

    # -- versions / derived ------------------------------------------------
    def begin_version(self, parent, label, fingerprint):
        vid = f"v{int(self.get_meta('version_counter', '0')) + 1}"
        self.set_meta("version_counter", str(int(vid[1:])))
        self.conn.execute(
            "INSERT INTO versions(id,parent,label,status,created,fingerprint)"
            " VALUES(?,?,?,?,?,?)",
            (vid, parent, label, "building", time.time(), fingerprint))
        self.conn.commit()
        return vid

    def put_derived(self, version, node, payload, inputs_hash, reason):
        self.conn.execute(
            "INSERT OR REPLACE INTO derived(version,node,payload,inputs_hash,reason)"
            " VALUES(?,?,?,?,?)",
            (version, node, json.dumps(payload), inputs_hash, reason))

    def commit_version(self, vid, failpoint=None):
        """Atomic flip: either the old current or a complete new one."""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute("UPDATE versions SET status='complete' WHERE id=?",
                              (vid,))
            if failpoint == "before_pointer_flip":
                raise RuntimeError("simulated crash before pointer flip")
            self.conn.execute("DELETE FROM current")
            self.conn.execute("INSERT INTO current(version) VALUES(?)", (vid,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return vid

    def version_info(self, vid):
        row = self.conn.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        return dict(row) if row else None

    def list_versions(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions ORDER BY created, id").fetchall()]

    def derived(self, version, node=None):
        if node:
            row = self.conn.execute(
                "SELECT * FROM derived WHERE version=? AND node=?",
                (version, node)).fetchone()
            return (dict(row) | {"payload": json.loads(row["payload"])}) if row else None
        rows = self.conn.execute(
            "SELECT * FROM derived WHERE version=? ORDER BY node", (version,)).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    # -- events ------------------------------------------------------------
    def record_event(self, kind, payload, base_version=None, result_version=None,
                     commit=True):
        cur = self.conn.execute(
            "INSERT INTO events(kind,payload,base_version,result_version,ts)"
            " VALUES(?,?,?,?,?)",
            (kind, json.dumps(payload), base_version, result_version, time.time()))
        seq = cur.lastrowid
        eid = f"ev-{seq:04d}"
        self.conn.execute("UPDATE events SET id=? WHERE seq=?", (eid, seq))
        if commit:
            self.conn.commit()
        return eid

    def events(self):
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in
                self.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()]

    def event(self, seq):
        row = self.conn.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        return dict(row) | {"payload": json.loads(row["payload"])} if row else None

    # -- decisions / conflicts --------------------------------------------
    def save_decision(self, did, scope, choice, author, base_version, payload=None):
        current = self.current_version()
        status = "applied" if base_version == current else "conflicted"
        with self.conn:
            self.conn.execute(
                "INSERT INTO decisions(id,scope,base_version,choice,author,status,"
                "payload,ts) VALUES(?,?,?,?,?,?,?,?)",
                (did, scope, base_version, choice, author, status,
                 json.dumps(payload or {}), time.time()))
        return status

    def resolve_decision(self, did):
        with self.conn:
            self.conn.execute(
                "UPDATE decisions SET status='superseded' WHERE scope="
                "(SELECT scope FROM decisions WHERE id=?) AND id!=? AND status='applied'",
                (did, did))
            self.conn.execute("UPDATE decisions SET status='applied' WHERE id=?", (did,))

    def decisions(self, scope=None, status=None):
        q, args = "SELECT * FROM decisions", []
        cond = []
        if scope:
            cond.append("scope=?"); args.append(scope)
        if status:
            cond.append("status=?"); args.append(status)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts, id"
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in
                self.conn.execute(q, args).fetchall()]

    # -- export / import ----------------------------------------------------
    TABLES = ["meta", "sources", "params", "param_current", "versions",
              "derived", "events", "decisions", "current"]

    def export_json(self):
        out = {"format": "adcp-evidence-export", "schema_version": SCHEMA_VERSION}
        for table in self.TABLES:
            rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
            out[table] = [dict(r) for r in rows]
        return out

    def export_file(self, path):
        with open(path, "w") as fh:
            json.dump(self.export_json(), fh, indent=1, sort_keys=True)

    @classmethod
    def import_file(cls, path, db_path):
        with open(path) as fh:
            data = json.load(fh)
        if os.path.exists(db_path):
            os.remove(db_path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
        store = cls(db_path)
        with store.conn:
            for table in cls.TABLES:
                store.conn.execute(f"DELETE FROM {table}")
                for row in data.get(table, []):
                    cols = list(row.keys())
                    store.conn.execute(
                        f"INSERT INTO {table}({','.join(cols)})"
                        f" VALUES({','.join('?' * len(cols))})",
                        [row[c] for c in cols])
        store.recover()
        return store
