"""SQLite storage with versioned derivations, atomic pointer switch,
crash recovery, conflict detection and schema migration.

Semantics
---------
- Every derivation run produces an independent version row.  Derived rows
  are written while the version is ``building``; only after a complete
  success does a single transaction mark it ``complete`` and move the
  ``current`` pointer.  A process that dies mid-write leaves either the old
  version or a complete new one visible -- never a partial one.
- Two users editing from the same base version produce a resolvable
  conflict: the second save becomes a branch version and a conflict record;
  resolving keeps the unselected branch instead of overwriting it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

SCHEMA_VERSION = 2

DDL_V2 = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS batch(
  id INTEGER PRIMARY KEY, label TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS source(
  id INTEGER PRIMARY KEY, batch_id INTEGER, kind TEXT,
  ping INTEGER, beam INTEGER, cell INTEGER, t REAL,
  frame TEXT, unit TEXT, raw TEXT, value REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS param(
  id INTEGER PRIMARY KEY, name TEXT, version INTEGER,
  value TEXT, unit TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS version(
  id INTEGER PRIMARY KEY, parent_id INTEGER, seq INTEGER,
  status TEXT, label TEXT, fingerprint TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS derived(
  version_id INTEGER, node_key TEXT, seq INTEGER, status TEXT,
  payload TEXT, fingerprint TEXT, inputs TEXT, reason TEXT,
  PRIMARY KEY(version_id, node_key));
CREATE TABLE IF NOT EXISTS pointer(id INTEGER PRIMARY KEY CHECK(id=1),
  version_id INTEGER);
CREATE TABLE IF NOT EXISTS event(
  id INTEGER PRIMARY KEY, version_id INTEGER, seq INTEGER,
  kind TEXT, payload TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS conflict(
  id INTEGER PRIMARY KEY, base_version INTEGER, branch_a INTEGER,
  branch_b INTEGER, status TEXT, resolution TEXT,
  created_at REAL, resolved_at REAL);
"""

TABLES = ["meta", "batch", "source", "param", "version", "derived",
          "pointer", "event", "conflict"]


class ConflictError(Exception):
    def __init__(self, conflict_id, base, current):
        super().__init__(f"edit base v{base} != current v{current}")
        self.conflict_id = conflict_id
        self.base = base
        self.current = current


class Store:
    def __init__(self, path):
        self.path = path
        first = not os.path.exists(path)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        if first:
            self.db.executescript(DDL_V2)
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES"
                "('schema_version',?)", (str(SCHEMA_VERSION),))
            self.db.commit()
        else:
            self.migrate()
        self.recover()

    # ---------------------------------------------------------- migration
    def migrate(self):
        row = self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        v = int(row["value"]) if row else 1
        if v < 2:
            # v1 -> v2: add derived.reason and the conflict table.
            cols = [r["name"] for r in
                    self.db.execute("PRAGMA table_info(derived)")]
            if "reason" not in cols:
                self.db.execute("ALTER TABLE derived ADD COLUMN reason TEXT")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conflict(
              id INTEGER PRIMARY KEY, base_version INTEGER, branch_a INTEGER,
              branch_b INTEGER, status TEXT, resolution TEXT,
              created_at REAL, resolved_at REAL);
            """)
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES"
                "('schema_version','2')")
            self.db.commit()

    def recover(self):
        """Mark interrupted building versions as aborted on startup."""
        self.db.execute(
            "UPDATE version SET status='aborted' WHERE status='building'")
        self.db.commit()

    def close(self):
        self.db.close()

    # ------------------------------------------------------------- helpers
    def _seq(self, table):
        row = self.db.execute(f"SELECT MAX(seq) s FROM {table}").fetchone()
        return (row["s"] or 0) + 1

    def current_version(self):
        row = self.db.execute(
            "SELECT version_id FROM pointer WHERE id=1").fetchone()
        return row["version_id"] if row else None

    # ------------------------------------------------------------- sources
    def add_batch(self, label):
        cur = self.db.execute(
            "INSERT INTO batch(label, created_at) VALUES(?,?)",
            (label, time.time()))
        self.db.commit()
        return cur.lastrowid

    def add_source(self, batch_id, kind, ping=None, beam=None, cell=None,
                   t=None, frame=None, unit=None, raw=None, value=None,
                   meta=None):
        cur = self.db.execute(
            "INSERT INTO source(batch_id,kind,ping,beam,cell,t,frame,unit,"
            "raw,value,meta) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (batch_id, kind, ping, beam, cell, t, frame, unit,
             None if raw is None else str(raw), value,
             json.dumps(meta or {}, ensure_ascii=False)))
        return cur.lastrowid

    def commit(self):
        self.db.commit()

    def sources(self, kind=None):
        q = "SELECT * FROM source"
        args = ()
        if kind:
            q += " WHERE kind=?"
            args = (kind,)
        rows = self.db.execute(q + " ORDER BY id", args).fetchall()
        return [dict(r) for r in rows]

    def latest_sources(self):
        """Latest row per coordinate (append-only corrections win)."""
        best = {}
        for r in self.sources():
            key = (r["kind"], r["ping"], r["beam"], r["cell"], r["t"])
            if key not in best or r["id"] > best[key]["id"]:
                best[key] = r
        return best

    # -------------------------------------------------------------- params
    def set_param(self, name, value, unit=None, meta=None):
        row = self.db.execute(
            "SELECT MAX(version) v FROM param WHERE name=?",
            (name,)).fetchone()
        ver = (row["v"] or 0) + 1
        self.db.execute(
            "INSERT INTO param(name,version,value,unit,meta) VALUES(?,?,?,?,?)",
            (name, ver, json.dumps(value), unit,
             json.dumps(meta or {}, ensure_ascii=False)))
        self.db.commit()
        return ver

    def params(self):
        """Latest version of every parameter."""
        out = {}
        for r in self.db.execute(
                "SELECT * FROM param ORDER BY name, version"):
            out[r["name"]] = {"version": r["version"],
                              "value": json.loads(r["value"]),
                              "unit": r["unit"],
                              "meta": json.loads(r["meta"] or "{}")}
        return out

    # ------------------------------------------------------------ versions
    def begin_version(self, parent_id, label):
        cur = self.db.execute(
            "INSERT INTO version(parent_id,seq,status,label,created_at)"
            " VALUES(?,?,?,?,?)",
            (parent_id, self._seq("version"), "building", label, time.time()))
        self.db.commit()
        return cur.lastrowid

    def put_derived(self, version_id, node_key, status, payload,
                    fingerprint, inputs, reason):
        self.db.execute(
            "INSERT OR REPLACE INTO derived(version_id,node_key,seq,status,"
            "payload,fingerprint,inputs,reason) VALUES(?,?,?,?,?,?,?,?)",
            (version_id, node_key, self._seq("derived"), status,
             json.dumps(payload, ensure_ascii=False), fingerprint,
             json.dumps(inputs, ensure_ascii=False), reason))

    def commit_version(self, version_id, fingerprint, move_pointer=True):
        """Atomically mark complete and (optionally) move the pointer."""
        with self.db:
            self.db.execute(
                "UPDATE version SET status='complete', fingerprint=?"
                " WHERE id=?", (fingerprint, version_id))
            if move_pointer:
                self.db.execute(
                    "INSERT OR REPLACE INTO pointer(id,version_id)"
                    " VALUES(1,?)", (version_id,))

    def versions(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM version ORDER BY seq")]

    def derived(self, version_id, node_key=None):
        if node_key:
            rows = self.db.execute(
                "SELECT * FROM derived WHERE version_id=? AND node_key=?",
                (version_id, node_key)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM derived WHERE version_id=? ORDER BY seq",
                (version_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            d["inputs"] = json.loads(d["inputs"])
            out.append(d)
        return out

    # -------------------------------------------------------------- events
    def add_event(self, version_id, kind, payload):
        cur = self.db.execute(
            "INSERT INTO event(version_id,seq,kind,payload,created_at)"
            " VALUES(?,?,?,?,?)",
            (version_id, self._seq("event"), kind,
             json.dumps(payload, ensure_ascii=False), time.time()))
        self.db.commit()
        return cur.lastrowid

    def events(self):
        out = []
        for r in self.db.execute("SELECT * FROM event ORDER BY seq"):
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # ------------------------------------------------------------ conflict
    def save_edit(self, base_version, label, build_fn):
        """Run ``build_fn(version_id)`` as a new version from ``base_version``.

        If the current pointer has moved past ``base_version`` since the
        caller loaded it, the edit is still preserved as a complete branch
        version and a resolvable conflict is recorded.
        """
        vid = self.begin_version(base_version, label)
        fingerprint = build_fn(vid)
        current = self.current_version()
        if current is not None and base_version != current:
            self.commit_version(vid, fingerprint, move_pointer=False)
            cur = self.db.execute(
                "INSERT INTO conflict(base_version,branch_a,branch_b,status,"
                "created_at) VALUES(?,?,?,?,?)",
                (base_version, current, vid, "open", time.time()))
            self.db.commit()
            conflict_id = cur.lastrowid
            self.add_event(vid, "conflict_detected", {
                "conflict_id": conflict_id, "base_version": base_version,
                "mainline_version": current, "branch_version": vid,
                "rule": "both branches retained; resolution picks one, "
                        "the unselected branch is never overwritten"})
            return vid, conflict_id
        self.commit_version(vid, fingerprint, move_pointer=True)
        return vid, None

    def resolve_conflict(self, conflict_id, keep):
        row = self.db.execute(
            "SELECT * FROM conflict WHERE id=?", (conflict_id,)).fetchone()
        if not row or row["status"] != "open":
            raise ValueError("conflict not open")
        chosen = row["branch_a"] if keep == "a" else row["branch_b"]
        dropped = row["branch_b"] if keep == "a" else row["branch_a"]
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO pointer(id,version_id) VALUES(1,?)",
                (chosen,))
            self.db.execute(
                "UPDATE conflict SET status='resolved', resolution=?,"
                " resolved_at=? WHERE id=?",
                (json.dumps({"kept": chosen, "retained_unselected": dropped}),
                 time.time(), conflict_id))
        self.add_event(chosen, "conflict_resolved", {
            "conflict_id": conflict_id, "kept_version": chosen,
            "unselected_version_retained": dropped})
        return chosen

    def conflicts(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM conflict ORDER BY id")]

    # -------------------------------------------------------- export/import
    def export(self):
        data = {"schema_version": SCHEMA_VERSION, "tables": {}}
        for t in TABLES:
            data["tables"][t] = [dict(r) for r in
                                 self.db.execute(f"SELECT * FROM {t}")]
        return data

    def import_dump(self, data):
        """Import an export into a fresh database, preserving identity
        (primary keys), order (seq columns) and lineage (parent/inputs)."""
        with self.db:
            for t in TABLES:
                for row in data["tables"].get(t, []):
                    cols = ",".join(row.keys())
                    ph = ",".join("?" for _ in row)
                    self.db.execute(
                        f"INSERT OR REPLACE INTO {t}({cols}) VALUES({ph})",
                        tuple(row.values()))
        self.recover()
