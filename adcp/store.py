"""SQLite storage: separate raw tables, immutable versions, atomic pointer.

Guarantees exercised by the automated tests:

* raw beam velocity, correlation, amplitude, bad-beam flags, attitude,
  bottom track and each transducer-config parameter live in their own table;
* a derivation produces a new immutable version, and the current pointer is
  moved inside the same transaction as the version contents -- a process
  killed while writing leaves either the old complete version or the new
  complete version, never a half-visible one;
* two editors saving from the same base version raise :class:`Conflict`; the
  losing candidate is retained (status ``"conflict"``) and the conflict is
  explicitly resolved, nothing is silently overwritten;
* export/import preserves stable ids, ordering and provenance edges.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

SCHEMA_VERSION = 3


class Conflict(Exception):
    def __init__(self, conflict_id: str, base_version: str,
                 current_version: str, candidate_version: str):
        super().__init__(
            "base %s is stale: current=%s candidate=%s"
            % (base_version, current_version, candidate_version))
        self.conflict_id = conflict_id
        self.base_version = base_version
        self.current_version = current_version
        self.candidate_version = candidate_version


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pointer(name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS raw_beam(
    sample_id TEXT NOT NULL, ping INTEGER NOT NULL, beam INTEGER NOT NULL,
    cell INTEGER NOT NULL, velocity REAL, unit TEXT NOT NULL,
    frame TEXT NOT NULL, source TEXT NOT NULL,
    PRIMARY KEY(sample_id, ping, beam, cell));
CREATE TABLE IF NOT EXISTS raw_correlation(
    sample_id TEXT NOT NULL, ping INTEGER NOT NULL, beam INTEGER NOT NULL,
    cell INTEGER NOT NULL, value REAL, unit TEXT NOT NULL,
    PRIMARY KEY(sample_id, ping, beam, cell));
CREATE TABLE IF NOT EXISTS raw_amplitude(
    sample_id TEXT NOT NULL, ping INTEGER NOT NULL, beam INTEGER NOT NULL,
    cell INTEGER NOT NULL, value REAL, unit TEXT NOT NULL,
    PRIMARY KEY(sample_id, ping, beam, cell));
CREATE TABLE IF NOT EXISTS raw_bad_beam(
    sample_id TEXT NOT NULL, ping INTEGER NOT NULL, beam INTEGER NOT NULL,
    flag INTEGER NOT NULL, source TEXT NOT NULL,
    PRIMARY KEY(sample_id, ping, beam));
CREATE TABLE IF NOT EXISTS raw_attitude(
    sample_id TEXT NOT NULL, t REAL NOT NULL,
    heading REAL, pitch REAL, roll REAL,
    h_unit TEXT NOT NULL, a_unit TEXT NOT NULL,
    PRIMARY KEY(sample_id, t));
CREATE TABLE IF NOT EXISTS raw_bottom_track(
    sample_id TEXT NOT NULL, ping INTEGER NOT NULL,
    u REAL, v REAL, w REAL, unit TEXT NOT NULL, frame TEXT NOT NULL,
    PRIMARY KEY(sample_id, ping));
CREATE TABLE IF NOT EXISTS raw_ping_time(
    sample_id TEXT NOT NULL, ping INTEGER PRIMARY KEY, t REAL NOT NULL);
CREATE TABLE IF NOT EXISTS raw_config(
    sample_id TEXT NOT NULL, key TEXT NOT NULL, version INTEGER NOT NULL,
    value REAL, text_value TEXT, unit TEXT NOT NULL,
    superseded_by INTEGER, recorded_at REAL NOT NULL,
    PRIMARY KEY(sample_id, key, version));
CREATE TABLE IF NOT EXISTS versions(
    version_id TEXT PRIMARY KEY, parent_id TEXT, status TEXT NOT NULL,
    message TEXT NOT NULL, created REAL NOT NULL,
    base_version TEXT);
CREATE TABLE IF NOT EXISTS derived(
    version_id TEXT NOT NULL, node_id TEXT NOT NULL, kind TEXT NOT NULL,
    ping INTEGER, cell INTEGER, candidate TEXT, status TEXT NOT NULL,
    payload TEXT NOT NULL, fingerprint TEXT NOT NULL,
    PRIMARY KEY(version_id, node_id));
CREATE TABLE IF NOT EXISTS node_deps(
    version_id TEXT NOT NULL, node_id TEXT NOT NULL, dep_id TEXT NOT NULL,
    ord INTEGER NOT NULL,
    PRIMARY KEY(version_id, node_id, dep_id));
CREATE TABLE IF NOT EXISTS events(
    event_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, version_id TEXT NOT NULL,
    kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
    decision_id TEXT PRIMARY KEY, event_id TEXT NOT NULL,
    version_id TEXT NOT NULL, action TEXT NOT NULL, target TEXT,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS conflicts(
    conflict_id TEXT PRIMARY KEY, base_version TEXT NOT NULL,
    current_version TEXT NOT NULL, candidate_version TEXT NOT NULL,
    status TEXT NOT NULL, resolution TEXT, created REAL NOT NULL,
    resolved_at REAL);
CREATE TABLE IF NOT EXISTS raw_snapshot(
    version_id TEXT NOT NULL, raw_key TEXT NOT NULL,
    payload TEXT NOT NULL, digest TEXT NOT NULL, ord INTEGER NOT NULL,
    PRIMARY KEY(version_id, raw_key));
"""

_RAW_TABLES = (
    "raw_beam", "raw_correlation", "raw_amplitude", "raw_bad_beam",
    "raw_attitude", "raw_bottom_track", "raw_ping_time", "raw_config",
)


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT version FROM schema_meta").fetchone()
    if row is None:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO schema_meta(version) VALUES (?)",
                     (SCHEMA_VERSION,))
        return
    current = int(row[0])
    if current == 1:
        conn.execute("ALTER TABLE derived ADD COLUMN fingerprint TEXT DEFAULT ''")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS node_deps("
            "version_id TEXT NOT NULL, node_id TEXT NOT NULL, "
            "dep_id TEXT NOT NULL, ord INTEGER NOT NULL, "
            "PRIMARY KEY(version_id, node_id, dep_id))")
        current = 2
    if current == 2:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS raw_ping_time("
            "sample_id TEXT NOT NULL, ping INTEGER PRIMARY KEY, "
            "t REAL NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS raw_snapshot("
            "version_id TEXT NOT NULL, raw_key TEXT NOT NULL, "
            "payload TEXT NOT NULL, digest TEXT NOT NULL, ord INTEGER NOT NULL, "
            "PRIMARY KEY(version_id, raw_key))")
        current = 3
    if current != SCHEMA_VERSION:
        raise RuntimeError("unknown schema version %s" % current)
    conn.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))


class Store:
    def __init__(self, path: str):
        self.path = path
        fresh = not os.path.exists(path)
        self.conn = _connect(path)
        if fresh:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                "INSERT INTO schema_meta(version) VALUES (?)",
                (SCHEMA_VERSION,))
        else:
            _migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ raw
    def put_raw_beam(self, sample_id: str, ping: int, beam: int, cell: int,
                     velocity, unit: str, frame: str, source: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_beam VALUES (?,?,?,?,?,?,?)",
            (sample_id, ping, beam, cell, velocity, unit, frame, source))

    def put_raw_correlation(self, sample_id, ping, beam, cell, value, unit):
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_correlation VALUES (?,?,?,?,?,?)",
            (sample_id, ping, beam, cell, value, unit))

    def put_raw_amplitude(self, sample_id, ping, beam, cell, value, unit):
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_amplitude VALUES (?,?,?,?,?,?)",
            (sample_id, ping, beam, cell, value, unit))

    def put_raw_bad_beam(self, sample_id, ping, beam, flag, source) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_bad_beam VALUES (?,?,?,?,?)",
            (sample_id, ping, beam, 1 if flag else 0, source))

    def put_raw_attitude(self, sample_id, t, heading, pitch, roll,
                         h_unit="deg_mag", a_unit="deg") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_attitude VALUES (?,?,?,?,?,?,?)",
            (sample_id, t, heading, pitch, roll, h_unit, a_unit))

    def put_raw_bottom_track(self, sample_id, ping, u, v, w,
                             unit="m/s", frame="ship") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_bottom_track VALUES (?,?,?,?,?,?,?)",
            (sample_id, ping, u, v, w, unit, frame))

    def put_ping_time(self, sample_id, ping, t) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_ping_time VALUES (?,?,?)",
            (sample_id, ping, t))

    def put_config(self, sample_id: str, key: str, version: int,
                   value, unit: str, text_value=None) -> None:
        """Store one parameter revision in its own versioned record."""
        prev = self.conn.execute(
            "SELECT version FROM raw_config WHERE sample_id=? AND key=? "
            "ORDER BY version DESC LIMIT 1", (sample_id, key)).fetchone()
        self.conn.execute(
            "INSERT OR REPLACE INTO raw_config VALUES (?,?,?,?,?,?,?,?)",
            (sample_id, key, version, value, text_value, unit,
             None, time.time()))
        if prev is not None and int(prev[0]) != version:
            self.conn.execute(
                "UPDATE raw_config SET superseded_by=? "
                "WHERE sample_id=? AND key=? AND version=?",
                (version, sample_id, key, int(prev[0])))

    def config_latest(self, sample_id: str, key: str):
        row = self.conn.execute(
            "SELECT value, text_value, unit, version FROM raw_config "
            "WHERE sample_id=? AND key=? ORDER BY version DESC LIMIT 1",
            (sample_id, key)).fetchone()
        if row is None:
            return None
        return {"value": row[0], "text_value": row[1], "unit": row[2],
                "version": row[3]}

    def config_history(self, sample_id: str, key: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT value, text_value, unit, version, superseded_by "
            "FROM raw_config WHERE sample_id=? AND key=? ORDER BY version",
            (sample_id, key)).fetchall()
        return [dict(r) for r in rows]

    def fetch(self, sql: str, args=()) -> List[sqlite3.Row]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # -------------------------------------------------------------- versions
    def _counter(self, name: str) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO counters(name, value) VALUES (?,0)",
            (name,))
        row = self.conn.execute(
            "SELECT value FROM counters WHERE name=?", (name,)).fetchone()
        return int(row[0])

    def _bump(self, name: str) -> int:
        n = self._counter(name) + 1
        self.conn.execute(
            "UPDATE counters SET value=? WHERE name=?", (n, name))
        return n

    def current_version(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM pointer WHERE name='current'").fetchone()
        return row[0] if row else None

    def writer(self, base_version: Optional[str], message: str) -> "VersionWriter":
        return VersionWriter(self, base_version, message)

    def _register_conflict(self, base, current, candidate, derived_rows,
                           deps_rows, events_rows, decisions_rows) -> Conflict:
        cid = "cf%03d" % self._bump("conflict")
        self.conn.execute(
            "INSERT INTO conflicts VALUES (?,?,?,?,?,?,?,?)",
            (cid, base or "", current or "", candidate, "open", None,
             time.time(), None))
        self.conn.execute(
            "INSERT INTO versions VALUES (?,?,?,?,?,?)",
            (candidate, base, "conflict",
             "conflicting edit from %s" % (base,), time.time(), base))
        self.conn.executemany(
            "INSERT INTO derived VALUES (?,?,?,?,?,?,?,?,?)", derived_rows)
        self.conn.executemany(
            "INSERT OR IGNORE INTO node_deps VALUES (?,?,?,?)", deps_rows)
        self.conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?)", events_rows)
        self.conn.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?)", decisions_rows)
        return Conflict(cid, base or "", current or "", candidate)

    def resolve_conflict(self, conflict_id: str, chosen: str,
                         resolution_note: str) -> None:
        """Resolve a save conflict; both versions remain, one is chosen."""
        row = self.conn.execute(
            "SELECT * FROM conflicts WHERE conflict_id=?",
            (conflict_id,)).fetchone()
        if row is None:
            raise KeyError(conflict_id)
        if row["status"] != "open":
            raise RuntimeError("conflict already resolved")
        choices = {row["current_version"], row["candidate_version"]}
        if chosen not in choices:
            raise ValueError("chosen version not part of conflict")
        loser = (choices - {chosen}).pop()
        self.conn.execute(
            "UPDATE conflicts SET status='resolved', resolution=?, "
            "resolved_at=? WHERE conflict_id=?",
            (resolution_note, time.time(), conflict_id))
        self.conn.execute(
            "UPDATE versions SET status='superseded' WHERE version_id=?",
            (loser,))
        self.conn.execute(
            "INSERT OR REPLACE INTO pointer(name, value) VALUES ('current',?)",
            (chosen,))

    def conflicts(self, status: Optional[str] = None) -> List[dict]:
        sql = "SELECT * FROM conflicts"
        args = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY created"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def versions(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions ORDER BY created, version_id").fetchall()]

    def children_of(self, version_id: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions WHERE parent_id=? ORDER BY created",
            (version_id,)).fetchall()]

    # ----------------------------------------------------------- snapshots
    def snapshot_keys(self, version_id: str) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT raw_key FROM raw_snapshot WHERE version_id=? ORDER BY ord",
            (version_id,)).fetchall()]

    def snapshot_digest(self, version_id: str, raw_key: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT digest FROM raw_snapshot WHERE version_id=? AND raw_key=?",
            (version_id, raw_key)).fetchone()
        return r[0] if r else None

    def snapshot_payload(self, version_id: str, raw_key: str):
        r = self.conn.execute(
            "SELECT payload FROM raw_snapshot WHERE version_id=? AND raw_key=?",
            (version_id, raw_key)).fetchone()
        return json.loads(r[0]) if r else None

    def save_snapshot(self, version_id: str, entries: List[tuple]) -> None:
        """entries: (raw_key, payload, digest) in stable order."""
        for ord_, (key, payload, digest) in enumerate(entries):
            self.conn.execute(
                "INSERT OR REPLACE INTO raw_snapshot VALUES (?,?,?,?,?)",
                (version_id, key,
                 json.dumps(payload, ensure_ascii=False), digest, ord_))

    # ---------------------------------------------------------------- reads
    def derived(self, version_id: str, kind: Optional[str] = None) -> List[dict]:
        sql = "SELECT * FROM derived WHERE version_id=?"
        args = [version_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY ping, cell, node_id"
        rows = self.conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def derived_node(self, version_id: str, node_id: str) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT * FROM derived WHERE version_id=? AND node_id=?",
            (version_id, node_id)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def deps_of(self, version_id: str, node_id: str) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT dep_id FROM node_deps WHERE version_id=? AND node_id=? "
            "ORDER BY ord", (version_id, node_id)).fetchall()]

    def events(self, version_id: Optional[str] = None) -> List[dict]:
        sql = "SELECT * FROM events"
        args = ()
        if version_id:
            sql += " WHERE version_id=?"
            args = (version_id,)
        sql += " ORDER BY seq"
        return [dict(r, payload=json.loads(r["payload"]))
                for r in self.conn.execute(sql, args).fetchall()]

    def event(self, event_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM events WHERE event_id=?",
                              (event_id,)).fetchone()
        if r is None:
            return None
        return dict(r, payload=json.loads(r["payload"]))

    def decisions(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM decisions ORDER BY created").fetchall()]

    # --------------------------------------------------------- export/import
    def export_dict(self) -> dict:
        def dump(table):
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM %s" % table).fetchall()]
        return {
            "archive": "flowbeam-evidence",
            "schema_version": SCHEMA_VERSION,
            "created": time.time(),
            "tables": {t: dump(t) for t in (
                "schema_meta", "counters", "pointer", *_RAW_TABLES,
                "versions", "derived", "node_deps", "raw_snapshot", "events",
                "decisions", "conflicts")},
        }

    def import_dict(self, data: dict) -> None:
        """Import an archive into an empty database; ids survive verbatim."""
        if data.get("archive") != "flowbeam-evidence":
            raise ValueError("not a flowbeam-evidence archive")
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM versions").fetchone()[0]
        if existing:
            raise RuntimeError("import target must be empty")
        tables = data["tables"]
        with self.conn:
            for name, rows in tables.items():
                if not rows:
                    continue
                cols = list(rows[0].keys())
                sql = "INSERT INTO %s(%s) VALUES (%s)" % (
                    name, ",".join(cols), ",".join("?" for _ in cols))
                self.conn.executemany(
                    sql, [tuple(r[c] for c in cols) for r in rows])
        _migrate(self.conn)


class VersionWriter:
    """Stages a complete new version; commit moves the pointer atomically.

    If anything goes wrong (including the process being killed) before
    ``commit`` returns, the transaction rolls back and readers keep seeing
    the previous version.
    """

    def __init__(self, store: Store, base_version: Optional[str],
                 message: str):
        self.store = store
        self.conn = store.conn
        self.base_version = base_version
        self.message = message
        self._derived: List[tuple] = []
        self._deps: List[tuple] = []
        self._events: List[tuple] = []
        self._decisions: List[tuple] = []
        self._snapshots: List[tuple] = []
        self._event_seq = 0
        self._decision_seq = 0
        self._finished = False

    def __enter__(self) -> "VersionWriter":
        return self

    def put_derived(self, node_id, kind, ping, cell, candidate, status,
                    payload: dict, fp: str, deps: List[str]) -> None:
        self._derived.append((
            None, node_id, kind, ping, cell, candidate, status,
            json.dumps(payload, ensure_ascii=False), fp))
        for i, dep in enumerate(deps):
            self._deps.append((None, node_id, dep, i))

    def add_event(self, kind: str, payload: dict) -> str:
        self._event_seq += 1
        local = "ev%04d" % self._event_seq
        self._events.append((local, self._event_seq, None, kind,
                             json.dumps(payload, ensure_ascii=False),
                             time.time()))
        return local

    def add_decision(self, action: str, target: Optional[str]) -> str:
        self._decision_seq += 1
        local = "dec%04d" % self._decision_seq
        last_ev = self._events[-1][0] if self._events else None
        self._decisions.append((local, last_ev, None, action, target,
                                time.time()))
        return local

    def set_snapshots(self, entries: List[tuple]) -> None:
        self._snapshots = list(entries)

    def commit(self, crash_before_pointer: bool = False) -> str:
        store = self.store
        """Commit staged content and move the pointer in one transaction."""
        if self._finished:
            raise RuntimeError("writer already committed")
        conn = store.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = store.current_version()
            vid = "v%d" % store._bump("version")
            status = "complete"
            conflict = None
            if self.base_version != current:
                status = "conflict"
                cid = "cf%03d" % store._bump("conflict")
            event_map = {}
            for (local, _, _, _, _, _) in self._events:
                gseq = store._bump("event")
                event_map[local] = ("ev%04d" % gseq, gseq)
            conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)",
                         (vid, self.base_version, status, self.message,
                          time.time(), self.base_version))
            for ord_, (key, payload, digest) in enumerate(self._snapshots):
                conn.execute(
                    "INSERT OR REPLACE INTO raw_snapshot "
                    "VALUES (?,?,?,?,?)",
                    (vid, key, json.dumps(payload, ensure_ascii=False),
                     digest, ord_))
            conn.executemany(
                "INSERT INTO derived VALUES (?,?,?,?,?,?,?,?,?)",
                [(vid, n, k, pi, ce, ca, st, pl, fp)
                 for (_, n, k, pi, ce, ca, st, pl, fp) in self._derived])
            conn.executemany(
                "INSERT OR IGNORE INTO node_deps VALUES (?,?,?,?)",
                [(vid, n, d, o) for (_, n, d, o) in self._deps])
            for (local, _, _, kind, payload, created) in self._events:
                ge, gs = event_map[local]
                conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                             (ge, gs, vid, kind, payload, created))
            for (local, ev_local, _, action, target, created) in \
                    self._decisions:
                gd = "dec%04d" % store._bump("decision")
                ge = event_map[ev_local][0] if ev_local else None
                conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?)",
                             (gd, ge, vid, action, target, created))
            if status == "conflict":
                conn.execute("INSERT INTO conflicts VALUES (?,?,?,?,?,?,?,?)",
                             (cid, self.base_version or "", current or "",
                              vid, "open", None, time.time(), None))
                conn.execute("COMMIT")
                self._finished = True
                conflict = Conflict(cid, self.base_version or "",
                                    current or "", vid)
            else:
                if crash_before_pointer:
                    conn.execute("ROLLBACK")
                    self._finished = True
                    raise RuntimeError("simulated crash before pointer move")
                conn.execute(
                    "INSERT OR REPLACE INTO pointer(name, value) "
                    "VALUES ('current',?)", (vid,))
                conn.execute("COMMIT")
                self._finished = True
                return vid
        except Exception:
            if not self._finished:
                conn.execute("ROLLBACK")
            raise
        raise conflict

    def rollback(self) -> None:
        self._finished = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self._finished:
            self.rollback()
