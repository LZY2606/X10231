"""Versioned SQLite storage for ADCP evidence.

Storage guarantees:

* Beam velocity, correlation, amplitude, bad-beam flags, heading, pitch,
  roll, magnetic declination and mounting angle are **separate records**.
* Raw records are append-only and version-ranged (``valid_from`` /
  ``valid_to``); they are never overwritten.
* Every derivation is an immutable version.  A version row starts as
  ``building``; the ``current_version`` pointer moves to it only after all
  nodes and events are written, inside one transaction.  A crash during
  writing therefore exposes either the old complete version or the new
  complete version - never a mixture.
* ``recover()`` purges defensively any ``building`` version left on disk.
* Two sessions opened against the same base version that both save get a
  :class:`ConflictError` carrying a resolvable node-level diff.
* Export/import preserves stable ids, order (``seq``) and provenance.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

DDL = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions(
  id INTEGER PRIMARY KEY,
  parent_id INTEGER,
  label TEXT NOT NULL,
  params TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_records(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  ping INTEGER,
  beam INTEGER,
  cell INTEGER,
  t REAL,
  value_raw TEXT,
  unit TEXT,
  frame TEXT,
  numeric REAL,
  status TEXT NOT NULL,
  diagnostic TEXT,
  source TEXT,
  valid_from INTEGER NOT NULL,
  valid_to INTEGER,
  seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY,
  version_id INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  payload TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  inputs TEXT NOT NULL,
  seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY,
  version_id INTEGER,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL,
  seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY,
  user TEXT NOT NULL,
  base_version INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
"""

# Schema as shipped in v1 of the tool; used by the migration test.
LEGACY_DDL_V1 = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE versions(
  id INTEGER PRIMARY KEY, parent_id INTEGER, label TEXT NOT NULL,
  params TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE raw_records(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL,
  ping INTEGER, beam INTEGER, cell INTEGER, t REAL,
  value_raw TEXT, unit TEXT, frame TEXT,
  numeric REAL, status TEXT NOT NULL, diagnostic TEXT,
  source TEXT, valid_from INTEGER NOT NULL, seq INTEGER NOT NULL);
CREATE TABLE nodes(
  id TEXT PRIMARY KEY, version_id INTEGER NOT NULL, node_key TEXT NOT NULL,
  fingerprint TEXT NOT NULL, status TEXT NOT NULL,
  payload TEXT NOT NULL, diagnostics TEXT NOT NULL, inputs TEXT NOT NULL, seq INTEGER NOT NULL);
CREATE TABLE events(
  id TEXT PRIMARY KEY, version_id INTEGER, kind TEXT NOT NULL,
  payload TEXT NOT NULL, created_at TEXT NOT NULL, seq INTEGER NOT NULL);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class ConflictError(Exception):
    """Raised when a session saves over a newer current version."""

    def __init__(self, session: Dict, current: int, diff: Dict):
        self.session = session
        self.current = current
        self.diff = diff
        super().__init__(
            f"session {session['id']} based on v{session['base_version']} "
            f"conflicts with current v{current}; nodes changed: "
            f"{sorted(diff.keys())}")


class Store:
    def __init__(self, path: str):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.recover()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # schema / recovery
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        tables = {
            r["name"]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "meta" not in tables:
            self.conn.executescript(DDL)
            self.conn.executemany(
                "INSERT INTO meta(key,value) VALUES(?,?)",
                [("schema_version", str(SCHEMA_VERSION)), ("seq", "0")])
            self.conn.commit()
            return
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row["value"]) if row else 1
        if version < 2:
            # v1 -> v2: bitemporal validity end + edit sessions.
            columns = [r["name"] for r in
                       self.conn.execute("PRAGMA table_info(raw_records)")]
            if "valid_to" not in columns:
                self.conn.execute(
                    "ALTER TABLE raw_records ADD COLUMN valid_to INTEGER")
            self.conn.executescript(
                "CREATE TABLE IF NOT EXISTS sessions("
                "id TEXT PRIMARY KEY, user TEXT NOT NULL, "
                "base_version INTEGER NOT NULL, created_at TEXT NOT NULL)")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) "
                "VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            self.conn.commit()

    def recover(self) -> int:
        """Remove half-written versions (and their orphan rows)."""
        purged = 0
        rows = list(self.conn.execute(
            "SELECT id FROM versions WHERE status='building'"))
        for row in rows:
            vid = row["id"]
            self.conn.execute("DELETE FROM nodes WHERE version_id=?", (vid,))
            self.conn.execute("DELETE FROM events WHERE version_id=?", (vid,))
            self.conn.execute("DELETE FROM versions WHERE id=?", (vid,))
            purged += 1
        if purged:
            self.conn.commit()
        return purged

    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------
    # meta / versions
    # ------------------------------------------------------------------
    def _next_seq(self) -> int:
        self.conn.execute(
            "UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='seq'")
        return int(self.conn.execute(
            "SELECT value FROM meta WHERE key='seq'").fetchone()["value"])

    def current_version(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='current_version_id'").fetchone()
        return int(row["value"]) if row else None

    def insert_version(self, parent_id: Optional[int], label: str,
                       params: Dict, status: str = "building") -> int:
        cur = self.conn.execute(
            "INSERT INTO versions(parent_id,label,params,status,created_at) "
            "VALUES(?,?,?,?,?)",
            (parent_id, label, json.dumps(params, sort_keys=True),
             status, _now()))
        return int(cur.lastrowid)

    def complete_version(self, vid: int) -> None:
        self.conn.execute(
            "UPDATE versions SET status='complete' WHERE id=?", (vid,))
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('current_version_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(vid),))

    def get_version(self, vid: int) -> Dict:
        row = self.conn.execute(
            "SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        if row is None:
            raise KeyError(f"version {vid}")
        out = dict(row)
        out["params"] = json.loads(out["params"])
        return out

    def versions(self) -> List[Dict]:
        return [self.get_version(r["id"]) for r in self.conn.execute(
            "SELECT id FROM versions ORDER BY id")]

    # ------------------------------------------------------------------
    # raw records (append-only, version-ranged)
    # ------------------------------------------------------------------
    def insert_raw(self, rec: Dict) -> None:
        seq = self._next_seq()
        self.conn.execute(
            "INSERT INTO raw_records(id,kind,ping,beam,cell,t,value_raw,"
            "unit,frame,numeric,status,diagnostic,source,valid_from,valid_to,seq) "
            "VALUES(:id,:kind,:ping,:beam,:cell,:t,:value_raw,:unit,:frame,"
            ":numeric,:status,:diagnostic,:source,:valid_from,:valid_to,:seq)",
            {"ping": None, "beam": None, "cell": None, "t": None,
             "frame": None, "source": None, "diagnostic": None,
             "valid_to": None, **rec, "seq": seq})

    def supersede_raw(self, old_id: str, new_rec: Dict, vid: int) -> None:
        """Mark old record superseded at v``vid`` and insert the replacement."""
        self.conn.execute(
            "UPDATE raw_records SET valid_to=? WHERE id=?", (vid, old_id))
        new_rec = {**new_rec, "id": new_rec.get("id") or f"{old_id}#r{vid}"}
        self.insert_raw({**new_rec, "valid_from": vid})

    def effective_raw(self, vid: int, kind: Optional[str] = None,
                      ping: Optional[int] = None) -> List[Dict]:
        sql = ("SELECT * FROM raw_records "
               "WHERE valid_from<=? AND (valid_to IS NULL OR valid_to>?)")
        args: List[Any] = [vid, vid]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if ping is not None:
            sql += " AND ping=?"
            args.append(ping)
        sql += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def get_raw(self, raw_id: str) -> Dict:
        return dict(self.conn.execute(
            "SELECT * FROM raw_records WHERE id=?", (raw_id,)).fetchone())

    def raw_kinds(self) -> List[str]:
        return [r["kind"] for r in self.conn.execute(
            "SELECT DISTINCT kind FROM raw_records ORDER BY kind")]

    # ------------------------------------------------------------------
    # derived nodes
    # ------------------------------------------------------------------
    def insert_node(self, node: Dict) -> None:
        seq = self._next_seq()
        self.conn.execute(
            "INSERT INTO nodes(id,version_id,node_key,fingerprint,status,"
            "payload,diagnostics,inputs,seq) VALUES(?,?,?,?,?,?,?,?,?)",
            (node["id"], node["version_id"], node["node_key"],
             node["fingerprint"], node["status"],
             json.dumps(node["payload"], sort_keys=True, ensure_ascii=False),
             json.dumps(node.get("diagnostics", []), sort_keys=True,
                        ensure_ascii=False),
             json.dumps(node["inputs"], sort_keys=True, ensure_ascii=False),
             seq))

    def nodes(self, vid: int) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE version_id=? ORDER BY seq", (vid,))
        return [self._parse_node(r) for r in rows]

    def node(self, vid: int, node_key: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE version_id=? AND node_key=?",
            (vid, node_key)).fetchone()
        return self._parse_node(row) if row else None

    @staticmethod
    def _parse_node(row) -> Dict:
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        out["diagnostics"] = json.loads(out["diagnostics"])
        out["inputs"] = json.loads(out["inputs"])
        return out

    # ------------------------------------------------------------------
    # machine-readable events
    # ------------------------------------------------------------------
    def insert_event(self, kind: str, version_id: Optional[int],
                     payload: Dict) -> Dict:
        seq = self._next_seq()
        event = {"id": f"evt-{seq:04d}", "version_id": version_id,
                 "kind": kind, "payload": payload,
                 "created_at": _now(), "seq": seq}
        self.conn.execute(
            "INSERT INTO events(id,version_id,kind,payload,created_at,seq) "
            "VALUES(?,?,?,?,?,?)",
            (event["id"], version_id, kind,
             json.dumps(payload, sort_keys=True, ensure_ascii=False),
             event["created_at"], seq))
        return event

    def events(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY seq")
        return [self._parse_event(r) for r in rows]

    def get_event(self, event_id: str) -> Dict:
        row = self.conn.execute(
            "SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._parse_event(row)

    @staticmethod
    def _parse_event(row) -> Dict:
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        return out

    def event_applied(self, step_id: str) -> bool:
        for event in self.events():
            if event["payload"].get("step_id") == step_id:
                return True
        return False

    # ------------------------------------------------------------------
    # edit sessions (optimistic concurrency)
    # ------------------------------------------------------------------
    def create_session(self, user: str, base_version: int) -> Dict:
        seq = self._next_seq()
        session = {"id": f"sess-{seq:04d}", "user": user,
                   "base_version": base_version, "created_at": _now()}
        self.conn.execute(
            "INSERT INTO sessions(id,user,base_version,created_at) "
            "VALUES(?,?,?,?)",
            (session["id"], user, base_version, session["created_at"]))
        self.conn.commit()
        return session

    def get_session(self, session_id: str) -> Dict:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return dict(row)

    # ------------------------------------------------------------------
    # export / import: identity, order and provenance preserved
    # ------------------------------------------------------------------
    def export_json(self) -> Dict:
        def dump(table: str) -> List[Dict]:
            return [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} ORDER BY seq,id")]
        return {
            "format": "adcp-evidence-export",
            "schema_version": SCHEMA_VERSION,
            "meta": [dict(r) for r in self.conn.execute(
                "SELECT * FROM meta ORDER BY key")],
            "versions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM versions ORDER BY id")],
            "raw_records": dump("raw_records"),
            "nodes": dump("nodes"),
            "events": dump("events"),
            "sessions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM sessions ORDER BY id")],
        }

    def import_json(self, data: Dict) -> None:
        with self.transaction():
            self.conn.executescript(
                "DELETE FROM nodes; DELETE FROM events; "
                "DELETE FROM raw_records; DELETE FROM versions; "
                "DELETE FROM sessions; DELETE FROM meta;")
            self.conn.executemany(
                "INSERT INTO meta(key,value) VALUES(:key,:value)",
                data["meta"])
            self.conn.executemany(
                "INSERT INTO versions(id,parent_id,label,params,status,created_at) "
                "VALUES(:id,:parent_id,:label,:params,:status,:created_at)",
                data["versions"])
            self.conn.executemany(
                "INSERT INTO raw_records(id,kind,ping,beam,cell,t,value_raw,"
                "unit,frame,numeric,status,diagnostic,source,valid_from,valid_to,seq) "
                "VALUES(:id,:kind,:ping,:beam,:cell,:t,:value_raw,:unit,:frame,"
                ":numeric,:status,:diagnostic,:source,:valid_from,:valid_to,:seq)",
                data["raw_records"])
            self.conn.executemany(
                "INSERT INTO nodes(id,version_id,node_key,fingerprint,status,"
                "payload,diagnostics,inputs,seq) "
                "VALUES(:id,:version_id,:node_key,:fingerprint,:status,"
                ":payload,:diagnostics,:inputs,:seq)", data["nodes"])
            self.conn.executemany(
                "INSERT INTO events(id,version_id,kind,payload,created_at,seq) "
                "VALUES(:id,:version_id,:kind,:payload,:created_at,:seq)",
                data["events"])
            if data.get("sessions"):
                self.conn.executemany(
                    "INSERT INTO sessions(id,user,base_version,created_at) "
                    "VALUES(:id,:user,:base_version,:created_at)",
                    data["sessions"])
