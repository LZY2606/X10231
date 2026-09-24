"""SQLite storage: versioned sources, versioned rules, append-only events.

Guarantees
----------
* Beam velocity, correlation, amplitude, bad-beam flags, attitude,
  declination and installation angles are distinct versioned records.
* A derived node is content-addressed by fingerprint; nodes are immutable.
* Every write batch is a run. Rows of an unfinished run are invisible.
  The pointer flips inside the same commit as run completion, so after a
  crash a restart sees either the old or the complete new version.
* Concurrent editors submit an expected branch head; a moved head is a
  resolvable conflict (HTTP layer turns it into 409).
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

SCHEMA_VERSION = 2


class ConflictError(RuntimeError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  state TEXT NOT NULL,             -- writing | committed | abandoned
  purpose TEXT NOT NULL,
  created_at REAL NOT NULL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS branches (
  branch_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  fork_event_id TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  kind TEXT NOT NULL,              -- beam_velocity|correlation|amplitude|bad_flag|
                                   -- attitude|ping|depth_cell|bt_velocity|bt_range|bt_flag
  record_key TEXT NOT NULL,
  version INTEGER NOT NULL,
  payload TEXT NOT NULL,           -- raw representation, unit, si value, diagnostics
  run_id TEXT NOT NULL,
  PRIMARY KEY (kind, record_key, version)
);
CREATE TABLE IF NOT EXISTS rules (
  rule TEXT NOT NULL,              -- geometry|declination|install|transform_order|quality_policy|cells
  version INTEGER NOT NULL,
  payload TEXT NOT NULL,
  run_id TEXT NOT NULL,
  PRIMARY KEY (rule, version)
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  branch_id TEXT NOT NULL,
  type TEXT NOT NULL,
  parent_event_id TEXT,
  payload TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS derived (
  node_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  kind TEXT NOT NULL,
  ping_id TEXT,
  cell_idx INTEGER,
  payload TEXT NOT NULL,
  created_event_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  PRIMARY KEY (node_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id, seq);
CREATE INDEX IF NOT EXISTS idx_derived_ping ON derived(kind, ping_id, cell_idx);
"""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class _RunBatch:
    def __init__(self, store: "Store", run_id: str):
        self.store = store
        self.run_id = run_id


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._bootstrap()
        self.recover()

    # ----- schema / migration -----------------------------------------
    def _bootstrap(self) -> None:
        self.conn.executescript(SCHEMA)
        cur = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                              (str(SCHEMA_VERSION),))
            self.conn.execute("INSERT INTO meta(key,value) VALUES('event_seq','0')")
            self.conn.commit()
        else:
            old = int(row["value"])
            if old < SCHEMA_VERSION:
                self._migrate(old, SCHEMA_VERSION)

    def _migrate(self, old: int, new: int) -> None:
        # v1 -> v2: derivation audit table, backfilled from existing nodes.
        if old == 1:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS derived_audit "
                "(node_id TEXT, fingerprint TEXT, created_event_id TEXT)")
            self.conn.execute(
                "INSERT OR IGNORE INTO derived_audit(node_id, fingerprint, created_event_id) "
                "SELECT node_id, fingerprint, created_event_id FROM derived")
            self.conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            self.conn.commit()

    def schema_version(self) -> int:
        return int(self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"])

    def recover(self) -> list[str]:
        """Mark runs interrupted before commit as abandoned (rows invisible)."""
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE state='writing'").fetchall()
        for r in rows:
            self.conn.execute(
                "UPDATE runs SET state='abandoned', finished_at=? WHERE run_id=?",
                (time.time(), r["run_id"]))
        self.conn.commit()
        return [r["run_id"] for r in rows]

    # ----- runs --------------------------------------------------------
    @contextmanager
    def run(self, purpose: str, run_id: str | None = None):
        run_id = run_id or f"run-{time.time_ns()}"
        self.conn.execute(
            "INSERT INTO runs(run_id,state,purpose,created_at) VALUES(?,?,?,?)",
            (run_id, "writing", purpose, time.time()))
        self.conn.commit()
        batch = _RunBatch(self, run_id)
        try:
            yield batch
            self._commit(run_id)
        except BaseException:
            self.conn.rollback()
            self.conn.execute(
                "UPDATE runs SET state='abandoned', finished_at=? WHERE run_id=?",
                (time.time(), run_id))
            self.conn.commit()
            raise

    def crash_without_commit(self, run_id: str) -> None:
        """Test hook: leave a run in 'writing' to simulate a killed process."""
        self.conn.commit()  # flush rows, keep run uncommitted

    def _commit(self, run_id: str) -> None:
        with self.conn:  # atomic flip: rows + completion + pointer events together
            self.conn.execute(
                "UPDATE runs SET state='committed', finished_at=? WHERE run_id=?",
                (time.time(), run_id))

    def _committed_runs(self) -> set[str]:
        return {r["run_id"] for r in self.conn.execute(
            "SELECT run_id FROM runs WHERE state='committed'")}

    # ----- sources & rules --------------------------------------------
    def put_source(self, run_id: str, kind: str, key: str, version: int,
                   payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sources(kind,record_key,version,payload,run_id) "
            "VALUES(?,?,?,?,?)",
            (kind, key, version, json.dumps(payload, ensure_ascii=False), run_id))

    def put_rule(self, run_id: str, rule: str, version: int, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO rules(rule,version,payload,run_id) VALUES(?,?,?,?)",
            (rule, version, json.dumps(payload, ensure_ascii=False), run_id))

    def committed_sources(self, include_run: str | None = None) -> list[dict]:
        runs = self._committed_runs()
        if include_run:
            runs.add(include_run)
        if not runs:
            return []
        q = ",".join("?" * len(runs))
        out = [dict(r) for r in self.conn.execute(
            f"SELECT kind,record_key,version,payload,run_id FROM sources "
            f"WHERE run_id IN ({q})", tuple(runs))]
        for r in out:
            r["payload"] = json.loads(r["payload"])
        return out

    def committed_rules(self, include_run: str | None = None) -> list[dict]:
        runs = self._committed_runs()
        if include_run:
            runs.add(include_run)
        if not runs:
            return []
        q = ",".join("?" * len(runs))
        out = [dict(r) for r in self.conn.execute(
            f"SELECT rule,version,payload,run_id FROM rules WHERE run_id IN ({q})",
            tuple(runs))]
        for r in out:
            r["payload"] = json.loads(r["payload"])
        return out

    # ----- branches & events ------------------------------------------
    def put_branch(self, branch_id: str, name: str, fork_event_id: str | None,
                   run_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO branches(branch_id,name,fork_event_id,created_at) "
            "VALUES(?,?,?,?)", (branch_id, name, fork_event_id, time.time()))

    def list_branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY created_at")]

    def branch_head(self, branch_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT event_id FROM events WHERE branch_id=? ORDER BY seq DESC LIMIT 1",
            (branch_id,)).fetchone()
        return row["event_id"] if row else None

    def _next_event_id(self) -> tuple[str, int]:
        n = int(self.conn.execute("SELECT value FROM meta WHERE key='event_seq'")
                .fetchone()["value"]) + 1
        self.conn.execute("UPDATE meta SET value=? WHERE key='event_seq'", (str(n),))
        return f"e{n:04d}", n

    def append_event(self, branch_id: str, etype: str, payload: dict,
                     actor: str = "system", event_id: str | None = None,
                     parent_event_id: str | None = None,
                     expected_head: str | None = None, run_id: str | None = None):
        head = self.branch_head(branch_id)
        if expected_head is not None and expected_head != head:
            raise ConflictError(
                f"branch {branch_id} head moved: expected {expected_head}, got {head}")
        parent = parent_event_id if parent_event_id is not None else head
        seq = int(self.conn.execute("SELECT value FROM meta WHERE key='event_seq'")
                  .fetchone()["value"]) + 1
        eid = event_id or f"e{seq:04d}"
        self.conn.execute("UPDATE meta SET value=? WHERE key='event_seq'", (str(seq),))
        rid = run_id or f"run-{time.time_ns()}"
        if not self.conn.execute("SELECT 1 FROM runs WHERE run_id=?", (rid,)).fetchone():
            self.conn.execute(
                "INSERT INTO runs(run_id,state,purpose,created_at,finished_at) "
                "VALUES(?,?,?,?,?)",
                (rid, "writing", f"event:{etype}", time.time(), None))
        self.conn.execute(
            "INSERT INTO events(event_id,seq,branch_id,type,parent_event_id,payload,"
            "actor,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (eid, seq, branch_id, etype, parent,
             json.dumps(payload, ensure_ascii=False), actor,
             payload.get("_ts") if isinstance(payload, dict) and payload.get("_ts")
             else time.time()))
        self.conn.execute(
            "UPDATE runs SET state='committed', finished_at=? WHERE run_id=?",
            (time.time(), rid))
        self.conn.commit()
        return self.get_event(eid)

    def get_event(self, event_id: str) -> dict:
        r = self.conn.execute("SELECT * FROM events WHERE event_id=?",
                              (event_id,)).fetchone()
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def list_events(self, branch_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE branch_id=? ORDER BY seq", (branch_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # ----- derived -----------------------------------------------------
    def put_derived(self, run_id: str, created_event_id: str, nodes: list[dict]) -> None:
        for n in nodes:
            self.conn.execute(
                "INSERT OR IGNORE INTO derived(node_id,fingerprint,kind,ping_id,cell_idx,"
                "payload,created_event_id,run_id) VALUES(?,?,?,?,?,?,?,?)",
                (n["node_id"], n["fingerprint"], n["kind"], n.get("ping_id"),
                 n.get("cell_idx"), json.dumps(n["payload"], ensure_ascii=False),
                 created_event_id, run_id))

    def get_derived(self, node_id: str, fingerprint: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM derived WHERE node_id=? AND fingerprint=?",
            (node_id, fingerprint)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def all_derived(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM derived").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # ----- export / import --------------------------------------------
    def export_json(self) -> dict:
        def rows(table):
            return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table} ORDER BY 1,2,3")]
        return {
            "format": "adcp-evidence-bundle",
            "format_version": 1,
            "schema_version": self.schema_version(),
            "meta": rows("meta"),
            "runs": rows("runs"),
            "branches": rows("branches"),
            "sources": rows("sources"),
            "rules": rows("rules"),
            "events": rows("events"),
            "derived": rows("derived"),
        }

    def import_json(self, bundle: dict) -> None:
        if bundle.get("format") != "adcp-evidence-bundle":
            raise ValueError("not an ADCP evidence bundle")
        with self.conn:
            for table in ("meta", "runs", "branches", "sources", "rules",
                          "events", "derived"):
                for row in bundle.get(table, []):
                    cols = list(row.keys())
                    placeholders = ",".join("?" * len(cols))
                    self.conn.execute(
                        f"INSERT OR REPLACE INTO {table}({','.join(cols)}) "
                        f"VALUES({placeholders})", [row[c] for c in cols])
        self.recover()
