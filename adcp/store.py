"""SQLite storage: separated raw records, content-addressed derivations,
atomic version pointers, an append-only event log and optimistic conflicts.

Crash semantics: a derivation is staged inside one transaction; the current
pointer is updated last.  A crash therefore exposes either the old version
or the complete new one, never a partial graph.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from .fingerprint import canonical_json, fingerprint

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_records (
  record_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  uid TEXT NOT NULL,
  kind TEXT NOT NULL,
  ping_index INTEGER,
  bin_index INTEGER,
  beam_index INTEGER,
  t REAL,
  raw_value TEXT NOT NULL,
  unit TEXT,
  datum TEXT,
  source TEXT NOT NULL,
  note TEXT,
  superseded_by TEXT,
  updated_at REAL NOT NULL,
  PRIMARY KEY (dataset_id, uid)
);
CREATE TABLE IF NOT EXISTS versions (
  version_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  base_version_id TEXT,
  created_at REAL NOT NULL,
  label TEXT
);
CREATE TABLE IF NOT EXISTS current_version (
  dataset_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  version_id TEXT NOT NULL,
  PRIMARY KEY (dataset_id, kind)
);
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  content TEXT NOT NULL,
  input_fingerprints TEXT NOT NULL,
  PRIMARY KEY (node_id, version_id)
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  at REAL NOT NULL,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  base_version_id TEXT,
  payload TEXT NOT NULL,
  UNIQUE (dataset_id, seq)
);
CREATE TABLE IF NOT EXISTS branches (
  dataset_id TEXT NOT NULL,
  branch_name TEXT NOT NULL,
  version_id TEXT NOT NULL,
  event_seq INTEGER NOT NULL,
  PRIMARY KEY (dataset_id, branch_name)
);
"""


class ConflictError(RuntimeError):
    """Raised when a save is based on a stale current version."""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    current = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if current is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    else:
        migrate(conn, int(current.value))


def migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Bring older databases up to :data:`SCHEMA_VERSION` in one transaction."""
    if from_version == SCHEMA_VERSION:
        return
    with conn:
        if from_version < 2:
            # v1 -> v2: branches table for named decision branches.
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='branches'"
                )
            }
            if "branches" not in existing:
                conn.execute(
                    """CREATE TABLE branches (
                      dataset_id TEXT NOT NULL,
                      branch_name TEXT NOT NULL,
                      version_id TEXT NOT NULL,
                      event_seq INTEGER NOT NULL,
                      PRIMARY KEY (dataset_id, branch_name)
                    )"""
                )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ----- datasets -----------------------------------------------------
    def create_dataset(self, dataset_id: str, name: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO datasets(dataset_id, name, created_at) VALUES (?,?,?)",
                (dataset_id, name, time.time()),
            )

    def list_datasets(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM datasets")]

    # ----- raw records --------------------------------------------------
    def upsert_record(self, dataset_id: str, record: dict) -> str:
        record_id = fingerprint("raw", record)
        with self.conn:
            self.conn.execute(
                """INSERT INTO raw_records(
                     record_id, dataset_id, uid, kind, ping_index, bin_index,
                     beam_index, t, raw_value, unit, datum, source, note,
                     superseded_by, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(dataset_id, uid) DO UPDATE SET
                     record_id=excluded.record_id,
                     kind=excluded.kind,
                     ping_index=excluded.ping_index,
                     bin_index=excluded.bin_index,
                     beam_index=excluded.beam_index,
                     t=excluded.t,
                     raw_value=excluded.raw_value,
                     unit=excluded.unit,
                     datum=excluded.datum,
                     source=excluded.source,
                     note=excluded.note,
                     updated_at=excluded.updated_at""",
                (
                    record_id, dataset_id, record["uid"], record["kind"],
                    record.get("ping_index"), record.get("bin_index"),
                    record.get("beam_index"), record.get("t"),
                    json.dumps(record["raw_value"], ensure_ascii=False),
                    record.get("unit"), record.get("datum"),
                    record["source"], record.get("note"), None, time.time(),
                ),
            )
        return record_id

    def list_records(self, dataset_id: str, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM raw_records WHERE dataset_id=?"
        params: list[Any] = [dataset_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        rows = self.conn.execute(sql + " ORDER BY uid", params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["raw_value"] = json.loads(item["raw_value"])
            out.append(item)
        return out

    def raw_fingerprint(self, dataset_id: str) -> str:
        rows = self.list_records(dataset_id)
        return fingerprint("rawset", [[r["record_id"], r["uid"], r["kind"]] for r in rows])

    # ----- derivations --------------------------------------------------
    def commit_derivation(
        self,
        dataset_id: str,
        kind: str,
        nodes: list[dict],
        base_version_id: str | None,
        label: str | None = None,
    ) -> str:
        """Stage every node, the version row, then flip the current pointer.

        One transaction => crash before COMMIT leaves the previous pointer
        and graph untouched.
        """
        node_payload = [
            {
                "scope": n["scope"],
                "scope_key": n["scope_key"],
                "content": n["content"],
                "input_fingerprints": n["input_fingerprints"],
            }
            for n in nodes
        ]
        version_id = fingerprint(
            "ver",
            {
                "dataset_id": dataset_id,
                "kind": kind,
                "base_version_id": base_version_id,
                "nodes": node_payload,
            },
        )
        with self.conn:
            existing = self.conn.execute(
                "SELECT 1 FROM versions WHERE version_id=?", (version_id,)
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO versions(version_id, dataset_id, kind, base_version_id, created_at, label)"
                    " VALUES (?,?,?,?,?,?)",
                    (version_id, dataset_id, kind, base_version_id, time.time(), label),
                )
                for n in nodes:
                    self.conn.execute(
                        "INSERT INTO nodes(node_id, version_id, dataset_id, scope, scope_key, content, input_fingerprints)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (
                            fingerprint("node", {**n["content"], "_scope": n["scope"]}),
                            version_id, dataset_id, n["scope"], n["scope_key"],
                            canonical_json(n["content"]),
                            canonical_json(n["input_fingerprints"]),
                        ),
                    )
            self.conn.execute(
                """INSERT INTO current_version(dataset_id, kind, version_id)
                   VALUES (?,?,?)
                   ON CONFLICT(dataset_id, kind)
                   DO UPDATE SET version_id=excluded.version_id""",
                (dataset_id, kind, version_id),
            )
        return version_id

    def current_version(self, dataset_id: str, kind: str = "derivation") -> str | None:
        row = self.conn.execute(
            "SELECT version_id FROM current_version WHERE dataset_id=? AND kind=?",
            (dataset_id, kind),
        ).fetchone()
        return row["version_id"] if row else None

    def version_nodes(self, version_id: str, scope: str | None = None) -> list[dict]:
        sql = "SELECT * FROM nodes WHERE version_id=?"
        params: list[Any] = [version_id]
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        out = []
        for row in self.conn.execute(sql + " ORDER BY scope_key", params):
            item = dict(row)
            item["content"] = json.loads(item["content"])
            item["input_fingerprints"] = json.loads(item["input_fingerprints"])
            out.append(item)
        return out

    def all_versions(self, dataset_id: str) -> list[dict]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM versions WHERE dataset_id=? ORDER BY created_at",
                (dataset_id,),
            )
        ]

    # ----- events -------------------------------------------------------
    def append_event(
        self,
        dataset_id: str,
        event_type: str,
        actor: str,
        payload: dict,
        base_version_id: str | None = None,
    ) -> dict:
        with self.conn:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM events WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            seq = row["m"] + 1
            event_id = fingerprint(
                "evt",
                {"dataset_id": dataset_id, "seq": seq, "event_type": event_type,
                 "actor": actor, "payload": payload, "base_version_id": base_version_id},
            )
            self.conn.execute(
                "INSERT INTO events(event_id, dataset_id, seq, at, event_type, actor,"
                " base_version_id, payload) VALUES (?,?,?,?,?,?,?,?)",
                (event_id, dataset_id, seq, time.time(), event_type, actor,
                 base_version_id, canonical_json(payload)),
            )
        return {"event_id": event_id, "seq": seq, "event_type": event_type,
                "actor": actor, "payload": payload, "base_version_id": base_version_id}

    def list_events(self, dataset_id: str) -> list[dict]:
        out = []
        for row in self.conn.execute(
            "SELECT * FROM events WHERE dataset_id=? ORDER BY seq", (dataset_id,)
        ):
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            out.append(item)
        return out

    def require_current(
        self, dataset_id: str, expected_version_id: str, kind: str = "derivation"
    ) -> None:
        """Optimistic-concurrency check for two editors on one base version."""
        current = self.current_version(dataset_id, kind)
        if current is not None and current != expected_version_id:
            raise ConflictError(
                f"base {expected_version_id} is stale; current is {current}"
            )

    # ----- branches -----------------------------------------------------
    def save_branch(self, dataset_id: str, name: str, version_id: str, event_seq: int) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO branches(dataset_id, branch_name, version_id, event_seq)
                   VALUES (?,?,?,?)
                   ON CONFLICT(dataset_id, branch_name) DO UPDATE SET
                     version_id=excluded.version_id, event_seq=excluded.event_seq""",
                (dataset_id, name, version_id, event_seq),
            )

    def list_branches(self, dataset_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches WHERE dataset_id=? ORDER BY branch_name", (dataset_id,))]

    # ----- export / import ---------------------------------------------
    def export_bundle(self, dataset_id: str) -> dict:
        """Self-describing bundle.  All ids are content hashes, so deleting
        the local database and importing preserves identity and ordering."""
        records = []
        for row in self.conn.execute(
            "SELECT * FROM raw_records WHERE dataset_id=? ORDER BY uid", (dataset_id,)
        ):
            item = dict(row)
            item["raw_value"] = json.loads(item["raw_value"])
            records.append(item)
        versions = [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions WHERE dataset_id=? ORDER BY created_at", (dataset_id,))]
        nodes = [dict(r) for r in self.conn.execute(
            "SELECT * FROM nodes WHERE dataset_id=? ORDER BY version_id, scope_key",
            (dataset_id,))]
        events = []
        for row in self.conn.execute(
            "SELECT * FROM events WHERE dataset_id=? ORDER BY seq", (dataset_id,)
        ):
            events.append(dict(row))
        currents = [dict(r) for r in self.conn.execute(
            "SELECT * FROM current_version WHERE dataset_id=?", (dataset_id,))]
        branches = self.list_branches(dataset_id)
        dataset = self.conn.execute(
            "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
        ).fetchone()
        return {
            "bundle": "adcp-evidence-v1",
            "schema_version": SCHEMA_VERSION,
            "dataset": dict(dataset) if dataset else None,
            "raw_records": records,
            "versions": versions,
            "nodes": nodes,
            "events": events,
            "current_version": currents,
            "branches": branches,
        }

    def import_bundle(self, bundle: dict) -> str:
        dataset = bundle["dataset"]
        dataset_id = dataset["dataset_id"]
        with self.conn:
            self.create_dataset(dataset_id, dataset["name"])
            for record in bundle["raw_records"]:
                self.conn.execute(
                    """INSERT OR REPLACE INTO raw_records(
                         record_id, dataset_id, uid, kind, ping_index, bin_index,
                         beam_index, t, raw_value, unit, datum, source, note,
                         superseded_by, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record["record_id"], dataset_id, record["uid"], record["kind"],
                        record["ping_index"], record["bin_index"], record["beam_index"],
                        record["t"],
                        record["raw_value"] if isinstance(record["raw_value"], str)
                        else canonical_json(record["raw_value"]),
                        record["unit"], record["datum"], record["source"],
                        record.get("note"), record.get("superseded_by"),
                        record["updated_at"],
                    ),
                )
            for version in bundle["versions"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?,?)",
                    (version["version_id"], version["dataset_id"], version["kind"],
                     version["base_version_id"], version["created_at"],
                     version.get("label")),
                )
            for node in bundle["nodes"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?)",
                    (node["node_id"], node["version_id"], node["dataset_id"],
                     node["scope"], node["scope_key"],
                     node["content"] if isinstance(node["content"], str)
                     else canonical_json(node["content"]),
                     node["input_fingerprints"] if isinstance(node["input_fingerprints"], str)
                     else canonical_json(node["input_fingerprints"])),
                )
            for event in bundle["events"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                    (event["event_id"], event["dataset_id"], event["seq"], event["at"],
                     event["event_type"], event["actor"], event.get("base_version_id"),
                     event["payload"] if isinstance(event["payload"], str)
                     else canonical_json(event["payload"])),
                )
            for current in bundle["current_version"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO current_version VALUES (?,?,?)",
                    (current["dataset_id"], current["kind"], current["version_id"]),
                )
            for branch in bundle.get("branches", []):
                self.conn.execute(
                    "INSERT OR REPLACE INTO branches VALUES (?,?,?,?)",
                    (branch["dataset_id"], branch["branch_name"],
                     branch["version_id"], branch["event_seq"]),
                )
        return dataset_id
