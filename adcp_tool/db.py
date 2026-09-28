"""SQLite storage: separated sources, versioned derivation and atomic pointers.

Guarantees:

* beam velocity, correlation, amplitude, bad-beam flag, heading, pitch, roll,
  declination and mount angles live in separate streams/params and retain the
  raw representation;
* derivations are content-addressed (a fingerprint is a stable identity) and
  node pointers switch only after the whole write succeeds in one transaction;
* a crash before COMMIT leaves the previous complete version visible;
* two saves from the same parent raise a :class:`Conflict` carrying enough
  context to resolve (rebase / choose fork);
* export -> delete -> import preserves ids, order and provenance edges.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dataset(
  id TEXT PRIMARY KEY, title TEXT, geom_json TEXT, doc_json TEXT);
CREATE TABLE IF NOT EXISTS source_slot(
  id TEXT PRIMARY KEY, dataset_id TEXT, stream TEXT,
  ping INTEGER, cell INTEGER, beam INTEGER, seq INTEGER);
CREATE TABLE IF NOT EXISTS source_revision(
  id TEXT PRIMARY KEY, slot_id TEXT, valid_from_event INTEGER,
  raw TEXT, unit TEXT, basis TEXT, value_json TEXT, kind TEXT,
  active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS param_revision(
  id TEXT PRIMARY KEY, dataset_id TEXT, kind TEXT,
  valid_from_ping INTEGER, valid_from_event INTEGER, content_json TEXT);
CREATE TABLE IF NOT EXISTS deriv_node(
  fingerprint TEXT PRIMARY KEY, dataset_id TEXT, scope TEXT,
  ping INTEGER, cell INTEGER, content_json TEXT,
  rule_version TEXT, created_event INTEGER);
CREATE TABLE IF NOT EXISTS node_candidate(
  fingerprint TEXT, candidate_id TEXT, mode TEXT,
  beams_json TEXT, content_json TEXT,
  PRIMARY KEY(fingerprint, candidate_id));
CREATE TABLE IF NOT EXISTS node_state(
  branch TEXT, event_seq INTEGER, key TEXT,
  node_fingerprint TEXT, chosen_candidate TEXT,
  PRIMARY KEY(branch, event_seq, key));
CREATE TABLE IF NOT EXISTS dep_edge(
  branch TEXT, key TEXT, source_kind TEXT, source_ref TEXT,
  slot_stream TEXT, event_seq INTEGER);
CREATE TABLE IF NOT EXISTS branch(
  name TEXT PRIMARY KEY, head_seq INTEGER,
  parent_branch TEXT, fork_seq INTEGER);
CREATE TABLE IF NOT EXISTS event(
  branch TEXT, seq INTEGER, kind TEXT, payload_json TEXT,
  result_json TEXT, parent_seq INTEGER,
  PRIMARY KEY(branch, seq));
CREATE TABLE IF NOT EXISTS decision(
  branch TEXT, seq INTEGER, ping INTEGER, cell INTEGER,
  action TEXT, candidate_id TEXT, payload_json TEXT,
  PRIMARY KEY(branch, seq));
"""


class Conflict(Exception):
    """Optimistic-concurrency conflict with the data needed to resolve it."""

    def __init__(self, expected_parent: int, actual_head: int, detail: dict):
        super().__init__(f"branch moved: expected {expected_parent}, "
                         f"found {actual_head}")
        self.expected_parent = expected_parent
        self.actual_head = actual_head
        self.detail = detail


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema','%d')"
                 % SCHEMA_VERSION)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection, crash_after_stage: bool = False):
    """Commit boundary. With crash flag we abort right before COMMIT so the
    restarted process sees the previous complete version only."""
    conn.execute("BEGIN")
    try:
        yield conn
        if crash_after_stage:
            conn.execute("ROLLBACK")
            raise RuntimeError("simulated crash before commit")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))


# ----- export / import -----------------------------------------------------

EXPORT_TABLES = ["meta", "dataset", "source_slot", "source_revision",
                 "param_revision", "deriv_node", "node_candidate",
                 "node_state", "dep_edge", "branch", "event", "decision"]


def export_json(conn: sqlite3.Connection) -> dict:
    dump = {"format": "adcp-evidence-pack", "schema_version": SCHEMA_VERSION,
            "tables": {}}
    for table in EXPORT_TABLES:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        rows = [dict(zip(cols, r)) for r in
                conn.execute(f"SELECT * FROM {table} ORDER BY 1,2,3,4")]
        dump["tables"][table] = {"columns": cols, "rows": rows}
    return dump


def import_json(conn: sqlite3.Connection, dump: dict) -> None:
    """Restore an exported pack preserving every id / sequence / ordering."""
    if dump.get("format") != "adcp-evidence-pack":
        raise ValueError("not an adcp evidence pack")
    with transaction(conn):
        conn.executescript(SCHEMA)
        for table, part in dump["tables"].items():
            cols = part["columns"]
            conn.execute(f"DELETE FROM {table}")
            ph = ",".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {table}({','.join(cols)}) VALUES({ph})",
                [tuple(r[c] for c in cols) for r in part["rows"]])
        # repaired/upgraded stores keep the current schema marker
        set_meta(conn, "schema", str(SCHEMA_VERSION))


def save_dump(dump: dict, path: str) -> None:
    Path(path).write_text(json.dumps(dump, ensure_ascii=False),
                          encoding="utf-8")


def load_dump(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
