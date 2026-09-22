"""SQLite connection management and old-version migration."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

from .schema import SCHEMA_STATEMENTS, SCHEMA_VERSION, SQLITE_PRAGMAS


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> int:
    """Create the schema (idempotent).  Returns schema version."""
    conn.executescript(SQLITE_PRAGMAS)
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                     (str(SCHEMA_VERSION),))
    conn.commit()
    return SCHEMA_VERSION


# -- legacy v1 support -------------------------------------------------------
# The v1 prototype stored every ping as a wide row in a single table and did
# not version sources or nodes.  migrate_v1 maps those wide rows onto v2
# source revisions in one baseline commit.

V1_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ping_rows (
  ping_id TEXT PRIMARY KEY,
  ts REAL,
  b1 REAL, b2 REAL, b3 REAL, b4 REAL,
  corr1 REAL, corr2 REAL, corr3 REAL, corr4 REAL,
  amp1 REAL, amp2 REAL, amp3 REAL, amp4 REAL,
  bad TEXT, heading REAL, pitch REAL, roll REAL
)"""


def is_v1_db(path: str) -> bool:
    if not os.path.exists(path):
        return False
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ping_rows'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def migrate_v1(path: str) -> None:
    """Upgrade a v1 file in place to v2 inside one transaction."""
    legacy = sqlite3.connect(path)
    legacy.row_factory = sqlite3.Row
    rows = [dict(r) for r in legacy.execute("SELECT * FROM ping_rows ORDER BY ts")]
    legacy.close()

    from .repository import Repository

    conn = connect(path)
    init_db(conn)
    repo = Repository(conn)
    sources = {}
    for r in rows:
        pid = r["ping_id"]
        sources["%s.time" % pid] = {"kind": "time", "raw": r["ts"], "unit": "unix_s"}
        for i, k in enumerate(("b1", "b2", "b3", "b4"), start=1):
            sources["%s.beam.%d.radial" % (pid, i)] = {
                "kind": "beam_radial", "raw": r[k], "unit": "m/s"}
            sources["%s.beam.%d.corr" % (pid, i)] = {
                "kind": "beam_corr", "raw": r["corr%d" % i], "unit": "count"}
            sources["%s.beam.%d.amp" % (pid, i)] = {
                "kind": "beam_amp", "raw": r["amp%d" % i], "unit": "dB"}
        bad = set(int(x) for x in json.loads(r["bad"] or "[]"))
        for i in range(1, 5):
            sources["%s.beam.%d.bad" % (pid, i)] = {
                "kind": "beam_bad", "raw": i in bad, "unit": "flag"}
        for name, k in (("heading", "heading"), ("pitch", "pitch"), ("roll", "roll")):
            sources["%s.att.%s" % (pid, name)] = {
                "kind": "att_" + name, "raw": r[k], "unit": "deg"}
    repo.commit(
        kind="migration", label="v1->v2 migration", branch="main",
        sources=sources, nodes=[], events=[], parent_commit=None,
        actor="migrator")
    conn.close()
