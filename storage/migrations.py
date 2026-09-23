"""Schema migrations.

v1 -> v2: adds the ``bottom_track`` table and the ``pointer_log`` time-travel
table. v1 had no pointer history, so backfilling seeds pointer_log with the
current pointers at the migration event.
"""
from __future__ import annotations

import json
import sqlite3
import time


def migrate(conn: sqlite3.Connection, from_version: int) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        if from_version == 1:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS bottom_track (
                    record_id TEXT PRIMARY KEY, ping_id TEXT NOT NULL, beam INTEGER NOT NULL,
                    range_unit TEXT NOT NULL, range_raw TEXT NOT NULL, range_m REAL,
                    vel_unit TEXT NOT NULL, vel_raw TEXT NOT NULL, vel_ms REAL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS pointer_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_type TEXT NOT NULL, ping_id TEXT NOT NULL, cell_index INTEGER,
                    derivation_id TEXT, revision INTEGER, event_seq INTEGER NOT NULL
                )"""
            )
            seq_row = conn.execute("SELECT COALESCE(MAX(seq),0) s FROM event").fetchone()
            eseq = int(seq_row[0]) + 1
            conn.execute(
                "INSERT INTO event(event_type,scope,payload_json,created_at_s) "
                "VALUES(?,?,?,?)",
                ("schema_migrated", None,
                 json.dumps({"from": 1, "to": 2}, separators=(",", ":")), time.time()),
            )
            for r in conn.execute("SELECT * FROM current_pointer").fetchall():
                conn.execute(
                    "INSERT INTO pointer_log(scope_type,ping_id,cell_index,derivation_id,"
                    "revision,event_seq) VALUES(?,?,?,?,?,?)",
                    (r["scope_type"], r["ping_id"], r["cell_index"], r["derivation_id"],
                     r["revision"], eseq),
                )
            conn.execute("PRAGMA user_version=2")
            conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
