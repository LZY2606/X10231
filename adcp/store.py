"""SQLite-backed versioned source records and derivation graph.

Every source fact has its own table and a version row. Events are the
append-only audit trail. A derivation run is written independently and the
current pointer is advanced only after the run's all-or-nothing transaction
commits.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2

SOURCE_TABLES = {
    "beam": ("beam_record_versions", "beam_record_id"),
    "bt": ("bt_record_versions", "bt_record_id"),
    "correlation": ("correlation_record_versions", "correlation_record_id"),
    "amplitude": ("amplitude_record_versions", "amplitude_record_id"),
    "flag": ("flag_record_versions", "flag_record_id"),
    "attitude": ("attitude_record_versions", "attitude_record_id"),
    "geom": ("geom_versions", "geom_id"),
    "install": ("install_versions", "install_id"),
    "declination": ("declination_versions", "declination_id"),
    "cell": ("cell_versions", "cell_id"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class OptimisticConflict(Exception):
    pass


def connect(path: str = "adcp.sqlite") -> sqlite3.Connection:
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if fresh:
        init_db(conn)
    elif user_version < SCHEMA_VERSION:
        migrate(conn, user_version)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    depth = getattr(conn, "_tx_depth", 0)
    if depth == 0:
        conn.execute("BEGIN IMMEDIATE")
        conn._tx_depth = 1
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn._tx_depth = 0
    else:
        yield conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE pings(dataset_id TEXT NOT NULL, ping_id TEXT PRIMARY KEY, ping_time REAL NOT NULL, ordinal INTEGER NOT NULL,
            UNIQUE(dataset_id, ping_time), FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));
        CREATE TABLE cells(dataset_id TEXT NOT NULL, cell_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, depth_m REAL NOT NULL,
            UNIQUE(dataset_id, depth_m), FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));

        CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, event_type TEXT NOT NULL,
            actor TEXT NOT NULL, payload_json TEXT NOT NULL, base_seq INTEGER, resolution TEXT,
            created_at TEXT NOT NULL, parent_event_seq INTEGER, UNIQUE(dataset_id, seq),
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));
        CREATE TABLE derivation_runs(run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, base_event_seq INTEGER NOT NULL,
            rule_json TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
            diagnostic_json TEXT NOT NULL DEFAULT '[]', replaced_run_id TEXT, selection_event_seq INTEGER);
        CREATE TABLE current_pointer(dataset_id TEXT PRIMARY KEY, run_id TEXT NOT NULL);

        CREATE TABLE derived_values(value_id TEXT NOT NULL, run_id TEXT NOT NULL, ping_id TEXT NOT NULL,
            cell_id TEXT NOT NULL, layer TEXT NOT NULL, candidate_id TEXT NOT NULL, status TEXT NOT NULL,
            value_json TEXT NOT NULL, fingerprint TEXT NOT NULL, PRIMARY KEY(run_id,value_id));
        CREATE TABLE derived_dependencies(value_id TEXT NOT NULL, run_id TEXT NOT NULL, source_table TEXT NOT NULL,
            source_record_id TEXT NOT NULL, source_version_hash TEXT NOT NULL, PRIMARY KEY(run_id,value_id,source_table,source_record_id));
        CREATE TABLE derived_inputs(run_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL);
        CREATE TABLE exclusions(exclusion_id TEXT NOT NULL, run_id TEXT NOT NULL, ping_id TEXT NOT NULL,
            cell_id TEXT NOT NULL, code TEXT NOT NULL, reason TEXT NOT NULL, evidence_json TEXT NOT NULL,
            PRIMARY KEY(run_id,exclusion_id));
        CREATE TABLE save_conflicts(conflict_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, attempted_seq INTEGER NOT NULL,
            expected_seq INTEGER NOT NULL, actual_seq INTEGER NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);

        CREATE TABLE beam_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, beam_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_id TEXT NOT NULL, cell_id TEXT NOT NULL, beam TEXT NOT NULL,
            raw_repr TEXT NOT NULL, raw_value REAL, raw_unit TEXT NOT NULL, first_seq INTEGER NOT NULL, last_seq INTEGER,
            version_hash TEXT NOT NULL, UNIQUE(beam_record_id, first_seq));
        CREATE TABLE bt_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, bt_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_id TEXT NOT NULL, beam TEXT NOT NULL, raw_repr TEXT NOT NULL, raw_value REAL, raw_unit TEXT NOT NULL,
            first_seq INTEGER NOT NULL, last_seq INTEGER, version_hash TEXT NOT NULL, UNIQUE(bt_record_id, first_seq));
        CREATE TABLE correlation_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, correlation_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_id TEXT NOT NULL, cell_id TEXT NOT NULL, beam TEXT NOT NULL,
            raw_repr TEXT NOT NULL, raw_value REAL, raw_unit TEXT NOT NULL, first_seq INTEGER NOT NULL, last_seq INTEGER,
            version_hash TEXT NOT NULL, UNIQUE(correlation_record_id, first_seq));
        CREATE TABLE amplitude_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, amplitude_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_id TEXT NOT NULL, cell_id TEXT NOT NULL, beam TEXT NOT NULL,
            raw_repr TEXT NOT NULL, raw_value REAL, raw_unit TEXT NOT NULL, first_seq INTEGER NOT NULL, last_seq INTEGER,
            version_hash TEXT NOT NULL, UNIQUE(amplitude_record_id, first_seq));
        CREATE TABLE flag_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, flag_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_id TEXT NOT NULL, cell_id TEXT, beam TEXT NOT NULL, flag TEXT NOT NULL,
            first_seq INTEGER NOT NULL, last_seq INTEGER, version_hash TEXT NOT NULL, UNIQUE(flag_record_id, first_seq));
        CREATE TABLE attitude_record_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, attitude_record_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, ping_time REAL NOT NULL, field TEXT NOT NULL, raw_repr TEXT NOT NULL, raw_value REAL, raw_unit TEXT NOT NULL,
            first_seq INTEGER NOT NULL, last_seq INTEGER, version_hash TEXT NOT NULL, UNIQUE(attitude_record_id, first_seq));
        CREATE TABLE geom_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, geom_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
            beam_angle REAL NOT NULL, angle_unit TEXT NOT NULL, geometry_version TEXT NOT NULL, first_seq INTEGER NOT NULL,
            last_seq INTEGER, version_hash TEXT NOT NULL, UNIQUE(geom_id, first_seq));
        CREATE TABLE install_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, install_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
            angle REAL NOT NULL, angle_unit TEXT NOT NULL, first_seq INTEGER NOT NULL, last_seq INTEGER,
            version_hash TEXT NOT NULL, UNIQUE(install_id, first_seq));
        CREATE TABLE declination_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, declination_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, degrees REAL NOT NULL, model_version TEXT NOT NULL, first_seq INTEGER NOT NULL,
            last_seq INTEGER, version_hash TEXT NOT NULL, UNIQUE(declination_id, first_seq));
        CREATE TABLE cell_versions(version_id INTEGER PRIMARY KEY AUTOINCREMENT, cell_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
            ping_id TEXT NOT NULL, valid INTEGER NOT NULL, reason TEXT NOT NULL, first_seq INTEGER NOT NULL, last_seq INTEGER,
            version_hash TEXT NOT NULL, UNIQUE(cell_id, ping_id, first_seq));

        CREATE INDEX idx_beam_lookup ON beam_record_versions(ping_id,cell_id,last_seq);
        CREATE INDEX idx_att_lookup ON attitude_record_versions(dataset_id,ping_time,field,last_seq);
        CREATE INDEX idx_flag_lookup ON flag_record_versions(ping_id,cell_id,beam,last_seq);
        CREATE INDEX idx_dv_lookup ON derived_values(run_id,ping_id,cell_id,layer);
        CREATE INDEX idx_dep_reverse ON derived_dependencies(source_table,source_record_id,run_id);
        """
    )
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def migrate(conn: sqlite3.Connection, old: int) -> None:
    # The shipped fixture includes a version-1 legacy database exporter.
    # Version 2 adds dependency graph indices; the migration is deliberately
    # explicit and tested with an offline database.
    if old == 1:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dep_reverse ON derived_dependencies(source_table,source_record_id,run_id)")
        conn.execute("PRAGMA user_version=2")
        return
    raise RuntimeError(f"unsupported database schema version {old}")
