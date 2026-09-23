"""SQLite schema and migration.

The schema keeps every direct source kind in its own table: beam velocity,
correlation, amplitude, bad-beam flags, heading/pitch/roll, declination and
install angle are never merged into one raw table.

Schema versioning is real: ``PRAGMA user_version`` starts at 1 and migrations to
higher versions are tested from an on-disk v1 fixture.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_batch (
    batch_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at_s REAL NOT NULL,
    seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ping (
    ping_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES source_batch(batch_id),
    seq INTEGER NOT NULL,
    time_s REAL NOT NULL,
    n_cells INTEGER NOT NULL,
    UNIQUE(batch_id, seq)
);

CREATE TABLE IF NOT EXISTS cell_meta (
    ping_id TEXT NOT NULL REFERENCES ping(ping_id),
    cell_index INTEGER NOT NULL,
    depth_unit TEXT NOT NULL,
    depth_raw TEXT NOT NULL,
    depth_m REAL,
    valid_cell INTEGER NOT NULL,
    PRIMARY KEY (ping_id, cell_index)
);

-- Each direct source kind lives in its own table.
CREATE TABLE IF NOT EXISTS beam_velocity (
    record_id TEXT PRIMARY KEY,
    ping_id TEXT NOT NULL, cell_index INTEGER NOT NULL, beam INTEGER NOT NULL,
    unit TEXT NOT NULL, raw TEXT NOT NULL, value REAL, normalized REAL,
    diagnostic TEXT
);
CREATE TABLE IF NOT EXISTS correlation (
    record_id TEXT PRIMARY KEY,
    ping_id TEXT NOT NULL, cell_index INTEGER NOT NULL, beam INTEGER NOT NULL,
    unit TEXT NOT NULL, raw TEXT NOT NULL, value REAL, normalized REAL,
    diagnostic TEXT
);
CREATE TABLE IF NOT EXISTS amplitude (
    record_id TEXT PRIMARY KEY,
    ping_id TEXT NOT NULL, cell_index INTEGER NOT NULL, beam INTEGER NOT NULL,
    unit TEXT NOT NULL, raw TEXT NOT NULL, value REAL, normalized REAL,
    diagnostic TEXT
);
CREATE TABLE IF NOT EXISTS bad_beam_flag (
    record_id TEXT PRIMARY KEY,
    ping_id TEXT NOT NULL, cell_index INTEGER NOT NULL, beam INTEGER NOT NULL,
    raw TEXT NOT NULL, flagged INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS heading (
    record_id TEXT PRIMARY KEY, ping_id TEXT NOT NULL,
    time_s REAL NOT NULL, unit TEXT NOT NULL, datum TEXT NOT NULL,
    raw TEXT NOT NULL, value REAL, normalized REAL, diagnostic TEXT
);
CREATE TABLE IF NOT EXISTS pitch (
    record_id TEXT PRIMARY KEY, ping_id TEXT NOT NULL,
    time_s REAL NOT NULL, unit TEXT NOT NULL, datum TEXT NOT NULL,
    raw TEXT NOT NULL, value REAL, normalized REAL, diagnostic TEXT
);
CREATE TABLE IF NOT EXISTS roll (
    record_id TEXT PRIMARY KEY, ping_id TEXT NOT NULL,
    time_s REAL NOT NULL, unit TEXT NOT NULL, datum TEXT NOT NULL,
    raw TEXT NOT NULL, value REAL, normalized REAL, diagnostic TEXT
);

CREATE TABLE IF NOT EXISTS declination (
    version_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL,
    valid_from_s REAL NOT NULL, unit TEXT NOT NULL, raw TEXT NOT NULL,
    value REAL, normalized REAL, diagnostic TEXT, superseded_by TEXT
);
CREATE TABLE IF NOT EXISTS install_angle (
    version_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL,
    valid_from_s REAL NOT NULL, unit TEXT NOT NULL, raw TEXT NOT NULL,
    value REAL, normalized REAL, diagnostic TEXT, superseded_by TEXT
);
CREATE TABLE IF NOT EXISTS transducer_config (
    version_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL,
    config_json TEXT NOT NULL, fingerprint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bottom_track (
    record_id TEXT PRIMARY KEY, ping_id TEXT NOT NULL, beam INTEGER NOT NULL,
    range_unit TEXT NOT NULL, range_raw TEXT NOT NULL, range_m REAL,
    vel_unit TEXT NOT NULL, vel_raw TEXT NOT NULL, vel_ms REAL
);

-- Derived layer: append-only candidates per scope.
CREATE TABLE IF NOT EXISTS derivation (
    derivation_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,           -- ping_cell | bottom_track
    ping_id TEXT NOT NULL,
    cell_index INTEGER,
    revision INTEGER NOT NULL,          -- recompute appends a new revision
    kind TEXT NOT NULL,                 -- four_beam | three_beam
    beams TEXT NOT NULL,                -- JSON list
    assumption TEXT NOT NULL,
    instrument TEXT NOT NULL,           -- JSON [u,v,w]
    ship TEXT NOT NULL,
    earth TEXT,                          -- JSON or null when unresolved
    error_velocity REAL,
    diagnostics TEXT NOT NULL,          -- JSON list
    fingerprint TEXT NOT NULL,
    rule_geom TEXT NOT NULL,
    rule_decl_version TEXT NOT NULL,
    rule_install_version TEXT NOT NULL,
    rule_order TEXT NOT NULL,
    created_event_seq INTEGER NOT NULL,
    PRIMARY KEY (derivation_id, revision)
);

-- Current pointer switches only after a full successful transaction.
CREATE TABLE IF NOT EXISTS current_pointer (
    scope_type TEXT NOT NULL, ping_id TEXT NOT NULL,
    cell_index INTEGER,
    derivation_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    updated_event_seq INTEGER NOT NULL,
    PRIMARY KEY (scope_type, ping_id, cell_index)
);

-- Pointer history powers "go back to any old event".
CREATE TABLE IF NOT EXISTS pointer_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type TEXT NOT NULL, ping_id TEXT NOT NULL, cell_index INTEGER,
    derivation_id TEXT, revision INTEGER,
    event_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    scope TEXT,
    payload_json TEXT NOT NULL,
    created_at_s REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deriv_ping ON derivation(ping_id);
CREATE INDEX IF NOT EXISTS idx_bt_ping ON bottom_track(ping_id);
CREATE INDEX IF NOT EXISTS idx_event_type ON event(event_type);
"""


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]
