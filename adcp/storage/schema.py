"""SQLite schema definitions and the legacy migration chain.

Version history:

* v0 (legacy fixture under ``fixtures/legacy_v0``): a flat single-table
  layout where each ping row carried beam values, attitude and QC flags in
  one record. It predates the separated-source requirement.
* v1 (current): separated sources, versioned configs, derived nodes,
  atomic versions and an append-only event log.
"""

from __future__ import annotations

CURRENT_SCHEMA = 1

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    ping_id     TEXT,
    cell_no     INTEGER,
    beam        INTEGER,
    stream_key  TEXT,
    raw         TEXT NOT NULL,
    unit        TEXT,
    datum       TEXT,
    numeric     REAL,
    diagnostic  TEXT,
    created_event INTEGER NOT NULL,
    superseded_by_source TEXT
);

CREATE TABLE IF NOT EXISTS config_versions (
    config_type TEXT NOT NULL,
    version_id  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    introduced_event INTEGER NOT NULL,
    retired_event INTEGER,
    PRIMARY KEY (config_type, version_id)
);

CREATE TABLE IF NOT EXISTS derived_nodes (
    node_id     TEXT NOT NULL,
    version_id  TEXT NOT NULL,
    scope       TEXT NOT NULL,
    ping_id     TEXT,
    cell_no     INTEGER,
    kind        TEXT NOT NULL,
    candidate_id TEXT,
    status      TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    value       TEXT NOT NULL,
    value_ids   TEXT NOT NULL,
    inputs      TEXT NOT NULL,
    evidence    TEXT NOT NULL,
    diagnostics TEXT NOT NULL,
    PRIMARY KEY (node_id, version_id)
);

CREATE TABLE IF NOT EXISTS versions (
    version_id  TEXT PRIMARY KEY,
    parent_id   TEXT,
    kind        TEXT NOT NULL,
    label       TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_event INTEGER NOT NULL,
    based_on_version TEXT
);

CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT UNIQUE NOT NULL,
    version_id  TEXT NOT NULL,
    code        TEXT NOT NULL,
    actor       TEXT,
    label       TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    version_id  TEXT NOT NULL,
    based_on_version TEXT NOT NULL,
    ping_id     TEXT NOT NULL,
    chosen_candidate_id TEXT,
    actor       TEXT NOT NULL,
    status      TEXT NOT NULL,
    conflicts_with TEXT,
    resolution  TEXT,
    created_event INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    version_id  TEXT NOT NULL,
    upstream_node_id TEXT NOT NULL,
    downstream_node_id TEXT NOT NULL,
    PRIMARY KEY (version_id, upstream_node_id, downstream_node_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_ver ON derived_nodes(version_id);
CREATE INDEX IF NOT EXISTS idx_nodes_scope ON derived_nodes(scope, ping_id, cell_no);
CREATE INDEX IF NOT EXISTS idx_events_ver ON events(version_id);
"""

# The legacy v0 layout (used by the migration test fixture). One flat row
# mixes beam velocities, attitude and quality flags - exactly the thing the
# new model forbids, so the migration must *split* it into separate sources.
SCHEMA_V0 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE ping_rows (
    ping_id TEXT PRIMARY KEY,
    t_s REAL,
    heading_deg REAL, pitch_deg REAL, roll_deg REAL,
    declination_deg REAL, mount_deg REAL,
    beam1_cmps REAL, beam2_cmps REAL, beam3_cmps REAL, beam4_cmps REAL,
    corr1 INTEGER, corr2 INTEGER, corr3 INTEGER, corr4 INTEGER,
    amp1 REAL, amp2 REAL, amp3 REAL, amp4 REAL,
    badflag1 INTEGER, badflag2 INTEGER, badflag3 INTEGER, badflag4 INTEGER,
    bottom_m REAL
);
"""


def initialize(conn) -> None:
    conn.executescript(SCHEMA_V1)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES('current_version', '')"
    )
