SQLITE_PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA recursive_triggers=ON;
"""

SCHEMA_STATEMENTS = [
"""
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
)""",
"""
CREATE TABLE IF NOT EXISTS commits (
  commit_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL UNIQUE,
  parent_ids TEXT NOT NULL,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  created_at REAL NOT NULL,
  branch TEXT NOT NULL,
  event_id TEXT,
  payload_json TEXT NOT NULL,
  content_hash TEXT NOT NULL
)""",
"""
CREATE TABLE IF NOT EXISTS commit_closure (
  ancestor_id TEXT NOT NULL,
  descendant_id TEXT NOT NULL,
  depth INTEGER NOT NULL,
  PRIMARY KEY (ancestor_id, descendant_id)
)""",
"""
CREATE TABLE IF NOT EXISTS branches (
  name TEXT PRIMARY KEY,
  head_commit TEXT NOT NULL
)""",
"""
CREATE TABLE IF NOT EXISTS source_revs (
  source_id TEXT NOT NULL,
  rev INTEGER NOT NULL,
  commit_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  datum TEXT,
  raw_json TEXT NOT NULL,
  prev_rev INTEGER,
  PRIMARY KEY (source_id, rev)
)""",
"""
CREATE TABLE IF NOT EXISTS node_revs (
  node_id TEXT NOT NULL,
  rev INTEGER NOT NULL,
  commit_id TEXT NOT NULL,
  status TEXT NOT NULL,
  fingerprint TEXT,
  payload_json TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  reason TEXT,
  prev_rev INTEGER,
  PRIMARY KEY (node_id, rev)
)""",
"""
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL UNIQUE,
  commit_id TEXT NOT NULL,
  parent_event_id TEXT,
  kind TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at REAL NOT NULL,
  payload_json TEXT NOT NULL,
  content_hash TEXT NOT NULL
)""",
"""
CREATE TABLE IF NOT EXISTS conflicts (
  conflict_id TEXT PRIMARY KEY,
  base_commit TEXT NOT NULL,
  commit_a TEXT NOT NULL,
  commit_b TEXT NOT NULL,
  edited_a_json TEXT NOT NULL,
  edited_b_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  resolved_commit TEXT,
  resolution TEXT
)""",
]

SCHEMA_VERSION = 2
