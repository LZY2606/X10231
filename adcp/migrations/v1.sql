-- Schema v1: initial shipped release.
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Each *kind* of source record lives in its own table (requirement: beams,
-- correlation, amplitude, bad flags, attitude, declination and mounting are
-- recorded separately).
CREATE TABLE IF NOT EXISTS src_beam (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER NOT NULL,
  cell INTEGER NOT NULL,
  beam INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  canonical_mps REAL,
  diagnostic_json TEXT,
  kind TEXT NOT NULL DEFAULT 'water'
);

CREATE TABLE IF NOT EXISTS src_correlation (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER NOT NULL,
  cell INTEGER NOT NULL,
  beam INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  canonical REAL,
  diagnostic_json TEXT
);

CREATE TABLE IF NOT EXISTS src_amplitude (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER NOT NULL,
  cell INTEGER NOT NULL,
  beam INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  canonical REAL,
  diagnostic_json TEXT
);

CREATE TABLE IF NOT EXISTS src_badflag (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER NOT NULL,
  cell INTEGER NOT NULL,
  beam INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  canonical INTEGER,
  diagnostic_json TEXT
);

CREATE TABLE IF NOT EXISTS src_attitude (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER,
  t REAL NOT NULL,
  raw_json TEXT NOT NULL,
  heading_deg REAL,
  pitch_deg REAL,
  roll_deg REAL,
  heading_datum TEXT,
  diagnostic_json TEXT
);

CREATE TABLE IF NOT EXISTS src_declination (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  raw_json TEXT NOT NULL,
  value_deg REAL,
  model TEXT,
  model_version TEXT
);

CREATE TABLE IF NOT EXISTS src_mounting (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  raw_json TEXT NOT NULL,
  heading_offset_deg REAL,
  pitch_offset_deg REAL,
  roll_offset_deg REAL,
  transform_order TEXT,
  revision TEXT
);

CREATE TABLE IF NOT EXISTS src_bottom (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  ping INTEGER NOT NULL,
  beam INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  range_m REAL,
  diagnostic_json TEXT
);

CREATE TABLE IF NOT EXISTS config (
  dataset_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (dataset_id, key)
);

CREATE TABLE IF NOT EXISTS derivations (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  supersedes_id TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  ping INTEGER,
  cell INTEGER,
  beam INTEGER,
  scope TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS node_pointers (
  node_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  current_derivation_id TEXT,
  PRIMARY KEY (dataset_id, node_id)
);

CREATE TABLE IF NOT EXISTS versions (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  parent_id TEXT,
  label TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  commit_seq INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  parent_event_id TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  choice TEXT NOT NULL,
  candidate_ids_json TEXT NOT NULL,
  rationale TEXT,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  supersedes_decision_id TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  body TEXT NOT NULL,
  rev INTEGER NOT NULL DEFAULT 1,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  UNIQUE(dataset_id, node_id)
);
