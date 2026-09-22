-- ADCP 证据库。原始测量四类分开记录；派生节点内容寻址；
-- 版本经暂存->原子切换 current 指针；事件只追加。
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 原始层：导入批次
CREATE TABLE IF NOT EXISTS source_batch (
  batch_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  original_encoding TEXT
);

-- ping 时间锚点（波束测量挂在 ping+cell 下；姿态是独立流）
CREATE TABLE IF NOT EXISTS ping (
  ping_id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL REFERENCES source_batch(batch_id),
  seq INTEGER NOT NULL,
  t_s REAL NOT NULL,
  raw_time_text TEXT,
  time_unit TEXT
);

CREATE TABLE IF NOT EXISTS cell (
  cell_id TEXT PRIMARY KEY,
  ping_id TEXT NOT NULL REFERENCES ping(ping_id),
  bin_index INTEGER NOT NULL,
  depth_raw REAL,
  depth_unit TEXT,
  UNIQUE(ping_id, bin_index)
);

-- 波束径向速度（每 beam 一行，保留原始表示与原始单位）
CREATE TABLE IF NOT EXISTS beam_velocity (
  id TEXT PRIMARY KEY,
  cell_id TEXT NOT NULL REFERENCES cell(cell_id),
  beam INTEGER NOT NULL,
  raw_value REAL,
  raw_unit TEXT,
  UNIQUE(cell_id, beam)
);

CREATE TABLE IF NOT EXISTS beam_correlation (
  id TEXT PRIMARY KEY,
  cell_id TEXT NOT NULL REFERENCES cell(cell_id),
  beam INTEGER NOT NULL,
  raw_value REAL,
  raw_unit TEXT,
  UNIQUE(cell_id, beam)
);

CREATE TABLE IF NOT EXISTS beam_amplitude (
  id TEXT PRIMARY KEY,
  cell_id TEXT NOT NULL REFERENCES cell(cell_id),
  beam INTEGER NOT NULL,
  raw_value REAL,
  raw_unit TEXT,
  UNIQUE(cell_id, beam)
);

CREATE TABLE IF NOT EXISTS beam_bad_flag (
  id TEXT PRIMARY KEY,
  cell_id TEXT NOT NULL REFERENCES cell(cell_id),
  beam INTEGER NOT NULL,
  flag INTEGER NOT NULL,
  UNIQUE(cell_id, beam)
);

-- bottom track：每个 ping 一套四波束
CREATE TABLE IF NOT EXISTS bottom_track (
  id TEXT PRIMARY KEY,
  ping_id TEXT NOT NULL REFERENCES ping(ping_id),
  beam INTEGER NOT NULL,
  velocity_raw REAL,
  velocity_unit TEXT,
  correlation_raw REAL,
  correlation_unit TEXT,
  amplitude_raw REAL,
  amplitude_unit TEXT,
  bad_flag INTEGER,
  range_raw REAL,
  range_unit TEXT,
  UNIQUE(ping_id, beam)
);

-- 姿态独立时间序列（heading 磁北、度；pitch/roll 度）
CREATE TABLE IF NOT EXISTS attitude_sample (
  id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL REFERENCES source_batch(batch_id),
  seq INTEGER NOT NULL,
  t_s REAL NOT NULL,
  heading_raw REAL,
  heading_unit TEXT,
  heading_datum TEXT,
  pitch_raw REAL,
  pitch_unit TEXT,
  roll_raw REAL,
  roll_unit TEXT
);

-- 配置块（波束几何/安装角/磁偏角/变换顺序/阈值/深度），各自带版本
CREATE TABLE IF NOT EXISTS config_blob (
  config_id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  version TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_event_seq INTEGER,
  UNIQUE(type, version)
);

-- 内容寻址派生节点
CREATE TABLE IF NOT EXISTS node (
  node_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  fingerprint TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  created_event_seq INTEGER
);

CREATE TABLE IF NOT EXISTS node_dep (
  node_id TEXT NOT NULL REFERENCES node(node_id),
  dep_node_id TEXT NOT NULL,
  dep_slot TEXT NOT NULL,
  PRIMARY KEY (node_id, dep_node_id, dep_slot)
);

-- 暂存版本（崩溃后由重启逻辑判为 abandoned）
CREATE TABLE IF NOT EXISTS version_stage (
  version_id TEXT PRIMARY KEY,
  created_event_seq INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('staged','committed','abandoned'))
);

-- 当前指针：单行，提交时原子更新
CREATE TABLE IF NOT EXISTS current_pointer (
  only_id INTEGER PRIMARY KEY CHECK(only_id=1),
  version_id TEXT NOT NULL,
  event_seq INTEGER NOT NULL
);

-- 槽位 -> 当前节点（版本内的可见清单）
CREATE TABLE IF NOT EXISTS version_slot (
  version_id TEXT NOT NULL,
  slot_key TEXT NOT NULL,
  node_id TEXT NOT NULL REFERENCES node(node_id),
  PRIMARY KEY (version_id, slot_key)
);

-- 只追加事件日志
CREATE TABLE IF NOT EXISTS event (
  seq INTEGER PRIMARY KEY,
  event_type TEXT NOT NULL,
  version_id TEXT,
  basis_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  actor TEXT NOT NULL,
  parent_seq INTEGER
);

-- 候选集合（三波束/四波束等）与人工裁定
CREATE TABLE IF NOT EXISTS candidate (
  candidate_id TEXT PRIMARY KEY,
  event_seq INTEGER NOT NULL,
  scope_key TEXT NOT NULL,          -- 如 ping:3/bin:2
  node_id TEXT NOT NULL REFERENCES node(node_id),
  label TEXT NOT NULL,
  adopted INTEGER NOT NULL DEFAULT 0,
  UNIQUE(scope_key, node_id)
);

-- 编辑乐观锁：同一 base_version 的两次编辑产生冲突行
CREATE TABLE IF NOT EXISTS edit_conflict (
  conflict_id TEXT PRIMARY KEY,
  scope_key TEXT NOT NULL,
  base_version_id TEXT NOT NULL,
  winner_version_id TEXT,
  loser_version_id TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  detail_json TEXT
);
