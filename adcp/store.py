"""版本化 SQLite 存储层。

保证：
- 波束测量 / correlation / amplitude / 坏 beam 标志 / heading,pitch,roll /
  磁偏角 / 安装角分成独立 raw 记录。
- 每次推导产生独立版本（derivation），只有完整成功后才切换 current 指针；
  写入中断后重启，要么看到旧版本，要么看到完整新版本。
- 两位使用者从同一版本编辑时产生可解决的冲突，两个分支都保留。
- 导出/导入保留稳定身份（id）、顺序（seq）与来源关系（deps）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid

SCHEMA_VERSION = 2


class InjectedFault(RuntimeError):
    """测试用：模拟写入中途断电。"""


class ConflictDetected(RuntimeError):
    """基于旧 current 版本保存时，当前指针已被别人推进。"""

    def __init__(self, conflict_id: str, current_id: str, attempted_id: str,
                 base_id: str):
        super().__init__(f"conflict:{conflict_id}")
        self.conflict_id = conflict_id
        self.current_id = current_id
        self.attempted_id = attempted_id
        self.base_id = base_id


SCHEMA = """
CREATE TABLE raw_records(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  ping_id TEXT,
  beam INTEGER,
  cell INTEGER,
  value TEXT,
  unit TEXT,
  frame TEXT,
  raw_repr TEXT,
  source TEXT,
  version INTEGER NOT NULL DEFAULT 1,
  active INTEGER NOT NULL DEFAULT 1,
  supersedes TEXT,
  attrs TEXT NOT NULL DEFAULT '{}',
  seq INTEGER NOT NULL
);
CREATE TABLE derivations(
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  label TEXT NOT NULL,
  params TEXT NOT NULL DEFAULT '{}',
  fingerprint TEXT,
  status TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE artifacts(
  id TEXT PRIMARY KEY,
  derivation_id TEXT NOT NULL,
  node_key TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  deps TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  reused INTEGER NOT NULL DEFAULT 0,
  seq INTEGER NOT NULL
);
CREATE TABLE events(
  id TEXT PRIMARY KEY,
  derivation_id TEXT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  seq INTEGER NOT NULL
);
CREATE TABLE current_pointer(id INTEGER PRIMARY KEY CHECK(id=1), derivation_id TEXT);
CREATE TABLE conflicts(
  id TEXT PRIMARY KEY,
  base_id TEXT NOT NULL,
  current_id TEXT NOT NULL,
  attempted_id TEXT NOT NULL,
  status TEXT NOT NULL,
  resolution TEXT,
  seq INTEGER NOT NULL
);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_raw_active ON raw_records(kind, active);
CREATE INDEX idx_art_der ON artifacts(derivation_id);
CREATE INDEX idx_ev_der ON events(derivation_id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Store:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.failpoint = 0  # 测试用故障注入计数
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False,
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._recover()

    # ---------- schema / migration / recovery ----------

    def _user_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def _init_schema(self) -> None:
        with self.lock:
            version = self._user_version()
            if version == 0:
                self.conn.executescript(SCHEMA)
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            else:
                self._migrate(version)

    def _migrate(self, version: int) -> None:
        """旧版本迁移：v1 -> v2（raw_records 增加 attrs，derivations 增加 params）。"""
        if version < 2:
            cols = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(raw_records)")}
            if "attrs" not in cols:
                self.conn.execute(
                    "ALTER TABLE raw_records ADD COLUMN attrs "
                    "TEXT NOT NULL DEFAULT '{}'")
            cols = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(derivations)")}
            if "params" not in cols:
                self.conn.execute(
                    "ALTER TABLE derivations ADD COLUMN params "
                    "TEXT NOT NULL DEFAULT '{}'")
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _recover(self) -> None:
        """崩溃恢复：未完整提交的 building 版本标记 abandoned；current 指针不动。"""
        with self.lock:
            rows = list(self.conn.execute(
                "SELECT id FROM derivations WHERE status='building'"))
            if rows:
                self.conn.execute("BEGIN IMMEDIATE")
                for row in rows:
                    self.conn.execute(
                        "UPDATE derivations SET status='abandoned' WHERE id=?",
                        (row["id"],))
                    self.add_event_no_tx(row["id"], "derivation_abandoned",
                                         {"reason": "crash_recovery"})
                self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()

    # ---------- sequences ----------

    def next_seq(self) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='next_seq'").fetchone()
            if row is None:
                seq = 1
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('next_seq','1')")
            else:
                seq = int(row["value"]) + 1
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='next_seq'", (str(seq),))
            return seq

    # ---------- raw records ----------

    def insert_raw(self, *, kind: str, ping_id: str | None, beam: int | None,
                   cell: int | None, value: str | None, unit: str | None,
                   frame: str | None = None, raw_repr: str | None = None,
                   source: str | None = None, attrs: dict | None = None) -> dict:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            record = self._insert_raw_no_tx(kind, ping_id, beam, cell, value,
                                            unit, frame, raw_repr, source,
                                            attrs or {}, supersedes=None)
            self.conn.execute("COMMIT")
            return record

    def _insert_raw_no_tx(self, kind, ping_id, beam, cell, value, unit, frame,
                          raw_repr, source, attrs, supersedes) -> dict:
        record_id = new_id("raw")
        version = 1
        if supersedes:
            row = self.conn.execute(
                "SELECT version FROM raw_records WHERE id=?",
                (supersedes,)).fetchone()
            version = int(row["version"]) + 1
        seq = self.next_seq()
        self.conn.execute(
            "INSERT INTO raw_records(id,kind,ping_id,beam,cell,value,unit,frame,"
            "raw_repr,source,version,active,supersedes,attrs,seq) VALUES("
            "?,?,?,?,?,?,?,?,?,?,1,1,?,?,?)",
            (record_id, kind, ping_id, beam, cell, value, unit, frame, raw_repr,
             source, supersedes,
             json.dumps(attrs, sort_keys=True, ensure_ascii=False), seq))
        return self.get_raw(record_id)

    def supersede_raw(self, record_id: str, **changes) -> dict:
        """用新版本取代一条 raw 记录；旧记录保留但 active=0。"""
        old = self.get_raw(record_id)
        for field in ("kind", "ping_id", "beam", "cell", "unit", "frame",
                      "raw_repr", "source"):
            changes.setdefault(field, old[field])
        changes.setdefault("value", old["value"])
        attrs = {**json.loads(old["attrs"] or "{}"),
                 **changes.pop("attrs", {})}
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "UPDATE raw_records SET active=0 WHERE id=?", (record_id,))
            record = self._insert_raw_no_tx(
                changes["kind"], changes["ping_id"], changes["beam"],
                changes["cell"], changes["value"], changes["unit"],
                changes["frame"], changes["raw_repr"], changes["source"],
                attrs, supersedes=record_id)
            self.conn.execute("COMMIT")
        return record

    def deactivate_raw(self, record_id: str) -> None:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "UPDATE raw_records SET active=0 WHERE id=?", (record_id,))
            self.conn.execute("COMMIT")

    def get_raw(self, record_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM raw_records WHERE id=?", (record_id,)).fetchone()
        return dict(row) if row else None

    def active_raw(self, kind: str | None = None) -> list[dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM raw_records WHERE active=1 AND kind=? ORDER BY seq",
                (kind,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM raw_records WHERE active=1 ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    def find_active(self, kind: str, ping_id=None, beam=None, cell=None):
        sql = "SELECT * FROM raw_records WHERE active=1 AND kind=?"
        params: list = [kind]
        if ping_id is not None:
            sql += " AND ping_id=?"
            params.append(ping_id)
        if beam is not None:
            sql += " AND beam=?"
            params.append(beam)
        if cell is not None:
            sql += " AND cell=?"
            params.append(cell)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
