"""SQLite 存储层：分开记录的源数据、版本化派生、事件溯源、原子指针、冲突。

关键保证
--------
- 波束测量 / correlation / amplitude / 坏 beam 标志 / heading,pitch,roll /
  磁偏角 / 安装角是 *分开* 的源记录（不同 source_kind），不混存。
- 每次推导产生独立 version，仅在同一事务内把当前指针从旧版本切到新版本；
  写一半崩溃后，外部只看到旧版本或完整新版本（绝不看到半成品）。
- 两个使用者从同一 base_version 提交时，后提交者收到 ``ConflictError``，
  可在重试时解决（rebase/选择），冲突不被静默覆盖。
- 导出（单文件 JSON）-> 删除本地数据库 -> 导入后，身份(stable_id)、
  顺序(seq) 与来源关系(source/derives) 完全保持。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 2

_lock = threading.RLock()


class ConflictError(Exception):
    def __init__(self, message: str, current_version: int, base_version: int):
        super().__init__(message)
        self.current_version = current_version
        self.base_version = base_version


def _digest(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def content_stable_id(kind: str, payload: Dict[str, Any]) -> str:
    """内容寻址身份：相同 kind+内容 -> 相同稳定身份。"""
    return kind + "_" + _digest([kind, payload])[:16]


def fingerprint(spec: Dict[str, Any]) -> str:
    return "fp_" + _digest([spec])[:20]


# --------------------------------------------------------------------------- #
# 模式
# --------------------------------------------------------------------------- #
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS meta(
       key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS sources(
       uid INTEGER PRIMARY KEY AUTOINCREMENT,
       stable_id TEXT NOT NULL,
       source_kind TEXT NOT NULL,
       ping TEXT, channel TEXT, bin_index INTEGER,
       payload TEXT NOT NULL,
       units TEXT,
       supersedes INTEGER,
       created_at REAL NOT NULL,
       UNIQUE(stable_id))""",
    """CREATE TABLE IF NOT EXISTS events(
       seq INTEGER PRIMARY KEY AUTOINCREMENT,
       event_id TEXT NOT NULL UNIQUE,
       at REAL NOT NULL, actor TEXT NOT NULL,
       type TEXT NOT NULL, base_version INTEGER, new_version INTEGER,
       subject TEXT, payload TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS versions(
       version_id INTEGER PRIMARY KEY AUTOINCREMENT,
       parent_version INTEGER,
       created_at REAL NOT NULL, label TEXT, kind TEXT)""",
    """CREATE TABLE IF NOT EXISTS version_items(
       version_id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
       PRIMARY KEY(version_id,key))""",
    """CREATE TABLE IF NOT EXISTS derivations(
       uid INTEGER PRIMARY KEY AUTOINCREMENT,
       stable_id TEXT NOT NULL UNIQUE,
       node_key TEXT NOT NULL,
       version_id INTEGER NOT NULL,
       fingerprint TEXT NOT NULL,
       kind TEXT NOT NULL, status TEXT NOT NULL,
       result TEXT, diagnostic TEXT,
       chosen INTEGER DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS depends(
       derivation_uid INTEGER NOT NULL,
       dep_stable_id TEXT NOT NULL,
       dep_kind TEXT NOT NULL,
       PRIMARY KEY(derivation_uid, dep_stable_id))""",
    """CREATE TABLE IF NOT EXISTS decisions(
       key TEXT PRIMARY KEY, value TEXT NOT NULL,
       decided_at REAL NOT NULL, actor TEXT NOT NULL,
       base_version INTEGER, event_seq INTEGER)""",
    """CREATE TABLE IF NOT EXISTS branch_refs(
       branch TEXT PRIMARY KEY, version_id INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_src_kind ON sources(source_kind, ping)",
    "CREATE INDEX IF NOT EXISTS idx_der_ver ON derivations(version_id)",
    "CREATE INDEX IF NOT EXISTS idx_dep_dep ON depends(dep_stable_id)",
]


def connect(path: str) -> sqlite3.Connection:
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with _txn(conn):
        cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),))
            conn.execute("INSERT INTO versions(version_id,parent_version,created_at,label,kind)"
                         " VALUES(0,NULL,?,'baseline-empty','root')", (time.time(),))
            conn.execute("INSERT INTO branch_refs(branch,version_id) VALUES('main',0)")
            conn.execute("INSERT INTO meta(key,value) VALUES('current_version','0')")
        else:
            migrate(conn, int(row["value"]))


def migrate(conn: sqlite3.Connection, from_v: int) -> None:
    """旧版本迁移：单调升级并记录。模拟 v1->v2 增加 branch_refs。"""
    v = from_v
    if v < 2:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
        if "event_seq" not in cols:
            conn.execute("ALTER TABLE decisions ADD COLUMN event_seq INTEGER")
        conn.execute("""CREATE TABLE IF NOT EXISTS branch_refs(
                        branch TEXT PRIMARY KEY, version_id INTEGER NOT NULL)""")
        exists = conn.execute("SELECT 1 FROM branch_refs WHERE branch='main'").fetchone()
        if not exists:
            cur = conn.execute("SELECT value FROM meta WHERE key='current_version'")
            cv = int(cur.fetchone()["value"]) if cur.fetchone() else 0
            conn.execute("INSERT OR IGNORE INTO branch_refs(branch,version_id) VALUES('main',?)", (cv,))
        v = 2
    conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(v),))


@contextmanager
def _txn(conn: sqlite3.Connection):
    with _lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, path: str, conn: Optional[sqlite3.Connection] = None):
        self.path = path
        self.conn = conn or connect(path)

    # ---- 基础读 ----------------------------------------------------------- #
    def current_version(self) -> int:
        r = self.conn.execute("SELECT value FROM meta WHERE key='current_version'").fetchone()
        return int(r["value"])

    def schema_version(self) -> int:
        r = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(r["value"])

    def close(self):
        self.conn.close()

    # ---- 源数据（分开记录）----------------------------------------------- #
    def add_source(self, source_kind: str, payload: Dict[str, Any], *,
                   ping: Optional[str] = None, channel: Optional[str] = None,
                   bin_index: Optional[int] = None, units: Optional[str] = None,
                   supersedes: Optional[str] = None, stable_id: Optional[str] = None) -> str:
        sid = stable_id or content_stable_id(source_kind, payload)
        exists = self.conn.execute("SELECT 1 FROM sources WHERE stable_id=?", (sid,)).fetchone()
        if exists:
            return sid
        sup_uid = None
        if supersedes:
            r = self.conn.execute("SELECT uid FROM sources WHERE stable_id=?", (supersedes,)).fetchone()
            sup_uid = r["uid"] if r else None
        with _txn(self.conn):
            self.conn.execute(
                """INSERT INTO sources(stable_id,source_kind,ping,channel,bin_index,
                   payload,units,supersedes,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (sid, source_kind, ping, channel, bin_index,
                 json.dumps(payload, ensure_ascii=False), units, sup_uid, time.time()))
        return sid

    def get_source(self, stable_id: str) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM sources WHERE stable_id=?", (stable_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def sources_by_kind(self, kind: str, ping: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM sources WHERE source_kind=?"
        args: list = [kind]
        if ping is not None:
            q += " AND ping=?"
            args.append(ping)
        q += " ORDER BY uid"
        rows = [dict(r) for r in self.conn.execute(q, args)]
        for d in rows:
            d["payload"] = json.loads(d["payload"])
        return rows

    def all_sources(self) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM sources ORDER BY uid")]
        for d in rows:
            d["payload"] = json.loads(d["payload"])
        return rows

    # ---- 事件（机器可读取、追加）----------------------------------------- #
    def append_event(self, etype: str, actor: str, payload: Dict[str, Any], *,
                     subject: Optional[str] = None, base_version: Optional[int] = None,
                     new_version: Optional[int] = None, event_id: Optional[str] = None) -> int:
        eid = event_id or ("evt_" + uuid.uuid4().hex[:16])
        with _txn(self.conn):
            cur = self.conn.execute(
                """INSERT INTO events(event_id,at,actor,type,base_version,new_version,
                   subject,payload) VALUES(?,?,?,?,?,?,?,?)""",
                (eid, time.time(), actor, etype, base_version, new_version,
                 subject, json.dumps(payload, ensure_ascii=False)))
            return int(cur.lastrowid)

    def events(self) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM events ORDER BY seq")]
        for d in rows:
            d["payload"] = json.loads(d["payload"])
        return rows

    def event_at(self, seq: int) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d
