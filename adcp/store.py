"""SQLite 版本化存储层。

语义:
- 每类来源(波束速度/correlation/amplitude/坏beam标志/姿态/磁偏角/安装角...)分开记录。
- 每次推导产生独立版本; 只有完整成功后才切换 current 指针(单事务)。
- 写入中断后重启: 要么看到旧版本, 要么看到完整新版本(building 版本被回收)。
- 两个使用者从同一版本编辑: 后提交者得到可解决的冲突, 双方分支都保留。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS versions(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE, parent_id TEXT, label TEXT, status TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS sources(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, kind TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS derived(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, kind TEXT, slot TEXT,
    fingerprint TEXT, inputs TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, type TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS pointer(name TEXT PRIMARY KEY, value TEXT);
"""

# 旧版 schema(v1): 无 events 表, derived 无 fingerprint/inputs 列。
V1_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE versions(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE, parent_id TEXT, label TEXT, status TEXT, payload TEXT);
CREATE TABLE sources(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, kind TEXT, payload TEXT);
CREATE TABLE derived(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, kind TEXT, slot TEXT, payload TEXT);
CREATE TABLE pointer(name TEXT PRIMARY KEY, value TEXT);
"""

MIGRATE_1_2 = """
ALTER TABLE derived ADD COLUMN fingerprint TEXT;
ALTER TABLE derived ADD COLUMN inputs TEXT;
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT, version_id TEXT, type TEXT, payload TEXT);
"""


class ConflictError(Exception):
    """base 版本已不再是 current: 提交被拒绝, 编辑方版本保留为 branch。"""

    def __init__(self, current, base, vid):
        super().__init__(f"conflict: current={current} base={base} vid={vid}")
        self.current = current
        self.base = base
        self.vid = vid


def content_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.fail_hook = None  # 测试用: 在 commit 事务中途抛错, 模拟写入中断
        self._init_or_migrate()

    def close(self):
        self.conn.close()

    def _init_or_migrate(self):
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self.conn.commit()
        else:
            version = int(self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"数据库 schema v{version} 比程序新, 无法打开")
            while version < SCHEMA_VERSION:
                version = self._migrate(version)
        self.recover()

    def _migrate(self, version):
        if version == 1:
            self.conn.executescript(MIGRATE_1_2)
            self.conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            self.conn.commit()
            return 2
        raise RuntimeError(f"未知 schema 版本 {version}")

    def recover(self):
        """回收中断的 building 版本: 重启后只能看到旧版本或完整新版本。"""
        building = [r[0] for r in self.conn.execute(
            "SELECT id FROM versions WHERE status='building'").fetchall()]
        for vid in building:
            for table in ("sources", "derived", "events"):
                self.conn.execute(f"DELETE FROM {table} WHERE version_id=?", (vid,))
            self.conn.execute("DELETE FROM versions WHERE id=?", (vid,))
        self.conn.commit()

    # ---- 版本与指针 ----

    def current(self):
        row = self.conn.execute(
            "SELECT value FROM pointer WHERE name='current'").fetchone()
        return row[0] if row else None

    def _next_id(self, table, prefix):
        n = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return f"{prefix}{n + 1:05d}"

    def new_version(self, parent_id, label, payload=None):
        vid = self._next_id("versions", "V")
        self.conn.execute(
            "INSERT INTO versions(id, parent_id, label, status, payload)"
            " VALUES(?,?,?,?,?)",
            (vid, parent_id, label, "building",
             json.dumps(payload or {}, ensure_ascii=False)))
        self.conn.commit()
        return vid

    def commit(self, vid, base):
        """单事务: 校验 base -> 标记 complete -> 切换 current 指针。

        中途任何失败(含进程崩溃)都会整体回滚, 指针保持旧版本。
        """
        try:
            self.conn.execute("BEGIN")
            cur = self.current()
            if cur != base:
                raise ConflictError(cur, base, vid)
            if self.fail_hook is not None:
                self.fail_hook()
            self.conn.execute(
                "UPDATE versions SET status='complete' WHERE id=?", (vid,))
            self.conn.execute(
                "INSERT INTO pointer(name, value) VALUES('current', ?)"
                " ON CONFLICT(name) DO UPDATE SET value=excluded.value", (vid,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def mark_branch(self, vid):
        self.conn.execute("UPDATE versions SET status='branch' WHERE id=?", (vid,))
        self.conn.commit()

    def resolve(self, chosen_vid, retained_vid, user="system"):
        """冲突解决: 指针指向被采纳版本, 未采纳分支保留。"""
        self.conn.execute(
            "UPDATE versions SET status='complete' WHERE id=?", (chosen_vid,))
        self.conn.execute(
            "INSERT INTO pointer(name, value) VALUES('current', ?)"
            " ON CONFLICT(name) DO UPDATE SET value=excluded.value", (chosen_vid,))
        self.conn.commit()
        self.add_event(chosen_vid, "conflict.resolved", {
            "chosen": chosen_vid, "retained": retained_vid, "user": user,
            "note": "未采纳分支保留在档案中, 不覆盖"})

    # ---- 记录写入 ----

    def add_source(self, vid, sid, kind, payload):
        self.conn.execute(
            "INSERT INTO sources(id, version_id, kind, payload) VALUES(?,?,?,?)",
            (sid, vid, kind, json.dumps(payload, ensure_ascii=False)))
        self.conn.commit()

    def add_derived(self, vid, rid, kind, slot, fingerprint, inputs, payload):
        self.conn.execute(
            "INSERT INTO derived(id, version_id, kind, slot, fingerprint, inputs, payload)"
            " VALUES(?,?,?,?,?,?,?)",
            (rid, vid, kind, slot, fingerprint,
             json.dumps(inputs, ensure_ascii=False),
             json.dumps(payload, ensure_ascii=False)))
        self.conn.commit()

    def add_event(self, vid, etype, payload):
        eid = self._next_id("events", "E")
        self.conn.execute(
            "INSERT INTO events(id, version_id, type, payload) VALUES(?,?,?,?)",
            (eid, vid, etype, json.dumps(payload, ensure_ascii=False)))
        self.conn.commit()
        return eid

    # ---- 读取 ----

    def _rows(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def versions(self):
        rows = self._rows("SELECT * FROM versions ORDER BY seq")
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    def version_payload(self, vid):
        row = self.conn.execute(
            "SELECT payload FROM versions WHERE id=?", (vid,)).fetchone()
        return json.loads(row[0] or "{}") if row else {}

    def sources(self, vid):
        rows = self._rows(
            "SELECT * FROM sources WHERE version_id=? ORDER BY seq", (vid,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def derived(self, vid):
        rows = self._rows(
            "SELECT * FROM derived WHERE version_id=? ORDER BY seq", (vid,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
            r["inputs"] = json.loads(r["inputs"])
        return rows

    def events(self, vid=None):
        if vid is None:
            rows = self._rows("SELECT * FROM events ORDER BY seq")
        else:
            rows = self._rows(
                "SELECT * FROM events WHERE version_id=? ORDER BY seq", (vid,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # ---- 导出 / 导入(身份、顺序、来源关系保持) ----

    def export(self):
        tables = {}
        for table in ("versions", "sources", "derived", "events", "pointer"):
            tables[table] = self._rows(f"SELECT * FROM {table} ORDER BY seq" if table != "pointer"
                                       else f"SELECT * FROM {table}")
        return {"schema_version": SCHEMA_VERSION, "tables": tables}

    def import_(self, data):
        if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError("导出文件 schema 版本不匹配")
        tables = data["tables"]
        with self.conn:
            for table in ("sources", "derived", "events", "versions", "pointer"):
                self.conn.execute(f"DELETE FROM {table}")
            for table in ("versions", "sources", "derived", "events", "pointer"):
                for row in tables.get(table, []):
                    cols = list(row.keys())
                    self.conn.execute(
                        f"INSERT INTO {table}({','.join(cols)})"
                        f" VALUES({','.join('?' for _ in cols)})",
                        [row[c] for c in cols])
        self.recover()
        return {"versions": len(tables.get("versions", []))}
