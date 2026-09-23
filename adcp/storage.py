"""SQLite 存储层:分源记录、版本化派生、原子切换、冲突检测、导出/导入。

- 波束测量 / correlation / amplitude / 坏 beam 标志 / heading/pitch/roll /
  磁偏角 / 安装角分别作为独立 source 行存储,保留原始表示(payload.raw)。
- 每次推导写入新 version;只有 commit_version 才在同一个事务里
  标记 complete 并切换 current 指针。写入中断后重启,building 版本被回滚,
  只能看到旧版本或完整新版本。
- 两人从同一版本编辑:save_decision 校验 base_version,不一致抛 Conflict,
  可通过 resolve_conflict 以 rebase 方式解决,未选候选保留。
"""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS versions (
  id INTEGER PRIMARY KEY, parent INTEGER, label TEXT, status TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY, version INTEGER, kind TEXT, ping INTEGER, beam INTEGER, cell INTEGER,
  identity TEXT, payload TEXT, origin TEXT);
CREATE TABLE IF NOT EXISTS derivations (
  id INTEGER PRIMARY KEY, version INTEGER, key TEXT, status TEXT, fingerprint TEXT,
  payload TEXT, deps TEXT);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, version INTEGER, kind TEXT, payload TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, version INTEGER, base_version INTEGER, branch TEXT,
  chosen TEXT, candidates TEXT, created_at REAL);
"""

TABLES = ["versions", "sources", "derivations", "events", "decisions"]


class Conflict(Exception):
    """两个使用者从同一版本编辑产生的可解决冲突。"""

    def __init__(self, base_version, current_version):
        self.base_version = base_version
        self.current_version = current_version
        super().__init__(f"base version {base_version} != current {current_version}")


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        fresh = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone() is None
        if fresh:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT INTO meta(key,value) VALUES ('schema_version',?)",
                              (str(SCHEMA_VERSION),))
            self.conn.execute("INSERT INTO meta(key,value) VALUES ('current_version','')")
            self.conn.commit()
        else:
            self._migrate()
        self._cleanup_interrupted()

    # ---------- schema / migration ----------
    def _migrate(self):
        v = self.get_meta("schema_version")
        v = int(v) if v else 1
        if v < 2:
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(sources)")]
            if "identity" not in cols:
                self.conn.execute("ALTER TABLE sources ADD COLUMN identity TEXT")
            if "origin" not in cols:
                self.conn.execute("ALTER TABLE sources ADD COLUMN origin TEXT")
            self.set_meta("schema_version", "2")
            self.conn.commit()

    def _cleanup_interrupted(self):
        """崩溃恢复:回滚所有未完成的 building 版本。"""
        stale = [r[0] for r in self.conn.execute(
            "SELECT id FROM versions WHERE status='building'")]
        for vid in stale:
            self._delete_version(vid)
        if stale:
            self.conn.commit()

    def _delete_version(self, vid):
        for t in ("sources", "derivations", "events", "decisions"):
            self.conn.execute(f"DELETE FROM {t} WHERE version=?", (vid,))
        self.conn.execute("DELETE FROM versions WHERE id=?", (vid,))

    # ---------- meta ----------
    def get_meta(self, key):
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def current_version(self):
        v = self.get_meta("current_version")
        return int(v) if v else None

    # ---------- version lifecycle ----------
    def begin_version(self, parent, label):
        cur = self.conn.execute(
            "INSERT INTO versions(parent,label,status,created_at) VALUES (?,?,?,?)",
            (parent, label, "building", time.time()))
        self.conn.commit()
        return cur.lastrowid

    def commit_version(self, vid):
        """原子切换:同事务标记 complete + 移动 current 指针。"""
        with self.conn:
            self.conn.execute("UPDATE versions SET status='complete' WHERE id=?", (vid,))
            self.set_meta("current_version", str(vid))

    def abort_version(self, vid):
        with self.conn:
            self._delete_version(vid)

    # ---------- rows ----------
    def add_source(self, version, kind, ping, beam, cell, identity, payload, origin=""):
        self.conn.execute(
            "INSERT INTO sources(version,kind,ping,beam,cell,identity,payload,origin)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (version, kind, ping, beam, cell, identity,
             json.dumps(payload, ensure_ascii=False, sort_keys=True), origin))

    def add_derivation(self, version, key, status, fingerprint, payload, deps):
        self.conn.execute(
            "INSERT INTO derivations(version,key,status,fingerprint,payload,deps)"
            " VALUES (?,?,?,?,?,?)",
            (version, key, status, fingerprint,
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             json.dumps(sorted(deps))))

    def add_event(self, version, kind, payload):
        cur = self.conn.execute(
            "INSERT INTO events(version,kind,payload,created_at) VALUES (?,?,?,?)",
            (version, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True),
             time.time()))
        return cur.lastrowid

    # ---------- queries ----------
    def _rows(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args)]

    def versions(self):
        return self._rows("SELECT * FROM versions ORDER BY id")

    def sources(self, version):
        rows = self._rows("SELECT * FROM sources WHERE version=? ORDER BY id", (version,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def derivations(self, version):
        rows = self._rows("SELECT * FROM derivations WHERE version=? ORDER BY key", (version,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
            r["deps"] = json.loads(r["deps"])
        return rows

    def derivation(self, version, key):
        rows = self._rows(
            "SELECT * FROM derivations WHERE version=? AND key=?", (version, key))
        if not rows:
            return None
        r = rows[0]
        r["payload"] = json.loads(r["payload"])
        r["deps"] = json.loads(r["deps"])
        return r

    def events(self, version=None):
        if version is None:
            rows = self._rows("SELECT * FROM events ORDER BY id")
        else:
            rows = self._rows("SELECT * FROM events WHERE version=? ORDER BY id", (version,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def event(self, event_id):
        rows = self._rows("SELECT * FROM events WHERE id=?", (event_id,))
        if not rows:
            return None
        rows[0]["payload"] = json.loads(rows[0]["payload"])
        return rows[0]

    def decisions(self, version=None):
        if version is None:
            rows = self._rows("SELECT * FROM decisions ORDER BY id")
        else:
            rows = self._rows(
                "SELECT * FROM decisions WHERE version=? ORDER BY id", (version,))
        for r in rows:
            r["candidates"] = json.loads(r["candidates"])
        return rows

    # ---------- decisions / conflict ----------
    def save_decision(self, base_version, branch, chosen, candidates, label=None):
        """从 base_version 保存人工决定;base 落后则抛 Conflict。"""
        current = self.current_version()
        if base_version != current:
            self.add_event(current, "conflict", {
                "base_version": base_version, "current_version": current,
                "branch": branch, "chosen": chosen,
                "reason": "base version is stale; decision rejected for rebase",
            })
            self.conn.commit()
            raise Conflict(base_version, current)
        vid = self.begin_version(current, label or f"decision:{chosen}")
        self.conn.execute(
            "INSERT INTO decisions(version,base_version,branch,chosen,candidates,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (vid, base_version, branch, chosen,
             json.dumps(candidates, ensure_ascii=False), time.time()))
        self.add_event(vid, "decision", {
            "branch": branch, "chosen": chosen, "candidates": candidates,
            "base_version": base_version,
        })
        return vid

    def resolve_conflict(self, branch, chosen, candidates, strategy="rebase"):
        """以 rebase 方式解决冲突:在当前版本之上重放决定,未选候选保留。"""
        current = self.current_version()
        vid = self.begin_version(current, f"conflict-resolution:{chosen}")
        self.conn.execute(
            "INSERT INTO decisions(version,base_version,branch,chosen,candidates,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (vid, current, branch, chosen,
             json.dumps(candidates, ensure_ascii=False), time.time()))
        self.add_event(vid, "conflict_resolved", {
            "strategy": strategy, "branch": branch, "chosen": chosen,
            "kept_candidates": candidates, "rebased_onto": current,
        })
        return vid

    # ---------- export / import ----------
    def export_archive(self):
        dump = {"schema_version": int(self.get_meta("schema_version")),
                "meta": {r[0]: r[1] for r in self.conn.execute("SELECT key,value FROM meta")},
                "tables": {}}
        for t in TABLES:
            dump["tables"][t] = self._rows(f"SELECT * FROM {t} ORDER BY id")
        return dump

    def export_to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_archive(), f, ensure_ascii=False, indent=1)

    def import_archive(self, dump):
        """导入到空库,保留全部 id / 顺序 / 来源关系。"""
        if self.versions():
            raise ValueError("import requires an empty database")
        with self.conn:
            for t in TABLES:
                rows = dump["tables"].get(t, [])
                for row in rows:
                    cols = list(row.keys())
                    self.conn.execute(
                        f"INSERT INTO {t}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [row[c] for c in cols])
            for k, v in dump.get("meta", {}).items():
                self.set_meta(k, v)

    def close(self):
        self.conn.close()
