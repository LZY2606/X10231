"""旧版本数据库迁移。仓库测试内置一份手工构造的 schema v1 库。"""
from __future__ import annotations

import sqlite3

from . import storage


def migrate(conn: sqlite3.Connection, from_version: int) -> None:
    if from_version < 2:
        # v1: sources 没有 replaced_by 链，derived 用 lineage_hash 而非 lineage。
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sources)").fetchall()}
        if "replaced_by" not in cols:
            conn.execute("ALTER TABLE sources ADD COLUMN replaced_by TEXT")
        dcols = {r[1] for r in conn.execute("PRAGMA table_info(derived)").fetchall()}
        if "lineage_hash" in dcols and "lineage" not in dcols:
            conn.execute("ALTER TABLE derived RENAME TO derived_v1")
            conn.execute("""CREATE TABLE derived (
                id TEXT PRIMARY KEY, node_key TEXT NOT NULL, lineage TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT NOT NULL,
                created_event TEXT NOT NULL, created_at TEXT NOT NULL)""")
            conn.execute(
                "INSERT INTO derived(id,node_key,lineage,status,result,created_event,created_at)"
                " SELECT id,node_key,lineage_hash,status,result,created_event,created_at"
                " FROM derived_v1")
            conn.execute("DROP TABLE derived_v1")
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if cur is None or int(cur[0]) < storage.SCHEMA_VERSION:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                     (str(storage.SCHEMA_VERSION),))
