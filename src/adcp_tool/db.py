"""SQLite 访问：原子版本指针、启动崩溃恢复、导出/导入、旧库迁移。"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

_TABLES = [
    "meta", "source_batch", "ping", "cell",
    "beam_velocity", "beam_correlation", "beam_amplitude", "beam_bad_flag",
    "bottom_track", "attitude_sample", "config_blob",
    "node", "node_dep", "version_stage", "current_pointer",
    "version_slot", "event", "candidate", "edit_conflict",
]


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _exec_script(conn: sqlite3.Connection) -> None:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())


def init_db(conn: sqlite3.Connection) -> Dict[str, Any]:
    """初始化或迁移旧库；并执行启动崩溃恢复。

    保证：崩溃中断后重启，current 指针要么指向旧版本，要么指向完整新版本，
    绝不会指向只写了一半的暂存版本。
    """
    _exec_script(conn)
    report = {"migrated_from": None, "abandoned_staged": [], "current": None}

    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
    else:
        old = int(row.value)
        if old < SCHEMA_VERSION:
            report["migrated_from"] = old
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

    # 崩溃恢复：所有仍处于 staged 的版本都不是完整新版本 -> 废弃，保留旧指针
    for r in conn.execute(
        "SELECT version_id FROM version_stage WHERE state='staged'"
    ).fetchall():
        conn.execute(
            "UPDATE version_stage SET state='abandoned' WHERE version_id=?",
            (r["version_id"],),
        )
        report["abandoned_staged"].append(r["version_id"])

    cur = conn.execute(
        "SELECT version_id FROM current_pointer WHERE only_id=1"
    ).fetchone()
    if cur is not None:
        report["current"] = cur["version_id"]
    return report


# ---------------- 导出 / 导入 ----------------

def export_all(conn: sqlite3.Connection) -> Dict[str, Any]:
    """导出全库为 JSON bundle；身份、顺序、来源关系全部保留。"""
    bundle: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "tables": {}}
    for table in _TABLES:
        cols = [r["name"] for r in conn.execute(
            f"PRAGMA table_info({table})").fetchall()]
        order = "seq" if "seq" in cols else (
            "rowid")
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM {table} ORDER BY {order}").fetchall()]
        bundle["tables"][table] = {"columns": cols, "rows": rows}
    bundle["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return bundle


def import_bundle(conn: sqlite3.Connection, bundle: Dict[str, Any],
                  replace: bool = True) -> Dict[str, Any]:
    """导入 bundle。默认整库替换（用于删除本地数据库后导入）。

    显式 ID 原样写回，保证身份、顺序、来源关系不变。
    """
    if replace:
        for table in reversed(_TABLES):
            conn.execute(f"DELETE FROM {table}")
    counts = {}
    for table in _TABLES:
        section = bundle.get("tables", {}).get(table)
        if not section:
            counts[table] = 0
            continue
        cols = section["columns"]
        placeholders = ",".join("?" for _ in cols)
        collist = ",".join(cols)
        n = 0
        for row in section["rows"]:
            values = [row.get(c) for c in cols]
            conn.execute(
                f"INSERT INTO {table}({collist}) VALUES({placeholders})", values)
            n += 1
        counts[table] = n
    return {"imported_rows": counts}
