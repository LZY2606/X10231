"""导出/导入整库快照。内容寻址 ID 使身份、顺序、来源关系跨库保持。"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict

from .identity import canonical_json
from . import storage

_TABLES = ["meta", "sources", "source_current", "derived", "pointers",
           "state_events", "decisions", "conflicts", "branches"]


def export_db(conn: sqlite3.Connection) -> Dict:
    snap = {"format": "flowbench-adcp", "format_version": 1, "tables": {}}
    for table in _TABLES:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        snap["tables"][table] = [dict(r) for r in rows]
    return snap


def to_json(snap: Dict) -> str:
    return canonical_json(snap)


def import_db(conn: sqlite3.Connection, snap: Dict) -> None:
    """导入到空库；不重新生成任何 ID/seq，保证身份与全序不变。"""
    tables = snap["tables"]
    with conn:
        conn.executescript(storage.SCHEMA)
        for table in _TABLES:
            rows = tables.get(table, [])
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" for _ in cols)
            sql = f"INSERT OR IGNORE INTO {table}({','.join(cols)}) VALUES({placeholders})"
            conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        # 自增序列对齐，导入后新事件的 seq 继续向后
        ev_rows = tables.get("state_events", [])
        if ev_rows:
            max_seq = max(r["seq"] for r in ev_rows)
            conn.execute("INSERT OR REPLACE INTO sqlite_sequence(name,seq) VALUES('state_events',?)",
                         (max_seq,))
