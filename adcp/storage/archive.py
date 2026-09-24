"""Whole-database export/import.

The archive is a single JSON document containing every table row. IDs are
stable content hashes, events keep their sequence, and provenance
references (``parent_id``, ``based_on_version``, ``inputs``, ``edges``)
resolve identically after import into an empty database.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .schema import initialize


TABLES = ("meta", "sources", "config_versions", "versions", "events",
          "decisions", "edges", "derived_nodes")


def export_to_dict(conn: sqlite3.Connection) -> dict[str, Any]:
    def rows(table: str) -> list[dict]:
        return [dict(r) for r in conn.execute(
            f"SELECT * FROM {table} ORDER BY 1"
            if table in {"meta", "sources", "config_versions", "versions",
                         "decisions"}
            else (f"SELECT * FROM {table} ORDER BY seq"
                  if table == "events" else f"SELECT * FROM {table}")
        ).fetchall()]

    return {"archive_version": 1, "tables": {t: rows(t) for t in TABLES}}


def export_file(conn: sqlite3.Connection, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export_to_dict(conn), fh, ensure_ascii=False, indent=1,
                  sort_keys=True)


def import_dict(conn: sqlite3.Connection, archive: dict[str, Any],
                *, allow_nonempty: bool = False) -> None:
    existing = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()[0]
    if existing:
        count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        if count and not allow_nonempty:
            raise ValueError("refusing to import into a non-empty database")
    else:
        initialize(conn)
    tables = archive["tables"]
    # Order matters for readability, not for SQLite (FK checks only between
    # existing committed rows at statement time; we postpone enforcement).
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in TABLES:
        for row in tables.get(table, []):
            keys = list(row.keys())
            placeholders = ",".join("?" for _ in keys)
            conn.execute(
                f"INSERT OR REPLACE INTO {table}({','.join(keys)}) "
                f"VALUES({placeholders})",
                [row[k] for k in keys],
            )
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()


def import_file(path_db: str, path_archive: str) -> None:
    with open(path_archive, encoding="utf-8") as fh:
        archive = json.load(fh)
    conn = sqlite3.connect(path_db)
    conn.row_factory = sqlite3.Row
    try:
        import_dict(conn, archive)
    finally:
        conn.close()
