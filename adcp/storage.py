from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 2


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def connect(path: str = "adcp.db") -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA busy_timeout=30000")
    migrate(db)
    return db


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        db.execute("ROLLBACK")
        raise
    else:
        db.execute("COMMIT")


def migrate(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    row = db.execute("SELECT value FROM schema_meta WHERE key=version").fetchone()
    version = int(row["value"]) if row else 1
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {version} is newer than code {SCHEMA_VERSION}")
    if version < 2:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases(
              case_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT , current_release_id TEXT
            );
            CREATE TABLE IF NOT EXISTS sources(
              record_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, stream TEXT NOT NULL,
              subject_id TEXT NOT NULL, version INTEGER NOT NULL, supersedes_id TEXT,
              valid_from_s REAL, valid_to_s REAL, created_event_id TEXT,
              raw_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, seq INTEGER NOT NULL,
              at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL,
              base_release_id TEXT, payload_json TEXT NOT NULL, result_release_id TEXT,
              status TEXT NOT NULL, conflict_with_event_id TEXT, resolution_event_id TEXT
            );
            CREATE TABLE IF NOT EXISTS releases(
              release_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, parent_release_id TEXT,
              label TEXT NOT NULL, event_id TEXT NOT NULL, created_at TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 0, fingerprint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes(
              node_uid TEXT PRIMARY KEY, release_id TEXT NOT NULL, stable_node_id TEXT NOT NULL,
              node_type TEXT NOT NULL, subject_id TEXT NOT NULL, status TEXT NOT NULL,
              fingerprint TEXT NOT NULL, value_json TEXT NOT NULL, diagnostics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS node_edges(
              release_id TEXT NOT NULL, from_node_uid TEXT NOT NULL, to_node_uid TEXT NOT NULL,
              role TEXT NOT NULL, changed INTEGER NOT NULL, reused_from_uid TEXT,
              reason TEXT NOT NULL, PRIMARY KEY(release_id, from_node_uid, to_node_uid, role)
            );
            """
        )
        existing = {r[0] for r in db.execute("PRAGMA table_info(sources)")}
        if "created_event_id" not in existing:
            db.execute("ALTER TABLE sources ADD COLUMN created_event_id TEXT")
        db.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES(version,2)"
        )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(text: str | bytes) -> Any:
    return json.loads(text)


def rowdict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def get_case(db: sqlite3.Connection, case_id: str) -> dict[str, Any] | None:
    return rowdict(db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())


def list_cases(db: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in db.execute("SELECT * FROM cases ORDER BY created_at").fetchall()]


def event_rows(db: sqlite3.Connection, case_id: str) -> list[sqlite3.Row]:
    return db.execute(
        "SELECT * FROM events WHERE case_id=? ORDER BY seq, event_id", (case_id,)
    ).fetchall()


def release_nodes(db: sqlite3.Connection, release_id: str) -> list[sqlite3.Row]:
    return db.execute(
        "SELECT * FROM nodes WHERE release_id=? ORDER BY stable_node_id", (release_id,)
    ).fetchall()


def release_node_by_stable(db: sqlite3.Connection, release_id: str, stable_node_id: str) -> dict[str, Any] | None:
    return rowdict(
        db.execute(
            "SELECT * FROM nodes WHERE release_id=? AND stable_node_id=?",
            (release_id, stable_node_id),
        ).fetchone()
    )


def node_by_uid(db: sqlite3.Connection, uid: str) -> dict[str, Any] | None:
    return rowdict(db.execute("SELECT * FROM nodes WHERE node_uid=?", (uid,)).fetchone())


def source_rows(db: sqlite3.Connection, case_id: str, include_inactive: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM sources WHERE case_id=?"
    if not include_inactive:
        sql += " AND active=1"
    return db.execute(sql + " ORDER BY stream, subject_id, version", (case_id,)).fetchall()


def source_by_id(db: sqlite3.Connection, record_id: str) -> dict[str, Any] | None:
    return rowdict(db.execute("SELECT * FROM sources WHERE record_id=?", (record_id,)).fetchone())


def edges_for(db: sqlite3.Connection, release_id: str, node_uid: str) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in db.execute(
            "SELECT * FROM node_edges WHERE release_id=? AND to_node_uid=? ORDER BY role,from_node_uid",
            (release_id, node_uid),
        ).fetchall()
    ]


def edge_chains(db: sqlite3.Connection, release_id: str) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in db.execute(
            "SELECT * FROM node_edges WHERE release_id=? ORDER BY to_node_uid,role,from_node_uid",
            (release_id,),
        ).fetchall()
    ]


def export_case(db: sqlite3.Connection, case_id: str) -> dict[str, Any]:
    case = get_case(db, case_id)
    if not case:
        raise KeyError(case_id)
    releases = [dict(r) for r in db.execute("SELECT * FROM releases WHERE case_id=?", (case_id,)).fetchall()]
    release_ids = [r["release_id"] for r in releases]
    nodes, edges = [], []
    for rid in release_ids:
        nodes.extend([dict(r) for r in release_nodes(db, rid)])
        edges.extend(
            dict(r)
            for r in db.execute("SELECT * FROM node_edges WHERE release_id=?", (rid,)).fetchall()
        )
    return {
        "format": "adcp-evidence-case/v2",
        "exported_at": now_id(),
        "case": case,
        "sources": [dict(r) for r in source_rows(db, case_id, include_inactive=True)],
        "events": [dict(r) for r in event_rows(db, case_id)],
        "releases": releases,
        "nodes": nodes,
        "edges": edges,
    }


def import_case(db: sqlite3.Connection, package: dict[str, Any], *, replace: bool = False) -> str:
    if package.get("format") != "adcp-evidence-case/v2":
        raise ValueError("unsupported case package format")
    case = package["case"]
    case_id = case["case_id"]
    with transaction(db):
        existing = get_case(db, case_id)
        if existing and not replace:
            raise ValueError(f"case already exists: {case_id}")
        if existing:
            for table in ("node_edges", "nodes", "releases", "events", "sources", "cases"):
                db.execute(f"DELETE FROM {table} WHERE case_id=?" if table != "cases" else "DELETE FROM cases WHERE case_id=?", (case_id,))
        db.execute(
            "INSERT INTO cases VALUES(:case_id,:title,:created_at,:description,:current_release_id)",
            case,
        )
        _insert_many(db, "sources", package["sources"])
        _insert_many(db, "events", package["events"])
        _insert_many(db, "releases", package["releases"])
        _insert_many(db, "nodes", package["nodes"])
        _insert_many(db, "node_edges", package["edges"])
    return case_id


def _insert_many(db: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {table}({,.join(columns)}) VALUES({placeholders})"
    db.executemany(sql, [[row.get(col) for col in columns] for row in rows])


def atomic_database_path() -> str:
    return os.environ.get("ADCP_DB", str(Path.cwd() / "adcp.db"))

