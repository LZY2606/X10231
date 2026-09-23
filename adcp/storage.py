from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(path: str | Path = "adcp_case.sqlite") -> sqlite3.Connection:
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def initialize(connection: sqlite3.Connection) -> None:
    current = connection.execute("PRAGMA user_version").fetchone()[0]
    if current not in (0, 1, SCHEMA_VERSION):
        raise RuntimeError(f"unsupported database schema version {current}")
    if current == SCHEMA_VERSION:
        return
    with transaction(connection):
        if current == 0:
            _create_schema(connection)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        else:
            _migrate_v1(connection)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE source_revisions (
            revision_id TEXT PRIMARY KEY,
            parent_revision_id TEXT,
            label TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('complete','abandoned')),
            UNIQUE(label, created_at)
        );
        CREATE TABLE sources (
            stable_id TEXT NOT NULL,
            revision_id TEXT NOT NULL REFERENCES source_revisions(revision_id),
            kind TEXT NOT NULL,
            ping_id TEXT,
            cell_id TEXT,
            beam INTEGER,
            sequence_no INTEGER NOT NULL,
            value_json TEXT NOT NULL,
            unit TEXT,
            datum TEXT,
            original_json TEXT NOT NULL,
            diagnostic TEXT,
            PRIMARY KEY (stable_id, revision_id)
        );
        CREATE INDEX sources_revision_kind ON sources(revision_id, kind);
        CREATE INDEX sources_ping ON sources(revision_id, ping_id);
        CREATE TABLE derivation_versions (
            version_id TEXT PRIMARY KEY,
            source_revision_id TEXT NOT NULL REFERENCES source_revisions(revision_id),
            parent_version_id TEXT REFERENCES derivation_versions(version_id),
            label TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('building','current','superseded','abandoned'))
        );
        CREATE TABLE current_pointer (
            scope TEXT PRIMARY KEY CHECK(scope='main'),
            version_id TEXT NOT NULL REFERENCES derivation_versions(version_id)
        );
        CREATE TABLE derived_nodes (
            node_id TEXT NOT NULL,
            version_id TEXT NOT NULL REFERENCES derivation_versions(version_id),
            ping_id TEXT,
            cell_id TEXT,
            stage TEXT NOT NULL,
            candidate_id TEXT,
            status TEXT NOT NULL,
            value_json TEXT NOT NULL,
            unit TEXT,
            coordinate_datum TEXT,
            fingerprint TEXT NOT NULL,
            reused_from_version TEXT REFERENCES derivation_versions(version_id),
            diagnostics_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY(version_id, node_id)
        );
        CREATE INDEX derived_version_stage ON derived_nodes(version_id, stage);
        CREATE INDEX derived_ping ON derived_nodes(version_id, ping_id);
        CREATE TABLE dependency_edges (
            version_id TEXT NOT NULL REFERENCES derivation_versions(version_id),
            node_id TEXT NOT NULL,
            source_stable_id TEXT,
            upstream_node_id TEXT,
            source_fingerprint TEXT NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY(version_id, node_id, role, COALESCE(source_stable_id,''), COALESCE(upstream_node_id,''))
        );
        CREATE INDEX edges_downstream ON dependency_edges(version_id, source_stable_id);
        CREATE INDEX edges_upstream ON dependency_edges(version_id, upstream_node_id);
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            source_revision_id TEXT,
            derivation_version_id TEXT,
            subject_id TEXT,
            payload_json TEXT NOT NULL,
            machine_judgment TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            base_version_id TEXT NOT NULL REFERENCES derivation_versions(version_id),
            actor TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('accept','reject','adjust')),
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            selected_candidate_id TEXT,
            rationale TEXT NOT NULL,
            created_at TEXT NOT NULL,
            superseded_by INTEGER REFERENCES decisions(decision_id),
            UNIQUE(case_id, branch_id, actor, subject_type, subject_id, superseded_by)
        );
        CREATE TABLE save_locks (
            lock_key TEXT PRIMARY KEY,
            holder TEXT NOT NULL,
            base_revision_id TEXT NOT NULL,
            acquired_at TEXT NOT NULL
        );
        """
    )


def _migrate_v1(connection: sqlite3.Connection) -> None:
    # V1 is the deliberately shipped old layout.  Data is copied, not mutated,
    # so an aborted migration still leaves the old database readable.
    connection.executescript(
        """
        ALTER TABLE derived_nodes RENAME TO derived_nodes_v1;
        CREATE TABLE derived_nodes (
            node_id TEXT NOT NULL,
            version_id TEXT NOT NULL REFERENCES derivation_versions(version_id),
            ping_id TEXT,
            cell_id TEXT,
            stage TEXT NOT NULL,
            candidate_id TEXT,
            status TEXT NOT NULL,
            value_json TEXT NOT NULL,
            unit TEXT,
            coordinate_datum TEXT,
            fingerprint TEXT NOT NULL,
            reused_from_version TEXT REFERENCES derivation_versions(version_id),
            diagnostics_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY(version_id, node_id)
        );
        INSERT INTO derived_nodes
        SELECT
            node_id, version_id, ping_id, cell_id, stage, candidate_id, status,
            value_json, unit, coordinate_datum, fingerprint, NULL, '[]'
        FROM derived_nodes_v1;
        DROP TABLE derived_nodes_v1;
        CREATE INDEX derived_version_stage ON derived_nodes(version_id, stage);
        CREATE INDEX derived_ping ON derived_nodes(version_id, ping_id);
        """
    )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(value: str) -> Any:
    return json.loads(value)


def record_event(
    connection: sqlite3.Connection,
    *,
    case_id: str,
    branch_id: str,
    event_type: str,
    actor: str,
    machine_judgment: str,
    payload: dict[str, Any],
    source_revision_id: str | None = None,
    derivation_version_id: str | None = None,
    subject_id: str | None = None,
    evidence: list[Any] | None = None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO events(case_id, branch_id, occurred_at, event_type, actor,
            source_revision_id, derivation_version_id, subject_id, payload_json,
            machine_judgment, evidence_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            case_id,
            branch_id,
            utc_now(),
            event_type,
            actor,
            source_revision_id,
            derivation_version_id,
            subject_id,
            dumps(payload),
            machine_judgment,
            dumps(evidence or []),
        ),
    )
    return int(cursor.lastrowid)


def export_database(connection: sqlite3.Connection, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    backup = sqlite3.connect(str(temporary))
    try:
        connection.backup(backup)
        backup.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        backup.close()
    temporary.replace(destination)
    return destination


def import_database(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with sqlite3.connect(str(source_path)) as probe:
        version = probe.execute("PRAGMA user_version").fetchone()[0]
        if version not in (1, SCHEMA_VERSION):
            raise RuntimeError(f"unsupported import schema version {version}")
    destination_path = Path(destination)
    temporary = destination_path.with_suffix(".import-tmp")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(temporary) + suffix)
        if candidate.exists():
            candidate.unlink()
    shutil.copy2(source_path, temporary)
    imported = connect(temporary)
    try:
        initialize(imported)
        imported.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        imported.close()
    except Exception:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(temporary) + suffix)
            if candidate.exists():
                candidate.unlink()
        raise
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(destination_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    temporary.replace(destination_path)


def begin_save(
    connection: sqlite3.Connection,
    lock_key: str,
    holder: str,
    base_revision_id: str,
) -> None:
    try:
        with transaction(connection):
            connection.execute(
                """
                INSERT INTO save_locks(lock_key, holder, base_revision_id, acquired_at)
                VALUES(?,?,?,?)
                """,
                (lock_key, holder, base_revision_id, utc_now()),
            )
    except sqlite3.IntegrityError as exc:
        row = connection.execute(
            "SELECT holder, base_revision_id FROM save_locks WHERE lock_key=?",
            (lock_key,),
        ).fetchone()
        raise ConcurrentSave(
            f"{holder} conflicts with {row['holder']} editing from {row['base_revision_id']}",
            conflicting_holder=row["holder"],
            base_revision_id=row["base_revision_id"],
        ) from exc


def finish_save(connection: sqlite3.Connection, lock_key: str) -> None:
    with transaction(connection):
        connection.execute("DELETE FROM save_locks WHERE lock_key=?", (lock_key,))


class ConcurrentSave(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        conflicting_holder: str,
        base_revision_id: str,
    ) -> None:
        super().__init__(message)
        self.conflicting_holder = conflicting_holder
        self.base_revision_id = base_revision_id
