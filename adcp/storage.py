from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_id(prefix: str, value: Any) -> str:
    return prefix + "_" + sha256_text(canonical_json(value))[:16]


def connect(path: str = ":memory:") -> sqlite3.Connection:
    if path != ":memory:":
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


SCHEMA = r"""
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_entity(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  stable_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(kind, stable_key)
);
CREATE TABLE IF NOT EXISTS source_revision(
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES source_entity(id),
  revision_no INTEGER NOT NULL,
  recorded_at TEXT NOT NULL,
  author TEXT NOT NULL,
  raw_value TEXT NOT NULL,
  normalized_value TEXT NOT NULL,
  unit_or_basis TEXT,
  status TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  UNIQUE(entity_id, revision_no)
);
CREATE TABLE IF NOT EXISTS event(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  derivation_version TEXT,
  before_version TEXT,
  checksum TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS derivation_version(
  id TEXT PRIMARY KEY,
  parent_version TEXT,
  created_at TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  event_id TEXT
);
CREATE TABLE IF NOT EXISTS node_revision(
  id TEXT PRIMARY KEY,
  node_kind TEXT NOT NULL,
  stable_key TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  value TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS version_node(
  version_id TEXT NOT NULL REFERENCES derivation_version(id),
  node_id TEXT NOT NULL REFERENCES node_revision(id),
  PRIMARY KEY(version_id, node_id)
);
CREATE TABLE IF NOT EXISTS node_dependency(
  node_id TEXT NOT NULL REFERENCES node_revision(id),
  dependency_id TEXT NOT NULL,
  role TEXT NOT NULL,
  PRIMARY KEY(node_id, dependency_id, role)
);
CREATE TABLE IF NOT EXISTS decision(
  subject_id TEXT NOT NULL,
  base_version TEXT NOT NULL,
  actor TEXT NOT NULL,
  choice TEXT NOT NULL,
  rationale TEXT NOT NULL,
  event_id TEXT NOT NULL,
  resolved_conflict_event_id TEXT,
  PRIMARY KEY(subject_id, base_version, actor)
);
CREATE TABLE IF NOT EXISTS current_pointer(
  id INTEGER PRIMARY KEY CHECK(id=1),
  version_id TEXT NOT NULL REFERENCES derivation_version(id)
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        conn.commit()
    elif int(row["value"]) < SCHEMA_VERSION:
        migrate(conn, int(row["value"]))
    elif int(row["value"]) > SCHEMA_VERSION:
        raise RuntimeError("database schema is newer than application")


def migrate(conn: sqlite3.Connection, old_version: int) -> None:
    if old_version == 1:
        # The offline v1 fixture intentionally stores raw and normalized values
        # together. Migration splits those fields without changing source ids.
        conn.executescript(SCHEMA)
        legacy_rows = conn.execute("SELECT * FROM legacy_measurement").fetchall()
        for row in legacy_rows:
            payload = json.loads(row["payload"])
            entity_id = row["entity_id"]
            rev_id = row["revision_id"]
            ensure_source(
                conn,
                entity_id=entity_id,
                kind=row["kind"],
                stable_key=row["stable_key"],
                raw_value=payload.get("raw", payload),
                normalized_value=payload.get("normalized", payload),
                unit_or_basis=payload.get("unit_or_basis"),
                author=row["author"],
                recorded_at=row["recorded_at"],
                status="accepted",
                diagnostics=[],
                revision_id=rev_id,
                revision_no=row["revision_no"],
            )
        conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        conn.commit()
        return
    raise RuntimeError("unsupported migration from schema %s" % old_version)


def create_legacy_v1_db(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE legacy_measurement(
          entity_id TEXT PRIMARY KEY,
          revision_id TEXT,
          kind TEXT,
          stable_key TEXT,
          revision_no INTEGER,
          recorded_at TEXT,
          author TEXT,
          payload TEXT
        );
        INSERT INTO meta VALUES('schema_version','1');
        """
    )
    payload = {
        "raw": {"ping": "p1", "beam": 1, "value": "0.42"},
        "normalized": {"ping": "p1", "beam": 1, "value": 0.42},
        "unit_or_basis": "m/s",
    }
    conn.execute(
        """INSERT INTO legacy_measurement VALUES(?,?,?,?,?,?,?,?)""",
        (
            "beam_p1_b1",
            "rev_legacy_beam_p1_b1",
            "beam_velocity",
            "p1#b1",
            1,
            "2026-01-01T00:00:00+00:00",
            "legacy",
            canonical_json(payload),
        ),
    )
    conn.commit()
    conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def ensure_source(
    conn: sqlite3.Connection,
    *,
    kind: str,
    stable_key: str,
    raw_value: Any,
    normalized_value: Any,
    unit_or_basis: Optional[str],
    author: str = "fixture",
    recorded_at: Optional[str] = None,
    status: str = "accepted",
    diagnostics: Optional[Sequence[Any]] = None,
    entity_id: Optional[str] = None,
    revision_id: Optional[str] = None,
    revision_no: Optional[int] = None,
) -> Tuple[str, str, int]:
    recorded_at = recorded_at or utc_now()
    entity_id = entity_id or ("src_" + sha256_text(kind + "|" + stable_key)[:16])
    conn.execute(
        "INSERT OR IGNORE INTO source_entity(id,kind,stable_key,created_at) VALUES(?,?,?,?)",
        (entity_id, kind, stable_key, recorded_at),
    )
    if revision_no is None:
        row = conn.execute(
            "SELECT COALESCE(MAX(revision_no),0)+1 AS n FROM source_revision WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        revision_no = int(row["n"])
    revision_payload = {
        "entity_id": entity_id,
        "revision_no": revision_no,
        "recorded_at": recorded_at,
        "author": author,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "unit_or_basis": unit_or_basis,
        "status": status,
        "diagnostics": diagnostics or [],
    }
    revision_id = revision_id or content_id("rev", revision_payload)
    conn.execute(
        """INSERT OR IGNORE INTO source_revision
           (id,entity_id,revision_no,recorded_at,author,raw_value,normalized_value,
            unit_or_basis,status,diagnostics)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            revision_id,
            entity_id,
            revision_no,
            recorded_at,
            author,
            canonical_json(raw_value),
            canonical_json(normalized_value),
            unit_or_basis,
            status,
            canonical_json(diagnostics or []),
        ),
    )
    return entity_id, revision_id, int(revision_no)


def current_sources(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT se.kind AS kind, se.stable_key AS stable_key, se.id AS entity_id,
               sr.* FROM source_entity se
        JOIN source_revision sr ON sr.entity_id=se.id
        WHERE sr.revision_no=(
          SELECT MAX(sr2.revision_no) FROM source_revision sr2 WHERE sr2.entity_id=se.id
        )
        ORDER BY se.kind, se.stable_key
        """
    ).fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        result[row["entity_id"]] = decode_source_row(row)
    return result


def decode_source_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "entity_id": row["entity_id"],
        "revision_id": row["id"],
        "kind": row["kind"],
        "stable_key": row["stable_key"],
        "revision_no": row["revision_no"],
        "recorded_at": row["recorded_at"],
        "author": row["author"],
        "raw_value": json.loads(row["raw_value"]),
        "normalized_value": json.loads(row["normalized_value"]),
        "unit_or_basis": row["unit_or_basis"],
        "status": row["status"],
        "diagnostics": json.loads(row["diagnostics"]),
    }


def append_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    actor: str,
    payload: Dict[str, Any],
    event_id: Optional[str] = None,
    at: Optional[str] = None,
    derivation_version: Optional[str] = None,
    before_version: Optional[str] = None,
) -> Dict[str, Any]:
    at = at or utc_now()
    previous = conn.execute("SELECT checksum FROM event ORDER BY seq DESC LIMIT 1").fetchone()
    parent_checksum = previous["checksum"] if previous else "GENESIS"
    event_body = {
        "at": at,
        "actor": actor,
        "type": event_type,
        "payload": payload,
        "derivation_version": derivation_version,
        "before_version": before_version,
        "previous_checksum": parent_checksum,
    }
    checksum = sha256_text(canonical_json(event_body))
    event_id = event_id or content_id("evt", {"at": at, "type": event_type, "checksum": checksum})
    cur = conn.execute(
        """INSERT INTO event(id,at,actor,type,payload,derivation_version,before_version,checksum)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            event_id,
            at,
            actor,
            event_type,
            canonical_json(payload),
            derivation_version,
            before_version,
            checksum,
        ),
    )
    return {
        "seq": cur.lastrowid,
        "id": event_id,
        "at": at,
        "actor": actor,
        "type": event_type,
        "payload": payload,
        "derivation_version": derivation_version,
        "before_version": before_version,
        "checksum": checksum,
    }


def current_version_id(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT version_id FROM current_pointer WHERE id=1").fetchone()
    return row["version_id"] if row else None


def insert_derivation_version(
    conn: sqlite3.Connection,
    *,
    title: str,
    nodes: Sequence[Dict[str, Any]],
    fingerprint: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    version_id: Optional[str] = None,
    parent_version: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    created_at = created_at or utc_now()
    parent_version = parent_version if parent_version is not None else current_version_id(conn)
    fp_basis = {
        "parent": parent_version,
        "nodes": [
            {"id": node["id"], "fingerprint": node["fingerprint"], "status": node["status"]}
            for node in nodes
        ],
    }
    fingerprint = fingerprint or sha256_text(canonical_json(fp_basis))
    version_id = version_id or content_id("ver", {"fingerprint": fingerprint, "at": created_at})
    summary_text = canonical_json(summary or {"node_count": len(nodes)})
    conn.execute(
        """INSERT OR IGNORE INTO derivation_version
           (id,parent_version,created_at,title,summary,fingerprint,event_id)
           VALUES(?,?,?,?,?,?,NULL)""",
        (version_id, parent_version, created_at, title, summary_text, fingerprint),
    )
    for node in nodes:
        conn.execute(
            """INSERT OR IGNORE INTO node_revision
               (id,node_kind,stable_key,fingerprint,status,value,diagnostics,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                node["id"],
                node["node_kind"],
                node["stable_key"],
                node["fingerprint"],
                node["status"],
                canonical_json(node.get("value")),
                canonical_json(node.get("diagnostics", [])),
                created_at,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO version_node(version_id,node_id) VALUES(?,?)",
            (version_id, node["id"]),
        )
        for dep in node.get("dependencies", []):
            conn.execute(
                "INSERT OR IGNORE INTO node_dependency(node_id,dependency_id,role) VALUES(?,?,?)",
                (node["id"], dep["id"], dep["role"]),
            )
    return version_id


def commit_derivation(
    conn: sqlite3.Connection,
    *,
    actor: str,
    title: str,
    nodes: Sequence[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
    event_type: str = "derivation.completed",
    payload: Optional[Dict[str, Any]] = None,
    parent_version: Optional[str] = None,
    switch: bool = True,
) -> Dict[str, Any]:
    before = parent_version if parent_version is not None else current_version_id(conn)
    version_id = insert_derivation_version(
        conn, title=title, nodes=nodes, summary=summary, parent_version=before
    )
    event = append_event(
        conn,
        event_type=event_type,
        actor=actor,
        payload=payload or {"title": title, "summary": summary or {"node_count": len(nodes)}},
        derivation_version=version_id,
        before_version=before,
    )
    conn.execute(
        "UPDATE derivation_version SET event_id=? WHERE id=?", (event["id"], version_id)
    )
    if switch:
        conn.execute(
            """INSERT INTO current_pointer(id,version_id) VALUES(1,?)
               ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id""",
            (version_id,),
        )
    return {"version_id": version_id, "event": event}


def nodes_for_version(conn: sqlite3.Connection, version_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT nr.* FROM node_revision nr
        JOIN version_node vn ON vn.node_id=nr.id
        WHERE vn.version_id=?
        ORDER BY nr.node_kind, nr.stable_key, nr.id
        """,
        (version_id,),
    ).fetchall()
    return [decode_node_row(row) for row in rows]


def decode_node_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "node_kind": row["node_kind"],
        "stable_key": row["stable_key"],
        "fingerprint": row["fingerprint"],
        "status": row["status"],
        "value": json.loads(row["value"]),
        "diagnostics": json.loads(row["diagnostics"]),
    }


def dependencies_for_node(conn: sqlite3.Connection, node_id: str) -> List[Dict[str, str]]:
    rows = conn.execute(
        "SELECT dependency_id,role FROM node_dependency WHERE node_id=? ORDER BY role,dependency_id",
        (node_id,),
    ).fetchall()
    return [{"id": row["dependency_id"], "role": row["role"]} for row in rows]


def events(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM event ORDER BY seq").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        result.append(item)
    return result


def version_at_event(conn: sqlite3.Connection, seq: int) -> Optional[str]:
    current = None
    for event in events(conn):
        if event["seq"] > seq:
            break
        if event["derivation_version"]:
            current = event["derivation_version"]
    return current


def record_decision(
    conn: sqlite3.Connection,
    *,
    subject_id: str,
    base_version: str,
    actor: str,
    choice: str,
    rationale: str,
) -> Dict[str, Any]:
    existing = conn.execute(
        """SELECT * FROM decision WHERE subject_id=? AND base_version=?
           ORDER BY event_id""",
        (subject_id, base_version),
    ).fetchall()
    conflict_with = next((row for row in existing if row["choice"] != choice), None)
    event_type = "decision.conflict" if conflict_with else "decision.recorded"
    event = append_event(
        conn,
        event_type=event_type,
        actor=actor,
        payload={
            "subject_id": subject_id,
            "base_version": base_version,
            "choice": choice,
            "rationale": rationale,
            "conflict_with": conflict_with["actor"] if conflict_with else None,
        },
        before_version=base_version,
    )
    conn.execute(
        """INSERT INTO decision(subject_id,base_version,actor,choice,rationale,event_id)
           VALUES(?,?,?,?,?,?)
           ON CONFLICT(subject_id,base_version,actor) DO UPDATE SET
             choice=excluded.choice,
             rationale=excluded.rationale,
             event_id=excluded.event_id""",
        (subject_id, base_version, actor, choice, rationale, event["id"]),
    )
    return {"event": event, "conflict": bool(conflict_with)}


def resolve_decision(
    conn: sqlite3.Connection,
    *,
    subject_id: str,
    base_version: str,
    actor: str,
    winning_actor: str,
    rationale: str,
) -> Dict[str, Any]:
    winner = conn.execute(
        "SELECT * FROM decision WHERE subject_id=? AND base_version=? AND actor=?",
        (subject_id, base_version, winning_actor),
    ).fetchone()
    if winner is None:
        raise KeyError("winning decision does not exist")
    event = append_event(
        conn,
        event_type="decision.resolved",
        actor=actor,
        payload={
            "subject_id": subject_id,
            "base_version": base_version,
            "winning_actor": winning_actor,
            "winning_choice": winner["choice"],
            "rationale": rationale,
        },
        before_version=base_version,
    )
    conn.execute(
        "UPDATE decision SET resolved_conflict_event_id=? WHERE subject_id=? AND base_version=?",
        (event["id"], subject_id, base_version),
    )
    return {"event": event}


def export_database(conn: sqlite3.Connection) -> Dict[str, Any]:
    tables = [
        "meta",
        "source_entity",
        "source_revision",
        "event",
        "derivation_version",
        "node_revision",
        "version_node",
        "node_dependency",
        "decision",
        "current_pointer",
    ]
    data: Dict[str, List[Dict[str, Any]]] = {}
    for table in tables:
        data[table] = [dict(row) for row in conn.execute("SELECT * FROM " + table).fetchall()]
    snapshot = {"format": "adcp-evidence-db", "format_version": 1, "tables": data}
    snapshot["snapshot_checksum"] = sha256_text(canonical_json(data))
    return snapshot


def import_database(path: str, snapshot: Dict[str, Any]) -> sqlite3.Connection:
    if snapshot.get("format") != "adcp-evidence-db":
        raise ValueError("unsupported snapshot format")
    data = snapshot["tables"]
    expected = snapshot.get("snapshot_checksum")
    if expected and sha256_text(canonical_json(data)) != expected:
        raise ValueError("snapshot checksum mismatch")
    if os.path.exists(path):
        os.remove(path)
    for suffix in ("-wal", "-shm"):
        sidecar = path + suffix
        if os.path.exists(sidecar):
            os.remove(sidecar)
    conn = connect(path)
    conn.executescript(SCHEMA)
    for table, rows in data.items():
        if table == "meta":
            conn.executemany("INSERT INTO meta(key,value) VALUES(:key,:value)", rows)
        elif table == "source_entity":
            conn.executemany(
                "INSERT INTO source_entity VALUES(:id,:kind,:stable_key,:created_at)", rows
            )
        elif table == "source_revision":
            conn.executemany(
                """INSERT INTO source_revision VALUES
                   (:id,:entity_id,:revision_no,:recorded_at,:author,:raw_value,
                    :normalized_value,:unit_or_basis,:status,:diagnostics)""",
                rows,
            )
        elif table == "event":
            conn.executemany(
                """INSERT INTO event
                   (seq,id,at,actor,type,payload,derivation_version,before_version,checksum)
                   VALUES
                   (:seq,:id,:at,:actor,:type,:payload,:derivation_version,
                    :before_version,:checksum)""",
                rows,
            )
        elif table == "derivation_version":
            conn.executemany(
                """INSERT INTO derivation_version
                   (id,parent_version,created_at,title,summary,fingerprint,event_id)
                   VALUES
                   (:id,:parent_version,:created_at,:title,:summary,:fingerprint,:event_id)""",
                rows,
            )
        elif table == "node_revision":
            conn.executemany(
                """INSERT INTO node_revision
                   (id,node_kind,stable_key,fingerprint,status,value,diagnostics,created_at)
                   VALUES
                   (:id,:node_kind,:stable_key,:fingerprint,:status,:value,
                    :diagnostics,:created_at)""",
                rows,
            )
        elif table == "version_node":
            conn.executemany(
                "INSERT INTO version_node(version_id,node_id) VALUES(:version_id,:node_id)",
                rows,
            )
        elif table == "node_dependency":
            conn.executemany(
                """INSERT INTO node_dependency(node_id,dependency_id,role)
                   VALUES(:node_id,:dependency_id,:role)""",
                rows,
            )
        elif table == "decision":
            conn.executemany(
                """INSERT INTO decision
                   (subject_id,base_version,actor,choice,rationale,event_id,
                    resolved_conflict_event_id)
                   VALUES
                   (:subject_id,:base_version,:actor,:choice,:rationale,:event_id,
                    :resolved_conflict_event_id)""",
                rows,
            )
        elif table == "current_pointer":
            conn.executemany(
                "INSERT INTO current_pointer(id,version_id) VALUES(:id,:version_id)", rows
            )
    verify_event_chain(conn)
    conn.commit()
    return conn


def verify_event_chain(conn: sqlite3.Connection) -> None:
    parent = "GENESIS"
    for row in conn.execute("SELECT * FROM event ORDER BY seq").fetchall():
        payload = json.loads(row["payload"])
        body = {
            "at": row["at"],
            "actor": row["actor"],
            "type": row["type"],
            "payload": payload,
            "derivation_version": row["derivation_version"],
            "before_version": row["before_version"],
            "previous_checksum": parent,
        }
        actual = sha256_text(canonical_json(body))
        if actual != row["checksum"]:
            raise ValueError("event checksum chain is invalid at seq %s" % row["seq"])
        parent = actual


def node_map(conn: sqlite3.Connection, version_id: str) -> Dict[str, Dict[str, Any]]:
    return {node["id"]: node for node in nodes_for_version(conn, version_id)}
