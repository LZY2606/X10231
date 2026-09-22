from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import SCHEMA_VERSION


SCHEMA = """
PRAGMA user_version = 2;
CREATE TABLE IF NOT EXISTS versions (
  version_id TEXT PRIMARY KEY,
  parent_version_id TEXT REFERENCES versions(version_id),
  label TEXT NOT NULL,
  kind TEXT NOT NULL,
  event_id TEXT,
  created_at TEXT NOT NULL,
  graph_run_id TEXT,
  is_complete INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  seq INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  summary TEXT NOT NULL,
  base_version_id TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(branch, seq)
);
CREATE TABLE IF NOT EXISTS source_records (
  record_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  logical_id TEXT NOT NULL,
  category TEXT NOT NULL,
  ping_id TEXT,
  ping_time_s REAL,
  beam INTEGER,
  depth_cell INTEGER,
  raw_repr TEXT NOT NULL,
  quantity TEXT,
  value REAL,
  unit TEXT,
  source TEXT NOT NULL,
  replaced_record_id TEXT,
  PRIMARY KEY(record_id, version_id)
);
CREATE TABLE IF NOT EXISTS derived_records (
  record_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  logical_id TEXT NOT NULL,
  category TEXT NOT NULL,
  ping_id TEXT,
  ping_time_s REAL,
  beam INTEGER,
  depth_cell INTEGER,
  candidate_id TEXT,
  value_json TEXT NOT NULL,
  unit TEXT,
  datum TEXT,
  status TEXT NOT NULL,
  diagnostic_json TEXT NOT NULL,
  fingerprint TEXT,
  source_event_id TEXT,
  PRIMARY KEY(record_id, version_id)
);
CREATE TABLE IF NOT EXISTS graph_nodes (
  graph_run_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  ping_id TEXT,
  depth_cell INTEGER,
  status TEXT NOT NULL,
  fingerprint TEXT,
  reasons_json TEXT NOT NULL,
  retained_record_id TEXT,
  PRIMARY KEY(graph_run_id, node_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
  graph_run_id TEXT NOT NULL,
  from_node_id TEXT NOT NULL,
  to_node_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  PRIMARY KEY(graph_run_id, from_node_id, to_node_id, relation)
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,
  base_version_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  decision_key TEXT NOT NULL,
  action TEXT NOT NULL,
  candidate_id TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflicts (
  conflict_key TEXT PRIMARY KEY,
  base_version_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  decision_key TEXT NOT NULL,
  first_decision_id TEXT NOT NULL,
  competing_decision_id TEXT NOT NULL,
  resolved_by_decision_id TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_version ON source_records(version_id);
CREATE INDEX IF NOT EXISTS idx_sources_logical ON source_records(logical_id, version_id);
CREATE INDEX IF NOT EXISTS idx_derived_version ON derived_records(version_id);
CREATE INDEX IF NOT EXISTS idx_derived_logical ON derived_records(logical_id, version_id);
CREATE INDEX IF NOT EXISTS idx_events_version ON events(version_id);
"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{digest(list(parts))[:16]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for key in ("payload_json", "value_json", "diagnostic_json", "reasons_json"):
        if key in data and data[key] is not None:
            data[key[:-5] if key.endswith("_json") else key] = json.loads(data[key])
    return data


class SaveConflict(Exception):
    def __init__(self, conflict_key: str, existing: Dict[str, Any], submitted: Dict[str, Any]):
        super().__init__(f"resolvable conflict for {conflict_key}")
        self.conflict_key = conflict_key
        self.existing = existing
        self.submitted = submitted


class Storage:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.commit()

    def current_version(self, branch: str = "main") -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT v.version_id FROM versions v
                JOIN events e ON e.event_id=v.event_id AND e.branch=?
                WHERE v.is_complete=1
                ORDER BY e.seq DESC LIMIT 1
                """,
                (branch,),
            ).fetchone()
            return row["version_id"] if row else None

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone()
            return row_to_dict(row) if row else None

    def event_branch(self, event_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT branch FROM events WHERE event_id=?", (event_id,)).fetchone()
            return row["branch"] if row else "main"

    def next_seq(self, branch: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS seq FROM events WHERE branch=?", (branch,)).fetchone()
            return int(row["seq"])

    def _copy_sources(self, conn: sqlite3.Connection, new_version: str, base_version: Optional[str]) -> None:
        if not base_version:
            return
        conn.execute(
            """
            INSERT INTO source_records
            SELECT record_id, ?, logical_id, category, ping_id, ping_time_s, beam, depth_cell,
                   raw_repr, quantity, value, unit, source, replaced_record_id
            FROM source_records WHERE version_id=?
            """,
            (new_version, base_version),
        )

    def _upsert_sources(self, conn: sqlite3.Connection, version_id: str, records: Sequence[Dict[str, Any]]) -> None:
        for record in records:
            old = conn.execute(
                "SELECT record_id FROM source_records WHERE version_id=? AND logical_id=? AND beam IS ? AND depth_cell IS ?",
                (version_id, record["logical_id"], record.get("beam"), record.get("depth_cell")),
            ).fetchone()
            replaced = old["record_id"] if old else record.get("replaced_record_id")
            record_id = record.get("record_id") or stable_id(
                "src",
                version_id,
                record["logical_id"],
                record.get("ping_id"),
                record.get("beam"),
                record.get("depth_cell"),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO source_records
                (record_id,version_id,logical_id,category,ping_id,ping_time_s,beam,depth_cell,
                 raw_repr,quantity,value,unit,source,replaced_record_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record_id,
                    version_id,
                    record["logical_id"],
                    record["category"],
                    record.get("ping_id"),
                    record.get("ping_time_s"),
                    record.get("beam"),
                    record.get("depth_cell"),
                    str(record.get("raw_repr", record.get("value", ""))),
                    record.get("quantity"),
                    record.get("value"),
                    record.get("unit"),
                    record.get("source", "manual"),
                    replaced,
                ),
            )

    def create_version(
        self,
        *,
        label: str,
        kind: str,
        actor: str,
        summary: str,
        event_type: str,
        payload: Dict[str, Any],
        source_changes: Sequence[Dict[str, Any]] = (),
        base_version: Optional[str] = None,
        branch: str = "main",
        event_id: Optional[str] = None,
        version_id: Optional[str] = None,
        created_at: Optional[str] = None,
        build_graph: Optional[Callable[[sqlite3.Connection, str, str], Dict[str, Any]]] = None,
        fail_after_sources: bool = False,
    ) -> Dict[str, Any]:
        created_at = created_at or now_iso()
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                seq = self.next_seq(branch)
                event_id = event_id or stable_id("evt", branch, seq, label, summary)
                version_id = version_id or stable_id("ver", event_id, label)
                graph_run_id = stable_id("run", version_id)
                conn.execute(
                    """
                    INSERT INTO versions
                    (version_id,parent_version_id,label,kind,event_id,created_at,graph_run_id,is_complete)
                    VALUES (?,?,?,?,?,?,?,0)
                    """,
                    (version_id, base_version, label, kind, event_id, created_at, graph_run_id),
                )
                self._copy_sources(conn, version_id, base_version)
                self._upsert_sources(conn, version_id, source_changes)
                if fail_after_sources:
                    raise RuntimeError("injected write interruption before commit")
                graph_summary: Dict[str, Any] = {"reused": [], "recomputed": [], "blocked": []}
                if build_graph is not None:
                    graph_summary = build_graph(conn, version_id, graph_run_id) or graph_summary
                conn.execute(
                    """
                    INSERT INTO events
                    (event_id,version_id,branch,seq,event_type,actor,summary,base_version_id,payload_json,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        event_id,
                        version_id,
                        branch,
                        seq,
                        event_type,
                        actor,
                        summary,
                        base_version,
                        canonical_json(payload),
                        created_at,
                    ),
                )
                conn.execute("UPDATE versions SET is_complete=1 WHERE version_id=?", (version_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"event_id": event_id, "version_id": version_id, "graph_run_id": graph_run_id, **graph_summary}

    def insert_graph_outputs(
        self,
        conn: sqlite3.Connection,
        *,
        graph_run_id: str,
        version_id: str,
        nodes: Sequence[Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
        derived: Sequence[Dict[str, Any]],
    ) -> None:
        for node in nodes:
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_nodes
                (graph_run_id,node_id,version_id,stage,ping_id,depth_cell,status,fingerprint,reasons_json,retained_record_id)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    graph_run_id,
                    node["node_id"],
                    version_id,
                    node["stage"],
                    node.get("ping_id"),
                    node.get("depth_cell"),
                    node["status"],
                    node.get("fingerprint"),
                    canonical_json(node.get("reasons", [])),
                    node.get("retained_record_id"),
                ),
            )
        for edge in edges:
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_edges(graph_run_id,from_node_id,to_node_id,relation)
                VALUES (?,?,?,?)
                """,
                (graph_run_id, edge["from"], edge["to"], edge["relation"]),
            )
        for record in derived:
            record_id = record.get("record_id") or stable_id(
                "drv",
                version_id,
                record["logical_id"],
                record.get("candidate_id"),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO derived_records
                (record_id,version_id,logical_id,category,ping_id,ping_time_s,beam,depth_cell,candidate_id,
                 value_json,unit,datum,status,diagnostic_json,fingerprint,source_event_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record_id,
                    version_id,
                    record["logical_id"],
                    record["category"],
                    record.get("ping_id"),
                    record.get("ping_time_s"),
                    record.get("beam"),
                    record.get("depth_cell"),
                    record.get("candidate_id"),
                    canonical_json(record.get("value")),
                    record.get("unit"),
                    record.get("datum"),
                    record["status"],
                    canonical_json(record.get("diagnostics", [])),
                    record.get("fingerprint"),
                    record.get("source_event_id"),
                ),
            )

    def sources_for_version(self, version_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_records WHERE version_id=? ORDER BY ping_time_s, ping_id, depth_cell, beam, category",
                (version_id,),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def source_by_logical(self, version_id: str, logical_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM source_records WHERE version_id=? AND logical_id=?",
                (version_id, logical_id),
            ).fetchone()
            return row_to_dict(row) if row else None

    def derived_for_version(self, version_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM derived_records WHERE version_id=? ORDER BY ping_time_s,ping_id,depth_cell,category,candidate_id",
                (version_id,),
            ).fetchall()
            return [self._decode_derived(row) for row in rows]

    @staticmethod
    def _decode_derived(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["value"] = json.loads(item.pop("value_json"))
        item["diagnostics"] = json.loads(item.pop("diagnostic_json"))
        item["stable_identity"] = item["logical_id"]
        return item

    def graph_for_version(self, version_id: str) -> Dict[str, List[Dict[str, Any]]]:
        with self.connect() as conn:
            run = conn.execute("SELECT graph_run_id FROM versions WHERE version_id=?", (version_id,)).fetchone()
            if not run:
                return {"nodes": [], "edges": []}
            graph_run_id = run["graph_run_id"]
            nodes = [
                {
                    **dict(row),
                    "reasons": json.loads(row["reasons_json"]),
                }
                for row in conn.execute("SELECT * FROM graph_nodes WHERE graph_run_id=? ORDER BY node_id", (graph_run_id,))
            ]
            edges = [dict(row) for row in conn.execute("SELECT * FROM graph_edges WHERE graph_run_id=?", (graph_run_id,))]
            return {"nodes": nodes, "edges": edges}

    def events(self, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM events"
        args: Tuple[Any, ...] = ()
        if branch:
            sql += " WHERE branch=?"
            args = (branch,)
        sql += " ORDER BY created_at, seq"
        with self.connect() as conn:
            return [
                {
                    **dict(row),
                    "payload": json.loads(row["payload_json"]),
                }
                for row in conn.execute(sql, args)
            ]

    def event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
            return {**dict(row), "payload": json.loads(row["payload_json"])} if row else None

    def decisions(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, e.version_id FROM decisions d
                JOIN events e ON e.event_id=d.event_id
                ORDER BY d.created_at
                """
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                result.append(item)
            return result

    def save_decision(
        self,
        *,
        decision_id: str,
        event_id: str,
        base_version_id: str,
        branch: str,
        decision_key: str,
        action: str,
        candidate_id: Optional[str],
        payload: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> str:
        created_at = created_at or now_iso()
        conflict_key = f"{base_version_id}:{decision_key}"
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT * FROM decisions WHERE base_version_id=? AND decision_key=?",
                    (base_version_id, decision_key),
                ).fetchone()
                if existing:
                    same = (
                        existing["action"] == action
                        and (existing["candidate_id"] or "") == (candidate_id or "")
                    )
                    if not same:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO conflicts
                            (conflict_key,base_version_id,branch,decision_key,first_decision_id,competing_decision_id,
                             resolved_by_decision_id,status,created_at)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                conflict_key,
                                base_version_id,
                                branch,
                                decision_key,
                                existing["decision_id"],
                                decision_id,
                                None,
                                "open",
                                created_at,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO decisions
                            (decision_id,event_id,base_version_id,branch,decision_key,action,candidate_id,payload_json,created_at)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                decision_id,
                                event_id,
                                base_version_id,
                                branch,
                                decision_key,
                                action,
                                candidate_id,
                                canonical_json(payload),
                                created_at,
                            ),
                        )
                        conn.commit()
                        return "conflict"
                    conn.commit()
                    return "duplicate"
                conn.execute(
                    """
                    INSERT INTO decisions
                    (decision_id,event_id,base_version_id,branch,decision_key,action,candidate_id,payload_json,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        decision_id,
                        event_id,
                        base_version_id,
                        branch,
                        decision_key,
                        action,
                        candidate_id,
                        canonical_json(payload),
                        created_at,
                    ),
                )
                conn.commit()
                return "saved"
            except Exception:
                conn.rollback()
                raise

    def resolve_conflict(self, conflict_key: str, decision_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conflicts SET resolved_by_decision_id=?, status='resolved' WHERE conflict_key=?",
                (decision_id, conflict_key),
            )
            conn.commit()

    def list_conflicts(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM conflicts ORDER BY created_at")]

    def export_db(self, path: str) -> None:
        payload = self.export_payload()
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def export_payload(self) -> Dict[str, Any]:
        with self.connect() as conn:
            tables = [
                "versions",
                "events",
                "source_records",
                "derived_records",
                "graph_nodes",
                "graph_edges",
                "decisions",
                "conflicts",
            ]
            return {
                "format": "adcp-replay",
                "schema_version": SCHEMA_VERSION,
                "exported_at": now_iso(),
                "tables": {table: [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1,2")] for table in tables},
            }

    def import_payload(self, payload: Dict[str, Any]) -> None:
        expected = {"format", "schema_version", "tables"}
        if not expected.issubset(payload) or payload["format"] != "adcp-replay":
            raise ValueError("unsupported archive format")
        tables = payload["tables"]
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for table in reversed(
                    ["conflicts", "decisions", "graph_edges", "graph_nodes", "derived_records", "source_records", "events", "versions"]
                ):
                    conn.execute(f"DELETE FROM {table}")
                conn.executemany(
                    """INSERT INTO versions(version_id,parent_version_id,label,kind,event_id,created_at,graph_run_id,is_complete)
                       VALUES(:version_id,:parent_version_id,:label,:kind,:event_id,:created_at,:graph_run_id,:is_complete)""",
                    tables["versions"],
                )
                conn.executemany(
                    """INSERT INTO events(event_id,version_id,branch,seq,event_type,actor,summary,base_version_id,payload_json,created_at)
                       VALUES(:event_id,:version_id,:branch,:seq,:event_type,:actor,:summary,:base_version_id,:payload_json,:created_at)""",
                    tables["events"],
                )
                conn.executemany(
                    """INSERT INTO source_records(record_id,version_id,logical_id,category,ping_id,ping_time_s,beam,depth_cell,
                       raw_repr,quantity,value,unit,source,replaced_record_id)
                       VALUES(:record_id,:version_id,:logical_id,:category,:ping_id,:ping_time_s,:beam,:depth_cell,
                       :raw_repr,:quantity,:value,:unit,:source,:replaced_record_id)""",
                    tables["source_records"],
                )
                conn.executemany(
                    """INSERT INTO derived_records(record_id,version_id,logical_id,category,ping_id,ping_time_s,beam,depth_cell,candidate_id,
                       value_json,unit,datum,status,diagnostic_json,fingerprint,source_event_id)
                       VALUES(:record_id,:version_id,:logical_id,:category,:ping_id,:ping_time_s,:beam,:depth_cell,:candidate_id,
                       :value_json,:unit,:datum,:status,:diagnostic_json,:fingerprint,:source_event_id)""",
                    tables["derived_records"],
                )
                conn.executemany(
                    """INSERT INTO graph_nodes(graph_run_id,node_id,version_id,stage,ping_id,depth_cell,status,fingerprint,reasons_json,retained_record_id)
                       VALUES(:graph_run_id,:node_id,:version_id,:stage,:ping_id,:depth_cell,:status,:fingerprint,:reasons_json,:retained_record_id)""",
                    tables["graph_nodes"],
                )
                conn.executemany(
                    "INSERT INTO graph_edges(graph_run_id,from_node_id,to_node_id,relation) VALUES(:graph_run_id,:from_node_id,:to_node_id,:relation)",
                    tables["graph_edges"],
                )
                conn.executemany(
                    """INSERT INTO decisions(decision_id,event_id,base_version_id,branch,decision_key,action,candidate_id,payload_json,created_at)
                       VALUES(:decision_id,:event_id,:base_version_id,:branch,:decision_key,:action,:candidate_id,:payload_json,:created_at)""",
                    tables["decisions"],
                )
                conn.executemany(
                    """INSERT INTO conflicts(conflict_key,base_version_id,branch,decision_key,first_decision_id,competing_decision_id,
                       resolved_by_decision_id,status,created_at)
                       VALUES(:conflict_key,:base_version_id,:branch,:decision_key,:first_decision_id,:competing_decision_id,
                       :resolved_by_decision_id,:status,:created_at)""",
                    tables["conflicts"],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def import_file(self, path: str) -> None:
        self.import_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def migrate_v1_database(path: str) -> None:
        legacy = sqlite3.connect(path)
        legacy.row_factory = sqlite3.Row
        try:
            version = legacy.execute("PRAGMA user_version").fetchone()[0]
            if version != 1:
                return
            legacy_rows = [dict(row) for row in legacy.execute("SELECT * FROM legacy_sources ORDER BY 1")]
        finally:
            legacy.close()
        fresh = Storage(path)
        timestamp = now_iso()

        def builder(conn: sqlite3.Connection, version_id: str, graph_run_id: str) -> Dict[str, Any]:
            return {"reused": [], "recomputed": [], "blocked": []}

        fresh.create_version(
            label="v1-migration",
            kind="migration",
            actor="system",
            summary="Migrated separated ADCP source records from schema v1",
            event_type="schema.migrated",
            payload={"from_schema": 1, "legacy_rows": len(legacy_rows)},
            source_changes=[
                {
                    "logical_id": row["logical_id"],
                    "category": row["category"],
                    "ping_id": row.get("ping_id"),
                    "ping_time_s": row.get("ping_time_s"),
                    "beam": row.get("beam"),
                    "depth_cell": row.get("depth_cell"),
                    "raw_repr": row["raw_repr"],
                    "quantity": row.get("quantity"),
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                    "source": row.get("source", "v1-import"),
                }
                for row in legacy_rows
            ],
            created_at=timestamp,
            build_graph=builder,
        )
