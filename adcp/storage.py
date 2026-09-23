from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = 4


class ConflictError(RuntimeError):
    pass


class Store:
    def __init__(self, path: str = "adcp.sqlite3"):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _migrate(self) -> None:
        with self.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            legacy = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if legacy and int(legacy["value"]) < 2:
                conn.execute("ALTER TABLE resources ADD COLUMN canonical TEXT")
            if legacy and int(legacy["value"]) < 4:
                conn.execute("UPDATE derivation_run_nodes SET reuse_state='recomputed' WHERE reuse_state IS NULL OR reuse_state=''")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS resources("
                "id TEXT NOT NULL, revision_id TEXT NOT NULL, kind TEXT NOT NULL, "
                "rev INTEGER NOT NULL, parent_revision_id TEXT, payload TEXT NOT NULL, "
                "canonical TEXT, created_at_s REAL NOT NULL, "
                "PRIMARY KEY(id, revision_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS versions("
                "version_id TEXT PRIMARY KEY, parent_version_id TEXT, label TEXT NOT NULL, "
                "created_at_s REAL NOT NULL, complete INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS version_resources("
                "version_id TEXT NOT NULL, resource_id TEXT NOT NULL, revision_id TEXT NOT NULL, "
                "PRIMARY KEY(version_id, resource_id), "
                "FOREIGN KEY(version_id) REFERENCES versions(version_id), "
                "FOREIGN KEY(resource_id, revision_id) REFERENCES resources(id, revision_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derivation_runs("
                "run_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, parent_run_id TEXT, "
                "rule_version TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                "created_at_s REAL NOT NULL, complete INTEGER NOT NULL DEFAULT 0, "
                "FOREIGN KEY(version_id) REFERENCES versions(version_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derivation_nodes("
                "node_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, subject_id TEXT NOT NULL, "
                "stage TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL, "
                "accepted INTEGER NOT NULL, value_hash TEXT, diagnostic TEXT, "
                "FOREIGN KEY(run_id) REFERENCES derivation_runs(run_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derivation_results("
                "result_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, candidate_key TEXT NOT NULL, "
                "kind TEXT NOT NULL, value TEXT NOT NULL, unit TEXT, frame TEXT, "
                "status TEXT NOT NULL, adopted INTEGER NOT NULL, diagnostic TEXT, "
                "FOREIGN KEY(node_id) REFERENCES derivation_nodes(node_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derivation_run_nodes("
                "run_id TEXT NOT NULL, node_id TEXT NOT NULL, reuse_state TEXT NOT NULL, "
                "reason TEXT, PRIMARY KEY(run_id, node_id), "
                "FOREIGN KEY(run_id) REFERENCES derivation_runs(run_id), "
                "FOREIGN KEY(node_id) REFERENCES derivation_nodes(node_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS derivation_deps("
                "node_id TEXT NOT NULL, dep_node_id TEXT NOT NULL, resource_revision_id TEXT, "
                "PRIMARY KEY(node_id, dep_node_id, resource_revision_id), "
                "FOREIGN KEY(node_id) REFERENCES derivation_nodes(node_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events("
                "event_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, run_id TEXT, "
                "seq INTEGER NOT NULL, event_type TEXT NOT NULL, actor TEXT NOT NULL, "
                "payload TEXT NOT NULL, created_at_s REAL NOT NULL, "
                "UNIQUE(version_id, seq))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS event_refs("
                "event_id TEXT NOT NULL, ref_type TEXT NOT NULL, ref_id TEXT NOT NULL, "
                "PRIMARY KEY(event_id, ref_type, ref_id), "
                "FOREIGN KEY(event_id) REFERENCES events(event_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pointers("
                "name TEXT PRIMARY KEY, version_id TEXT, run_id TEXT, updated_at_s REAL NOT NULL)"
            )
            conn.execute("INSERT OR IGNORE INTO pointers(name, version_id, run_id, updated_at_s) VALUES('current', NULL, NULL, 0)")
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))

    @staticmethod
    def digest(parts: Sequence[Any]) -> str:
        encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get_current(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pointers WHERE name='current'").fetchone()

    def current_version_id(self) -> str | None:
        return self.get_current()["version_id"]

    def current_run_id(self) -> str | None:
        return self.get_current()["run_id"]

    def revision_id(self, resource_id: str, rev: int) -> str:
        return f"{resource_id}:r{rev}"

    def add_resource_revision(self, conn: sqlite3.Connection, resource_id: str, kind: str, payload: dict[str, Any], parent_revision_id: str | None = None) -> str:
        previous = conn.execute("SELECT MAX(rev) AS rev FROM resources WHERE id=?", (resource_id,)).fetchone()
        rev = int(previous["rev"] or 0) + 1
        revision_id = self.revision_id(resource_id, rev)
        canonical = canonical_json(payload)
        conn.execute(
            "INSERT INTO resources(id, revision_id, kind, rev, parent_revision_id, payload, canonical, created_at_s) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (resource_id, revision_id, kind, rev, parent_revision_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), canonical, time.time()),
        )
        return revision_id

    def get_resource_revision(self, revision_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM resources WHERE revision_id=?", (revision_id,)).fetchone()
        if row is None:
            raise KeyError(revision_id)
        payload = json.loads(row["payload"])
        return {"id": row["id"], "revision_id": row["revision_id"], "kind": row["kind"], "rev": row["rev"], "parent_revision_id": row["parent_revision_id"], "payload": payload}

    def version_resource_map(self, version_id: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT resource_id, revision_id FROM version_resources WHERE version_id=?", (version_id,)).fetchall()
        return {row["resource_id"]: row["revision_id"] for row in rows}

    def create_version(self, resources: dict[str, str], label: str, parent_version_id: str | None = None, complete: bool = True) -> str:
        version_id = self.digest(["version", parent_version_id, resources, label])
        with self.transaction() as conn:
            existing = conn.execute("SELECT version_id FROM versions WHERE version_id=?", (version_id,)).fetchone()
            if existing is None:
                conn.execute("INSERT INTO versions(version_id, parent_version_id, label, created_at_s, complete) VALUES(?,?,?,?,?)", (version_id, parent_version_id, label, time.time(), 1 if complete else 0))
                for resource_id, revision_id in sorted(resources.items()):
                    conn.execute("INSERT INTO version_resources(version_id, resource_id, revision_id) VALUES(?,?,?)", (version_id, resource_id, revision_id))
            return version_id

    def set_current(self, version_id: str, run_id: str | None, conn: sqlite3.Connection | None = None) -> None:
        def work(connection: sqlite3.Connection) -> None:
            row = connection.execute("SELECT complete FROM versions WHERE version_id=?", (version_id,)).fetchone()
            if row is None or not row["complete"]:
                raise ValueError("cannot point at incomplete version")
            connection.execute("UPDATE pointers SET version_id=?, run_id=?, updated_at_s=? WHERE name='current'", (version_id, run_id, time.time()))
        if conn is None:
            with self.transaction() as transaction:
                work(transaction)
        else:
            work(conn)

    def latest_seq(self, conn: sqlite3.Connection, version_id: str) -> int:
        row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS seq FROM events WHERE version_id=?", (version_id,)).fetchone()
        return int(row["seq"])

    def add_event(self, event_type: str, payload: dict[str, Any], actor: str = "system", refs: Sequence[tuple[str, str]] | None = None, version_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        version_id = version_id or self.current_version_id()
        if not version_id:
            raise ValueError("no current version")
        event_id = self.digest(["event", version_id, event_type, actor, payload, time.time_ns()])
        with self.transaction() as conn:
            seq = self.latest_seq(conn, version_id) + 1
            conn.execute(
                "INSERT INTO events(event_id, version_id, run_id, seq, event_type, actor, payload, created_at_s) VALUES(?,?,?,?,?,?,?,?)",
                (event_id, version_id, run_id, seq, event_type, actor, json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time()),
            )
            for ref_type, ref_id in refs or []:
                conn.execute("INSERT INTO event_refs(event_id, ref_type, ref_id) VALUES(?,?,?)", (event_id, ref_type, ref_id))
        return self.get_event(event_id)

    def get_event(self, event_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        refs = self.conn.execute("SELECT ref_type, ref_id FROM event_refs WHERE event_id=?", (event_id,)).fetchall()
        return {"event_id": event_id, "version_id": row["version_id"], "run_id": row["run_id"], "seq": row["seq"], "event_type": row["event_type"], "actor": row["actor"], "payload": json.loads(row["payload"]), "created_at_s": row["created_at_s"], "refs": [dict(ref) for ref in refs]}

    def list_events(self, version_id: str | None = None) -> list[dict[str, Any]]:
        version_id = version_id or self.current_version_id()
        rows = self.conn.execute("SELECT event_id FROM events WHERE version_id=? ORDER BY seq", (version_id,)).fetchall()
        return [self.get_event(row["event_id"]) for row in rows]

    def insert_derivation_run(self, conn: sqlite3.Connection, version_id: str, parent_run_id: str | None, rule_version: str, fingerprint: str) -> str:
        run_id = self.digest(["run", version_id, parent_run_id, rule_version, fingerprint])
        conn.execute(
            "INSERT OR IGNORE INTO derivation_runs(run_id, version_id, parent_run_id, rule_version, fingerprint, created_at_s, complete) "
            "VALUES(?,?,?,?,?,?,0)",
            (run_id, version_id, parent_run_id, rule_version, fingerprint, time.time()),
        )
        return run_id

    def complete_run(self, conn: sqlite3.Connection, run_id: str) -> None:
        conn.execute("UPDATE derivation_runs SET complete=1 WHERE run_id=?", (run_id,))

    def link_run_node(self, conn: sqlite3.Connection, run_id: str, node_id: str, reuse_state: str, reason: str | None = None) -> None:
        conn.execute("INSERT OR REPLACE INTO derivation_run_nodes(run_id, node_id, reuse_state, reason) VALUES(?,?,?,?)", (run_id, node_id, reuse_state, reason))

    def insert_node(self, conn: sqlite3.Connection, node: dict[str, Any]) -> None:
        diagnostic = None if node.get("diagnostic") is None else json.dumps(node["diagnostic"], ensure_ascii=False, sort_keys=True)
        conn.execute(
            "INSERT OR REPLACE INTO derivation_nodes(node_id, run_id, subject_id, stage, fingerprint, status, accepted, value_hash, diagnostic) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (node["node_id"], node["run_id"], node["subject_id"], node["stage"], node["fingerprint"], node["status"], 1 if node["accepted"] else 0, node.get("value_hash"), diagnostic),
        )
        conn.execute("DELETE FROM derivation_deps WHERE node_id=?", (node["node_id"],))
        for dep_node_id, resource_revision_id in node.get("deps", []):
            conn.execute("INSERT INTO derivation_deps(node_id, dep_node_id, resource_revision_id) VALUES(?,?,?)", (node["node_id"], dep_node_id, resource_revision_id))
        conn.execute("DELETE FROM derivation_results WHERE node_id=?", (node["node_id"],))
        for result in node.get("results", []):
            diagnostic = None if result.get("diagnostic") is None else json.dumps(result["diagnostic"], ensure_ascii=False, sort_keys=True)
            conn.execute(
                "INSERT INTO derivation_results(result_id, node_id, candidate_key, kind, value, unit, frame, status, adopted, diagnostic) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (result["result_id"], result["node_id"], result["candidate_key"], result["kind"], json.dumps(result["value"], ensure_ascii=False, sort_keys=True), result.get("unit"), result.get("frame"), result["status"], 1 if result["adopted"] else 0, diagnostic),
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM derivation_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def list_runs(self, version_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM derivation_runs WHERE version_id=? ORDER BY created_at_s", (version_id,)).fetchall()]

    def list_nodes(self, run_id: str) -> list[dict[str, Any]]:
        nodes = []
        rows = self.conn.execute(
            "SELECT n.*, m.reuse_state, m.reason AS reuse_reason FROM derivation_run_nodes m "
            "JOIN derivation_nodes n ON n.node_id=m.node_id WHERE m.run_id=? ORDER BY n.subject_id, n.stage",
            (run_id,),
        ).fetchall()
        for row in rows:
            node = dict(row)
            node["accepted"] = bool(node["accepted"])
            node["diagnostic"] = None if row["diagnostic"] is None else json.loads(row["diagnostic"])
            node["results"] = []
            for result_row in self.conn.execute("SELECT * FROM derivation_results WHERE node_id=? ORDER BY candidate_key", (row["node_id"],)).fetchall():
                result = dict(result_row)
                result["value"] = json.loads(result_row["value"])
                result["adopted"] = bool(result_row["adopted"])
                result["diagnostic"] = None if result_row["diagnostic"] is None else json.loads(result_row["diagnostic"])
                node["results"].append(result)
            node["deps"] = [dict(dep) for dep in self.conn.execute("SELECT dep_node_id, resource_revision_id FROM derivation_deps WHERE node_id=? ORDER BY dep_node_id", (row["node_id"],)).fetchall()]
            nodes.append(node)
        return nodes

    def node_map(self, run_id: str) -> dict[str, dict[str, Any]]:
        return {(node["subject_id"], node["stage"]): node for node in self.list_nodes(run_id)}

    def export_bundle(self) -> dict[str, Any]:
        def rows(table: str) -> list[dict[str, Any]]:
            return [dict(row) for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()]
        resources = rows("resources")
        versions = rows("versions")
        mappings = rows("version_resources")
        events = rows("events")
        refs = rows("event_refs")
        runs = rows("derivation_runs")
        nodes = rows("derivation_nodes")
        results = rows("derivation_results")
        deps = rows("derivation_deps")
        pointer = self.get_current()
        return {
            "bundle_version": 1,
            "schema_version": SCHEMA_VERSION,
            "resources": resources,
            "versions": versions,
            "version_resources": mappings,
            "derivation_runs": runs,
            "derivation_nodes": nodes,
            "derivation_results": results,
            "derivation_deps": deps,
            "events": events,
            "event_refs": refs,
            "current": {"version_id": pointer["version_id"], "run_id": pointer["run_id"]},
        }

    def import_bundle(self, bundle: dict[str, Any], path: str | None = None) -> str:
        target = path or self.path
        temporary = f"{target}.import-tmp"
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(temporary + suffix)
            except FileNotFoundError:
                pass
        imported = Store(temporary)
        with imported.transaction() as conn:
            for table in [
                "resources",
                "versions",
                "version_resources",
                "derivation_runs",
                "derivation_nodes",
                "derivation_results",
                "derivation_deps",
                "derivation_run_nodes",
                "events",
                "event_refs",
            ]:
                for row in bundle.get(table, []):
                    columns = list(row.keys())
                    placeholders = ",".join("?" for _ in columns)
                    conn.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders})", [row[column] for column in columns])
            current = bundle.get("current", {})
            conn.execute("UPDATE pointers SET version_id=?, run_id=?, updated_at_s=? WHERE name='current'", (current.get("version_id"), current.get("run_id"), time.time()))
        imported.close()
        if path is None:
            self.close()
        os.replace(temporary, target)
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(temporary + suffix)
            except FileNotFoundError:
                pass
        return target


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def find_node_by_identity(self, subject_id: str, stage: str, fingerprint: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT n.*, m.reuse_state, m.reason AS reuse_reason FROM derivation_nodes n "
            "JOIN derivation_run_nodes m ON m.node_id=n.node_id "
            "WHERE n.subject_id=? AND n.stage=? AND n.fingerprint=? ORDER BY n.rowid DESC LIMIT 1",
            (subject_id, stage, fingerprint),
        ).fetchone()
        if row is None:
            return None
        node = dict(row)
        node["accepted"] = bool(node["accepted"])
        node["diagnostic"] = None if row["diagnostic"] is None else __import__("json").loads(row["diagnostic"])
        node["results"] = []
        for result_row in self.conn.execute("SELECT * FROM derivation_results WHERE node_id=? ORDER BY candidate_key", (row["node_id"],)).fetchall():
            import json
            result = dict(result_row)
            result["value"] = json.loads(result_row["value"])
            result["adopted"] = bool(result_row["adopted"])
            result["diagnostic"] = None if result_row["diagnostic"] is None else json.loads(result_row["diagnostic"])
            node["results"].append(result)
        node["deps"] = [dict(dep) for dep in self.conn.execute("SELECT dep_node_id, resource_revision_id FROM derivation_deps WHERE node_id=? ORDER BY dep_node_id", (row["node_id"],)).fetchall()]
        return node
