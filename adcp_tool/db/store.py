"""Versioned, crash-safe SQLite storage.

Durability model:

* source records, derivation records and branches are independent,
  append-only, versioned entities;
* a derivation is built inside one transaction and only flipped from
  ``building`` -> ``ready`` *after* every node committed - the current branch
  pointer is updated in the same transaction;
* a crash mid-write leaves a ``building`` derivation that the branch never
  points at (recovered as ``aborted`` at startup): readers see either the
  previous ready derivation or the complete new one, never a hybrid;
* two editors starting from the same source/config base both submit: the
  first commit wins, the second gets a resolvable ``Conflict`` carrying both
  base versions.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

SCHEMA_VERSION = 2


class Conflict(Exception):
    """Raised when an edit bases on a superseded version (resolvable)."""

    def __init__(self, message, base_version, current_version, server_payload=None):
        super().__init__(message)
        self.base_version = base_version
        self.current_version = current_version
        self.server_payload = server_payload


def connect(path):
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def transaction(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    schema_name TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'import',
    created_at REAL NOT NULL,
    PRIMARY KEY (case_id, source_id, version)
);
CREATE TABLE IF NOT EXISTS source_heads (
    case_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (case_id, source_id)
);
CREATE TABLE IF NOT EXISTS config_versions (
    case_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    label TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (case_id, version)
);
CREATE TABLE IF NOT EXISTS derivations (
    derivation_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    base_sources TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at REAL NOT NULL,
    committed_at REAL,
    propagation TEXT,
    crash_marker INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (case_id, version)
);
CREATE TABLE IF NOT EXISTS nodes (
    case_id TEXT NOT NULL,
    derivation_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    ping_id TEXT,
    cell_id TEXT,
    status TEXT NOT NULL,
    rule TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    upstream TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    payload TEXT NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY (case_id, derivation_version, node_id)
);
CREATE TABLE IF NOT EXISTS branches (
    case_id TEXT NOT NULL,
    name TEXT NOT NULL,
    derivation_version INTEGER NOT NULL,
    base_version INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (case_id, name)
);
CREATE TABLE IF NOT EXISTS decisions (
    case_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    derivation_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    chosen_candidate_id TEXT,
    parameter_overrides TEXT,
    basis_versions TEXT NOT NULL,
    created_at REAL NOT NULL,
    note TEXT,
    PRIMARY KEY (case_id, decision_id)
);
CREATE TABLE IF NOT EXISTS events (
    case_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    branch_name TEXT,
    derivation_version INTEGER,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (case_id, seq)
);
CREATE TABLE IF NOT EXISTS tags (
    case_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    derivation_version INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (case_id, tag)
);
CREATE TABLE IF NOT EXISTS node_identity (
    case_id TEXT NOT NULL,
    stable_key TEXT NOT NULL,
    node_id TEXT NOT NULL,
    first_seen_derivation INTEGER NOT NULL,
    PRIMARY KEY (case_id, stable_key)
);
"""


class Store:
    def __init__(self, conn):
        self.conn = conn
        self.init_schema()
        self.recover_aborted()

    # ---------- schema / migration ----------
    def init_schema(self):
        with self.conn:
            self.conn.executescript(SCHEMA)
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                self.migrate(int(row["value"]))

    def migrate(self, old):
        """Idempotent in-place upgrade; keeps all cases and history."""
        if old < 2:
            cols = {
                r["name"] for r in self.conn.execute("PRAGMA table_info(derivations)")
            }
            if "crash_marker" not in cols:
                self.conn.execute(
                    "ALTER TABLE derivations ADD COLUMN crash_marker INTEGER NOT NULL DEFAULT 0"
                )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS tags ("
                "case_id TEXT NOT NULL, tag TEXT NOT NULL, branch_name TEXT NOT NULL,"
                "derivation_version INTEGER NOT NULL, created_at REAL NOT NULL,"
                "PRIMARY KEY (case_id, tag))"
            )
            self.conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

    def recover_aborted(self):
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE derivations SET status='aborted', committed_at=? "
                "WHERE status='building'", (time.time(),)
            )

    # ---------- events ----------
    def _next_event_seq(self, case_id):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS s FROM events WHERE case_id=?",
            (case_id,),
        ).fetchone()
        return int(row["s"])

    def add_event(self, case_id, etype, payload, branch_name=None,
                  derivation_version=None):
        seq = self._next_event_seq(case_id)
        self.conn.execute(
            "INSERT INTO events(case_id,seq,type,branch_name,derivation_version,"
            "payload,created_at) VALUES(?,?,?,?,?,?,?)",
            (case_id, seq, etype, branch_name, derivation_version,
             json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time()),
        )
        return seq

    def list_events(self, case_id):
        rows = self.conn.execute(
            "SELECT * FROM events WHERE case_id=? ORDER BY seq", (case_id,)
        ).fetchall()
        return [self._event_dict(r) for r in rows]

    @staticmethod
    def _event_dict(r):
        return {
            "seq": r["seq"], "type": r["type"], "branch_name": r["branch_name"],
            "derivation_version": r["derivation_version"],
            "payload": json.loads(r["payload"]), "created_at": r["created_at"],
        }

    # ---------- sources ----------
    def record_sources(self, case_id, source_payloads, origin="import"):
        """Insert version 1 for each stable source id. Idempotent on import."""
        now = time.time()
        with transaction(self.conn):
            for sid, payload in source_payloads.items():
                exists = self.conn.execute(
                    "SELECT 1 FROM sources WHERE case_id=? AND source_id=?",
                    (case_id, sid),
                ).fetchone()
                if exists:
                    continue
                self.conn.execute(
                    "INSERT INTO sources(source_id,case_id,kind,scope,version,"
                    "payload,origin,created_at) VALUES(?,?,?,?,1,?,?,?)",
                    (sid, case_id, payload["kind"], payload["scope"],
                     json.dumps(payload["data"], ensure_ascii=False, sort_keys=True),
                     origin, now),
                )
                self.conn.execute(
                    "INSERT INTO source_heads(case_id,source_id,version) VALUES(?,?,1)",
                    (case_id, sid),
                )

    def update_source(self, case_id, source_id, data, editor, base_version,
                      origin="edit"):
        """Append a new source version; raise Conflict on stale base."""
        head = self._source_head(case_id, source_id)
        if head is not None and head != base_version:
            raise Conflict(
                "source %s changed since your read" % source_id,
                base_version, head,
                server_payload=self._source(case_id, source_id, head),
            )
        new_version = (head or 0) + 1
        payload = self._source(case_id, source_id, head or 1, required=False)
        kind = payload["kind"] if payload else data.get("kind", "unknown")
        scope = payload["scope"] if payload else data.get("scope", "")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO sources(source_id,case_id,kind,scope,version,payload,"
                "origin,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (source_id, case_id, kind, scope, new_version,
                 json.dumps(data, ensure_ascii=False, sort_keys=True), origin,
                 time.time()),
            )
            self.conn.execute(
                "INSERT INTO source_heads(case_id,source_id,version) VALUES(?,?,?) "
                "ON CONFLICT(case_id,source_id) DO UPDATE SET version=excluded.version",
                (case_id, source_id, new_version),
            )
            self.add_event(case_id, "source-revision",
                           {"source_id": source_id, "base_version": base_version,
                            "new_version": new_version, "editor": editor})
        return new_version

    def _source_head(self, case_id, source_id):
        row = self.conn.execute(
            "SELECT version FROM source_heads WHERE case_id=? AND source_id=?",
            (case_id, source_id),
        ).fetchone()
        return None if row is None else int(row["version"])

    def _source(self, case_id, source_id, version, required=True):
        row = self.conn.execute(
            "SELECT * FROM sources WHERE case_id=? AND source_id=? AND version=?",
            (case_id, source_id, version),
        ).fetchone()
        if row is None and required:
            raise KeyError((case_id, source_id, version))
        if row is None:
            return None
        return {"source_id": source_id, "kind": row["kind"], "scope": row["scope"],
                "version": version, "data": json.loads(row["payload"])}

    def head_sources(self, case_id):
        rows = self.conn.execute(
            "SELECT s.* FROM sources s JOIN source_heads h ON s.case_id=h.case_id "
            "AND s.source_id=h.source_id AND s.version=h.version WHERE s.case_id=?",
            (case_id,),
        ).fetchall()
        return {r["source_id"]: {"source_id": r["source_id"], "kind": r["kind"],
                                 "scope": r["scope"], "version": r["version"],
                                 "data": json.loads(r["payload"])}
                for r in rows}

    def source_versions(self, case_id):
        rows = self.conn.execute(
            "SELECT source_id, MAX(version) AS v FROM sources WHERE case_id=? "
            "GROUP BY source_id", (case_id,),
        ).fetchall()
        return {r["source_id"]: int(r["v"]) for r in rows}

    # ---------- config versions ----------
    def put_config(self, case_id, payload, label=None, editor=None):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0) AS v FROM config_versions WHERE case_id=?",
            (case_id,),
        ).fetchone()
        version = int(row["v"]) + 1
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO config_versions(case_id,version,payload,label,created_at)"
                " VALUES(?,?,?,?,?)",
                (case_id, version,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 label, time.time()),
            )
            self.add_event(case_id, "config-revision",
                           {"version": version, "label": label, "editor": editor})
        return version

    def get_config(self, case_id, version=None):
        if version is None:
            row = self.conn.execute(
                "SELECT * FROM config_versions WHERE case_id=? ORDER BY version DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM config_versions WHERE case_id=? AND version=?",
                (case_id, version),
            ).fetchone()
        if row is None:
            return None
        return {"version": row["version"], "label": row["label"],
                "payload": json.loads(row["payload"])}

    # ---------- derivations ----------
    def next_derivation_version(self, case_id):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0)+1 AS v FROM derivations WHERE case_id=?",
            (case_id,),
        ).fetchone()
        return int(row["v"])

    def current_derivation_version(self, case_id, branch="main"):
        row = self.conn.execute(
            "SELECT derivation_version FROM branches WHERE case_id=? AND name=?",
            (case_id, branch),
        ).fetchone()
        return None if row is None else int(row["derivation_version"])

    def commit_derivation(self, case_id, config_version, nodes, base_sources,
                          fingerprint_value, branch="main", label=None,
                          propagation=None, editor=None, crash_after_nodes=False):
        """Atomically build a derivation and flip the branch pointer.

        With ``crash_after_nodes=True`` the connection is killed after nodes
        are written but before status/branch update, simulating a power loss.
        """
        version = self.next_derivation_version(case_id)
        derivation_id = "der-" + uuid.uuid4().hex[:12]
        base = (branch, self.current_derivation_version(case_id, branch))
        now = time.time()
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO derivations(derivation_id,case_id,version,status,"
                "base_sources,config_version,fingerprint,created_at,crash_marker)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (derivation_id, case_id, version, "building",
                 json.dumps(base_sources, sort_keys=True), config_version,
                 fingerprint_value, now, 1 if crash_after_nodes else 0),
            )
            for seq, node in enumerate(nodes):
                conn.execute(
                    "INSERT INTO nodes(case_id,derivation_version,node_id,kind,"
                    "ping_id,cell_id,status,rule,fingerprint,upstream,depends_on,"
                    "source_ids,payload,seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (case_id, version, node["node_id"], node["kind"],
                     node.get("ping"), node.get("cell"), node["status"],
                     node["rule"], node["fingerprint"],
                     json.dumps(node.get("upstream", [])),
                     json.dumps(node.get("depends_on", [])),
                     json.dumps(node.get("source_ids", [])),
                     json.dumps(node.get("value", {}), ensure_ascii=False, sort_keys=True),
                     seq),
                )
                stable_key = node["kind"] + ":" + (node.get("ping") or "-") + ":" + (
                    node.get("cell") or "-")
                conn.execute(
                    "INSERT OR IGNORE INTO node_identity(case_id,stable_key,node_id,"
                    "first_seen_derivation) VALUES(?,?,?,?)",
                    (case_id, stable_key, node["node_id"], version),
                )

            if crash_after_nodes:
                # Simulate process death: abandon the transaction abruptly.
                conn.interrupt()
                conn.close()
                raise RuntimeError("simulated crash during derivation write")

            conn.execute(
                "UPDATE derivations SET status='ready', committed_at=?, "
                "propagation=? WHERE case_id=? AND version=?",
                (time.time(),
                 json.dumps(propagation, ensure_ascii=False, sort_keys=True)
                 if propagation else None, case_id, version),
            )
            conn.execute(
                "INSERT INTO branches(case_id,name,derivation_version,base_version,"
                "updated_at) VALUES(?,?,?,?,?) ON CONFLICT(case_id,name) DO UPDATE SET "
                "derivation_version=excluded.derivation_version, updated_at=excluded.updated_at",
                (case_id, branch, version,
                 base[1] if base[1] is not None else version, time.time()),
            )
            conn.execute(
                "INSERT INTO events(case_id,seq,type,branch_name,derivation_version,"
                "payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (case_id, self._next_event_seq(case_id), "derivation",
                 branch, version,
                 json.dumps({"version": version, "fingerprint": fingerprint_value,
                             "label": label, "editor": editor,
                             "propagation": propagation}, ensure_ascii=False, sort_keys=True),
                 time.time()),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        return version

    def derivation(self, case_id, version):
        row = self.conn.execute(
            "SELECT * FROM derivations WHERE case_id=? AND version=?",
            (case_id, version),
        ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"], "status": row["status"],
            "config_version": row["config_version"],
            "fingerprint": row["fingerprint"],
            "base_sources": json.loads(row["base_sources"]),
            "propagation": json.loads(row["propagation"]) if row["propagation"] else None,
            "created_at": row["created_at"], "committed_at": row["committed_at"],
        }

    def list_derivations(self, case_id):
        rows = self.conn.execute(
            "SELECT * FROM derivations WHERE case_id=? ORDER BY version", (case_id,)
        ).fetchall()
        return [self.derivation(case_id, r["version"]) for r in rows]

    def list_branches(self, case_id):
        rows = self.conn.execute(
            "SELECT * FROM branches WHERE case_id=? ORDER BY name", (case_id,)
        ).fetchall()
        return [{"name": r["name"], "derivation_version": r["derivation_version"],
                 "base_version": r["base_version"], "updated_at": r["updated_at"]}
                for r in rows]

    def set_branch(self, case_id, name, derivation_version, base_version=None):
        cur = self.current_derivation_version(case_id, name)
        if cur is not None and base_version is not None and cur != base_version:
            raise Conflict("branch moved since your edit", base_version, cur)
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO branches(case_id,name,derivation_version,base_version,"
                "updated_at) VALUES(?,?,?,?,?) ON CONFLICT(case_id,name) DO UPDATE SET "
                "derivation_version=excluded.derivation_version, updated_at=excluded.updated_at",
                (case_id, name, derivation_version,
                 base_version if base_version is not None else derivation_version,
                 time.time()),
            )

    # ---------- nodes ----------
    def nodes(self, case_id, derivation_version):
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE case_id=? AND derivation_version=? ORDER BY seq",
            (case_id, derivation_version),
        ).fetchall()
        return [self._node_dict(r) for r in rows]

    def node(self, case_id, derivation_version, node_id):
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE case_id=? AND derivation_version=? AND node_id=?",
            (case_id, derivation_version, node_id),
        ).fetchone()
        return None if row is None else self._node_dict(row)

    @staticmethod
    def _node_dict(r):
        return {
            "node_id": r["node_id"], "kind": r["kind"], "ping": r["ping_id"],
            "cell": r["cell_id"], "status": r["status"], "rule": r["rule"],
            "fingerprint": r["fingerprint"], "upstream": json.loads(r["upstream"]),
            "depends_on": json.loads(r["depends_on"]),
            "source_ids": json.loads(r["source_ids"]),
            "value": json.loads(r["payload"]), "seq": r["seq"],
            "derivation_version": r["derivation_version"],
        }

    # ---------- decisions ----------
    def add_decision(self, case_id, node_id, outcome, branch, derivation_version,
                     chosen_candidate_id=None, parameter_overrides=None, note=None,
                     editor=None):
        decision_id = "dec-" + uuid.uuid4().hex[:12]
        basis = {
            "branch": branch,
            "derivation_version": derivation_version,
            "node_fingerprint": self._node_fp(case_id, derivation_version, node_id),
        }
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO decisions(case_id,decision_id,branch_name,"
                "derivation_version,node_id,outcome,chosen_candidate_id,"
                "parameter_overrides,basis_versions,created_at,note) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, decision_id, branch, derivation_version, node_id, outcome,
                 chosen_candidate_id,
                 json.dumps(parameter_overrides, ensure_ascii=False, sort_keys=True)
                 if parameter_overrides is not None else None,
                 json.dumps(basis, sort_keys=True), time.time(), note),
            )
            self.add_event(case_id, "decision",
                           {"decision_id": decision_id, "node_id": node_id,
                            "outcome": outcome,
                            "chosen_candidate_id": chosen_candidate_id,
                            "parameter_overrides": parameter_overrides,
                            "basis": basis, "editor": editor, "note": note},
                           branch, derivation_version)
        return decision_id

    def _node_fp(self, case_id, version, node_id):
        row = self.conn.execute(
            "SELECT fingerprint FROM nodes WHERE case_id=? AND derivation_version=? "
            "AND node_id=?", (case_id, version, node_id),
        ).fetchone()
        return None if row is None else row["fingerprint"]

    def decisions(self, case_id):
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE case_id=? ORDER BY created_at", (case_id,)
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "decision_id": r["decision_id"], "node_id": r["node_id"],
                "outcome": r["outcome"], "branch_name": r["branch_name"],
                "derivation_version": r["derivation_version"],
                "chosen_candidate_id": r["chosen_candidate_id"],
                "parameter_overrides": json.loads(r["parameter_overrides"])
                if r["parameter_overrides"] else None,
                "basis_versions": json.loads(r["basis_versions"]),
                "note": r["note"], "created_at": r["created_at"],
            })
        return out

    def add_tag(self, case_id, tag, branch=None, derivation_version=None):
        branch = branch or "main"
        if derivation_version is None:
            derivation_version = self.current_derivation_version(case_id, branch)
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO tags(case_id,tag,branch_name,derivation_version,created_at)"
                " VALUES(?,?,?,?,?) ON CONFLICT(case_id,tag) DO UPDATE SET "
                "branch_name=excluded.branch_name, derivation_version=excluded.derivation_version",
                (case_id, tag, branch, derivation_version, time.time()),
            )
            self.add_event(case_id, "tag", {"tag": tag, "branch": branch,
                                            "derivation_version": derivation_version})

    def tags(self, case_id):
        rows = self.conn.execute("SELECT * FROM tags WHERE case_id=? ORDER BY tag",
                                 (case_id,)).fetchall()
        return [{"tag": r["tag"], "branch_name": r["branch_name"],
                 "derivation_version": r["derivation_version"]} for r in rows]

    # ---------- case lifecycle ----------
    def ensure_case(self, case_id, title, schema_name):
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR IGNORE INTO cases(case_id,title,schema_name,created_at) "
                "VALUES(?,?,?,?)", (case_id, title, schema_name, time.time()),
            )

    def list_cases(self):
        rows = self.conn.execute("SELECT * FROM cases ORDER BY created_at").fetchall()
        return [{"case_id": r["case_id"], "title": r["title"],
                 "schema_name": r["schema_name"], "created_at": r["created_at"]}
                for r in rows]

    def export_case(self, case_id):
        """Full deterministic snapshot; node identities are explicit, never rowids."""
        return {
            "export_schema": "adcp-case-export-v1",
            "case": next((c for c in self.list_cases() if c["case_id"] == case_id), None),
            "sources": [
                {"source_id": sid, "versions": self._source_versions(case_id, sid)}
                for sid in sorted(self.source_versions(case_id))
            ],
            "config_versions": [self.get_config(case_id, v["version"])
                                 for v in self.conn.execute(
                                     "SELECT version FROM config_versions WHERE case_id=? "
                                     "ORDER BY version", (case_id,)).fetchall()],
            "derivations": [self._export_derivation(case_id, d["version"])
                            for d in self.list_derivations(case_id)
                            if d["status"] == "ready"],
            "branches": self.list_branches(case_id),
            "decisions": self.decisions(case_id),
            "events": self.list_events(case_id),
            "tags": self.tags(case_id),
            "node_identity": [
                {"stable_key": r["stable_key"], "node_id": r["node_id"],
                 "first_seen_derivation": r["first_seen_derivation"]}
                for r in self.conn.execute(
                    "SELECT * FROM node_identity WHERE case_id=? ORDER BY stable_key",
                    (case_id,)).fetchall()],
        }

    def _source_versions(self, case_id, sid):
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE case_id=? AND source_id=? ORDER BY version",
            (case_id, sid),
        ).fetchall()
        return [{"version": r["version"], "kind": r["kind"], "scope": r["scope"],
                 "origin": r["origin"], "data": json.loads(r["payload"]),
                 "created_at": r["created_at"]} for r in rows]

    def _export_derivation(self, case_id, version):
        d = self.derivation(case_id, version)
        d["nodes"] = self.nodes(case_id, version)
        return d

    def import_case(self, snapshot, case_id_override=None, origin="import"):
        case_id = case_id_override or snapshot["case"]["case_id"]
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR IGNORE INTO cases(case_id,title,schema_name,created_at) "
                "VALUES(?,?,?,?)",
                (case_id, snapshot["case"]["title"], snapshot["case"].get("schema_name"),
                 snapshot["case"].get("created_at", time.time())),
            )
            for src in snapshot["sources"]:
                for v in src["versions"]:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO sources(source_id,case_id,kind,scope,"
                        "version,payload,origin,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (src["source_id"], case_id, v["kind"], v["scope"], v["version"],
                         json.dumps(v["data"], ensure_ascii=False, sort_keys=True),
                         v.get("origin", origin), v.get("created_at", time.time())),
                    )
                self.conn.execute(
                    "INSERT INTO source_heads(case_id,source_id,version) VALUES(?,?,?) "
                    "ON CONFLICT(case_id,source_id) DO UPDATE SET version=excluded.version",
                    (case_id, src["source_id"], src["versions"][-1]["version"]),
                )
            for cfg in snapshot["config_versions"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO config_versions(case_id,version,payload,"
                    "label,created_at) VALUES(?,?,?,?,?)",
                    (case_id, cfg["version"],
                     json.dumps(cfg["payload"], ensure_ascii=False, sort_keys=True),
                     cfg.get("label"), time.time()),
                )
            for der in snapshot["derivations"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO derivations(derivation_id,case_id,version,"
                    "status,base_sources,config_version,fingerprint,created_at,"
                    "committed_at,propagation) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("der-imp-%03d" % der["version"], case_id, der["version"], "ready",
                     json.dumps(der["base_sources"], sort_keys=True),
                     der["config_version"], der["fingerprint"],
                     der.get("created_at", time.time()), der.get("committed_at"),
                     json.dumps(der.get("propagation"), sort_keys=True)
                     if der.get("propagation") else None),
                )
                for node in der["nodes"]:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO nodes(case_id,derivation_version,node_id,"
                        "kind,ping_id,cell_id,status,rule,fingerprint,upstream,"
                        "depends_on,source_ids,payload,seq) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (case_id, der["version"], node["node_id"], node["kind"],
                         node.get("ping"), node.get("cell"), node["status"],
                         node["rule"], node["fingerprint"],
                         json.dumps(node.get("upstream", [])),
                         json.dumps(node.get("depends_on", [])),
                         json.dumps(node.get("source_ids", [])),
                         json.dumps(node.get("value", {}), ensure_ascii=False, sort_keys=True),
                         node.get("seq", 0)),
                    )
            for b in snapshot["branches"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO branches(case_id,name,derivation_version,"
                    "base_version,updated_at) VALUES(?,?,?,?,?)",
                    (case_id, b["name"], b["derivation_version"], b["base_version"],
                     b.get("updated_at", time.time())),
                )
            for dec in snapshot.get("decisions", []):
                self.conn.execute(
                    "INSERT OR REPLACE INTO decisions(case_id,decision_id,branch_name,"
                    "derivation_version,node_id,outcome,chosen_candidate_id,"
                    "parameter_overrides,basis_versions,created_at,note) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (case_id, dec["decision_id"], dec["branch_name"],
                     dec["derivation_version"], dec["node_id"], dec["outcome"],
                     dec.get("chosen_candidate_id"),
                     json.dumps(dec["parameter_overrides"], ensure_ascii=False, sort_keys=True)
                     if dec.get("parameter_overrides") is not None else None,
                     json.dumps(dec["basis_versions"], sort_keys=True),
                     dec.get("created_at", time.time()), dec.get("note")),
                )
            for ev in snapshot["events"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO events(case_id,seq,type,branch_name,"
                    "derivation_version,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                    (case_id, ev["seq"], ev["type"], ev.get("branch_name"),
                     ev.get("derivation_version"),
                     json.dumps(ev["payload"], ensure_ascii=False, sort_keys=True),
                     ev.get("created_at", time.time())),
                )
            for ni in snapshot.get("node_identity", []):
                self.conn.execute(
                    "INSERT OR IGNORE INTO node_identity(case_id,stable_key,node_id,"
                    "first_seen_derivation) VALUES(?,?,?,?)",
                    (case_id, ni["stable_key"], ni["node_id"],
                     ni["first_seen_derivation"]),
                )
        return case_id
