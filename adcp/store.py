"""SQLite 版本化存储层。

核心保证:
- 来源分表: beam_measurement / correlation / amplitude / bad_beam_flag /
  heading_pitch_roll / declination / mount_config / bottom_track / transducer_config
  各自独立记录, 修订只追加新版本 (supersedes 链), 永不就地覆盖原始表示。
- 每次推导产生独立 version 行, 所有 version_sources/results/events 在**一个
  事务**内写入; 指针 current_version 只在完整成功后切换。写入中断后重启,
  读到的是旧版本或完整新版本, 不存在半成品指针 (WAL + 事务回滚保证)。
- 所有行携带显式稳定字符串 id; 导出 JSON 再导入新库后 id/顺序/来源关系不变。
- 两位使用者从同一 parent_version 编辑 -> 第二个 commit 抛 ConcurrentEdit,
  必须经 resolve_conflict 显式合并 (三方: base/theirs/ours), 冲突事件入库。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fingerprint import content_hash

SCHEMA_VERSION = 2

SOURCE_KINDS = (
    "beam_measurement",
    "correlation",
    "amplitude",
    "bad_beam_flag",
    "heading_pitch_roll",
    "declination",
    "mount_config",
    "bottom_track",
    "transducer_config",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    seq INTEGER NOT NULL,
    supersedes TEXT,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    raw_digest TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS idx_sources_kind_scope ON sources(kind, scope);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    parent_version TEXT,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    author TEXT NOT NULL,
    status TEXT NOT NULL,
    basis_event_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS version_sources (
    version_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (version_id, source_id)
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    candidate_id TEXT,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    excluded_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_results_version ON results(version_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_identity
    ON results(version_id, node_type, scope, seq, COALESCE(candidate_id, ''));
CREATE TABLE IF NOT EXISTS dependencies (
    version_id TEXT NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (version_id, from_node, to_node, relation)
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL UNIQUE,
    seq INTEGER NOT NULL,
    parent_event_id TEXT,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    assertions_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_events_seq ON events(seq);
CREATE TABLE IF NOT EXISTS pointers (
    name TEXT PRIMARY KEY,
    version_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    node_scope TEXT NOT NULL,
    chosen_candidate_id TEXT,
    rejected_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
"""


class ConcurrentEdit(Exception):
    """两个使用者从同一 parent_version 分叉时抛出, 必须显式解决。"""


class MissingVersion(KeyError):
    pass


def _now() -> float:
    return time.time()


class Store:
    def __init__(self, path: str = ":memory:", *, clock=_now):
        self.path = path
        self._clock = clock
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                        exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = FULL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self.transaction() as c:
            c.executescript(SCHEMA_SQL)
            row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                c.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                          (str(SCHEMA_VERSION),))
            else:
                self._migrate(c, int(row["value"]))

    def _migrate(self, c: sqlite3.Connection, found: int) -> None:
        if found < 2:
            # v1 -> v2: 增加 decisions 表与 basis_event_id 列; 旧数据不丢,
            # 缺失字段填默认值。测试 tests/test_migration.py 实际走这条路径。
            cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
            if "basis_event_id" not in cols:
                c.execute("ALTER TABLE versions ADD COLUMN basis_event_id TEXT")
            c.execute(
                "CREATE TABLE IF NOT EXISTS decisions ("
                "id TEXT PRIMARY KEY, version_id TEXT NOT NULL, node_scope TEXT NOT NULL, "
                "chosen_candidate_id TEXT, rejected_candidate_ids_json TEXT NOT NULL DEFAULT '[]', "
                "note TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)")
            c.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                      (str(SCHEMA_VERSION),))

    # ------------------------------------------------------------------ sources
    def add_source(self, *, source_id: str, kind: str, scope: str,
                   payload: Dict[str, Any], supersedes: Optional[str] = None,
                   author: str = "system",
                   created_at: Optional[float] = None) -> Dict[str, Any]:
        if kind not in SOURCE_KINDS:
            raise ValueError("unknown source kind %r" % kind)
        created_at = self._clock() if created_at is None else created_at
        seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM sources WHERE kind=? AND scope=?",
            (kind, scope)).fetchone()
        record = {
            "id": source_id,
            "kind": kind,
            "scope": scope,
            "seq": seq_row["n"],
            "supersedes": supersedes,
            "created_at": created_at,
            "payload": payload,
            "raw_digest": content_hash(payload),
            "author": author,
        }
        with self.transaction() as c:
            c.execute(
                "INSERT INTO sources(id, kind, scope, seq, supersedes, created_at, "
                "payload_json, raw_digest, author) VALUES (?,?,?,?,?,?,?,?,?)",
                (record["id"], record["kind"], record["scope"], record["seq"],
                 record["supersedes"], record["created_at"],
                 json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 record["raw_digest"], author))
        return record

    def get_source(self, source_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM sources WHERE id=?",
                                (source_id,)).fetchone()
        if row is None:
            raise KeyError(source_id)
        return self._source_row(row)

    @staticmethod
    def _source_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"], "kind": row["kind"], "scope": row["scope"],
            "seq": row["seq"], "supersedes": row["supersedes"],
            "created_at": row["created_at"], "author": row["author"],
            "raw_digest": row["raw_digest"],
            "payload": json.loads(row["payload_json"]),
        }

    def sources_for_version(self, version_id: str,
                            kind: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = ("SELECT s.* FROM sources s JOIN version_sources vs "
               "ON s.id = vs.source_id WHERE vs.version_id=?")
        args: List[Any] = [version_id]
        if kind:
            sql += " AND s.kind=?"
            args.append(kind)
        sql += " ORDER BY s.kind, s.seq"
        return [self._source_row(r) for r in self.conn.execute(sql, args)]

    def latest_source(self, kind: str, scope: str = "default") -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE kind=? AND scope=? ORDER BY seq DESC LIMIT 1",
            (kind, scope)).fetchone()
        return self._source_row(row) if row else None

    # ----------------------------------------------------------------- versions
    def current_version_id(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT version_id FROM pointers WHERE name='current'").fetchone()
        return row["version_id"] if row else None

    def get_version(self, version_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM versions WHERE id=?",
                                (version_id,)).fetchone()
        if row is None:
            raise MissingVersion(version_id)
        return {
            "id": row["id"], "seq": row["seq"], "parent_version": row["parent_version"],
            "label": row["label"], "kind": row["kind"], "created_at": row["created_at"],
            "author": row["author"], "status": row["status"],
            "basis_event_id": row["basis_event_id"],
            "payload": json.loads(row["payload_json"]),
        }

    def list_versions(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM versions ORDER BY seq").fetchall()
        return [self.get_version(r["id"]) for r in rows]

    def commit_plan(self, *, version_id: str, label: str, kind: str,
                    parent_version: Optional[str], author: str,
                    source_ids: Sequence[str],
                    results: Iterable[Dict[str, Any]],
                    dependencies: Iterable[Tuple[str, str, str]] = (),
                    events_payload: Optional[Dict[str, Any]] = None,
                    event_type: str = "derivation",
                    assertions: Optional[List[Dict[str, Any]]] = None,
                    basis_event_id: Optional[str] = None,
                    make_current: bool = True,
                    payload: Optional[Dict[str, Any]] = None,
                    created_at: Optional[float] = None) -> Dict[str, Any]:
        """原子地写入一个完整版本。

        并发规则: parent_version 必须等于当前指针 (分叉点), 否则 ConcurrentEdit。
        全部写入在一个事务内, 指针最后切换; 任一步失败整体回滚, 旧版本仍在。
        """
        created_at = self._clock() if created_at is None else created_at
        current = self.current_version_id()
        if parent_version != current:
            raise ConcurrentEdit(
                "parent %r is not current %r" % (parent_version, current))
        seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM versions").fetchone()
        event_seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM events").fetchone()
        event_id = "ev-" + version_id
        version = {
            "id": version_id,
            "seq": seq_row["n"],
            "parent_version": parent_version,
            "label": label,
            "kind": kind,
            "created_at": created_at,
            "author": author,
            "status": "complete",
            "basis_event_id": basis_event_id,
            "payload": payload or {},
        }
        results = list(results)
        assertions = assertions or []
        events_payload = events_payload or {}
        try:
            with self.transaction() as c:
                c.execute(
                    "INSERT INTO versions(id, seq, parent_version, label, kind, "
                    "created_at, author, status, basis_event_id, payload_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (version["id"], version["seq"], version["parent_version"],
                     version["label"], version["kind"], version["created_at"],
                     version["author"], version["status"], version["basis_event_id"],
                     json.dumps(version["payload"], ensure_ascii=False, sort_keys=True)))
                c.executemany(
                    "INSERT INTO version_sources(version_id, source_id) VALUES (?,?)",
                    [(version_id, sid) for sid in source_ids])
                c.executemany(
                    "INSERT INTO results(id, version_id, node_type, scope, seq, "
                    "candidate_id, fingerprint, status, payload_json, excluded_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(r["id"], version_id, r["node_type"], r["scope"],
                      r.get("seq", 0), r.get("candidate_id"), r["fingerprint"],
                      r["status"], json.dumps(r.get("payload", {}), ensure_ascii=False,
                                              sort_keys=True),
                      json.dumps(r.get("excluded", []), ensure_ascii=False))
                     for r in results])
                c.executemany(
                    "INSERT OR REPLACE INTO dependencies(version_id, from_node, "
                    "to_node, relation) VALUES (?,?,?,?)",
                    [(version_id, a, b, rel) for (a, b, rel) in dependencies])
                c.execute(
                    "INSERT INTO events(id, version_id, seq, parent_event_id, "
                    "event_type, actor, created_at, payload_json, assertions_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (event_id, version_id, event_seq_row["n"],
                     ("ev-" + parent_version) if parent_version else None,
                     event_type, author, created_at,
                     json.dumps(events_payload, ensure_ascii=False, sort_keys=True),
                     json.dumps(assertions, ensure_ascii=False, sort_keys=True)))
                if make_current:
                    c.execute(
                        "INSERT INTO pointers(name, version_id) VALUES ('current', ?) "
                        "ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                        (version_id,))
        except Exception:
            raise
        return version

    # ------------------------------------------------------------------ results
    @staticmethod
    def _result_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"], "version_id": row["version_id"],
            "node_type": row["node_type"], "scope": row["scope"],
            "seq": row["seq"], "candidate_id": row["candidate_id"],
            "fingerprint": row["fingerprint"], "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "excluded": json.loads(row["excluded_json"]),
        }

    def results(self, version_id: str,
                node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM results WHERE version_id=?"
        args: List[Any] = [version_id]
        if node_type:
            sql += " AND node_type=?"
            args.append(node_type)
        sql += " ORDER BY scope, seq, COALESCE(candidate_id, '')"
        return [self._result_row(r) for r in self.conn.execute(sql, args)]

    def result(self, result_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM results WHERE id=?",
                                (result_id,)).fetchone()
        if row is None:
            raise KeyError(result_id)
        return self._result_row(row)

    def dependencies(self, version_id: str) -> List[Dict[str, str]]:
        rows = self.conn.execute(
            "SELECT * FROM dependencies WHERE version_id=?", (version_id,))
        return [{"from": r["from_node"], "to": r["to_node"],
                 "relation": r["relation"]} for r in rows]

    # ------------------------------------------------------------------- events
    @staticmethod
    def _event_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"], "version_id": row["version_id"], "seq": row["seq"],
            "parent_event_id": row["parent_event_id"],
            "event_type": row["event_type"], "actor": row["actor"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload_json"]),
            "assertions": json.loads(row["assertions_json"]),
        }

    def events(self) -> List[Dict[str, Any]]:
        return [self._event_row(r) for r in
                self.conn.execute("SELECT * FROM events ORDER BY seq")]

    def event_for_version(self, version_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM events WHERE version_id=?",
                                (version_id,)).fetchone()
        if row is None:
            raise KeyError(version_id)
        return self._event_row(row)

    def version_at_event(self, event_id: str) -> Dict[str, Any]:
        """时间旅行: 事件对应的版本快照 + 当时指针所依据的来源集合。"""
        ev = self.conn.execute("SELECT * FROM events WHERE id=?",
                               (event_id,)).fetchone()
        if ev is None:
            raise KeyError(event_id)
        return self.get_version(ev["version_id"])

    # ---------------------------------------------------------------- decisions
    def add_decision(self, *, decision_id: str, version_id: str, node_scope: str,
                     chosen_candidate_id: Optional[str],
                     rejected_candidate_ids: Sequence[str] = (),
                     note: str = "", created_at: Optional[float] = None) -> None:
        created_at = self._clock() if created_at is None else created_at
        with self.transaction() as c:
            c.execute(
                "INSERT INTO decisions(id, version_id, node_scope, "
                "chosen_candidate_id, rejected_candidate_ids_json, note, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (decision_id, version_id, node_scope, chosen_candidate_id,
                 json.dumps(list(rejected_candidate_ids)), note, created_at))

    def decisions(self, version_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM decisions WHERE version_id=?",
                                 (version_id,))
        return [{"id": r["id"], "version_id": r["version_id"],
                 "node_scope": r["node_scope"], "chosen_candidate_id": r["chosen_candidate_id"],
                 "rejected_candidate_ids": json.loads(r["rejected_candidate_ids_json"]),
                 "note": r["note"], "created_at": r["created_at"]} for r in rows]

    # ------------------------------------------------------------- conflict mgmt
    def force_pointer(self, version_id: str) -> None:
        """冲突解决合并提交后, 显式把指针指向合并版本。"""
        with self.transaction() as c:
            c.execute("INSERT INTO pointers(name, version_id) VALUES ('current', ?) "
                      "ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                      (version_id,))

    def merge_commit(self, *, version_id: str, label: str, author: str,
                     base_version: str, parent_version: str,
                     source_ids: Sequence[str],
                     results: Iterable[Dict[str, Any]],
                     dependencies: Iterable[Tuple[str, str, str]] = (),
                     events_payload: Optional[Dict[str, Any]] = None,
                     assertions: Optional[List[Dict[str, Any]]] = None,
                     resolution_note: str = "",
                     payload: Optional[Dict[str, Any]] = None,
                     created_at: Optional[float] = None) -> Dict[str, Any]:
        """冲突解决: 绕过 parent==current 的检查 (parent 是分叉点), 但留 conflict
        事件记录 base/theirs/ours, 指针显式切换到合并结果。"""
        created_at = self._clock() if created_at is None else created_at
        seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM versions").fetchone()
        event_seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM events").fetchone()
        event_id = "ev-" + version_id
        payload = dict(payload or {})
        payload.update({"merge": {"base": base_version, "parents": [parent_version],
                                  "resolution_note": resolution_note}})
        results = list(results)
        with self.transaction() as c:
            c.execute(
                "INSERT INTO versions(id, seq, parent_version, label, kind, "
                "created_at, author, status, basis_event_id, payload_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (version_id, seq_row["n"], parent_version, label, "conflict-resolution",
                 created_at, author, "complete", None,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True)))
            c.executemany(
                "INSERT INTO version_sources(version_id, source_id) VALUES (?,?)",
                [(version_id, sid) for sid in source_ids])
            c.executemany(
                "INSERT INTO results(id, version_id, node_type, scope, seq, "
                "candidate_id, fingerprint, status, payload_json, excluded_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(r["id"], version_id, r["node_type"], r["scope"], r.get("seq", 0),
                  r.get("candidate_id"), r["fingerprint"], r["status"],
                  json.dumps(r.get("payload", {}), ensure_ascii=False, sort_keys=True),
                  json.dumps(r.get("excluded", []), ensure_ascii=False))
                 for r in results])
            c.executemany(
                "INSERT OR REPLACE INTO dependencies(version_id, from_node, "
                "to_node, relation) VALUES (?,?,?,?)",
                [(version_id, a, b, rel) for (a, b, rel) in dependencies])
            c.execute(
                "INSERT INTO events(id, version_id, seq, parent_event_id, "
                "event_type, actor, created_at, payload_json, assertions_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (event_id, version_id, event_seq_row["n"], "ev-" + parent_version,
                 "conflict_resolution", author, created_at,
                 json.dumps(events_payload or {"resolution_note": resolution_note},
                            ensure_ascii=False, sort_keys=True),
                 json.dumps(assertions or [], ensure_ascii=False, sort_keys=True)))
            c.execute("INSERT INTO pointers(name, version_id) VALUES ('current', ?) "
                      "ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                      (version_id,))
        return self.get_version(version_id)

    # ----------------------------------------------------------- export/import
    def export_snapshot(self) -> Dict[str, Any]:
        """整库导出为自描述 JSON; 行按 seq/主键排序, 导入后身份与顺序一致。"""
        def rows(table: str, order: str) -> List[sqlite3.Row]:
            return list(self.conn.execute(
                "SELECT * FROM %s ORDER BY %s" % (table, order)))
        return {
            "format": "adcp-evidence-snapshot",
            "format_version": 1,
            "schema_version": SCHEMA_VERSION,
            "exported_at": self._clock(),
            "meta": [{"key": r["key"], "value": r["value"]}
                     for r in rows("meta", "key")],
            "sources": [dict(r) for r in rows("sources", "kind, scope, seq")],
            "versions": [dict(r) for r in rows("versions", "seq")],
            "version_sources": [dict(r)
                                for r in rows("version_sources", "version_id, source_id")],
            "results": [dict(r) for r in rows(
                "results", "version_id, scope, seq, node_type, candidate_id")],
            "dependencies": [dict(r)
                             for r in rows("dependencies", "version_id, from_node")],
            "events": [dict(r) for r in rows("events", "seq")],
            "pointers": [dict(r) for r in rows("pointers", "name")],
            "decisions": [dict(r) for r in rows("decisions", "version_id")],
        }

    @staticmethod
    def import_snapshot(snapshot: Dict[str, Any], path: str = ":memory:") -> "Store":
        """导入到全新库。显式 id 原样恢复, 因此删除本地数据库再导入后,
        所有显示值的稳定身份、顺序与来源关系保持不变。"""
        if snapshot.get("format") != "adcp-evidence-snapshot":
            raise ValueError("not an adcp evidence snapshot")
        store = Store(path)
        with store.transaction() as c:
            for r in snapshot["meta"]:
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
                          (r["key"], r["value"]))
            for r in snapshot["sources"]:
                c.execute("INSERT INTO sources(id, kind, scope, seq, supersedes, "
                          "created_at, payload_json, raw_digest, author) "
                          "VALUES (?,?,?,?,?,?,?,?,?)",
                          (r["id"], r["kind"], r["scope"], r["seq"], r["supersedes"],
                           r["created_at"], r["payload_json"], r["raw_digest"],
                           r.get("author", "system")))
            for r in snapshot["versions"]:
                c.execute("INSERT INTO versions(id, seq, parent_version, label, kind, "
                          "created_at, author, status, basis_event_id, payload_json) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (r["id"], r["seq"], r["parent_version"], r["label"], r["kind"],
                           r["created_at"], r["author"], r["status"],
                           r.get("basis_event_id"), r["payload_json"]))
            for r in snapshot["version_sources"]:
                c.execute("INSERT INTO version_sources(version_id, source_id) "
                          "VALUES (?,?)", (r["version_id"], r["source_id"]))
            for r in snapshot["results"]:
                c.execute("INSERT INTO results(id, version_id, node_type, scope, seq, "
                          "candidate_id, fingerprint, status, payload_json, excluded_json) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (r["id"], r["version_id"], r["node_type"], r["scope"], r["seq"],
                           r["candidate_id"], r["fingerprint"], r["status"],
                           r["payload_json"], r.get("excluded_json", "[]")))
            for r in snapshot["dependencies"]:
                c.execute("INSERT OR REPLACE INTO dependencies(version_id, from_node, "
                          "to_node, relation) VALUES (?,?,?,?)",
                          (r["version_id"], r["from_node"], r["to_node"], r["relation"]))
            for r in snapshot["events"]:
                c.execute("INSERT INTO events(id, version_id, seq, parent_event_id, "
                          "event_type, actor, created_at, payload_json, assertions_json) "
                          "VALUES (?,?,?,?,?,?,?,?,?)",
                          (r["id"], r["version_id"], r["seq"], r["parent_event_id"],
                           r["event_type"], r["actor"], r["created_at"],
                           r["payload_json"], r.get("assertions_json", "[]")))
            for r in snapshot["pointers"]:
                c.execute("INSERT INTO pointers(name, version_id) VALUES (?,?) "
                          "ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                          (r["name"], r["version_id"]))
            for r in snapshot.get("decisions", []):
                c.execute("INSERT INTO decisions(id, version_id, node_scope, "
                          "chosen_candidate_id, rejected_candidate_ids_json, note, "
                          "created_at) VALUES (?,?,?,?,?,?,?)",
                          (r["id"], r["version_id"], r["node_scope"],
                           r["chosen_candidate_id"],
                           r.get("rejected_candidate_ids_json", "[]"),
                           r.get("note", ""), r["created_at"]))
        return store
