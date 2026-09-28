"""SQLite 存储层：分开记录直接来源，内容寻址派生，当前指针原子切换。

一致性模型：
- 每次推导产生独立版本（新 content_id 的不可变派生行）；
- 只有一个事务完整提交后才更新 pointers；写入中断后重启，
  要么看到旧版本（旧指针仍在），要么看到完整新版本（新指针已切换）；
- 同版本编辑用 expected_revision 做乐观并发，冲突写入 conflicts 表，可解决；
- state_events 是只追加的机器可读事件流；(seq, event_id) 保持全序。
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .identity import canonical_json, content_id

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    rev INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    replaced_by TEXT,
    PRIMARY KEY (id)
);
CREATE TABLE IF NOT EXISTS source_current (
  source_type TEXT NOT NULL,
    key TEXT NOT NULL,
    id TEXT NOT NULL,
    rev INTEGER NOT NULL,
    PRIMARY KEY (source_type, key)
);
CREATE TABLE IF NOT EXISTS derived (
  id TEXT PRIMARY KEY,
    node_key TEXT NOT NULL,
    lineage TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT NOT NULL,
    created_event TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pointers (
  node_key TEXT PRIMARY KEY,
    derived_id TEXT NOT NULL,
    event_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    selected_candidate_id TEXT,
    action TEXT NOT NULL,
    based_on_event TEXT NOT NULL,
    rev INTEGER NOT NULL,
    superseded_by TEXT,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflicts (
  id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    winning_event TEXT NOT NULL,
    lost_event TEXT NOT NULL,
    detail TEXT NOT NULL,
    resolved_by TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
  name TEXT PRIMARY KEY,
    at_event TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(SCHEMA)
        cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                         (str(SCHEMA_VERSION),))
        else:
            from .migration import migrate
            migrate(conn, int(row[0]))


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --------------------------------------------------------------- 来源记录

def put_source(conn: sqlite3.Connection, source_type: str, key: str,
               payload: Dict[str, Any], event_id: str,
               expected_rev: Optional[int] = None,
               created_at: Optional[str] = None) -> Tuple[str, int]:
    """写入一版直接来源。expected_rev 给出时做乐观并发检查。

    返回 (source_id, new_rev)。冲突时抛 ConcurrentEdit，原版本保持当前。
    """
    created_at = created_at or _now()
    with conn:
        row = conn.execute(
            "SELECT id, rev FROM source_current WHERE source_type=? AND key=?",
            (source_type, key)).fetchone()
        if row is not None:
            new_rev = row["rev"] + 1
            if expected_rev is not None and expected_rev != row["rev"]:
                raise ConcurrentEdit(source_type, key, row["rev"], expected_rev)
            prev_id = row["id"]
        else:
            if expected_rev not in (None, 0):
                raise ConcurrentEdit(source_type, key, 0, expected_rev)
            new_rev = 1
            prev_id = None
        sid = content_id("src", {
            "source_type": source_type, "key": key, "rev": new_rev,
            "payload": payload, "rule": _rule_anchor(payload),
        })
        conn.execute(
            "INSERT INTO sources(id,source_type,key,revision,rev,payload,created_at,replaced_by)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (sid, source_type, key, new_rev, new_rev,
             canonical_json(payload), created_at, None))
        if prev_id:
            conn.execute("UPDATE sources SET replaced_by=? WHERE id=?", (sid, prev_id))
        conn.execute(
            "INSERT INTO source_current(source_type,key,id,rev) VALUES(?,?,?,?)"
            " ON CONFLICT(source_type,key) DO UPDATE SET id=excluded.id, rev=excluded.rev",
            (source_type, key, sid, new_rev))
    return sid, new_rev


def _rule_anchor(payload: Dict[str, Any]) -> str:
    return str(payload.get("_rule_version", ""))


def get_source_current(conn: sqlite3.Connection, source_type: str, key: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT s.* FROM sources s JOIN source_current c ON c.id=s.id"
        " WHERE c.source_type=? AND c.key=?", (source_type, key)).fetchone()
    return _source_view(row) if row else None


def list_sources_current(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute(
        "SELECT s.* FROM sources s JOIN source_current c ON c.id=s.id"
        " ORDER BY s.source_type, s.key").fetchall()
    return [_source_view(r) for r in rows]


def _source_view(row: sqlite3.Row) -> Dict:
    return {
        "id": row["id"], "source_type": row["source_type"], "key": row["key"],
        "rev": row["rev"], "payload": json.loads(row["payload"]),
        "created_at": row["created_at"],
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ConcurrentEdit(Exception):
    def __init__(self, source_type: str, key: str, current_rev: int, expected_rev: int):
        self.source_type = source_type
        self.key = key
        self.current_rev = current_rev
        self.expected_rev = expected_rev
        super().__init__(f"concurrent edit on {source_type}:{key}: "
                         f"current rev {current_rev}, expected {expected_rev}")


# --------------------------------------------------------------- 事件流

def append_event(conn: sqlite3.Connection, kind: str, payload: Dict[str, Any],
                 event_id: Optional[str] = None,
                 created_at: Optional[str] = None) -> Tuple[int, str]:
    """追加一条机器可读事件。payload 不含时钟，保证可重放。"""
    created_at = created_at or _now()
    eid = event_id or content_id("evt", {"kind": kind, "payload": payload})
    with conn:
        cur = conn.execute(
            "INSERT INTO state_events(event_id,kind,payload,created_at) VALUES(?,?,?,?)",
            (eid, kind, canonical_json(payload), created_at))
        seq = cur.lastrowid
    return seq, eid


def list_events(conn: sqlite3.Connection, upto_event: Optional[str] = None) -> List[Dict]:
    rows = conn.execute("SELECT * FROM state_events ORDER BY seq").fetchall()
    out = []
    for r in rows:
        out.append({"seq": r["seq"], "event_id": r["event_id"], "kind": r["kind"],
                    "payload": json.loads(r["payload"]), "created_at": r["created_at"]})
        if upto_event and r["event_id"] == upto_event:
            break
    return out


# ----------------------------------------------------- 派生与指针原子切换

def commit_derivation(conn: sqlite3.Connection, event_id: str,
                      nodes: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """整批写入不可变派生节点，随后在同一事务内统一切换指针。

    任何异常都会回滚：指针仍指向上一批完整版本。
    节点 status: ok | blocked | multi。
    """
    node_map: Dict[str, str] = {}
    created_at = _now()
    with conn:
        for node in nodes:
            did = content_id("der", {
                "node_key": node["node_key"],
                "lineage": node["lineage"],
                "status": node["status"],
                "result": node["result"],
            })
            conn.execute(
                "INSERT OR IGNORE INTO derived(id,node_key,lineage,status,result,"
                "created_event,created_at) VALUES(?,?,?,?,?,?,?)",
                (did, node["node_key"], canonical_json(node["lineage"]),
                 node["status"], canonical_json(node["result"]), event_id, created_at))
            conn.execute(
                "INSERT INTO pointers(node_key,derived_id,event_id) VALUES(?,?,?)"
                " ON CONFLICT(node_key) DO UPDATE SET derived_id=excluded.derived_id,"
                " event_id=excluded.event_id",
                (node["node_key"], did, event_id))
            node_map[node["node_key"]] = did
    return node_map


def get_current_derived(conn: sqlite3.Connection, node_key: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT d.*, p.event_id AS pointer_event FROM pointers p JOIN derived d ON d.id=p.derived_id"
        " WHERE p.node_key=?", (node_key,)).fetchone()
    return _derived_view(row) if row else None


def list_current_derived(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute(
        "SELECT d.*, p.event_id AS pointer_event FROM pointers p JOIN derived d ON d.id=p.derived_id"
        " ORDER BY d.node_key").fetchall()
    return [_derived_view(r) for r in rows]


def get_derived(conn: sqlite3.Connection, derived_id: str) -> Optional[Dict]:
    row = conn.execute("SELECT * FROM derived WHERE id=?", (derived_id,)).fetchone()
    return _derived_view(row) if row else None


def _derived_view(row: sqlite3.Row) -> Dict:
    return {"id": row["id"], "node_key": row["node_key"],
            "lineage": json.loads(row["lineage"]), "status": row["status"],
            "result": json.loads(row["result"]),
            "created_event": row["created_event"]}


# --------------------------------------------------------------- 人工决定

def put_decision(conn: sqlite3.Connection, scope_key: str, action: str,
                 based_on_event: str, selected_candidate_id: Optional[str],
                 payload: Dict[str, Any], expected_rev: Optional[int] = None,
                 created_at: Optional[str] = None) -> Tuple[str, int]:
    created_at = created_at or _now()
    with conn:
        row = conn.execute(
            "SELECT id, rev FROM decisions WHERE scope_key=? AND superseded_by IS NULL",
            (scope_key,)).fetchone()
        if row is not None:
            new_rev = row["rev"] + 1
            if expected_rev is not None and expected_rev != row["rev"]:
                raise ConcurrentEdit("decision", scope_key, row["rev"], expected_rev)
            did = content_id("dec", {"scope_key": scope_key, "rev": new_rev, "action": action,
                                     "selected_candidate_id": selected_candidate_id,
                                     "payload": payload, "based_on_event": based_on_event})
            conn.execute("UPDATE decisions SET superseded_by=? WHERE id=?", (did, row["id"]))
        else:
            if expected_rev not in (None, 0):
                raise ConcurrentEdit("decision", scope_key, 0, expected_rev)
            new_rev = 1
            did = content_id("dec", {"scope_key": scope_key, "rev": 1, "action": action,
                                     "selected_candidate_id": selected_candidate_id,
                                     "payload": payload, "based_on_event": based_on_event})
        conn.execute(
            "INSERT INTO decisions(id,scope_key,selected_candidate_id,action,based_on_event,"
            "rev,superseded_by,created_at,payload) VALUES(?,?,?,?,?,?,?,?,?)",
            (did, scope_key, selected_candidate_id, action, based_on_event, new_rev, None,
             created_at, canonical_json(payload)))
    return did, new_rev


def get_decision_history(conn: sqlite3.Connection, scope_key: str) -> List[Dict]:
    rows = conn.execute("SELECT * FROM decisions WHERE scope_key=? ORDER BY rev",
                        (scope_key,)).fetchall()
    return [{"id": r["id"], "rev": r["rev"], "action": r["action"],
             "selected_candidate_id": r["selected_candidate_id"],
             "based_on_event": r["based_on_event"], "superseded_by": r["superseded_by"],
             "payload": json.loads(r["payload"]), "created_at": r["created_at"]}
            for r in rows]


def record_conflict(conn: sqlite3.Connection, scope: str, winning_event: str,
                    lost_event: str, detail: Dict[str, Any]) -> str:
    cid = content_id("conflict", {"scope": scope, "winning_event": winning_event,
                                  "lost_event": lost_event, "detail": detail})
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO conflicts(id,scope,winning_event,lost_event,detail,"
            "resolved_by,created_at) VALUES(?,?,?,?,?,NULL,?)",
            (cid, scope, winning_event, lost_event, canonical_json(detail), _now()))
    return cid


def resolve_conflict(conn: sqlite3.Connection, conflict_id: str, resolution_event: str) -> None:
    with conn:
        conn.execute("UPDATE conflicts SET resolved_by=? WHERE id=?",
                     (resolution_event, conflict_id))


def list_conflicts(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM conflicts ORDER BY created_at").fetchall()
    return [{"id": r["id"], "scope": r["scope"], "winning_event": r["winning_event"],
             "lost_event": r["lost_event"], "detail": json.loads(r["detail"]),
             "resolved_by": r["resolved_by"], "created_at": r["created_at"]} for r in rows]


def save_branch(conn: sqlite3.Connection, name: str, at_event: str, note: str = "") -> None:
    with conn:
        conn.execute("INSERT OR REPLACE INTO branches(name,at_event,note,created_at)"
                     " VALUES(?,?,?,?)", (name, at_event, note, _now()))


def list_branches(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM branches ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]
