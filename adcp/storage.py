"""版本化 SQLite 存储。

提交语义：一次推导的全部 facts/events 在单个事务里写入；
revision 行最后插入，HEAD 在同一事务内切换。写入中断后重启，
调用方只能看到旧 HEAD，或看到完整新 revision（无半截版本）。

并发：commit 要求 base_revision == 当前 HEAD；冲突时不写入，
抛 ConflictError，由上层构造可解决的冲突事件。

历史：revisions 永不删除；事件链可回溯任意旧版本当时的结论。
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid as uuidlib
from datetime import datetime, timezone

SCHEMA_VERSION = 2


def _now():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
  revision_id TEXT PRIMARY KEY,
  parent_id TEXT,
  branch TEXT NOT NULL DEFAULT 'main',
  author TEXT NOT NULL,
  message TEXT NOT NULL,
  created_at TEXT NOT NULL,
  replay_step TEXT,
  derived_fingerprint TEXT
);
CREATE TABLE IF NOT EXISTS facts (
  revision_id TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  ftype TEXT NOT NULL,
  ping INTEGER, cell INTEGER, beam INTEGER, time_s REAL,
  raw TEXT, raw_unit TEXT, datum TEXT,
  payload TEXT NOT NULL,
  fhash TEXT NOT NULL,
  supersedes TEXT,
  seq INTEGER NOT NULL,
  PRIMARY KEY (revision_id, fact_id)
);
CREATE TABLE IF NOT EXISTS events (
  revision_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  event_id TEXT PRIMARY KEY,
  etype TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at TEXT NOT NULL,
  scope TEXT,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS heads (
  branch TEXT PRIMARY KEY,
  revision_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_loc ON facts(ftype, ping, cell, beam);
CREATE INDEX IF NOT EXISTS idx_events_rev ON events(revision_id, seq);
CREATE INDEX IF NOT EXISTS idx_rev_parent ON revisions(parent_id);
"""


class ConflictError(Exception):
    """base_revision 与当前 HEAD 不一致：两位使用者基于同一版本编辑。"""


def connect(path: str):
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
    if ver.fetchone() is None:
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','2')")
        conn.execute("INSERT INTO meta(key,value) VALUES('created_at',?)",
                     (_now(),))
    return conn


def head(conn, branch="main"):
    row = conn.execute("SELECT revision_id FROM heads WHERE branch=?",
                       (branch,)).fetchone()
    return row["revision_id"] if row else None


def ancestry(conn, rev_id):
    """从某版本沿 parent 链到根（含自身）。"""
    chain = []
    cur = rev_id
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        row = conn.execute(
            "SELECT parent_id FROM revisions WHERE revision_id=?",
            (cur,)).fetchone()
        chain.append(cur)
        cur = row["parent_id"] if row else None
    return chain


def _decode_fact(row: sqlite3.Row) -> dict:
    f = dict(row)
    f["payload"] = json.loads(f["payload"] or "{}")
    return f


def visible_facts(conn, rev_id):
    """沿祖先链物化某版本可见事实：靠近该版本的同 fact_id 覆盖旧值。"""
    out = {}
    for rid in reversed(ancestry(conn, rev_id)):
        rows = conn.execute(
            "SELECT * FROM facts WHERE revision_id=? ORDER BY seq", (rid,))
        for row in rows:
            f = _decode_fact(row)
            out[f["fact_id"]] = f
    return list(out.values())


def events_until(conn, rev_id):
    """到达该版本为止（祖先链）发生的全部事件，按 (revision顺序,seq)。"""
    events = []
    for rid in reversed(ancestry(conn, rev_id)):
        rows = conn.execute(
            "SELECT * FROM events WHERE revision_id=? ORDER BY seq", (rid,))
        for row in rows:
            ev = dict(row)
            ev["payload"] = json.loads(ev["payload"] or "{}")
            events.append(ev)
    return events


def commit(conn, *, author, message, new_facts=None, events=None,
           base_revision=None, branch="main", replay_step=None,
           derived_fingerprint=None, _crash_before_revision=False):
    """原子提交。冲突 -> ConflictError（不写任何东西）。

    base_revision 必须等于当前分支 HEAD（首次提交时二者都为 None）。
    """
    new_facts = new_facts or []
    events = events or []
    cur_head = head(conn, branch)
    if base_revision != cur_head:
        raise ConflictError(
            f"expected base {cur_head}, client based on {base_revision}")

    revision_id = "r_" + uuidlib.uuid4().hex[:16]
    try:
        conn.execute("BEGIN IMMEDIATE")
        # 1) 先写所有事实
        for seq, f in enumerate(new_facts):
            conn.execute(
                "INSERT INTO facts(revision_id,fact_id,ftype,ping,cell,beam,"
                "time_s,raw,raw_unit,datum,payload,fhash,supersedes,seq) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (revision_id, f["fact_id"], f["ftype"], f.get("ping"),
                 f.get("cell"), f.get("beam"), f.get("time_s"),
                 None if f.get("raw") is None else str(f["raw"]),
                 f.get("raw_unit"), f.get("datum"),
                 json.dumps(f.get("payload") or {}, ensure_ascii=False),
                 f["fhash"], f.get("supersedes"), seq))
        # 2) 写事件
        for seq, ev in enumerate(events):
            conn.execute(
                "INSERT INTO events(revision_id,seq,event_id,etype,actor,"
                "created_at,scope,payload) VALUES(?,?,?,?,?,?,?,?)",
                (revision_id, seq, ev.get("event_id",
                                         "e_" + uuidlib.uuid4().hex[:16]),
                 ev["etype"], ev.get("actor", author),
                 ev.get("created_at", _now()), ev.get("scope"),
                 json.dumps(ev.get("payload") or {}, ensure_ascii=False)))

        if _crash_before_revision:
            raise RuntimeError("simulated write interruption before revision")

        # 3) 最后写 revision 并切换 HEAD
        conn.execute(
            "INSERT INTO revisions(revision_id,parent_id,branch,author,"
            "message,created_at,replay_step,derived_fingerprint) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (revision_id, cur_head, branch, author, message, _now(),
             replay_step, derived_fingerprint))
        conn.execute(
            "INSERT INTO heads(branch,revision_id) VALUES(?,?) "
            "ON CONFLICT(branch) DO UPDATE SET revision_id=excluded.revision_id",
            (branch, revision_id))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return revision_id


def list_revisions(conn, branch="main"):
    rows = conn.execute(
        "SELECT * FROM revisions WHERE branch=? ORDER BY created_at",
        (branch,)).fetchall()
    return [dict(r) for r in rows]


def revision(conn, rev_id):
    row = conn.execute("SELECT * FROM revisions WHERE revision_id=?",
                       (rev_id,)).fetchone()
    return dict(row) if row else None


def export_bundle(conn):
    """确定性导出：稳定身份、顺序与来源关系全部保留。"""
    revisions = [dict(r) for r in conn.execute(
        "SELECT * FROM revisions ORDER BY created_at,rowid")]
    heads = [dict(r) for r in conn.execute(
        "SELECT * FROM heads ORDER BY branch")]
    facts, events = [], []
    for r in revisions:
        for row in conn.execute(
                "SELECT * FROM facts WHERE revision_id=? ORDER BY seq",
                (r["revision_id"],)):
            facts.append(_decode_fact(row))
        for row in conn.execute(
                "SELECT * FROM events WHERE revision_id=? ORDER BY seq",
                (r["revision_id"],)):
            ev = dict(row)
            ev["payload"] = json.loads(ev["payload"] or "{}")
            events.append(ev)
    return {
        "bundle_format": "adcp-evidence-bundle/1",
        "schema_version": SCHEMA_VERSION,
        "exported_at": _now(),
        "revisions": revisions,
        "heads": heads,
        "facts": facts,
        "events": events,
    }


def import_bundle(conn, bundle: dict):
    """导入：按导出顺序重建，身份/fact_id/parent 关系不重新生成。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for r in bundle["revisions"]:
            conn.execute(
                "INSERT OR REPLACE INTO revisions(revision_id,parent_id,"
                "branch,author,message,created_at,replay_step,"
                "derived_fingerprint) VALUES(?,?,?,?,?,?,?,?)",
                (r["revision_id"], r.get("parent_id"), r["branch"],
                 r["author"], r["message"], r["created_at"],
                 r.get("replay_step"), r.get("derived_fingerprint")))
        for f in bundle["facts"]:
            conn.execute(
                "INSERT OR REPLACE INTO facts(revision_id,fact_id,ftype,"
                "ping,cell,beam,time_s,raw,raw_unit,datum,payload,fhash,"
                "supersedes,seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f["revision_id"], f["fact_id"], f["ftype"], f.get("ping"),
                 f.get("cell"), f.get("beam"), f.get("time_s"),
                 None if f.get("raw") is None else str(f["raw"]),
                 f.get("raw_unit"), f.get("datum"),
                 json.dumps(f.get("payload") or {}, ensure_ascii=False),
                 f["fhash"], f.get("supersedes"), f["seq"]))
        for ev in bundle["events"]:
            conn.execute(
                "INSERT OR REPLACE INTO events(revision_id,seq,event_id,"
                "etype,actor,created_at,scope,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (ev["revision_id"], ev["seq"], ev["event_id"], ev["etype"],
                 ev["actor"], ev["created_at"], ev.get("scope"),
                 json.dumps(ev.get("payload") or {}, ensure_ascii=False)))
        for h in bundle["heads"]:
            conn.execute(
                "INSERT OR REPLACE INTO heads(branch,revision_id) "
                "VALUES(?,?)", (h["branch"], h["revision_id"]))
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','2')")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return head(conn)


def migrate_legacy_v1(path: str):
    """旧版本迁移：把 v1 单表 raw_beam 搬到分开记录的事实模型。"""
    conn = connect(path)
    ver_row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    ver = int(ver_row["value"]) if ver_row else 1
    if ver >= 2:
        return conn
    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_beam'"
    ).fetchone()
    new_facts = []
    if legacy:
        rows = conn.execute(
            "SELECT * FROM raw_beam ORDER BY ping,cell,beam").fetchall()
        from .model import make_fact
        for i, row in enumerate(rows):
            d = dict(row)
            new_facts.append(make_fact(
                "beam", raw=d.get("radial"), raw_unit=d.get("unit", "m/s"),
                ping=d.get("ping"), cell=d.get("cell"),
                beam=d.get("beam"), time_s=d.get("time_s"),
                fact_id=f"mig_b_p{d.get('ping')}_c{d.get('cell')}"
                        f"_b{d.get('beam')}"))
    rid = commit(conn, author="migrator",
                 message="migrate schema v1 -> v2 (split facts)",
                 new_facts=new_facts, base_revision=head(conn),
                 replay_step="migration_v1_v2")
    conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    return conn, rid
