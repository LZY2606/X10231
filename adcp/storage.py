"""SQLite 存储模型。

分开记录：波束测量 / correlation / amplitude / 坏 beam 标志 /
heading-pitch-roll / 磁偏角 / 安装角各自独立行与独立版本。

版本与指针
----------
- source / rule / derivation 全部只追加、不可变；每次推导独立版本。
- current_pointers 在同一事务内随新版本一起提交，因此
  "只有完整成功后才切换当前指针"，写入中断（事务回滚）后重启，
  读者要么看到旧版本，要么看到完整新版本。
- 派生版本带 status=ok|indeterminate|excluded，绝不吞掉多解/不确定。
- events 为追加事件日志（机器可读）；branches 记录分叉，
  人工裁定产生 decision 事件，两个使用者从同一版本编辑产生冲突记录，
  未选候选保留，不被覆盖。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager

from .fingerprint import stable_id

SCHEMA_VERSION = 2  # 旧版本迁移测试从 v1 升到 v2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sid TEXT NOT NULL,                 -- 稳定身份（跨导入导出）
  kind TEXT NOT NULL,                -- beam_velocity|correlation|amplitude|
                                     -- bad_beam|attitude|bottom_track|cell_depth
  ping INTEGER,
  cell INTEGER,
  beam INTEGER,
  scope_json TEXT NOT NULL,          -- 稳定坐标键
  data TEXT NOT NULL,                -- 规范化 JSON（含 raw/unit/datum/diagnostic）
  revision INTEGER NOT NULL,
  superseded_by INTEGER,
  created_event TEXT,
  created_at REAL NOT NULL,
  UNIQUE(sid, revision)
);
CREATE INDEX IF NOT EXISTS idx_sources_scope ON sources(kind, ping, cell, beam);

CREATE TABLE IF NOT EXISTS rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rid TEXT NOT NULL,
  kind TEXT NOT NULL,                -- beam_geometry|declination|mount|quality|order
  data TEXT NOT NULL,
  version TEXT NOT NULL,
  revision INTEGER NOT NULL,
  superseded_by INTEGER,
  created_event TEXT,
  created_at REAL NOT NULL,
  UNIQUE(rid, revision)
);

CREATE TABLE IF NOT EXISTS derivations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL,             -- 逻辑节点（ping/cell/layer 等）
  fingerprint TEXT NOT NULL,
  op TEXT NOT NULL,
  op_version TEXT NOT NULL,
  inputs_json TEXT NOT NULL,         -- 进入指纹的输入身份快照
  payload_json TEXT NOT NULL,        -- 结果（含全部合法候选）
  status TEXT NOT NULL,              -- ok|indeterminate|excluded
  branch_id TEXT NOT NULL,
  created_event TEXT,
  created_at REAL NOT NULL,
  UNIQUE(branch_id, node_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_deriv_node ON derivations(branch_id, node_id);

CREATE TABLE IF NOT EXISTS current_pointers (
  branch_id TEXT NOT NULL,
  kind TEXT NOT NULL,                -- source|rule|derivation
  key TEXT NOT NULL,                 -- sid|rid|node_id
  value_rowid INTEGER NOT NULL,
  updated_event TEXT,
  PRIMARY KEY (branch_id, kind, key)
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  seq INTEGER NOT NULL,
  branch_id TEXT NOT NULL,
  parent_event_id TEXT,
  type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  scope_json TEXT,
  affected_json TEXT,
  time REAL NOT NULL,
  UNIQUE(branch_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id, seq);

CREATE TABLE IF NOT EXISTS branches (
  branch_id TEXT PRIMARY KEY,
  parent_branch_id TEXT,
  forked_from_event TEXT,
  name TEXT NOT NULL,
  base_seq INTEGER NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conflict_id TEXT NOT NULL UNIQUE,
  branch_id TEXT NOT NULL,
  node_id TEXT,
  decision_a_event TEXT,
  decision_b_event TEXT,
  status TEXT NOT NULL,              -- open|resolved
  resolution_event TEXT,
  created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._tls = threading.local()
        self._init_lock = threading.Lock()
        self.init_schema()

    # ---- connection ----------------------------------------------------
    def _conn(self):
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._tls.conn = conn
        return conn

    @contextmanager
    def tx(self, immediate=False):
        conn = self._conn()
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ---- schema / migration -------------------------------------------
    def init_schema(self):
        with self._init_lock, self.tx() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta(key,value) VALUES('schema_version',?)",
                             (str(SCHEMA_VERSION),))
            else:
                self._migrate(conn, int(row["value"]))
            conn.execute("INSERT OR IGNORE INTO branches(branch_id,parent_branch_id,"
                         "forked_from_event,name,base_seq,created_at) "
                         "VALUES('main',NULL,NULL,'main',0,0)")

    def _migrate(self, conn, from_version):
        # v1 -> v2：derivations 增加 op_version（迁移测试覆盖）
        if from_version < 2:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(derivations)")]
            if "op_version" not in cols:
                conn.execute("ALTER TABLE derivations ADD COLUMN op_version TEXT "
                             "NOT NULL DEFAULT 'op:legacy:v0'")
            conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")

    # ---- write helpers (调用方在一个事务内完成 append + 指针切换) -------
    def _clock(self):
        import time
        return time.time()

    def put_source(self, conn, kind, scope, data, event_id, branch_id="main",
                   created_at=None):
        sid = stable_id("src", kind, *scope)
        rev_row = conn.execute("SELECT COALESCE(MAX(revision),0)+1 AS r FROM sources "
                               "WHERE sid=?", (sid,)).fetchone()
        revision = rev_row["r"]
        cur = conn.execute(
            "INSERT INTO sources(sid,kind,ping,cell,beam,scope_json,data,revision,"
            "created_event,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (sid, kind, scope[0] if len(scope) > 0 else None,
             scope[1] if len(scope) > 1 else None,
             scope[2] if len(scope) > 2 else None,
             json.dumps(list(scope), ensure_ascii=False),
             json.dumps(data, ensure_ascii=False), revision,
             event_id, created_at or self._clock()))
        rowid = cur.lastrowid
        if revision > 1:
            conn.execute("UPDATE sources SET superseded_by=? WHERE sid=? AND revision=?",
                         (rowid, sid, revision - 1))
        conn.execute(
            "INSERT INTO current_pointers(branch_id,kind,key,value_rowid,updated_event) "
            "VALUES('main','source',?,?,?) "
            "ON CONFLICT(branch_id,kind,key) DO UPDATE SET value_rowid=excluded.value_rowid,"
            "updated_event=excluded.updated_event", (sid, rowid, event_id))
        return {"sid": sid, "revision": revision, "rowid": rowid}

    def put_rule(self, conn, kind, rid, data, version, event_id, created_at=None):
        rev_row = conn.execute("SELECT COALESCE(MAX(revision),0)+1 AS r FROM rules "
                               "WHERE rid=?", (rid,)).fetchone()
        revision = rev_row["r"]
        cur = conn.execute(
            "INSERT INTO rules(rid,kind,data,version,revision,created_event,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (rid, kind, json.dumps(data, ensure_ascii=False), version, revision,
             event_id, created_at or self._clock()))
        rowid = cur.lastrowid
        if revision > 1:
            conn.execute("UPDATE rules SET superseded_by=? WHERE rid=? AND revision=?",
                         (rowid, rid, revision - 1))
        conn.execute(
            "INSERT INTO current_pointers(branch_id,kind,key,value_rowid,updated_event) "
            "VALUES('main','rule',?,?,?) "
            "ON CONFLICT(branch_id,kind,key) DO UPDATE SET value_rowid=excluded.value_rowid,"
            "updated_event=excluded.updated_event", (rid, rowid, event_id))
        return {"rid": rid, "revision": revision, "rowid": rowid, "version": version}

    def put_derivation(self, conn, branch_id, node_id, op, op_version, inputs,
                       fingerprint, payload, status, event_id, created_at=None):
        existing = conn.execute(
            "SELECT id FROM derivations WHERE branch_id=? AND node_id=? AND fingerprint=?",
            (branch_id, node_id, fingerprint)).fetchone()
        if existing:
            rowid = existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO derivations(node_id,fingerprint,op,op_version,inputs_json,"
                "payload_json,status,branch_id,created_event,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (node_id, fingerprint, op, op_version,
                 json.dumps(inputs, ensure_ascii=False),
                 json.dumps(payload, ensure_ascii=False), status, branch_id,
                 event_id, created_at or self._clock()))
            rowid = cur.lastrowid
        conn.execute(
            "INSERT INTO current_pointers(branch_id,kind,key,value_rowid,updated_event) "
            "VALUES(?,'derivation',?,?,?) "
            "ON CONFLICT(branch_id,kind,key) DO UPDATE SET value_rowid=excluded.value_rowid,"
            "updated_event=excluded.updated_event", (branch_id, node_id, rowid, event_id))
        return rowid

    # ---- events / branches / conflicts --------------------------------
    def add_event(self, conn, event_id, type, actor, payload, scope=None,
                  affected=None, branch_id="main", created_at=None):
        parent = conn.execute(
            "SELECT event_id FROM events WHERE branch_id=? ORDER BY seq DESC LIMIT 1",
            (branch_id,)).fetchone()
        seq_row = conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS s FROM events "
                               "WHERE branch_id=?", (branch_id,)).fetchone()
        conn.execute(
            "INSERT INTO events(event_id,seq,branch_id,parent_event_id,type,actor,"
            "payload_json,scope_json,affected_json,time) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, seq_row["s"], branch_id,
             parent["event_id"] if parent else None, type, actor,
             json.dumps(payload, ensure_ascii=False),
             json.dumps(scope, ensure_ascii=False) if scope is not None else None,
             json.dumps(affected, ensure_ascii=False) if affected is not None else None,
             created_at or self._clock()))
        return seq_row["s"]

    def create_branch(self, conn, branch_id, name, from_branch="main",
                      forked_from_event=None, created_at=None):
        base = conn.execute("SELECT COALESCE(MAX(seq),0) AS s FROM events "
                            "WHERE branch_id=?", (from_branch,)).fetchone()["s"]
        conn.execute("INSERT OR IGNORE INTO branches(branch_id,parent_branch_id,"
                     "forked_from_event,name,base_seq,created_at) VALUES(?,?,?,?,?,?)",
                     (branch_id, from_branch, forked_from_event, name, base,
                      created_at or self._clock()))
        return base

    def record_conflict(self, conn, conflict_id, node_id, ev_a, ev_b,
                        branch_id="main", created_at=None):
        conn.execute("INSERT OR IGNORE INTO conflicts(conflict_id,branch_id,node_id,"
                     "decision_a_event,decision_b_event,status,created_at) "
                     "VALUES(?,?,?,?,?, 'open', ?)",
                     (conflict_id, branch_id, node_id, ev_a, ev_b,
                      created_at or self._clock()))

    def resolve_conflict(self, conn, conflict_id, resolution_event):
        conn.execute("UPDATE conflicts SET status='resolved',resolution_event=? "
                     "WHERE conflict_id=?", (resolution_event, conflict_id))

    # ---- reads ---------------------------------------------------------
    def get_current_source(self, sid, branch_id="main"):
        row = self._conn().execute(
            "SELECT s.* FROM current_pointers p JOIN sources s ON s.id=p.value_rowid "
            "WHERE p.branch_id=? AND p.kind='source' AND p.key=?",
            (branch_id, sid)).fetchone()
        if row is None and branch_id != "main":
            return self.get_current_source(sid, "main")
        return self._source_row(row) if row else None

    def get_current_rule(self, rid):
        row = self._conn().execute(
            "SELECT r.* FROM current_pointers p JOIN rules r ON r.id=p.value_rowid "
            "WHERE p.kind='rule' AND p.key=?", (rid,)).fetchone()
        return self._rule_row(row) if row else None

    def scan_current_sources(self, kind, branch_id="main"):
        rows = self._conn().execute(
            "SELECT s.* FROM current_pointers p JOIN sources s ON s.id=p.value_rowid "
            "WHERE p.branch_id='main' AND p.kind='source' AND s.kind=? ORDER BY s.ping,s.cell,s.beam",
            (kind,)).fetchall()
        return [self._source_row(r) for r in rows]

    @staticmethod
    def _source_row(row):
        return {"sid": row["sid"], "kind": row["kind"], "ping": row["ping"],
                "cell": row["cell"], "beam": row["beam"],
                "scope": json.loads(row["scope_json"]),
                "data": json.loads(row["data"]), "revision": row["revision"],
                "created_event": row["created_event"]}

    @staticmethod
    def _rule_row(row):
        return {"rid": row["rid"], "kind": row["kind"],
                "data": json.loads(row["data"]), "version": row["version"],
                "revision": row["revision"], "created_event": row["created_event"]}

    def get_derivation(self, node_id, branch_id="main"):
        rowid = self._effective_pointer(branch_id, "derivation", node_id)
        if rowid is None:
            return None
        row = self._conn().execute("SELECT * FROM derivations WHERE id=?",
                                   (rowid,)).fetchone()
        return self._deriv_row(row) if row else None

    def _effective_pointer(self, branch_id, kind, key):
        conn = self._conn()
        while branch_id:
            row = conn.execute(
                "SELECT value_rowid FROM current_pointers WHERE branch_id=? AND kind=? "
                "AND key=?", (branch_id, kind, key)).fetchone()
            if row:
                return row["value_rowid"]
            br = conn.execute("SELECT parent_branch_id FROM branches WHERE branch_id=?",
                              (branch_id,)).fetchone()
            branch_id = br["parent_branch_id"] if br else None
        return None

    @staticmethod
    def _deriv_row(row):
        return {"node_id": row["node_id"], "fingerprint": row["fingerprint"],
                "op": row["op"], "op_version": row["op_version"],
                "inputs": json.loads(row["inputs_json"]),
                "payload": json.loads(row["payload_json"]),
                "status": row["status"], "branch_id": row["branch_id"],
                "created_event": row["created_event"]}

    def events(self, branch_id="main", limit=None):
        q = ("SELECT * FROM events WHERE branch_id=? ORDER BY seq ASC"
             + (f" LIMIT {int(limit)}" if limit else ""))
        return [self._event_row(r) for r in self._conn().execute(q, (branch_id,))]

    def latest_event_id(self, branch_id="main"):
        row = self._conn().execute("SELECT event_id FROM events WHERE branch_id=? "
                                   "ORDER BY seq DESC LIMIT 1", (branch_id,)).fetchone()
        return row["event_id"] if row else None

    @staticmethod
    def _event_row(r):
        return {"event_id": r["event_id"], "seq": r["seq"],
                "branch_id": r["branch_id"], "parent_event_id": r["parent_event_id"],
                "type": r["type"], "actor": r["actor"],
                "payload": json.loads(r["payload_json"]),
                "scope": json.loads(r["scope_json"]) if r["scope_json"] else None,
                "affected": json.loads(r["affected_json"]) if r["affected_json"] else None,
                "time": r["time"]}

    def list_branches(self):
        return [dict(b) for b in self._conn().execute(
            "SELECT * FROM branches ORDER BY created_at")]

    def list_conflicts(self, only_open=False):
        q = "SELECT * FROM conflicts" + (" WHERE status='open'" if only_open else "")
        return [dict(r) for r in self._conn().execute(q)]

    # ---- as-of 还原：历史事件决定当时可见的派生版本 --------------------
    def ancestry_events(self, branch_id, upto_seq=None):
        """返回分支祖先链上 upto_seq（含）之前的全部事件（全局创建序）。"""
        conn = self._conn()
        chain, b = [], branch_id
        while b:
            row = conn.execute("SELECT * FROM branches WHERE branch_id=?", (b,)).fetchone()
            chain.append((b, row["parent_branch_id"], row["base_seq"]))
            b = row["parent_branch_id"]
        allowed = set()
        # 从根向叶：祖先分支只取分叉点之前的事件
        for bid, _, base in reversed(chain):
            cap = None
            # 父分支的截止 = 子分叉时记录的 base_seq
            for child_bid, _, child_base in chain:
                if self._parent_of(child_bid) == bid:
                    cap = child_base if cap is None else min(cap, child_base)
            if bid == branch_id:
                cap = upto_seq
            rows = conn.execute("SELECT event_id FROM events WHERE branch_id=? "
                                + ("AND seq<=? " if cap is not None else "")
                                + "ORDER BY seq",
                                ((bid, cap) if cap is not None else (bid,))).fetchall()
            allowed.update(r["event_id"] for r in rows)
        return allowed

    def _parent_of(self, bid):
        r = self._conn().execute("SELECT parent_branch_id FROM branches WHERE branch_id=?",
                                 (bid,)).fetchone()
        return r["parent_branch_id"] if r else None

    def derivations_at(self, branch_id, upto_seq=None):
        """as-of 视图：node_id -> 该事件点可见的派生版本（不可变）。"""
        allowed = self.ancestry_events(branch_id, upto_seq)
        if not allowed:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" for _ in allowed)
        rows = conn.execute(
            f"SELECT * FROM derivations WHERE created_event IN ({placeholders})",
            tuple(allowed)).fetchall()
        latest = {}
        order = {eid: i for i, eid in enumerate(sorted(allowed))}
        for r in rows:
            node = r["node_id"]
            cur = latest.get(node)
            if cur is None or r["id"] > cur["id"]:
                latest[node] = r
        return {k: self._deriv_row(v) for k, v in latest.items()}

    def derivations_current(self, branch_id="main"):
        rows = self._conn().execute(
            "SELECT d.* FROM current_pointers p JOIN derivations d ON d.id=p.value_rowid "
            "WHERE p.kind='derivation' AND p.branch_id=?", (branch_id,)).fetchall()
        return {r["node_id"]: self._deriv_row(r) for r in rows}

    # ---- 导出 / 删除本地库再导入：身份、顺序、来源关系不变 -------------
    def export_bundle(self):
        conn = self._conn()
        def rows(table):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            return [dict(zip(cols, r)) for r in conn.execute(f"SELECT * FROM {table}")]
        return {"bundle": "adcp-evidence", "format_version": 1,
                "tables": {t: rows(t) for t in
                           ("meta", "sources", "rules", "derivations",
                            "current_pointers", "events", "branches", "conflicts")},
                "order": {"events": [r["event_id"] for r in
                         conn.execute("SELECT event_id FROM events ORDER BY id")]}}

    @classmethod
    def import_bundle(cls, path, bundle):
        """导入到一个全新/空库；稳定身份、事件顺序、指针引用全部保留。"""
        store = cls(path)
        tables = bundle["tables"]
        with store.tx(immediate=True) as conn:
            for table, rows_ in tables.items():
                if not rows_:
                    continue
                cols = list(rows_[0].keys())
                ph = ",".join("?" for _ in cols)
                conn.executemany(
                    f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({ph})",
                    [tuple(r.get(c) for c in cols) for r in rows_])
        return store

    def close(self):
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None
