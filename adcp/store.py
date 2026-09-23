"""SQLite 存储层:分源记录 + 版本化派生 + 原子指针 + 可解决冲突。

语义:
- beam velocity / correlation / amplitude / bad beam flag / heading / pitch /
  roll / 磁偏角 / 安装角均为独立 source 行,各自保留原始表示与单位。
- 每次推导产生独立 version;只有完整成功后同事务切换 current pointer;
  写入中断回滚 => 重启只能看到旧版本或完整新版本。
- 两位使用者从同一版本保存 => 第二条记录为 conflict 分支,两者都保留,
  可经 resolve 选择,未选分支不删除。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 2


class WriteInterrupted(RuntimeError):
    """模拟写入中断(测试用)。"""


_DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, unit TEXT NOT NULL,
  frame TEXT NOT NULL DEFAULT '', raw TEXT NOT NULL, value TEXT,
  diagnostic TEXT, supersedes TEXT, superseded INTEGER NOT NULL DEFAULT 0,
  seq INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE versions(
  id TEXT PRIMARY KEY, parent TEXT, label TEXT NOT NULL, status TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}', seq INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE nodes(
  version_id TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL,
  status TEXT NOT NULL, payload TEXT NOT NULL, deps TEXT NOT NULL,
  seq INTEGER NOT NULL, PRIMARY KEY(version_id, key));
CREATE TABLE pointer(id INTEGER PRIMARY KEY CHECK(id=1), version_id TEXT NOT NULL);
CREATE TABLE events(
  id TEXT PRIMARY KEY, version_id TEXT, kind TEXT NOT NULL,
  payload TEXT NOT NULL, seq INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE decisions(
  id TEXT PRIMARY KEY, base_version TEXT NOT NULL, branch_version TEXT NOT NULL,
  actor TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL, seq INTEGER NOT NULL, created_at TEXT NOT NULL);
"""

_TABLES = ("sources", "versions", "nodes", "events", "decisions")
_PREFIX = {"source": ("S", 4), "version": ("v", 4), "event": ("E", 4),
          "decision": ("D", 4), "node": ("n", 6)}


class Store:
    def __init__(self, path=":memory:", clock=None):
        self.path = path
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self):
        row = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            self.db.executescript(_DDL)
            self.db.execute("INSERT INTO meta(key,value) VALUES('schema_version','2')")
            for kind in _PREFIX:
                self.db.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?)", (f"seq:{kind}", "0")
                )
            self.db.commit()
            return
        v = int(self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"])
        if v == 1:
            # v1 -> v2:sources 增加坐标基准列 frame
            self.db.execute(
                "ALTER TABLE sources ADD COLUMN frame TEXT NOT NULL DEFAULT ''")
            self.db.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            self.db.commit()
        v = int(self.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"])
        if v != SCHEMA_VERSION:
            raise RuntimeError("未知 schema 版本")

    def is_empty(self):
        n = self.db.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]
        ptr = self.db.execute("SELECT COUNT(*) c FROM pointer").fetchone()["c"]
        return n == 0 and ptr == 0

    def close(self):
        self.db.close()

    def _now(self):
        return self.clock()

    def _next(self, kind):
        prefix, width = _PREFIX[kind]
        row = self.db.execute(
            "SELECT value FROM meta WHERE key=?", (f"seq:{kind}",)
        ).fetchone()
        n = int(row["value"]) + 1
        self.db.execute(
            "UPDATE meta SET value=? WHERE key=?", (str(n), f"seq:{kind}")
        )
        return f"{prefix}{n:0{width}d}"

    def _seq_after(self, kind):
        prefix, _ = _PREFIX[kind]
        return int(self._next(kind).lstrip(prefix))

    # ---------- sources ----------
    def add_source(self, kind, unit, raw, value, frame="", diagnostic=None,
                   supersedes=None, sid=None):
        sid = sid or self._next("source")
        if supersedes:
            self.db.execute(
                "UPDATE sources SET superseded=1 WHERE id=?", (supersedes,)
            )
        self.db.execute(
            "INSERT INTO sources(id,kind,unit,frame,raw,value,diagnostic,"
            "supersedes,superseded,seq,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (sid, kind, unit, frame, raw,
             json.dumps(value, ensure_ascii=False, sort_keys=True)
             if value is not None else None,
             diagnostic, supersedes, 0, self._seq_after("source"), self._now()),
        )
        self.db.commit()
        return sid

    def sources(self, include_superseded=False):
        sql = "SELECT * FROM sources"
        if not include_superseded:
            sql += " WHERE superseded=0"
        sql += " ORDER BY seq"
        return [dict(r) for r in self.db.execute(sql)]

    def source(self, sid):
        r = self.db.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    # ---------- versions / atomic commit ----------
    def current_version(self):
        row = self.db.execute("SELECT version_id FROM pointer WHERE id=1").fetchone()
        return row["version_id"] if row else None

    def version(self, vid):
        r = self.db.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        return dict(r) if r else None

    def versions(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM versions ORDER BY seq")]

    def nodes(self, vid):
        rows = self.db.execute(
            "SELECT * FROM nodes WHERE version_id=? ORDER BY seq", (vid,)
        ).fetchall()
        return {r["key"]: {
            "fingerprint": r["fingerprint"], "status": r["status"],
            "payload": json.loads(r["payload"]), "deps": json.loads(r["deps"]),
            "seq": r["seq"]} for r in rows}

    def commit_version(self, label, parent, payload, nodes, events=(),
                       switch=True, status="complete", crash_after=None):
        """单事务写入 version+nodes+events,最后一步才切换指针。

        crash_after='nodes' 时模拟节点写完、指针切换前断电 => 整体回滚。
        """
        vid = self._next("version")
        cur = self.db.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "INSERT INTO versions(id,parent,label,status,payload,seq,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (vid, parent, label, "building",
                 json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 int(vid.lstrip("v")), self._now()),
            )
            for key, node in nodes.items():
                cur.execute(
                    "INSERT INTO nodes(version_id,key,fingerprint,status,payload,"
                    "deps,seq) VALUES(?,?,?,?,?,?,?)",
                    (vid, key, node["fingerprint"], node["status"],
                     json.dumps(node["payload"], ensure_ascii=False, sort_keys=True),
                     json.dumps(node.get("deps", []), ensure_ascii=False),
                     int(self._next("node").lstrip("n"))),
                )
            if crash_after == "nodes":
                raise WriteInterrupted("模拟崩溃:节点已写、指针未切换")
            for ev in events:
                eid = self._next("event")
                cur.execute(
                    "INSERT INTO events(id,version_id,kind,payload,seq,created_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (eid, ev.get("version_id", vid), ev["kind"],
                     json.dumps(ev["payload"], ensure_ascii=False, sort_keys=True),
                     int(eid.lstrip("E")), self._now()),
                )
            cur.execute("UPDATE versions SET status=? WHERE id=?", (status, vid))
            if switch:
                cur.execute(
                    "INSERT INTO pointer(id,version_id) VALUES(1,?) "
                    "ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id",
                    (vid,),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return vid

    def add_event(self, version_id, kind, payload):
        eid = self._next("event")
        self.db.execute(
            "INSERT INTO events(id,version_id,kind,payload,seq,created_at)"
            " VALUES(?,?,?,?,?,?)",
            (eid, version_id, kind,
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             int(eid.lstrip("E")), self._now()),
        )
        self.db.commit()
        return eid

    def events(self, version_id=None):
        if version_id is None:
            rows = self.db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM events WHERE version_id=? ORDER BY seq", (version_id,)
            ).fetchall()
        out = []
        for r in rows:
            out.append({"id": r["id"], "version_id": r["version_id"],
                        "kind": r["kind"], "payload": json.loads(r["payload"]),
                        "seq": r["seq"], "created_at": r["created_at"]})
        return out

    def event(self, eid):
        r = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "version_id": r["version_id"], "kind": r["kind"],
                "payload": json.loads(r["payload"]), "seq": r["seq"],
                "created_at": r["created_at"]}

    def decisions(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM decisions ORDER BY seq")]

    def decision(self, did):
        r = self.db.execute("SELECT * FROM decisions WHERE id=?", (did,)).fetchone()
        return dict(r) if r else None

    def save_decision(self, actor, action, payload, base_version, build):
        """从 base_version 派生人工决定分支。

        build(sources, overrides, previous) -> (nodes, events)。
        若 base_version 已不是当前指针,分支保留但不切换指针,状态 conflict,
        两位使用者的结果都在,事后可 resolve。
        """
        current = self.current_version()
        applied = base_version == current
        base = self.version(base_version)
        base_payload = json.loads(base["payload"]) if base["payload"] else {}
        source_ids = base_payload.get("source_ids", [s["id"] for s in self.sources()])
        snapshot = [s for s in self.sources() if s["id"] in set(source_ids)]
        overrides = base_payload.get("overrides", []) + [{
            "actor": actor, "action": action, **payload,
        }]
        nodes, events = build(snapshot, overrides, self.nodes(base_version))
        status = "complete" if applied else "conflict"
        branch = self.commit_version(
            label=f"decision({actor}): {action} {payload}",
            parent=base_version,
            payload={"source_ids": source_ids, "overrides": overrides},
            nodes=nodes, events=events, switch=applied, status=status,
        )
        did = self._next("decision")
        self.db.execute(
            "INSERT INTO decisions(id,base_version,branch_version,actor,action,"
            "payload,status,seq,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (did, base_version, branch, actor, action,
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             "applied" if applied else "conflict",
             int(did.lstrip("D")), self._now()),
        )
        self.db.commit()
        self.add_event(
            branch,
            "decision_applied" if applied else "decision_conflict",
            {"decision": did, "actor": actor, "action": action, "payload": payload,
             "base_version": base_version, "branch_version": branch,
             "current_version": current},
        )
        return did, branch

    def resolve_decision(self, decision_id, choose):
        """解决冲突:choose='branch' 切到冲突分支,choose='current' 保留当前。

        无论选哪个,未选版本都保留在 versions 中,不做不可逆覆盖。
        """
        d = self.decision(decision_id)
        if d is None:
            raise KeyError(decision_id)
        current = self.current_version()
        chosen = d["branch_version"] if choose == "branch" else current
        if choose == "branch":
            self.db.execute("UPDATE pointer SET version_id=? WHERE id=1", (chosen,))
        new_status = f"resolved_{choose}"
        self.db.execute(
            "UPDATE decisions SET status=? WHERE id=?", (new_status, decision_id)
        )
        self.db.commit()
        unchosen = current if choose == "branch" else d["branch_version"]
        eid = self.add_event(chosen, "conflict_resolved", {
            "decision": decision_id, "choose": choose, "chosen_version": chosen,
            "unchosen_version_kept": unchosen,
        })
        return chosen, unchosen, eid

    # ---------- export / import ----------
    def export_archive(self):
        tables = {}
        for t in _TABLES:
            tables[t] = [dict(r) for r in self.db.execute(
                f"SELECT * FROM {t} ORDER BY seq")]
        meta = {r["key"]: r["value"] for r in self.db.execute("SELECT * FROM meta")}
        pointer = None
        row = self.db.execute("SELECT version_id FROM pointer WHERE id=1").fetchone()
        if row:
            pointer = row["version_id"]
        return {"format": "adcp-evidence/1", "schema": SCHEMA_VERSION,
                "meta": meta, "pointer": pointer, "tables": tables}

    def import_archive(self, archive):
        """导入档案:身份(id)、顺序(seq)、来源关系(deps/supersedes)原样保留。"""
        if not self.is_empty():
            raise RuntimeError("仅可向空库导入;删除本地数据库后再导入")
        if archive.get("format") != "adcp-evidence/1":
            raise ValueError("档案格式不支持")
        cur = self.db.cursor()
        cur.execute("BEGIN IMMEDIATE")
        for key, value in archive["meta"].items():
            if key == "schema_version":
                continue
            cur.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value)
            )
        cur.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
        for t in _TABLES:
            for row in archive["tables"].get(t, []):
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
                vals = [row.get(c) for c in cols]
                placeholders = ",".join("?" for _ in cols)
                cur.execute(
                    f"INSERT INTO {t} ({','.join(cols)}) "
                    f"VALUES({placeholders})", vals,
                )
        if archive.get("pointer"):
            cur.execute(
                "INSERT INTO pointer(id,version_id) VALUES(1,?) "
                "ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id",
                (archive["pointer"],),
            )
        self.db.commit()

    def wipe(self):
        self.db.executescript(
            "DELETE FROM sources; DELETE FROM versions; DELETE FROM nodes;"
            "DELETE FROM events; DELETE FROM decisions; DELETE FROM pointer;"
        )
        for kind in _PREFIX:
            self.db.execute(
                "UPDATE meta SET value='0' WHERE key=?", (f"seq:{kind}",)
            )
        self.db.commit()
