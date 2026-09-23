"""版本化、原子、可审计的 SQLite 存储层。

存储模型要点：
  * 波束测量 / correlation / amplitude / 坏 beam 标志 / heading-pitch-roll /
    磁偏角 / 安装角分别是独立 stream，分开记录、分开版本化。
  * 每次推导产生独立版本（derivation_versions，append-only），只有整次提交
    完整成功后才移动 current（current_*）。崩溃时调用 crash_hook 让进程退出，
    重启后要么看到旧版本，要么看到完整新版本（单事务 + 提交后立刻崩溃）。
  * 两名使用者从同一版本编辑：edit_stream 带 base_version，落后者得到 ConflictError，
    可通过 rebase（重试）或 force（记录覆盖决定）解决。
  * 事件与人工决定 append-only；视图可投影到任意历史 seq，恢复当时的依赖版本与结论。
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 2


class ConflictError(RuntimeError):
    def __init__(self, stream_id: str, base: int, current: int):
        super().__init__(f"stream {stream_id}: base v{base} is stale, current is v{current}")
        self.stream_id = stream_id
        self.base_version = base
        self.current_version = current


class NotFound(KeyError):
    pass


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS streams(
    stream_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 0,
    created_seq INTEGER NOT NULL,
    updated_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stream_versions(
    stream_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    author TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY(stream_id, version)
);
CREATE TABLE IF NOT EXISTS derivations(
    derivation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS derivation_versions(
    derivation_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    basis TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(derivation_id, version)
);
CREATE TABLE IF NOT EXISTS current_derivations(
    derivation_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS commits(
    seq INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    message TEXT NOT NULL,
    changes TEXT NOT NULL,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS events(
    event_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    subject TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions(
    decision_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    scope TEXT NOT NULL,
    subject TEXT NOT NULL,
    chosen_candidate_id TEXT,
    action TEXT NOT NULL,
    rationale TEXT,
    base_seq INTEGER
);
CREATE TABLE IF NOT EXISTS selection_log(
    scope TEXT NOT NULL,
    subject TEXT NOT NULL,
    seq INTEGER NOT NULL,
    decision_id TEXT NOT NULL,
    PRIMARY KEY(scope, subject, seq)
);
"""


class Repository:
    def __init__(self, path: str, *, crash_hook: Callable[[str], None] | None = None):
        self.path = path
        self.crash_hook = crash_hook
        fresh = not os.path.exists(path)
        self.conn = _connect(path)
        if fresh:
            self._init_schema()
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                              (str(SCHEMA_VERSION),))

    def _migrate(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row["value"]) if row else 1
        if version == 1:
            # v1: derivations 仅保存单一当前值，没有版本/指针表。
            with self.conn:
                self.conn.executescript(SCHEMA)
                self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                                  (str(SCHEMA_VERSION),))
                self.conn.execute(
                    "INSERT INTO commits(seq,kind,actor,message,changes) VALUES(0,'migration','system',?, '[]')",
                    ("migrate schema v1->v2",))

    # ---------- 基础读取 ----------
    def _next_seq(self, cur) -> int:
        row = cur.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM commits").fetchone()
        return int(row["n"])

    def current_state(self) -> dict[str, Any]:
        """投影当前指针：每条 stream 只返回其当前版本记录。"""
        rows = self.conn.execute(
            """SELECT s.stream_id, s.kind, s.scope, s.current_version AS version,
                      v.payload, v.author, v.note, s.updated_seq AS seq
               FROM streams s JOIN stream_versions v
                 ON v.stream_id=s.stream_id AND v.version=s.current_version""").fetchall()
        records = []
        for r in rows:
            records.append({"stream_id": r["stream_id"], "kind": r["kind"], "scope": r["scope"],
                            "version": r["version"], "seq": r["seq"],
                            "payload": json.loads(r["payload"]),
                            "author": r["author"], "note": r["note"]})
        return {"records": records,
                "head_seq": self.head_seq(),
                "schema_version": SCHEMA_VERSION}

    def head_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0) AS n FROM commits").fetchone()
        return int(row["n"])

    def current_derivations(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT d.derivation_id, d.kind, cd.version, cd.seq,
                      dv.fingerprint, dv.basis, dv.payload
               FROM derivations d
               JOIN current_derivations cd ON cd.derivation_id=d.derivation_id
               JOIN derivation_versions dv ON dv.derivation_id=d.derivation_id AND dv.version=cd.version"""
        ).fetchall()
        out = []
        for r in rows:
            out.append({"derivation_id": r["derivation_id"], "kind": r["kind"],
                        "version": r["version"], "seq": r["seq"],
                        "fingerprint": r["fingerprint"], "basis": json.loads(r["basis"]),
                        "payload": json.loads(r["payload"])})
        return out

    # ---------- 写入：流版本 ----------
    def ingest(self, records: Iterable[dict[str, Any]], *, actor: str = "importer",
               message: str = "baseline import") -> dict[str, Any]:
        records = list(records)
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            changes = []
            for rec in records:
                sid = rec["stream_id"]
                kind, scope = rec["kind"], rec.get("scope", "global")
                payload = json.dumps(rec["payload"], ensure_ascii=False, sort_keys=True)
                exists = cur.execute("SELECT 1 FROM streams WHERE stream_id=?", (sid,)).fetchone()
                if exists:
                    raise RuntimeError(f"ingest refuses to overwrite existing stream {sid}")
                cur.execute(
                    "INSERT INTO streams(stream_id,kind,scope,current_version,created_seq,updated_seq)"
                    " VALUES(?,?,?,1,?,?)", (sid, kind, scope, seq, seq))
                cur.execute(
                    "INSERT INTO stream_versions(stream_id,version,seq,payload,author,note)"
                    " VALUES(?,1,?,?,?,?)", (sid, seq, payload, actor, rec.get("note")))
                changes.append({"stream_id": sid, "kind": kind, "from": 0, "to": 1})
            cur.execute("INSERT INTO commits(seq,kind,actor,message,changes) VALUES(?,?,?,?,?)",
                        (seq, "ingest", actor, message, json.dumps(changes)))
            return {"seq": seq, "changes": changes}

    def edit_stream(self, stream_id: str, payload: dict[str, Any], *,
                    base_version: int, actor: str = "reviewer", note: str | None = None,
                    force: bool = False, recompute_fn=None) -> dict[str, Any]:
        """从指定基础版本编辑单条流。落后版本 -> ConflictError（可解决）。

        recompute_fn(state)->list[derivation]：在同一事务内完成推导与指针切换。
        """
        with self.conn:
            cur = self.conn.cursor()
            row = cur.execute("SELECT current_version, kind, scope FROM streams WHERE stream_id=?",
                              (stream_id,)).fetchone()
            if row is None:
                raise NotFound(stream_id)
            cur_version = int(row["current_version"])
            if base_version != cur_version:
                raise ConflictError(stream_id, base_version, cur_version)
            seq = self._next_seq(cur)
            new_version = cur_version + 1
            cur.execute("INSERT INTO stream_versions(stream_id,version,seq,payload,author,note)"
                        " VALUES(?,?,?,?,?,?)",
                        (stream_id, new_version, seq,
                         json.dumps(payload, ensure_ascii=False, sort_keys=True),
                         actor if not force else f"{actor}(force)", note))
            cur.execute("UPDATE streams SET current_version=?, updated_seq=? WHERE stream_id=?",
                        (new_version, seq, stream_id))
            change = {"stream_id": stream_id, "kind": row["kind"],
                      "from": cur_version, "to": new_version, "forced": force}
            deriv_summary = self._apply_recompute(cur, seq, recompute_fn)
            cur.execute("INSERT INTO commits(seq,kind,actor,message,changes) VALUES(?,?,?,?,?)",
                        (seq, "edit", actor, note or f"edit {stream_id}",
                         json.dumps([change])))
            if self.crash_hook is not None:
                # 提交已在 with 内登记；hook 模拟“写入中断”：退出进程。
                self.crash_hook("after_edit_commit")
            return {"seq": seq, "change": change, "derivations": deriv_summary}

    def decision(self, scope: str, subject: str, action: str, *,
                 chosen_candidate_id: str | None = None, actor: str = "reviewer",
                 rationale: str | None = None) -> dict[str, Any]:
        """接受/否定/选择候选/参数调整都是 append-only 决定，绝不覆盖历史。"""
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            decision_id = f"dec-{seq}-{scope}-{subject}".replace(":", "_")
            base_seq = None
            log_row = cur.execute(
                "SELECT seq FROM selection_log WHERE scope=? AND subject=? ORDER BY seq DESC LIMIT 1",
                (scope, subject)).fetchone()
            if log_row:
                base_seq = int(log_row["seq"])
            cur.execute("INSERT INTO decisions(decision_id,seq,scope,subject,chosen_candidate_id,action,"
                        "rationale,base_seq) VALUES(?,?,?,?,?,?,?,?)",
                        (decision_id, seq, scope, subject, chosen_candidate_id, action, rationale, base_seq))
            cur.execute("INSERT INTO selection_log(scope,subject,seq,decision_id) VALUES(?,?,?,?)",
                        (scope, subject, seq, decision_id))
            cur.execute("INSERT INTO commits(seq,kind,actor,message,changes) VALUES(?,?,?,?,?)",
                        (seq, "decision", actor,
                         f"{action} {subject} -> {chosen_candidate_id or '-'}",
                         json.dumps([{"scope": scope, "subject": subject, "action": action}])))
            return {"seq": seq, "decision_id": decision_id, "base_seq": base_seq}

    def _apply_recompute(self, cur, seq: int, recompute_fn) -> dict[str, Any]:
        if recompute_fn is None:
            return {"updated": [], "unchanged": 0}
        state = self.current_state()
        fresh = recompute_fn(state)
        updated, unchanged = [], 0
        for d in fresh:
            cur.execute("INSERT OR IGNORE INTO derivations(derivation_id,kind) VALUES(?,?)",
                        (d["derivation_id"], d["kind"]))
            row = cur.execute(
                """SELECT dv.fingerprint, cd.version FROM current_derivations cd
                   JOIN derivation_versions dv ON dv.derivation_id=cd.derivation_id AND dv.version=cd.version
                   WHERE cd.derivation_id=?""", (d["derivation_id"],)).fetchone()
            if row is not None and row["fingerprint"] == d["fingerprint"]:
                unchanged += 1
                continue
            nv = int(row["version"]) + 1 if row is not None else 1
            cur.execute("INSERT INTO derivation_versions(derivation_id,version,seq,fingerprint,basis,payload)"
                        " VALUES(?,?,?,?,?,?)",
                        (d["derivation_id"], nv, seq, d["fingerprint"],
                         json.dumps(d["basis"], sort_keys=True),
                         json.dumps(d["payload"], ensure_ascii=False, sort_keys=True)))
            cur.execute("""INSERT INTO current_derivations(derivation_id,version,seq)
                           VALUES(?,?,?) ON CONFLICT(derivation_id)
                           DO UPDATE SET version=excluded.version, seq=excluded.seq""",
                        (d["derivation_id"], nv, seq))
            updated.append({"derivation_id": d["derivation_id"], "version": nv,
                            "fingerprint": d["fingerprint"]})
        return {"updated": updated, "unchanged": unchanged, "seq": seq}

    def add_events(self, events: list[dict[str, Any]], *, seq: int | None = None) -> int:
        n = 0
        with self.conn:
            cur = self.conn.cursor()
            useq = seq or self._next_seq(cur)
            for i, ev in enumerate(events):
                eid = ev.get("event_id") or f"ev-{useq}-{i}-{ev['event_type']}-{ev.get('subject','')}".replace(":", "_")
                cur.execute("INSERT OR IGNORE INTO events(event_id,seq,event_type,scope,subject,detail)"
                            " VALUES(?,?,?,?,?,?)",
                            (eid, useq, ev["event_type"], ev.get("scope", "global"),
                             ev.get("subject", ""), json.dumps(ev.get("detail", {}), ensure_ascii=False)))
                n += 1
        return n

    def list_events(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY seq, event_id").fetchall()
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]

    def list_commits(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM commits ORDER BY seq").fetchall()
        return [dict(r) | {"changes": json.loads(r["changes"])} for r in rows]

    def stream_history(self, stream_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM stream_versions WHERE stream_id=? ORDER BY version", (stream_id,)).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def derivation_history(self, derivation_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM derivation_versions WHERE derivation_id=? ORDER BY version",
            (derivation_id,)).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"]), "basis": json.loads(r["basis"])} for r in rows]

    def decisions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM decisions ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    def decision_at(self, scope: str, subject: str, seq: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """SELECT d.* FROM selection_log l JOIN decisions d ON d.decision_id=l.decision_id
               WHERE l.scope=? AND l.subject=? AND l.seq<=? ORDER BY l.seq DESC LIMIT 1""",
            (scope, subject, seq)).fetchone()
        return dict(row) if row else None

    # ---------- 历史投影 ----------
    def state_at(self, seq: int) -> dict[str, Any]:
        """投影到任意提交：返回当时各 stream 的版本与内容，不被最新值覆盖。"""
        rows = self.conn.execute(
            """SELECT sv.stream_id, st.kind, st.scope, sv.version, sv.seq, sv.payload, sv.author, sv.note
               FROM stream_versions sv JOIN streams st ON st.stream_id=sv.stream_id
               WHERE sv.seq<=? AND sv.version=(
                   SELECT MAX(version) FROM stream_versions x
                   WHERE x.stream_id=sv.stream_id AND x.seq<=?)
               ORDER BY sv.stream_id""", (seq, seq)).fetchall()
        records = [{"stream_id": r["stream_id"], "kind": r["kind"], "scope": r["scope"],
                    "version": r["version"], "seq": r["seq"],
                    "payload": json.loads(r["payload"]), "author": r["author"], "note": r["note"]}
                   for r in rows]
        drows = self.conn.execute(
            """SELECT dv.* FROM derivation_versions dv
               WHERE dv.seq<=? AND dv.version=(
                   SELECT MAX(version) FROM derivation_versions x
                   WHERE x.derivation_id=dv.derivation_id AND x.seq<=?)
               ORDER BY dv.derivation_id""", (seq, seq)).fetchall()
        derivations = [{"derivation_id": r["derivation_id"], "version": r["version"], "seq": r["seq"],
                        "fingerprint": r["fingerprint"], "basis": json.loads(r["basis"]),
                        "payload": json.loads(r["payload"])} for r in drows]
        return {"seq": seq, "records": records, "derivations": derivations}

    # ---------- 导出 / 导入（身份、顺序、来源关系保持） ----------
    def export_bundle(self) -> dict[str, Any]:
        def rows(table: str) -> list[dict]:
            return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table} ORDER BY seq").fetchall()]
        streams = [dict(r) for r in self.conn.execute("SELECT * FROM streams ORDER BY stream_id").fetchall()]
        sv = [dict(r) for r in self.conn.execute(
            "SELECT * FROM stream_versions ORDER BY seq, stream_id, version").fetchall()]
        deriv = [dict(r) for r in self.conn.execute("SELECT * FROM derivations ORDER BY derivation_id").fetchall()]
        dv = [dict(r) for r in self.conn.execute(
            "SELECT * FROM derivation_versions ORDER BY seq, derivation_id, version").fetchall()]
        curd = [dict(r) for r in self.conn.execute("SELECT * FROM current_derivations ORDER BY derivation_id").fetchall()]
        return {"bundle": "adcp-evidence", "bundle_version": 2, "schema_version": SCHEMA_VERSION,
                "head_seq": self.head_seq(),
                "streams": streams, "stream_versions": sv,
                "derivations": deriv, "derivation_versions": dv,
                "current_derivations": curd,
                "commits": rows("commits"), "events": rows("events"),
                "decisions": rows("decisions"), "selection_log": rows("selection_log")}

    @staticmethod
    def import_bundle(path: str, bundle: dict[str, Any]) -> "Repository":
        if bundle.get("bundle") != "adcp-evidence":
            raise ValueError("not an adcp-evidence bundle")
        if os.path.exists(path):
            os.remove(path)
        repo = Repository(path)
        with repo.conn:
            c = repo.conn.cursor()
            c.executescript("DELETE FROM events;DELETE FROM selection_log;DELETE FROM decisions;"
                            "DELETE FROM current_derivations;DELETE FROM derivation_versions;"
                            "DELETE FROM derivations;DELETE FROM stream_versions;DELETE FROM streams;"
                            "DELETE FROM commits;")
            for r in bundle["streams"]:
                c.execute("INSERT INTO streams VALUES(?,?,?,?,?,?)",
                          (r["stream_id"], r["kind"], r["scope"], r["current_version"],
                           r["created_seq"], r["updated_seq"]))
            for r in bundle["stream_versions"]:
                c.execute("INSERT INTO stream_versions VALUES(?,?,?,?,?,?)",
                          (r["stream_id"], r["version"], r["seq"], r["payload"], r["author"], r["note"]))
            for r in bundle["derivations"]:
                c.execute("INSERT INTO derivations VALUES(?,?)", (r["derivation_id"], r["kind"]))
            for r in bundle["derivation_versions"]:
                c.execute("INSERT INTO derivation_versions VALUES(?,?,?,?,?,?)",
                          (r["derivation_id"], r["version"], r["seq"], r["fingerprint"],
                           r["basis"], r["payload"]))
            for r in bundle["current_derivations"]:
                c.execute("INSERT INTO current_derivations VALUES(?,?,?)",
                          (r["derivation_id"], r["version"], r["seq"]))
            for r in bundle["commits"]:
                c.execute("INSERT INTO commits VALUES(?,?,?,?,?,?)",
                          (r["seq"], r["kind"], r["actor"], r["message"], r["changes"], r["ts"]))
            for r in bundle["events"]:
                c.execute("INSERT INTO events VALUES(?,?,?,?,?,?)",
                          (r["event_id"], r["seq"], r["event_type"], r["scope"], r["subject"], r["detail"]))
            for r in bundle["decisions"]:
                c.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?)",
                          (r["decision_id"], r["seq"], r["scope"], r["subject"],
                           r["chosen_candidate_id"], r["action"], r["rationale"], r["base_seq"]))
            for r in bundle["selection_log"]:
                c.execute("INSERT INTO selection_log VALUES(?,?,?,?)",
                          (r["scope"], r["subject"], r["seq"], r["decision_id"]))
        return repo

    # ---------- 依赖图说明 ----------
    def explain(self, derivation_id: str) -> dict[str, Any]:
        d = next((x for x in self.current_derivations() if x["derivation_id"] == derivation_id), None)
        if d is None:
            raise NotFound(derivation_id)
        state = self.current_state()
        current_refs = {}
        for r in state["records"]:
            current_refs[f'{r["stream_id"]}@v{r["version"]}'] = r["version"]
        checks = []
        for key, ref in d["basis"].items():
            refs = ref if isinstance(ref, list) else [ref]
            for one in refs:
                if isinstance(one, str) and "@v" in one:
                    sid, ver = one.rsplit("@v", 1)
                    row = self.conn.execute(
                        "SELECT current_version FROM streams WHERE stream_id=?", (sid,)).fetchone()
                    state_flag = "still_valid" if row and int(row["current_version"]) == int(ver) else "stale"
                    checks.append({"input": key, "ref": one, "state": state_flag})
                else:
                    checks.append({"input": key, "ref": one, "state": "parameter"})
        return {"derivation_id": derivation_id, "fingerprint": d["fingerprint"],
                "version": d["version"], "basis_checks": checks,
                "needs_recompute": any(x["state"] == "stale" for x in checks)}
