"""SQLite 存储层。

语义：
- 波束测量、correlation、amplitude、坏 beam 标志、heading/pitch/roll、
  磁偏角、安装角分开记录（raw_records.kind），原始表示保留；
- 每次推导产生独立版本，只有完整成功后才切换当前指针；
- 写入中断后重启：要么看到旧版本，要么看到完整新版本（building -> aborted）；
- 两位使用者从同一版本编辑时出现可解决的冲突，未选分支保留。
"""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_records(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  ping INTEGER, cell INTEGER, beam INTEGER, t REAL,
  value TEXT NOT NULL,
  source TEXT NOT NULL,
  event_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  base_version INTEGER,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions(
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL,
  parent INTEGER,
  fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  label TEXT
);
CREATE TABLE IF NOT EXISTS derived(
  id INTEGER PRIMARY KEY,
  version_id INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  value TEXT,
  fingerprint TEXT NOT NULL,
  inputs TEXT NOT NULL,
  status TEXT NOT NULL,
  diagnostics TEXT NOT NULL DEFAULT '[]',
  UNIQUE(version_id, node_key)
);
CREATE TABLE IF NOT EXISTS pointer(name TEXT PRIMARY KEY, version_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS conflicts(
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL,
  base_version INTEGER NOT NULL,
  winner_version INTEGER,
  loser_version INTEGER,
  status TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS params(
  id INTEGER PRIMARY KEY,
  key TEXT NOT NULL,
  version INTEGER NOT NULL,
  value TEXT NOT NULL,
  event_id INTEGER NOT NULL,
  UNIQUE(key, version)
);
"""

TABLES = ["raw_records", "events", "versions", "derived", "pointer", "conflicts", "params"]


class Conflict(Exception):
    def __init__(self, base_version, current_version):
        super().__init__(f"base v{base_version} != current v{current_version}")
        self.base_version = base_version
        self.current_version = current_version


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._abort_orphans()

    # ---------- schema / migration ----------
    def _migrate(self):
        cur = self.conn.execute("PRAGMA user_version")
        v = cur.fetchone()[0]
        if v == 0:
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
            return
        if v == 1:
            # v1 -> v2: derived 增加 diagnostics；新增 conflicts 与 params 表。
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(derived)")]
            if "diagnostics" not in cols:
                self.conn.execute("ALTER TABLE derived ADD COLUMN diagnostics TEXT NOT NULL DEFAULT '[]'")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS conflicts(
                     id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL,
                     base_version INTEGER NOT NULL, winner_version INTEGER,
                     loser_version INTEGER, status TEXT NOT NULL, payload TEXT NOT NULL)"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS params(
                     id INTEGER PRIMARY KEY, key TEXT NOT NULL, version INTEGER NOT NULL,
                     value TEXT NOT NULL, event_id INTEGER NOT NULL, UNIQUE(key, version))"""
            )
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
            return
        if v != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported schema version {v}")

    def _abort_orphans(self):
        """重启时把未完成的 building 版本标记为 aborted：指针仍指向旧版本。"""
        with self.conn:
            self.conn.execute("UPDATE versions SET status='aborted' WHERE status='building'")

    # ---------- events ----------
    def add_event(self, kind, base_version, payload) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO events(kind, base_version, payload, created_at) VALUES(?,?,?,?)",
                (kind, base_version, json.dumps(payload, ensure_ascii=False), time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        return cur.lastrowid

    def events(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM events ORDER BY id")]

    def event(self, eid):
        r = self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        return dict(r) if r else None

    # ---------- raw records ----------
    def insert_raw(self, records, event_id):
        with self.conn:
            for rec in records:
                self.conn.execute(
                    "INSERT INTO raw_records(kind,ping,cell,beam,t,value,source,event_id) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        rec["kind"], rec.get("ping"), rec.get("cell"), rec.get("beam"), rec.get("t"),
                        json.dumps(rec["value"], ensure_ascii=False),
                        rec.get("source", "casepack"), event_id,
                    ),
                )

    def raw_as_of(self, event_seq):
        """取截至某事件可见的原始记录；同一自然键取最新事件，旧记录保留不删。"""
        rows = self.conn.execute(
            """SELECT r.*, e.id AS seq FROM raw_records r JOIN events e ON e.id=r.event_id
               WHERE e.id<=? ORDER BY r.id""",
            (event_seq,),
        ).fetchall()
        latest = {}
        for r in rows:
            d = dict(r)
            d["value"] = json.loads(d["value"])
            if d["kind"] in ("beam_velocity", "correlation", "amplitude", "bad_beam_flag"):
                key = (d["kind"], d["ping"], d["cell"], d["beam"])
            elif d["kind"] in ("heading", "pitch", "roll"):
                key = (d["kind"], d["t"])
            elif d["kind"] == "bottom_track":
                key = (d["kind"], d["ping"])
            else:
                key = (d["kind"],)
            if key not in latest or latest[key]["event_id"] <= d["event_id"]:
                latest[key] = d
        out = {}
        for d in latest.values():
            out.setdefault(d["kind"], []).append(d)
        return out

    def raw_counts(self):
        return {r[0]: r[1] for r in self.conn.execute("SELECT kind, COUNT(*) FROM raw_records GROUP BY kind")}

    # ---------- params ----------
    def insert_param(self, key, value, event_id):
        row = self.conn.execute("SELECT MAX(version) FROM params WHERE key=?", (key,)).fetchone()
        nxt = (row[0] or 0) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO params(key,version,value,event_id) VALUES(?,?,?,?)",
                (key, nxt, json.dumps(value, ensure_ascii=False), event_id),
            )
        return nxt

    def params_as_of(self, event_seq):
        rows = self.conn.execute(
            """SELECT p.* FROM params p JOIN events e ON e.id=p.event_id
               WHERE e.id<=? ORDER BY p.key, p.version""",
            (event_seq,),
        ).fetchall()
        out = {}
        for r in rows:
            out[r["key"]] = {"value": json.loads(r["value"]), "version": r["version"], "param_id": r["id"]}
        return out

    # ---------- versions ----------
    def begin_version(self, event_id, parent, fp, label) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO versions(event_id,parent,fingerprint,status,label) VALUES(?,?,?,?,?)",
                (event_id, parent, fp, "building", label),
            )
        return cur.lastrowid

    def write_derived(self, version_id, nodes):
        with self.conn:
            for n in nodes:
                self.conn.execute(
                    "INSERT INTO derived(version_id,node_key,value,fingerprint,inputs,status,diagnostics)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (
                        version_id, n["key"],
                        json.dumps(n.get("value"), ensure_ascii=False),
                        n["fingerprint"],
                        json.dumps(n.get("inputs", {}), ensure_ascii=False),
                        n.get("status", "valid"),
                        json.dumps(n.get("diagnostics", []), ensure_ascii=False),
                    ),
                )

    def commit_version(self, version_id, expected_base=None, adopt=True) -> bool:
        """原子切换：标记 complete 并（可选）移动当前指针。

        若 expected_base 与当前指针不一致，返回 False（冲突），版本仍保留为分支。
        """
        with self.conn:
            self.conn.execute("UPDATE versions SET status='complete' WHERE id=?", (version_id,))
            if not adopt:
                return True
            row = self.conn.execute("SELECT version_id FROM pointer WHERE name='current'").fetchone()
            if row is None:
                if expected_base is not None:
                    return False
                self.conn.execute("INSERT INTO pointer(name,version_id) VALUES('current',?)", (version_id,))
                return True
            cur = row[0]
            if expected_base is not None and cur != expected_base:
                return False
            self.conn.execute("UPDATE pointer SET version_id=? WHERE name='current'", (version_id,))
            return True

    def current_version(self):
        row = self.conn.execute("SELECT version_id FROM pointer WHERE name='current'").fetchone()
        return row[0] if row else None

    def set_pointer(self, version_id):
        with self.conn:
            self.conn.execute(
                "INSERT INTO pointer(name,version_id) VALUES('current',?)"
                " ON CONFLICT(name) DO UPDATE SET version_id=excluded.version_id",
                (version_id,),
            )

    def version(self, vid):
        r = self.conn.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        return dict(r) if r else None

    def versions(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM versions ORDER BY id")]

    def version_at_event(self, event_seq):
        """截至某事件时指针所对应的可见版本（用于回放当时结论）。"""
        row = self.conn.execute(
            "SELECT * FROM versions WHERE event_id<=? AND status='complete' ORDER BY event_id DESC, id DESC LIMIT 1",
            (event_seq,),
        ).fetchone()
        return dict(row) if row else None

    # ---------- derived ----------
    def derived(self, version_id):
        rows = self.conn.execute(
            "SELECT * FROM derived WHERE version_id=? ORDER BY id", (version_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["value"] = json.loads(d["value"]) if d["value"] else None
            d["inputs"] = json.loads(d["inputs"])
            d["diagnostics"] = json.loads(d["diagnostics"])
            out.append(d)
        return out

    def derived_map(self, version_id):
        return {d["node_key"]: d for d in self.derived(version_id)}

    def node(self, version_id, key):
        r = self.conn.execute(
            "SELECT * FROM derived WHERE version_id=? AND node_key=?", (version_id, key)
        ).fetchone()
        if not r:
            return None
        d = dict(r)
        d["value"] = json.loads(d["value"]) if d["value"] else None
        d["inputs"] = json.loads(d["inputs"])
        d["diagnostics"] = json.loads(d["diagnostics"])
        return d

    # ---------- conflicts ----------
    def record_conflict(self, event_id, base_version, winner_version, loser_version, payload) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO conflicts(event_id,base_version,winner_version,loser_version,status,payload)"
                " VALUES(?,?,?,?,?,?)",
                (event_id, base_version, winner_version, loser_version, "open",
                 json.dumps(payload, ensure_ascii=False)),
            )
        return cur.lastrowid

    def resolve_conflict(self, conflict_id, adopt):
        """adopt: 'winner' 或 'loser'。未选分支保留，不做不可逆覆盖。"""
        c = self.conn.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
        if not c:
            raise KeyError(conflict_id)
        chosen = c["winner_version"] if adopt == "winner" else c["loser_version"]
        with self.conn:
            self.conn.execute("UPDATE conflicts SET status=? WHERE id=?",
                              (f"resolved:{adopt}", conflict_id))
        self.set_pointer(chosen)
        return chosen

    def conflicts(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM conflicts ORDER BY id")]

    # ---------- export / import ----------
    def export_json(self):
        """完整导出：身份、顺序与来源关系全部保留。"""
        data = {"schema_version": SCHEMA_VERSION, "tables": {}}
        for t in TABLES:
            data["tables"][t] = [dict(r) for r in self.conn.execute(f"SELECT * FROM {t} ORDER BY id")]
        return data

    def import_json(self, data):
        """导入到空库：保留原 id、顺序与来源关系。"""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("unsupported export schema_version")
        with self.conn:
            for t in TABLES:
                self.conn.execute(f"DELETE FROM {t}")
            for t in TABLES:
                for row in data["tables"].get(t, []):
                    cols = ",".join(row.keys())
                    qs = ",".join("?" for _ in row)
                    self.conn.execute(f"INSERT INTO {t}({cols}) VALUES({qs})", tuple(row.values()))
        self._abort_orphans()
