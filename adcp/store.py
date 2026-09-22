"""Versioned SQLite storage.

Guarantees exercised by the test suite:

* sources of different kinds (beam velocity, correlation, amplitude, bad-beam
  flags, heading, pitch, roll, declination, mount angles, bottom track) are
  stored as separate immutable records; corrections add new revisions;
* every derivation is an independent run version.  Nodes are written to a
  staging area; the *current pointer* is switched in the same transaction that
  promotes the complete run, so a process killed mid-write leaves the previous
  complete version (or a finished new one) - never a partial version;
* two reviewers editing from the same version get a resolvable conflict.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

SCHEMA_VERSION = 3


class ConflictError(Exception):
    """Raised when an edit is based on a stale edit version."""


def _uid(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:12]


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        fresh = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if fresh:
            self._create_schema(SCHEMA_VERSION)
        else:
            self._migrate_and_recover()

    # ---------------------------------------------------------------- schema
    def _create_schema(self, version: int) -> None:
        with self._tx() as c:
            c.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE sources (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    unit TEXT,
                    revision INTEGER NOT NULL,
                    supersedes_id TEXT,
                    withdrawn INTEGER NOT NULL DEFAULT 0,
                    origin TEXT NOT NULL,
                    created_event_seq INTEGER,
                    created_at_s REAL NOT NULL,
                    UNIQUE(kind, target, revision)
                );
                CREATE TABLE rules (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    supersedes_id TEXT,
                    withdrawn INTEGER NOT NULL DEFAULT 0,
                    created_event_seq INTEGER,
                    created_at_s REAL NOT NULL,
                    UNIQUE(type, version)
                );
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    label TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT 'main',
                    status TEXT NOT NULL,
                    manifest_json TEXT,
                    fp TEXT,
                    rationale TEXT,
                    trigger_event_seq INTEGER,
                    created_at_s REAL NOT NULL,
                    completed_at_s REAL
                );
                CREATE TABLE run_nodes_staging (
                    run_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deps_json TEXT NOT NULL,
                    value_json TEXT,
                    basis TEXT,
                    unit TEXT,
                    val_id TEXT,
                    candidates_json TEXT,
                    diagnostics_json TEXT,
                    source_refs_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, slot)
                );
                CREATE TABLE run_nodes (
                    run_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deps_json TEXT NOT NULL,
                    value_json TEXT,
                    basis TEXT,
                    unit TEXT,
                    val_id TEXT,
                    candidates_json TEXT,
                    diagnostics_json TEXT,
                    source_refs_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, slot)
                );
                CREATE TABLE current_run (
                    scope TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL
                );
                CREATE TABLE events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    at_s REAL NOT NULL,
                    run_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE decisions (
                    event_seq INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    node_slot TEXT,
                    actor TEXT NOT NULL,
                    note TEXT
                );
                CREATE TABLE decision_overlays (
                    run_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_event_seq INTEGER NOT NULL,
                    PRIMARY KEY(run_id, slot)
                );
                CREATE TABLE edit_conflicts (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    base_edit_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at_s REAL NOT NULL,
                    resolved_event_seq INTEGER,
                    resolution TEXT
                );
                """
            )
            c.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(version),))
            c.execute("INSERT INTO meta VALUES ('edit_version_main', '0')")
            for key in ("dataset_id", "dataset_title", "current_run_id"):
                c.execute("INSERT INTO meta(key,value) VALUES(?, '')", (key,))

    def _migrate_and_recover(self) -> None:
        ver = int(self._meta("schema_version", "1"))
        if ver == 1:
            with self._tx() as c:
                # v1 derivations lacked the fingerprint column and conflicts.
                c.execute("ALTER TABLE runs ADD COLUMN fp TEXT")
                c.execute(
                    """CREATE TABLE IF NOT EXISTS edit_conflicts (
                        id TEXT PRIMARY KEY, scope TEXT NOT NULL, run_id TEXT NOT NULL,
                        base_edit_version INTEGER NOT NULL, payload_json TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open', created_at_s REAL NOT NULL,
                        resolved_event_seq INTEGER, resolution TEXT)"""
                )
                c.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            ver = 2
        if ver == 2:
            with self._tx() as c:
                # v2 kept only one decision; v3 keeps full overlays + events.
                c.execute(
                    """CREATE TABLE IF NOT EXISTS decisions (
                        event_seq INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                        action TEXT NOT NULL, node_slot TEXT,
                        actor TEXT NOT NULL, note TEXT)"""
                )
                c.execute(
                    """CREATE TABLE IF NOT EXISTS decision_overlays (
                        run_id TEXT NOT NULL, slot TEXT NOT NULL, action TEXT NOT NULL,
                        from_event_seq INTEGER NOT NULL, PRIMARY KEY(run_id, slot))"""
                )
                c.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
            ver = 3
        self.recover()

    def recover(self) -> List[str]:
        """Drop staging/``building`` remnants of interrupted writes.

        The current pointer never moved (that happens only at complete_run),
        so readers still see the previous complete version.
        """
        removed: List[str] = []
        with self._tx() as c:
            rows = c.execute(
                "SELECT id FROM runs WHERE status='building'"
            ).fetchall()
            for r in rows:
                c.execute("DELETE FROM run_nodes_staging WHERE run_id=?", (r["id"],))
                c.execute("DELETE FROM runs WHERE id=?", (r["id"],))
                removed.append(r["id"])
        return removed

    # ------------------------------------------------------------------- util
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    @staticmethod
    def _j(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)

    @staticmethod
    def _d(text: Optional[str]) -> Any:
        return json.loads(text) if text is not None else None

    def _meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return row["value"]

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def init_dataset(self, dataset_id: str, title: str) -> None:
        self.set_meta("dataset_id", dataset_id)
        self.set_meta("dataset_title", title)

    # ---------------------------------------------------------------- sources
    def add_source(self, kind: str, target: str, payload: dict,
                   unit: Optional[str], origin: str,
                   supersedes_id: Optional[str] = None,
                   event_seq: Optional[int] = None) -> str:
        revision = 1 + (
            self.conn.execute(
                "SELECT COALESCE(MAX(revision),0) AS m FROM sources "
                "WHERE kind=? AND target=?",
                (kind, target),
            ).fetchone()["m"]
        )
        sid = _uid("src")
        with self._tx() as c:
            c.execute(
                """INSERT INTO sources
                   (id,kind,target,payload_json,unit,revision,supersedes_id,
                    withdrawn,origin,created_event_seq,created_at_s)
                   VALUES(?,?,?,?,?,?,?,0,?,?,?,?)""",
                (sid, kind, target, self._j(payload), unit, revision,
                 supersedes_id, origin, event_seq, time.time()),
            )
        return sid

    def get_source(self, sid: str) -> dict:
        r = self.conn.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        out = dict(r)
        out["payload"] = self._d(r["payload_json"])
        return out

    def latest_sources(self) -> Dict[tuple, str]:
        """Map ``(kind,target)`` to id of newest non-withdrawn source id."""
        out: Dict[tuple, str] = {}
        rows = self.conn.execute(
            "SELECT id,kind,target,revision FROM sources WHERE withdrawn=0"
        ).fetchall()
        best: Dict[tuple, int] = {}
        for r in rows:
            key = (r["kind"], r["target"])
            if key not in best or r["revision"] > best[key]:
                best[key] = r["revision"]
                out[key] = r["id"]
        return out

    def withdraw_source(self, sid: str, event_seq: Optional[int] = None) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE sources SET withdrawn=1, created_event_seq=COALESCE(created_event_seq,?) WHERE id=?",
                (event_seq, sid),
            )

    # ------------------------------------------------------------------ rules
    def add_rule(self, type_: str, config: dict,
                 supersedes_id: Optional[str] = None,
                 event_seq: Optional[int] = None) -> str:
        version = 1 + (
            self.conn.execute(
                "SELECT COALESCE(MAX(version),0) m FROM rules WHERE type=?",
                (type_,),
            ).fetchone()["m"]
        )
        rid = _uid("rule")
        with self._tx() as c:
            c.execute(
                """INSERT INTO rules
                   (id,type,version,config_json,supersedes_id,withdrawn,
                    created_event_seq,created_at_s)
                   VALUES(?,?,?,?,?,0,?,?)""",
                (rid, type_, version, self._j(config), supersedes_id,
                 event_seq, time.time()),
            )
        return rid

    def get_rule(self, rid: str) -> dict:
        r = self.conn.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        out = dict(r)
        out["config"] = self._d(r["config_json"])
        return out

    def active_rules(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        rows = self.conn.execute(
            "SELECT * FROM rules WHERE withdrawn=0 ORDER BY version DESC"
        ).fetchall()
        for r in rows:
            if r["type"] not in out:
                out[r["type"]] = self.get_rule(r["id"])
        return out

    def rule_at_event(self, type_: str, seq: int) -> Optional[dict]:
        """Rule version that was active immediately at/after event ``seq``."""
        rows = self.conn.execute(
            """SELECT id FROM rules
               WHERE type=? AND withdrawn=0
                 AND COALESCE(created_event_seq,0)<=?
               ORDER BY version DESC""",
            (type_, seq),
        ).fetchall()
        return self.get_rule(rows[0]["id"]) if rows else None

    # ------------------------------------------------------------------- runs
    def begin_run(self, label: str, parent_run_id: Optional[str],
                  branch: str, manifest: dict, fp: Optional[str],
                  rationale: str, trigger_event_seq: Optional[int] = None) -> str:
        rid = _uid("run")
        with self._tx() as c:
            c.execute(
                """INSERT INTO runs
                   (id,parent_run_id,label,branch,status,manifest_json,fp,
                    rationale,trigger_event_seq,created_at_s)
                   VALUES(?,?,?,?,'building',?,?,?,?,?)""",
                (rid, parent_run_id, label, branch, self._j(manifest), fp,
                 rationale, trigger_event_seq, time.time()),
            )
        return rid

    def stage_nodes(self, run_id: str, nodes: List[dict]) -> None:
        """Write nodes to the staging table only; not visible until complete."""
        rows = [
            (
                run_id, n["slot"], n["kind"], n["status"],
                self._j(n.get("deps", [])),
                self._j(n["value"]) if n.get("value") is not None else None,
                n.get("basis"), n.get("unit"), n.get("val_id"),
                self._j(n["candidates"]) if n.get("candidates") is not None else None,
                self._j(n.get("diagnostics", [])),
                self._j(n.get("source_refs", {})),
            )
            for n in nodes
        ]
        with self._tx() as c:
            c.executemany(
                """INSERT INTO run_nodes_staging
                   (run_id,slot,kind,status,deps_json,value_json,basis,unit,
                    val_id,candidates_json,diagnostics_json,source_refs_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def complete_run(self, run_id: str) -> None:
        """Atomic promotion: copy staging -> live and move the pointer."""
        with self._tx() as c:
            n = c.execute("SELECT COUNT(*) AS n FROM run_nodes_staging WHERE run_id=?",
                          (run_id,)).fetchone()["n"]
            if n == 0:
                raise RuntimeError("refusing to complete empty run %s" % run_id)
            row = c.execute("SELECT branch FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            c.execute(
                """INSERT INTO run_nodes
                   SELECT * FROM run_nodes_staging WHERE run_id=?""",
                (run_id,),
            )
            c.execute("DELETE FROM run_nodes_staging WHERE run_id=?", (run_id,))
            c.execute(
                "UPDATE runs SET status='complete', completed_at_s=? WHERE id=?",
                (time.time(), run_id),
            )
            c.execute(
                """INSERT INTO current_run(scope,run_id) VALUES(?,?)
                   ON CONFLICT(scope) DO UPDATE SET run_id=excluded.run_id""",
                (row["branch"], run_id),
            )
            c.execute("UPDATE meta SET value=? WHERE key='current_run_id'", (run_id,))

    def current_run_id(self, branch: str = "main") -> Optional[str]:
        r = self.conn.execute("SELECT run_id FROM current_run WHERE scope=?",
                              (branch,)).fetchone()
        return r["run_id"] if r else None

    def run_status(self, run_id: str) -> str:
        return self.conn.execute("SELECT status FROM runs WHERE id=?",
                                 (run_id,)).fetchone()["status"]

    def get_run(self, run_id: str) -> dict:
        r = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        out = dict(r)
        out["manifest"] = self._d(r["manifest_json"])
        return out

    def list_runs(self) -> List[dict]:
        return [self.get_run(r["id"]) for r in
                self.conn.execute("SELECT id FROM runs ORDER BY created_at_s").fetchall()]

    def get_node(self, run_id: str, slot: str) -> Optional[dict]:
        table = "run_nodes" if self.run_status(run_id) == "complete" \
            else "run_nodes_staging"
        r = self.conn.execute(
            f"SELECT * FROM {table} WHERE run_id=? AND slot=?", (run_id, slot)
        ).fetchone()
        return self._node_row(r) if r else None

    @staticmethod
    def _node_row(r: sqlite3.Row) -> dict:
        return {
            "slot": r["slot"], "kind": r["kind"], "status": r["status"],
            "deps": json.loads(r["deps_json"]),
            "value": json.loads(r["value_json"]) if r["value_json"] else None,
            "basis": r["basis"], "unit": r["unit"], "val_id": r["val_id"],
            "candidates": json.loads(r["candidates_json"])
                if r["candidates_json"] else None,
            "diagnostics": json.loads(r["diagnostics_json"]),
            "source_refs": json.loads(r["source_refs_json"]),
        }

    def all_nodes(self, run_id: str) -> List[dict]:
        r = self.conn.execute("SELECT status FROM runs WHERE id=?",
                              (run_id,)).fetchone()
        table = "run_nodes" if r and r["status"] == "complete" else "run_nodes_staging"
        return [self._node_row(x) for x in
                self.conn.execute(f"SELECT * FROM {table} WHERE run_id=? ORDER BY slot",
                                  (run_id,)).fetchall()]

    def staging_count(self, run_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) n FROM run_nodes_staging WHERE run_id=?", (run_id,)
        ).fetchone()["n"]

    # ----------------------------------------------------------------- events
    def append_event(self, type_: str, actor: str, payload: dict,
                     run_id: Optional[str] = None,
                     eid: Optional[str] = None) -> dict:
        eid = eid or _uid("evt")
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO events(id,type,actor,at_s,run_id,payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (eid, type_, actor, time.time(), run_id, self._j(payload)),
            )
            seq = cur.lastrowid
        return {"seq": seq, "id": eid, "type": type_, "actor": actor,
                "run_id": run_id, "payload": payload}

    def list_events(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        return [
            {"seq": r["seq"], "id": r["id"], "type": r["type"],
             "actor": r["actor"], "at_s": r["at_s"], "run_id": r["run_id"],
             "payload": self._d(r["payload_json"])}
            for r in rows
        ]

    # -------------------------------------------------------------- decisions
    def edit_version(self, branch: str = "main") -> int:
        return int(self._meta("edit_version_" + branch, "0"))

    def _bump_edit_version(self, c: sqlite3.Connection, branch: str) -> int:
        key = "edit_version_" + branch
        nv = int(self._meta(key, "0")) + 1
        c.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(nv)),
        )
        return nv

    def apply_decision(self, run_id: str, action: str, actor: str,
                       base_edit_version: int, slot: Optional[str] = None,
                       note: str = "", branch: str = "main") -> dict:
        """Accept / reject / adjust one slot or a whole branch.

        Two reviewers acting from the same ``base_edit_version`` collide: the
        first wins, the second gets a recorded, resolvable conflict.
        """
        if base_edit_version != self.edit_version(branch):
            cid = _uid("conf")
            payload = {"run_id": run_id, "action": action, "slot": slot,
                       "actor": actor, "note": note,
                       "their_base": base_edit_version,
                       "current": self.edit_version(branch)}
            with self._tx() as c:
                c.execute(
                    """INSERT INTO edit_conflicts
                       (id,scope,run_id,base_edit_version,payload_json,
                        status,created_at_s)
                       VALUES(?,?,?,?,?,'open',?)""",
                    (cid, branch, run_id, base_edit_version,
                     self._j(payload), time.time()),
                )
            raise ConflictError(cid)
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO events(id,type,actor,at_s,run_id,payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (_uid("evt"), "decision", actor, time.time(), run_id,
                 self._j({"action": action, "slot": slot, "note": note,
                          "base_edit_version": base_edit_version}))),
            seq = cur.lastrowid
            c.execute(
                "INSERT INTO decisions(event_seq,run_id,action,node_slot,actor,note) "
                "VALUES(?,?,?,?,?,?)",
                (seq, run_id, action, slot, actor, note),
            )
            if slot is not None:
                c.execute(
                    """INSERT INTO decision_overlays(run_id,slot,action,from_event_seq)
                       VALUES(?,?,?,?)
                       ON CONFLICT(run_id,slot) DO UPDATE SET action=excluded.action,
                           from_event_seq=excluded.from_event_seq""",
                    (run_id, slot, action, seq),
                )
            nv = self._bump_edit_version(c, branch)
        return {"event_seq": seq, "new_edit_version": nv}

    def overlays(self, run_id: str) -> Dict[str, dict]:
        return {
            r["slot"]: {"action": r["action"], "from_event_seq": r["from_event_seq"]}
            for r in self.conn.execute(
                "SELECT slot,action,from_event_seq FROM decision_overlays WHERE run_id=?",
                (run_id,)).fetchall()
        }

    def list_conflicts(self, status: str = "open") -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edit_conflicts WHERE status=? ORDER BY created_at_s",
            (status,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = self._d(r["payload_json"])
            out.append(d)
        return out

    def resolve_conflict(self, conflict_id: str, resolution: str,
                         actor: str, run_id: str, action: str,
                         slot: Optional[str] = None, note: str = "",
                         branch: str = "main") -> dict:
        """Resolve a conflict by re-applying a decision at the new version."""
        base = self.edit_version(branch)
        result = self.apply_decision(run_id, action, actor, base, slot=slot,
                                     note=note + " [resolution:%s]" % resolution,
                                     branch=branch)
        with self._tx() as c:
            c.execute(
                "UPDATE edit_conflicts SET status='resolved', "
                "resolved_event_seq=?, resolution=? WHERE id=?",
                (result["event_seq"], resolution, conflict_id),
            )
        return result

    # --------------------------------------------------------- export/import
    def export_bundle(self) -> dict:
        """Portable bundle. Identity strings are content/addressed by
        construction, so re-import keeps every id, order and provenance link."""
        def rows(table: str) -> List[dict]:
            return [dict(r) for r in
                    self.conn.execute(f"SELECT * FROM {table}").fetchall()]

        return {
            "bundle_format": 1,
            "exported_at_s": time.time(),
            "meta": rows("meta"),
            "sources": rows("sources"),
            "rules": rows("rules"),
            "runs": rows("runs"),
            "run_nodes": rows("run_nodes"),
            "current_run": rows("current_run"),
            "events": rows("events"),
            "decisions": rows("decisions"),
            "decision_overlays": rows("decision_overlays"),
            "edit_conflicts": rows("edit_conflicts"),
            "sqlite_sequence": rows("sqlite_sequence"),
        }

    def import_bundle(self, bundle: dict) -> None:
        with self._tx() as c:
            c.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name LIKE 'run%' OR name IN
                   ('meta','sources','rules','events','decisions',
                    'decision_overlays','edit_conflicts')""")
            for (name,) in c.fetchall():
                c.execute("DELETE FROM %s" % name)
            c.execute("DELETE FROM sqlite_sequence")
            for r in bundle["meta"]:
                c.execute("INSERT INTO meta VALUES(:key,:value)", r)
            for r in bundle["sources"]:
                c.execute(
                    """INSERT INTO sources
                       (id,kind,target,payload_json,unit,revision,supersedes_id,
                        withdrawn,origin,created_event_seq,created_at_s)
                       VALUES(:id,:kind,:target,:payload_json,:unit,:revision,
                              :supersedes_id,:withdrawn,:origin,
                              :created_event_seq,:created_at_s)""", r)
            for r in bundle["rules"]:
                c.execute(
                    """INSERT INTO rules
                       (id,type,version,config_json,supersedes_id,withdrawn,
                        created_event_seq,created_at_s)
                       VALUES(:id,:type,:version,:config_json,:supersedes_id,
                              :withdrawn,:created_event_seq,:created_at_s)""", r)
            for r in bundle["runs"]:
                c.execute(
                    """INSERT INTO runs
                       (id,parent_run_id,label,branch,status,manifest_json,fp,
                        rationale,trigger_event_seq,created_at_s,completed_at_s)
                       VALUES(:id,:parent_run_id,:label,:branch,:status,
                              :manifest_json,:fp,:rationale,:trigger_event_seq,
                              :created_at_s,:completed_at_s)""", r)
            for r in bundle["run_nodes"]:
                c.execute(
                    """INSERT INTO run_nodes
                       (run_id,slot,kind,status,deps_json,value_json,basis,unit,
                        val_id,candidates_json,diagnostics_json,source_refs_json)
                       VALUES(:run_id,:slot,:kind,:status,:deps_json,:value_json,
                              :basis,:unit,:val_id,:candidates_json,
                              :diagnostics_json,:source_refs_json)""", r)
            for r in bundle.get("current_run", []):
                c.execute("INSERT INTO current_run VALUES(:scope,:run_id)", r)
            for r in bundle["events"]:
                c.execute(
                    "INSERT INTO events(seq,id,type,actor,at_s,run_id,payload_json) "
                    "VALUES(:seq,:id,:type,:actor,:at_s,:run_id,:payload_json)", r)
            for r in bundle.get("decisions", []):
                c.execute(
                    """INSERT INTO decisions
                       (event_seq,run_id,action,node_slot,actor,note)
                       VALUES(:event_seq,:run_id,:action,:node_slot,:actor,:note)""",
                    r)
            for r in bundle.get("decision_overlays", []):
                c.execute(
                    """INSERT INTO decision_overlays
                       (run_id,slot,action,from_event_seq)
                       VALUES(:run_id,:slot,:action,:from_event_seq)""", r)
            for r in bundle.get("edit_conflicts", []):
                c.execute(
                    """INSERT INTO edit_conflicts
                       (id,scope,run_id,base_edit_version,payload_json,status,
                        created_at_s,resolved_event_seq,resolution)
                       VALUES(:id,:scope,:run_id,:base_edit_version,:payload_json,
                              :status,:created_at_s,:resolved_event_seq,:resolution)""",
                    r)
            for r in bundle.get("sqlite_sequence", []):
                c.execute("INSERT INTO sqlite_sequence VALUES(:name,:seq)", r)

    def close(self) -> None:
        self.conn.close()
