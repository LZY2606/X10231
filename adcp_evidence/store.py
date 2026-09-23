"""Versioned SQLite storage.

Only two things are mutable: the per-branch commit pointer (only moves
forward) and the working set of uncommitted edits. Derivations are content
addressed; a commit inserts all rows in one transaction and moves the pointer
last, so a crash leaves either the previous commit or the complete new one.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from .fingerprint import canonical_json, content_id

SCHEMA_VERSION = 2


SCHEMA_V2 = """
CREATE TABLE meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE branches(
  name TEXT PRIMARY KEY,
  head_seq INTEGER NOT NULL REFERENCES commits(seq)
);
CREATE TABLE commits(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  branch TEXT NOT NULL,
  parent_seq INTEGER REFERENCES commits(seq),
  author TEXT NOT NULL,
  label TEXT NOT NULL,
  created_s REAL NOT NULL,
  event_log TEXT NOT NULL,
  changes TEXT NOT NULL
);
CREATE TABLE source_versions(
  record_id TEXT NOT NULL,
  version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  introduced_seq INTEGER NOT NULL REFERENCES commits(seq),
  superseded_seq INTEGER REFERENCES commits(seq),
  payload TEXT NOT NULL,
  PRIMARY KEY (record_id, version)
);
CREATE TABLE derivations(
  fingerprint TEXT PRIMARY KEY,
  slot TEXT NOT NULL,
  status TEXT NOT NULL,
  payload TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  transform_order TEXT NOT NULL,
  created_seq INTEGER NOT NULL REFERENCES commits(seq)
);
CREATE TABLE commit_slots(
  seq INTEGER NOT NULL,
  slot TEXT NOT NULL,
  status TEXT NOT NULL,
  fingerprint TEXT,
  PRIMARY KEY (seq, slot)
);
CREATE TABLE decisions(
  id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL REFERENCES commits(seq),
  subject TEXT NOT NULL,
  verdict TEXT NOT NULL,
  rejected_candidates TEXT NOT NULL,
  note TEXT NOT NULL
);
CREATE TABLE work_edits(
  record_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  base_record_version TEXT NOT NULL,
  updated_s REAL NOT NULL
);
CREATE TABLE work_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX idx_sv_record ON source_versions(record_id, superseded_seq);
CREATE INDEX idx_der_slot ON derivations(slot);
CREATE INDEX idx_slots_fp ON commit_slots(fingerprint);
"""


class ConflictError(RuntimeError):
    """An edit was based on a stale source version."""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> int:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return version
    if version == 0:
        conn.executescript(SCHEMA_V2)
        conn.execute("PRAGMA user_version=2")
        conn.commit()
        return 2
    if version == 1:
        _migrate_1_to_2(conn)
        return 2
    raise RuntimeError(f"unsupported schema version {version}")


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """Honest legacy migration from a flat active-snapshot schema.

    v1 stored the active payload in ``sources`` and a flat pointer list in
    ``v1_pointers``. Immutable version rows are replayed and the row count is
    verified before the version stamp advances.
    """
    old_sources = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
    old_pointers = conn.execute(
        "SELECT COUNT(*) AS n FROM v1_pointers"
    ).fetchone()["n"]
    conn.executescript(SCHEMA_V2)
    migrated = 0
    for row in conn.execute("SELECT * FROM sources"):
        payload_obj = json.loads(row["payload"])
        version = content_id("src", payload_obj)
        conn.execute(
            "INSERT INTO source_versions"
            "(record_id,version,record_kind,introduced_seq,payload) "
            "VALUES(?,?,?,-1,?)",
            (row["record_id"], version,
             payload_obj.get("record_kind", "unknown"), row["payload"]),
        )
        migrated += 1
    if migrated != old_sources:
        raise RuntimeError("migration row count mismatch")
    conn.execute("PRAGMA user_version=2")
    conn.commit()
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('v1_pointer_count',?)",
        (str(old_pointers),),
    )


def backfill_migration_bootstrap(conn: sqlite3.Connection, seq: int) -> None:
    """Link v1-replayed rows to the bootstrap commit and verify pointer count."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key='v1_pointer_count'"
    ).fetchone()
    conn.execute(
        "UPDATE source_versions SET introduced_seq=? WHERE introduced_seq=-1",
        (seq,),
    )
    if row is not None:
        expected = int(row["value"])
        actual = conn.execute(
            "SELECT COUNT(*) AS n FROM commit_slots WHERE seq=?", (seq,)
        ).fetchone()["n"]
        if actual != expected:
            raise RuntimeError("migration pointer count mismatch")
        conn.execute("DELETE FROM meta WHERE key='v1_pointer_count'")
        conn.commit()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = connect(path)
        init_db(self.conn)

    # ---------- working set of uncommitted edits ----------
    def stage_edit(self, record_id: str, payload: dict,
                   base_record_version: str,
                   allow_stale: bool = False) -> None:
        """Stage an edit with optimistic locking on the source version."""
        current = self.active_source_version(record_id)
        if current is not None and current != base_record_version:
            if not allow_stale:
                raise ConflictError(
                    f"{record_id}: base {base_record_version[:12]} stale, "
                    f"current {current[:12]}"
                )
        self._write_edit(record_id, payload, base_record_version)

    def _write_edit(self, record_id: str, payload: dict,
                    base_record_version: str) -> None:
        self.conn.execute(
            "INSERT INTO work_edits(record_id,payload,base_record_version,"
            "updated_s) VALUES(?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET "
            "payload=excluded.payload,base_record_version=excluded."
            "base_record_version,updated_s=excluded.updated_s",
            (record_id, canonical_json(payload), base_record_version,
             time.time()),
        )
        self.conn.commit()

    def staged_edits(self) -> dict[str, dict]:
        return {
            row["record_id"]: {
                "payload": json.loads(row["payload"]),
                "base_version": row["base_record_version"],
            }
            for row in self.conn.execute("SELECT * FROM work_edits")
        }

    def discard_edits(self, record_ids: Optional[set[str]] = None) -> None:
        if record_ids is None:
            self.conn.execute("DELETE FROM work_edits")
        else:
            self.conn.executemany(
                "DELETE FROM work_edits WHERE record_id=?",
                [(rid,) for rid in record_ids],
            )
        self.conn.commit()

    def edit_conflicts(self) -> list[dict]:
        out = []
        for rid, edit in self.staged_edits().items():
            current = self.active_source_version(rid)
            if current is not None and current != edit["base_version"]:
                out.append(
                    {"record_id": rid, "base": edit["base_version"],
                     "current": current, "resolvable": True}
                )
        return out

    # ---------- commits ----------
    def head_seq(self, branch: str = "main") -> Optional[int]:
        row = self.conn.execute(
            "SELECT head_seq FROM branches WHERE name=?", (branch,)
        ).fetchone()
        return None if row is None else row["head_seq"]

    def branches(self) -> list[str]:
        return [r["name"] for r in self.conn.execute(
            "SELECT name FROM branches ORDER BY name")]

    def active_source_version(self, record_id: str,
                              branch: str = "main") -> Optional[str]:
        seq = self.head_seq(branch)
        if seq is None:
            return None
        row = self.conn.execute(
            "SELECT version FROM source_versions sv JOIN commits c "
            "ON c.seq=sv.introduced_seq WHERE sv.record_id=? AND c.branch=? "
            "AND (sv.superseded_seq IS NULL OR sv.superseded_seq>?) "
            "ORDER BY sv.introduced_seq DESC LIMIT 1",
            (record_id, branch, seq),
        ).fetchone()
        if row:
            return row["version"]
        # Record may live on the parent branch ancestry only.
        row = self.conn.execute(
            "SELECT version FROM source_versions WHERE record_id=? "
            "AND introduced_seq<=? AND (superseded_seq IS NULL "
            "OR superseded_seq>?) ORDER BY introduced_seq DESC LIMIT 1",
            (record_id, seq, seq),
        ).fetchone()
        return row["version"] if row else None

    def commit(
        self,
        *,
        branch: str,
        author: str,
        label: str,
        source_changes: dict[str, dict],
        derivations: list[dict],
        slot_status: dict[str, dict],
        events: list[dict],
        decisions: Optional[list[dict]] = None,
        parent_branch: Optional[str] = None,
        crash_after_rows: bool = False,
    ) -> int:
        """Atomically write one commit; pointer moves only at the end."""
        parent_seq = self.head_seq(branch)
        if parent_seq is None and parent_branch:
            parent_seq = self.head_seq(parent_branch)
        if parent_seq is None and self.head_seq("main") is not None:
            raise RuntimeError(f"branch {branch} needs an explicit parent")
        commit_id = content_id(
            "commit",
            {"branch": branch, "parent": parent_seq, "label": label,
             "changes": sorted(source_changes),
             "events": events},
        )
        cur = self.conn
        cur.execute("BEGIN IMMEDIATE")
        try:
            created_s = time.time()
            cur.execute(
                "INSERT INTO commits(id,branch,parent_seq,author,label,"
                "created_s,event_log,changes) VALUES(?,?,?,?,?,?,?,?)",
                (commit_id, branch, parent_seq, author, label, created_s,
                 canonical_json(events), canonical_json(
                     {rid: v["version"] for rid, v in source_changes.items()})),
            )
            seq = cur.execute("SELECT seq FROM commits WHERE id=?",
                              (commit_id,)).fetchone()["seq"]
            for rid, change in source_changes.items():
                if change.get("action") == "delete":
                    cur.execute(
                        "UPDATE source_versions SET superseded_seq=? "
                        "WHERE record_id=? AND superseded_seq IS NULL",
                        (seq, rid),
                    )
                    continue
                payload = change["payload"]
                version = change["version"]
                kind = payload.get("record_kind", "unknown")
                cur.execute(
                    "UPDATE source_versions SET superseded_seq=? "
                    "WHERE record_id=? AND superseded_seq IS NULL",
                    (seq, rid),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO source_versions(record_id,version,"
                    "record_kind,introduced_seq,payload) VALUES(?,?,?,?,?)",
                    (rid, version, kind, seq, canonical_json(payload)),
                )
            for der in derivations:
                cur.execute(
                    "INSERT OR IGNORE INTO derivations(fingerprint,slot,status,"
                    "payload,rule_version,transform_order,created_seq) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (der["fingerprint"], der["slot"], der["status"],
                     canonical_json(der["payload"]), der["rule_version"],
                     der["transform_order"], seq),
                )
            for slot, spec in sorted(slot_status.items()):
                cur.execute(
                    "INSERT INTO commit_slots(seq,slot,status,fingerprint) "
                    "VALUES(?,?,?,?)",
                    (seq, slot, spec["status"], spec.get("fingerprint")),
                )
            for dec in decisions or []:
                cur.execute(
                    "INSERT INTO decisions(id,seq,subject,verdict,"
                    "rejected_candidates,note) VALUES(?,?,?,?,?,?)",
                    (dec["id"], seq, dec["subject"], dec["verdict"],
                     canonical_json(dec.get("rejected_candidates", [])),
                     dec.get("note", "")),
                )
            if crash_after_rows:
                # Test hook: rows are in the transaction but the pointer has
                # not moved and no COMMIT is issued.
                cur.execute("ROLLBACK")
                raise RuntimeError("simulated crash before pointer switch")
            cur.execute(
                "INSERT INTO branches(name,head_seq) VALUES(?,?) "
                "ON CONFLICT(name) DO UPDATE SET head_seq=excluded.head_seq",
                (branch, seq),
            )
            cur.execute("DELETE FROM work_edits")
            cur.execute("COMMIT")
        except Exception:
            if cur.in_transaction:
                cur.execute("ROLLBACK")
            raise
        return seq

    # ---------- read / replay ----------
    def commit_row(self, seq: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM commits WHERE seq=?",
                                 (seq,)).fetchone()

    def timeline(self, branch: str = "main") -> list[dict]:
        seq = self.head_seq(branch)
        out = []
        while seq is not None:
            row = self.commit_row(seq)
            out.append({
                "seq": row["seq"], "id": row["id"], "branch": row["branch"],
                "parent_seq": row["parent_seq"], "author": row["author"],
                "label": row["label"], "created_s": row["created_s"],
                "events": json.loads(row["event_log"]),
            })
            seq = row["parent_seq"]
        return list(reversed(out))

    def sources_at(self, seq: int) -> dict[str, dict]:
        out = {}
        for row in self.conn.execute(
            "SELECT * FROM source_versions WHERE introduced_seq<=? "
            "AND (superseded_seq IS NULL OR superseded_seq>?)",
            (seq, seq),
        ):
            out[row["record_id"]] = {
                "version": row["version"],
                "record_kind": row["record_kind"],
                "payload": json.loads(row["payload"]),
            }
        return out

    def slots_at(self, seq: int) -> dict[str, dict]:
        out = {}
        for row in self.conn.execute(
            "SELECT * FROM commit_slots WHERE seq=?", (seq,)
        ):
            spec = {"status": row["status"], "fingerprint": row["fingerprint"]}
            if row["fingerprint"]:
                d = self.conn.execute(
                    "SELECT * FROM derivations WHERE fingerprint=?",
                    (row["fingerprint"],),
                ).fetchone()
                if d:
                    spec["derivation"] = {
                        "slot": d["slot"], "status": d["status"],
                        "payload": json.loads(d["payload"]),
                        "rule_version": d["rule_version"],
                        "transform_order": d["transform_order"],
                        "fingerprint": d["fingerprint"],
                    }
            out[row["slot"]] = spec
        return out

    def decisions_at(self, seq: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE seq<=? ORDER BY seq", (seq,)
        ).fetchall()
        return [{
            "id": r["id"], "seq": r["seq"], "subject": r["subject"],
            "verdict": r["verdict"],
            "rejected_candidates": json.loads(r["rejected_candidates"]),
            "note": r["note"],
        } for r in rows]

    def derivation(self, fingerprint: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM derivations WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if not row:
            return None
        return {
            "slot": row["slot"], "status": row["status"],
            "payload": json.loads(row["payload"]),
            "rule_version": row["rule_version"],
            "transform_order": row["transform_order"],
            "fingerprint": row["fingerprint"],
        }

    def active_state(self, branch: str = "main") -> dict:
        seq = self.head_seq(branch)
        if seq is None:
            return {"seq": None, "sources": {}, "slots": {}, "events": [],
                    "decisions": []}
        return {
            "seq": seq,
            "sources": self.sources_at(seq),
            "slots": self.slots_at(seq),
            "events": self.timeline(branch),
            "decisions": self.decisions_at(seq),
        }

    # ---------- export / import ----------
    def export_bundle(self) -> dict:
        def rows(table: str) -> list[dict]:
            return [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} ORDER BY seq")]

        bundle = {
            "format": "adcp-evidence-bundle",
            "format_version": 1,
            "schema_version": SCHEMA_VERSION,
            "branches": [dict(r) for r in self.conn.execute(
                "SELECT * FROM branches ORDER BY name")],
            "commits": rows("commits"),
            "source_versions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM source_versions ORDER BY record_id,version")],
            "derivations": [dict(r) for r in self.conn.execute(
                "SELECT * FROM derivations ORDER BY fingerprint")],
            "commit_slots": [dict(r) for r in self.conn.execute(
                "SELECT * FROM commit_slots ORDER BY seq,slot")],
            "decisions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM decisions ORDER BY seq")],
        }
        body = canonical_json({k: v for k, v in bundle.items()})
        bundle["checksum"] = __import__("hashlib").sha256(
            body.encode("utf-8")).hexdigest()
        return bundle

    @staticmethod
    def import_bundle(bundle: dict, path: str = ":memory:") -> "Store":
        import hashlib

        checksum = bundle.pop("checksum")
        body = canonical_json(bundle)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != checksum:
            raise ValueError("bundle checksum mismatch")
        if bundle["schema_version"] != SCHEMA_VERSION:
            raise ValueError("schema version mismatch on import")
        store = Store(path)
        conn = store.conn
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO commits(seq,id,branch,parent_seq,author,label,"
            "created_s,event_log,changes) VALUES(?,?,?,?,?,?,?,?,?)",
            [tuple(r[k] for k in ("seq", "id", "branch", "parent_seq",
             "author", "label", "created_s", "event_log", "changes"))
             for r in bundle["commits"]],
        )
        conn.executemany(
            "INSERT INTO source_versions(record_id,version,record_kind,"
            "introduced_seq,superseded_seq,payload) VALUES(?,?,?,?,?,?)",
            [(r["record_id"], r["version"], r["record_kind"],
              r["introduced_seq"], r["superseded_seq"], r["payload"])
             for r in bundle["source_versions"]],
        )
        conn.executemany(
            "INSERT INTO derivations(fingerprint,slot,status,payload,"
            "rule_version,transform_order,created_seq) "
            "VALUES(?,?,?,?,?,?,?)",
            [(r["fingerprint"], r["slot"], r["status"], r["payload"],
              r["rule_version"], r["transform_order"], r["created_seq"])
             for r in bundle["derivations"]],
        )
        conn.executemany(
            "INSERT INTO commit_slots(seq,slot,status,fingerprint) "
            "VALUES(?,?,?,?)",
            [(r["seq"], r["slot"], r["status"], r["fingerprint"])
             for r in bundle["commit_slots"]],
        )
        conn.executemany(
            "INSERT INTO decisions(id,seq,subject,verdict,"
            "rejected_candidates,note) VALUES(?,?,?,?,?,?)",
            [(r["id"], r["seq"], r["subject"], r["verdict"],
              r["rejected_candidates"], r["note"])
             for r in bundle["decisions"]],
        )
        conn.executemany(
            "INSERT INTO branches(name,head_seq) VALUES(?,?)",
            [(r["name"], r["head_seq"]) for r in bundle["branches"]],
        )
        conn.execute("COMMIT")
        return store
