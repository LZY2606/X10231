"""SQLite storage: separated sources, atomic version switching, events.

Durability model
----------------
* Source tables are append-only key/value tables with a revision id per key;
  deletes are tombstones (so history is replayable).
* A recomputation writes all derived rows inside ONE transaction; the branch's
  current calc pointer moves only after COMMIT. A crash mid-write therefore
  leaves readers on the previous complete calc (the partial transaction is
  rolled back by SQLite on recovery).
* Events are append-only; walking events from a branch root reproduces any
  past state (time travel).

Concurrency
-----------
* Branch updates require the client's known base event id. Concurrent edits
  from the same branch head raise a 409-style conflict carrying both sides;
  the user resolves by choosing a branch, merging, or re-basing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

SCHEMA_VERSION = 2

_SOURCE_KINDS = (
    "BeamVelocity",
    "Correlation",
    "Amplitude",
    "BadBeamFlag",
    "AttitudeRecord",
    "Declination",
    "MountAngles",
    "BottomTrack",
    "GeometryConfig",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_rows (
  rev TEXT PRIMARY KEY,
  key_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  event_id TEXT,
  op TEXT NOT NULL CHECK (op IN ('put','delete')),
  data TEXT,
  created_s REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_key ON source_rows(key_id, created_s);

CREATE TABLE IF NOT EXISTS branches (
  name TEXT PRIMARY KEY,
  root_event TEXT,
  head_event TEXT,
  updated_s REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  branch TEXT NOT NULL,
  parent TEXT,
  seq INTEGER NOT NULL,
  type TEXT NOT NULL,
  data TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_s REAL NOT NULL,
  base_head TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch, seq);

CREATE TABLE IF NOT EXISTS calcs (
  id TEXT PRIMARY KEY,
  branch TEXT NOT NULL,
  event_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('complete','broken')),
  rules_fingerprint TEXT NOT NULL,
  created_s REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS calc_nodes (
  calc_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  target TEXT NOT NULL,
  status TEXT NOT NULL,
  payload TEXT,
  reasons TEXT,
  inputs_fingerprint TEXT,
  PRIMARY KEY (calc_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_target ON calc_nodes(calc_id, target);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    # -- schema ------------------------------------------------------------
    def _init_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
        elif version < SCHEMA_VERSION:
            self._migrate(version)
        self._recover_pending()

    def _migrate(self, version: int) -> None:
        """Upgrade an old on-disk database to the current schema.

        Migration is deliberately conservative: v1 keeps source tables per
        kind with stable ids; we copy each row into the append-only
        ``source_rows`` table preserving its stable identity, then add the
        calc/event/branch tables. See fixtures/migrations/v1.sql.
        """
        if version == 1:
            self.conn.executescript(SCHEMA)
            if self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v1_records'"
            ).fetchone():
                for row in self.conn.execute(
                    "SELECT id, kind, data FROM v1_records"
                ):
                    rev = new_id("rev")
                    self.conn.execute(
                        "INSERT INTO source_rows(rev,key_id,kind,event_id,op,data,created_s)"
                        " VALUES(?,?,?,NULL,'put',?,?)",
                        (rev, row["id"], row["kind"], row["data"], time.time()),
                    )
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
        else:
            raise RuntimeError(f"cannot migrate schema version {version}")

    def _recover_pending(self) -> None:
        """No-op placeholder documenting recovery: uncommitted write
        transactions are automatically rolled back by SQLite at connection
        open; only committed (complete) calcs are ever referenced by events.
        Broken calcs are retained for forensics but never become a pointer."""
        pass

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def close(self):
        self.conn.close()

    # -- branches ----------------------------------------------------------
    def ensure_branch(self, name: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM branches WHERE name=?", (name,)
        ).fetchone()
        if row:
            return dict(row)
        now = time.time()
        self.conn.execute(
            "INSERT INTO branches(name,root_event,head_event,updated_s) VALUES(?,NULL,NULL,?)",
            (name, now),
        )
        return dict(
            self.conn.execute("SELECT * FROM branches WHERE name=?", (name,)).fetchone()
        )

    def list_branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM branches ORDER BY name")]

    def head_event(self, branch: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM branches b WHERE b.name=?", (branch,)
        ).fetchone()
        if not row or not row["head_event"]:
            return None
        return dict(
            self.conn.execute(
                "SELECT * FROM events WHERE id=?", (row["head_event"],)
            ).fetchone()
        )

    # -- events ------------------------------------------------------------
    def append_event(
        self,
        branch: str,
        etype: str,
        data: dict,
        actor: str = "system",
        base_head: Optional[str] = None,
        expect_head: Optional[str] = None,
    ) -> dict:
        """Append an event with optimistic concurrency control.

        ``expect_head``: the head event id the editor based its change on.
        A mismatch raises :class:`BranchConflict` carrying both heads so the
        conflict is resolvable (choose/continue/rebase), never silent.
        """
        self.ensure_branch(branch)
        with self.transaction():
            branch_row = self.conn.execute(
                "SELECT * FROM branches WHERE name=?", (branch,)
            ).fetchone()
            current_head = branch_row["head_event"]
            if expect_head is not None and expect_head != current_head:
                other = self.conn.execute(
                    "SELECT * FROM events WHERE id=?", (current_head,)
                ).fetchone()
                raise BranchConflict(
                    branch=branch,
                    your_base=expect_head,
                    current_head=current_head,
                    other_event=dict(other) if other else None,
                    attempted_type=etype,
                    attempted_data=data,
                    actor=actor,
                )
            seq = (
                self.conn.execute(
                    "SELECT COALESCE(MAX(seq),0)+1 AS n FROM events WHERE branch=?",
                    (branch,),
                ).fetchone()["n"]
            )
            eid = new_id("evt")
            self.conn.execute(
                "INSERT INTO events(id,branch,parent,seq,type,data,actor,created_s,base_head)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    eid, branch, current_head, seq, etype, json.dumps(data, sort_keys=True),
                    actor, time.time(), base_head,
                ),
            )
            self.conn.execute(
                "UPDATE branches SET head_event=?, updated_s=? WHERE name=?",
                (eid, time.time(), branch),
            )
            if branch_row["root_event"] is None:
                self.conn.execute(
                    "UPDATE branches SET root_event=? WHERE name=?", (eid, branch)
                )
        return dict(self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone())

    def walk_events(self, branch: str, upto_event: Optional[str] = None) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE branch=? ORDER BY seq", (branch,)
        ).fetchall()
        result = []
        for row in rows:
            result.append(dict(row))
            if upto_event and row["id"] == upto_event:
                break
        return result

    def branch_lineage_at(self, branch: str, event_id: str) -> list[dict]:
        return self.walk_events(branch, event_id)

    # -- sources -----------------------------------------------------------
    def put_source(self, kind: str, record: dict, event_id: str) -> str:
        if kind not in _SOURCE_KINDS:
            raise ValueError(f"unknown source kind {kind}")
        key_id = record["id"]
        rev = new_id("rev")
        self.conn.execute(
            "INSERT INTO source_rows(rev,key_id,kind,event_id,op,data,created_s)"
            " VALUES(?,?,?,?, 'put', ?, ?)",
            (rev, key_id, kind, event_id, json.dumps(record, sort_keys=True), time.time()),
        )
        return rev

    def delete_source(self, kind: str, key_id: str, event_id: str) -> str:
        rev = new_id("rev")
        self.conn.execute(
            "INSERT INTO source_rows(rev,key_id,kind,event_id,op,data,created_s)"
            " VALUES(?,?,?,?,'delete',NULL,?)",
            (rev, key_id, kind, event_id, time.time()),
        )
        return rev

    def sources_at(self, branch: str, upto_event: Optional[str] = None) -> dict[str, dict]:
        """Replay source-mutating events to reconstruct sources at a point.

        Returns {key_id: {kind, data}}. Tombstones remove the key.
        """
        sources: dict[str, dict] = {}
        for ev in self.walk_events(branch, upto_event):
            payload = json.loads(ev["data"])
            if ev["type"] in ("source_put", "import"):
                for item in payload.get("sources", []):
                    sources[item["data"]["id"]] = {"kind": item["kind"], "data": item["data"]}
            elif ev["type"] == "source_delete":
                sources.pop(payload["key_id"], None)
            elif ev["type"] == "case_seed":
                for item in payload.get("sources", []):
                    sources[item["data"]["id"]] = {"kind": item["kind"], "data": item["data"]}
        return sources
