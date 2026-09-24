"""The evidence store.

Atomicity contract
------------------

Every derivation writes a *new* version. The ``current`` pointer in
``meta`` moves to the new version_id only inside the same SQLite transaction
that inserts all of the new version's nodes. A process killed mid-write
therefore leaves either the old pointer/old version or the complete new one;
a partial version (``status='partial'``) is never pointed at and is ignored
by readers and cleaned on open.

Concurrency contract
--------------------

Two users editing from the same base version both record their own derived
version; the second human *decision* on the same scope carries the same
``based_on_version`` and is stored as a conflict with status
``conflict`` instead of silently overwriting. Resolving is an explicit new
event.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from ..fingerprints import stable_id
from .schema import CURRENT_SCHEMA, initialize
from .migration import needs_migration, migrate_legacy_v0


CONCURRENT_BASE = "__concurrent_base__"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class Store:
    def __init__(self, path: str = ":memory:", *, auto_migrate: bool = True):
        self.path = path
        if path != ":memory:" and os.path.exists(path):
            if needs_migration(path):
                if not auto_migrate:
                    raise RuntimeError("database needs migration")
                migrate_legacy_v0(path)
        self.conn = connect(path)
        self._tx_conn = None
        self._cleanup_partial()
        self._ensure_schema()

    # ------------------------------------------------------------------ basic

    def _ensure_schema(self) -> None:
        initialize(self.conn)
        self.conn.commit()

    def _cleanup_partial(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='versions'"
        )
        if not cur.fetchone():
            return
        with self.conn:
            self.conn.execute(
                "DELETE FROM derived_nodes WHERE version_id IN "
                "(SELECT version_id FROM versions WHERE status='partial')"
            )
            self.conn.execute(
                "DELETE FROM edges WHERE version_id IN "
                "(SELECT version_id FROM versions WHERE status='partial')"
            )
            self.conn.execute("DELETE FROM versions WHERE status='partial'")

    def close(self) -> None:
        self.conn.close()

    def meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def current_version(self) -> str | None:
        value = self.meta("current_version")
        return value or None

    # ----------------------------------------------------------------- events

    def add_event(
        self,
        conn,
        *,
        version_id: str,
        code: str,
        payload: dict[str, Any],
        actor: str | None = None,
        label: str | None = None,
    ) -> tuple[int, str]:
        event_id = stable_id("evt", version_id, code,
                             payload.get("seq_marker"), payload)
        created = _now()
        cur = conn.execute(
            "INSERT INTO events(event_id, version_id, code, actor, label, "
            "payload, created_at) VALUES(?,?,?,?,?,?,?)",
            (event_id, version_id, code, actor, label,
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             created),
        )
        return cur.lastrowid, event_id

    def list_events(self, version_id: str | None = None) -> list[dict]:
        if version_id:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE version_id=? ORDER BY seq",
                (version_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY seq"
            ).fetchall()
        return [self._event_row(r) for r in rows]

    @staticmethod
    def _event_row(r: sqlite3.Row) -> dict:
        return {
            "seq": r["seq"],
            "event_id": r["event_id"],
            "version_id": r["version_id"],
            "code": r["code"],
            "actor": r["actor"],
            "label": r["label"],
            "payload": json.loads(r["payload"]),
            "created_at": r["created_at"],
        }

    def get_event(self, seq: int) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM events WHERE seq=?", (seq,)
        ).fetchone()
        return self._event_row(r) if r else None

    # ---------------------------------------------------------------- sources

    def insert_source(self, conn, *, created_event: int, **cols) -> None:
        cols.setdefault("ping_id", None)
        cols.setdefault("cell_no", None)
        cols.setdefault("beam", None)
        cols.setdefault("stream_key", None)
        cols.setdefault("unit", None)
        cols.setdefault("datum", None)
        cols.setdefault("numeric", None)
        cols.setdefault("diagnostic", None)
        cols.setdefault("superseded_by_source", None)
        conn.execute(
            "INSERT OR REPLACE INTO sources(source_id, kind, ping_id, cell_no,"
            " beam, stream_key, raw, unit, datum, numeric, diagnostic, "
            "created_event, superseded_by_source) VALUES("
            "?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cols["source_id"], cols["kind"], cols["ping_id"],
             cols["cell_no"], cols["beam"], cols["stream_key"],
             str(cols["raw"]), cols["unit"], cols["datum"], cols["numeric"],
             cols["diagnostic"], created_event,
             cols["superseded_by_source"]),
        )

    def list_sources(self, *, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM sources"
        args: tuple = ()
        if kind:
            sql += " WHERE kind=?"
            args = (kind,)
        sql += " ORDER BY source_id"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def get_source(self, source_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
        return dict(r) if r else None

    def upsert_config(self, conn, config_type: str, version_id: str,
                      payload: dict, *, introduced_event: int,
                      retired_event: int | None = None) -> None:
        conn.execute(
            "INSERT INTO config_versions(config_type, version_id, payload, "
            "introduced_event, retired_event) VALUES(?,?,?,?,?) "
            "ON CONFLICT(config_type, version_id) DO UPDATE SET "
            "payload=excluded.payload, retired_event=excluded.retired_event",
            (config_type, version_id,
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             introduced_event, retired_event),
        )

    def active_config(self, config_type: str,
                      at_event: int | None = None) -> tuple[str, dict] | None:
        rows = self.conn.execute(
            "SELECT * FROM config_versions WHERE config_type=? "
            "ORDER BY introduced_event", (config_type,)
        ).fetchall()
        chosen = None
        for r in rows:
            if at_event is None or r["introduced_event"] <= at_event:
                if r["retired_event"] is None or (
                    at_event is not None and at_event < r["retired_event"]
                ):
                    chosen = r
        if chosen is None:
            return None
        return chosen["version_id"], json.loads(chosen["payload"])

    # ------------------------------------------------------------- versions

    def create_version(
        self,
        conn,
        *,
        version_id: str,
        parent_id: str | None,
        kind: str,
        label: str,
        created_event: int,
        based_on_version: str | None = None,
        status: str = "partial",
    ) -> None:
        conn.execute(
            "INSERT INTO versions(version_id, parent_id, kind, label, status, "
            "created_event, based_on_version) VALUES(?,?,?,?,?,?,?)",
            (version_id, parent_id, kind, label, status, created_event,
             based_on_version),
        )

    def insert_node(self, conn, node: dict) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO derived_nodes(node_id, version_id, scope,"
            " ping_id, cell_no, kind, candidate_id, status, fingerprint, "
            "value, value_ids, inputs, evidence, diagnostics) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node["node_id"], node["version_id"], node["scope"],
             node.get("ping_id"), node.get("cell_no"), node["kind"],
             node.get("candidate_id"), node["status"], node["fingerprint"],
             json.dumps(node["value"], ensure_ascii=False),
             json.dumps(node.get("value_ids", []), ensure_ascii=False),
             json.dumps(node["inputs"], ensure_ascii=False, sort_keys=True),
             json.dumps(node.get("evidence", {}), ensure_ascii=False),
             json.dumps(node.get("diagnostics", []), ensure_ascii=False)),
        )

    def insert_edge(self, conn, version_id: str,
                    upstream: str, downstream: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO edges(version_id, upstream_node_id, "
            "downstream_node_id) VALUES(?,?,?)",
            (version_id, upstream, downstream),
        )

    def commit_version(
        self,
        conn,
        *,
        version_id: str,
        make_current: bool = True,
    ) -> None:
        """Switch a built version to committed and (optionally) current.

        Must run inside the caller's single transaction so the pointer move
        is atomic with every node insert.
        """
        conn.execute(
            "UPDATE versions SET status='committed' WHERE version_id=?",
            (version_id,),
        )
        if make_current:
            conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'",
            (version_id,),
        )

    def get_version(self, version_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM versions WHERE version_id=?", (version_id,)
        ).fetchone()
        return dict(r) if r else None

    def list_versions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions ORDER BY created_event"
        ).fetchall()]

    def nodes(self, version_id: str, *, ping_id: str | None = None,
              cell_no: int | None = None, kind: str | None = None,
              scope: str | None = None) -> list[dict]:
        sql = "SELECT * FROM derived_nodes WHERE version_id=?"
        args: list[Any] = [version_id]
        if ping_id is not None:
            sql += " AND ping_id=?"
            args.append(ping_id)
        if cell_no is not None:
            sql += " AND cell_no=?"
            args.append(cell_no)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if scope is not None:
            sql += " AND scope=?"
            args.append(scope)
        sql += " ORDER BY ping_id, cell_no, kind, candidate_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            d = dict(r)
            for key in ("value", "value_ids", "inputs", "evidence",
                        "diagnostics"):
                d[key] = json.loads(d[key])
            out.append(d)
        return out

    def get_node(self, node_id: str, version_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM derived_nodes WHERE node_id=? AND version_id=?",
            (node_id, version_id),
        ).fetchone()
        if not r:
            return None
        d = dict(r)
        for key in ("value", "value_ids", "inputs", "evidence",
                    "diagnostics"):
            d[key] = json.loads(d[key])
        return d

    def edges(self, version_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM edges WHERE version_id=?", (version_id,)
        ).fetchall()]

    # -------------------------------------------------------------- decisions

    def add_decision(self, conn, decision: dict) -> str:
        """Record a human decision, detecting same-base conflicts.

        Returns ``"ok"`` or ``"conflict"``. A conflict never blocks storage;
        the new decision is kept with status ``conflict`` and linked to the
        prior decision. Resolution is a separate explicit action.
        """
        prior = conn.execute(
            "SELECT * FROM decisions WHERE ping_id=? AND status IN "
            "('accepted','conflict') ORDER BY created_event",
            (decision["ping_id"],),
        ).fetchall()
        conflict_id = None
        status = "accepted"
        for row in prior:
            if (row["based_on_version"] == decision["based_on_version"]
                    and row["actor"] != decision["actor"]
                    and row["chosen_candidate_id"]
                    != decision["chosen_candidate_id"]):
                conflict_id = row["decision_id"]
                status = "conflict"
                break
        conn.execute(
            "INSERT INTO decisions(decision_id, version_id, based_on_version,"
            " ping_id, chosen_candidate_id, actor, status, conflicts_with, "
            "resolution, created_event) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (decision["decision_id"], decision["version_id"],
             decision["based_on_version"], decision["ping_id"],
             decision["chosen_candidate_id"], decision["actor"], status,
             conflict_id, None, decision["created_event"]),
        )
        return status

    def resolve_conflict(self, conn, *, decision_id: str, resolution: str,
                         event_seq: int) -> None:
        conn.execute(
            "UPDATE decisions SET status='resolved', resolution=?,"
            "created_event=? WHERE decision_id=?",
            (resolution, event_seq, decision_id),
        )

    def list_decisions(self, ping_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM decisions"
        args: tuple = ()
        if ping_id:
            sql += " WHERE ping_id=?"
            args = (ping_id,)
        sql += " ORDER BY created_event"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # --------------------------------------------------------- raw connection

    def begin(self) -> sqlite3.Connection:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        self._tx_conn = conn
        return conn

    def commit(self) -> None:
        self.conn.commit()
        self._tx_conn = None

    def rollback(self) -> None:
        self.conn.rollback()
        self._tx_conn = None

    @property
    def db(self) -> sqlite3.Connection:
        """Connection visible inside an open transaction."""
        return self._tx_conn or self.conn
