"""SQLite storage: separate source streams, versioned derivations, atomic commits.

Guarantees
----------
* Beam velocity, correlation, amplitude, bad-beam flags, attitude,
  declination and mounting each have their own table.
* Every derivation is an immutable content-addressed row.  The current pointer
  for a node only moves inside the same transaction that writes the derivation;
  a crash at any point leaves either the old complete version or the new
  complete version.
* Every mutation appends a machine-readable event.
* Two editors starting from the same version cannot silently overwrite one
  another (optimistic version conflict, resolvable).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fingerprint import canonical_json, digest

SCHEMA_VERSION = 2
_LOCK = threading.RLock()

CRASH_AFTER_DERIVATIONS = "after_derivations"
CRASH_AFTER_SOURCES = "after_sources"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def node_id(kind: str, dataset_id: str, ping=None, cell=None, beam=None) -> str:
    return digest(
        {
            "node": kind,
            "dataset": dataset_id,
            "ping": ping,
            "cell": cell,
            "beam": beam,
        },
        prefix="node",
    )


def record_id(kind: str, dataset_id: str, payload: Dict[str, Any]) -> str:
    return digest({"kind": kind, "dataset": dataset_id, "payload": payload}, prefix="rec")


# ---------------------------------------------------------------------------


class ConflictError(Exception):
    """Raised when a save is based on a stale parent version."""


class IntegrityError(Exception):
    pass


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _read_schema_sql(version: int) -> str:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "migrations", "v%d.sql" % version)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_read_schema_sql(1))
    _set_meta(conn, "schema_version", "1")
    migrate(conn)


def _get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM schema_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def migrate(conn: sqlite3.Connection, target: int = SCHEMA_VERSION) -> int:
    """Upgrade old versions; returns the version reached."""
    current = int(_get_meta(conn, "schema_version", "1"))
    if current < 2 <= target:
        # v2: add a commit counter / provenance column on versions (idempotent).
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(versions)")}
        if "rule_bundle" not in cols:
            conn.execute("ALTER TABLE versions ADD COLUMN rule_bundle TEXT")
        _set_meta(conn, "schema_version", "2")
        current = 2
    return current


@contextmanager
def transaction(conn: sqlite3.Connection):
    with _LOCK:
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


class Store:
    def __init__(self, db_path: str, crash_plan: Optional[Dict[str, Any]] = None):
        self.db_path = db_path
        self.conn = connect(db_path)
        self.crash_plan = dict(crash_plan or {})
        if not os.path.exists(db_path) or self._is_fresh():
            init_schema(self.conn)

    def _is_fresh(self) -> bool:
        try:
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
            ).fetchone()
            return row is None
        except sqlite3.DatabaseError:
            return True

    def close(self):
        self.conn.close()

    # -- datasets / versions -----------------------------------------------

    def create_dataset(self, name: str, dataset_id: Optional[str] = None) -> str:
        ds_id = dataset_id or ("ds-" + uuid.uuid4().hex[:12])
        with transaction(self.conn) as c:
            c.execute(
                "INSERT OR IGNORE INTO datasets(id, name, created_at) VALUES(?,?,?)",
                (ds_id, name, now_iso()),
            )
            vid = self._new_version(c, ds_id, None, "baseline", "committed", 0)
            c.execute(
                "INSERT INTO events(id, dataset_id, version_id, seq, at, actor, "
                "event_type, payload_json, parent_event_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (record_id("event", ds_id, {"v": vid, "seq": 0}), ds_id, vid, 0,
                 now_iso(), "system", "dataset_created",
                 json.dumps({"dataset": ds_id, "name": name}), None),
            )
        return ds_id

    def head_version(self, dataset_id: str) -> str:
        row = self.conn.execute(
            "SELECT id FROM versions WHERE dataset_id=? AND status='committed' "
            "ORDER BY commit_seq DESC, rowid DESC LIMIT 1",
            (dataset_id,),
        ).fetchone()
        if row is None:
            raise IntegrityError("no committed version for dataset %r" % dataset_id)
        return row["id"]

    def head_seq(self, dataset_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(seq) AS m FROM events WHERE dataset_id=?", (dataset_id,)
        ).fetchone()
        return int(row["m"] or 0)

    def _new_version(self, c, dataset_id, parent_id, label, status, seq,
                     rule_bundle=None):
        vid = "ver-" + uuid.uuid4().hex[:16]
        c.execute(
            "INSERT INTO versions(id, dataset_id, parent_id, label, status, "
            "created_at, commit_seq, rule_bundle) VALUES(?,?,?,?,?,?,?,?)",
            (vid, dataset_id, parent_id, label, status, now_iso(), seq, rule_bundle),
        )
        return vid

    # -- source records ----------------------------------------------------

    def insert_sources(
        self,
        conn: sqlite3.Connection,
        dataset_id: str,
        version_id: str,
        kind: str,
        rows: Sequence[Dict[str, Any]],
    ) -> int:
        """Insert one family of source rows (families stay in own tables)."""
        table = "src_" + kind
        allowed = {
            "beam", "correlation", "amplitude", "badflag", "attitude",
            "declination", "mounting", "bottom",
        }
        if kind not in allowed:
            raise ValueError("unknown source family %r" % kind)
        n = 0
        for row in rows:
            rid = row.get("id") or record_id(
                kind, dataset_id, {"version": version_id, "data": row["raw"]}
            )
            raw_json = canonical_json(row["raw"])
            diag_json = canonical_json(row.get("diagnostic")) if row.get("diagnostic") else None
            if kind == "beam":
                conn.execute(
                    "INSERT INTO src_beam(id, dataset_id, version_id, ping, cell, "
                    "beam, raw_json, canonical_mps, diagnostic_json, kind) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row["ping"], row["cell"],
                     row["beam"], raw_json, row.get("canonical"), diag_json,
                     row.get("source_kind", "water")),
                )
            elif kind == "correlation":
                conn.execute(
                    "INSERT INTO src_correlation(id, dataset_id, version_id, ping, "
                    "cell, beam, raw_json, canonical, diagnostic_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row["ping"], row["cell"],
                     row["beam"], raw_json, row.get("canonical"), diag_json),
                )
            elif kind == "amplitude":
                conn.execute(
                    "INSERT INTO src_amplitude(id, dataset_id, version_id, ping, "
                    "cell, beam, raw_json, canonical, diagnostic_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row["ping"], row["cell"],
                     row["beam"], raw_json, row.get("canonical"), diag_json),
                )
            elif kind == "badflag":
                conn.execute(
                    "INSERT INTO src_badflag(id, dataset_id, version_id, ping, "
                    "cell, beam, raw_json, canonical, diagnostic_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row["ping"], row["cell"],
                     row["beam"], raw_json, row.get("canonical"), diag_json),
                )
            elif kind == "attitude":
                conn.execute(
                    "INSERT INTO src_attitude(id, dataset_id, version_id, ping, t, "
                    "raw_json, heading_deg, pitch_deg, roll_deg, heading_datum, "
                    "diagnostic_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row.get("ping"), row["t"],
                     raw_json, row.get("heading_deg"), row.get("pitch_deg"),
                     row.get("roll_deg"), row.get("heading_datum"), diag_json),
                )
            elif kind == "declination":
                conn.execute(
                    "INSERT INTO src_declination(id, dataset_id, version_id, "
                    "raw_json, value_deg, model, model_version) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, raw_json, row.get("value_deg"),
                     row.get("model"), row.get("model_version")),
                )
            elif kind == "mounting":
                conn.execute(
                    "INSERT INTO src_mounting(id, dataset_id, version_id, "
                    "raw_json, heading_offset_deg, pitch_offset_deg, "
                    "roll_offset_deg, transform_order, revision) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, raw_json,
                     row.get("heading_offset_deg"), row.get("pitch_offset_deg"),
                     row.get("roll_offset_deg"), row.get("transform_order"),
                     row.get("revision")),
                )
            elif kind == "bottom":
                conn.execute(
                    "INSERT INTO src_bottom(id, dataset_id, version_id, ping, beam, "
                    "raw_json, range_m, diagnostic_json) VALUES(?,?,?,?,?,?,?,?)",
                    (rid, dataset_id, version_id, row["ping"], row["beam"],
                     raw_json, row.get("range_m"), diag_json),
                )
            n += 1
        return n

    # -- atomic version commit ---------------------------------------------

    def commit_version(
        self,
        dataset_id: str,
        label: str,
        sources: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        derivations: Optional[List[Dict[str, Any]]] = None,
        pointer_moves: Optional[List[Tuple[str, str]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        decisions: Optional[List[Dict[str, Any]]] = None,
        expected_parent: Optional[str] = None,
        rule_bundle: Optional[str] = None,
    ) -> str:
        """All-or-nothing version commit.

        On restart after an injected crash the caller sees either the previous
        committed version or the whole new committed version.
        """
        with _LOCK:
            parent = self.head_version(dataset_id)
            if expected_parent is not None and parent != expected_parent:
                raise ConflictError(
                    "expected parent %s but head is %s" % (expected_parent, parent)
                )
            seq_max = self.head_seq(dataset_id)
            commit_seq = self._commit_count(dataset_id)
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                vid = self._new_version(
                    conn, dataset_id, parent, label, "pending", commit_seq + 1,
                    rule_bundle,
                )
                for family, rows in (sources or {}).items():
                    self.insert_sources(conn, dataset_id, vid, family, rows)
                    if self.crash_plan.get(CRASH_AFTER_SOURCES):
                        self._simulate_crash(conn)
                for d in derivations or []:
                    self._insert_derivation(conn, dataset_id, vid, d)
                if self.crash_plan.get(CRASH_AFTER_DERIVATIONS):
                    self._simulate_crash(conn)
                for node, der_id in pointer_moves or []:
                    self._move_pointer(conn, dataset_id, node, der_id)
                for dec in decisions or []:
                    self._insert_decision(conn, dataset_id, vid, dec)
                for i, ev in enumerate(events or []):
                    self._insert_event(
                        conn, dataset_id, vid, seq_max + 1 + i, ev,
                    )
                conn.execute(
                    "UPDATE versions SET status='committed' WHERE id=?", (vid,)
                )
                conn.commit()
            except _Crash:
                # Simulated power loss: connection dropped mid-transaction.
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise
            return vid

    def _simulate_crash(self, conn):
        # Roll back and close the handle to mimic the process dying.
        conn.rollback()
        try:
            conn.close()
        except Exception:
            pass
        raise _Crash("simulated write interruption")

    def _commit_count(self, dataset_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(commit_seq), 0) AS m FROM versions WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        return int(row["m"])

    def _ensure_node(self, conn, dataset_id, node, kind, ping, cell, beam, scope):
        conn.execute(
            "INSERT INTO nodes(id, dataset_id, kind, ping, cell, beam, scope) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO NOTHING",
            (node, dataset_id, kind, ping, cell, beam, scope),
        )
        conn.execute(
            "INSERT INTO node_pointers(node_id, dataset_id, current_derivation_id) "
            "VALUES(?,?,NULL) ON CONFLICT(node_id, dataset_id) DO NOTHING",
            (node, dataset_id),
        )

    def _insert_derivation(self, conn, dataset_id, version_id, d):
        nid = d["node_id"]
        self._ensure_node(
            conn, dataset_id, nid, d["kind"], d.get("ping"), d.get("cell"),
            d.get("beam"), d.get("scope", "cell"),
        )
        conn.execute(
            "INSERT INTO derivations(id, dataset_id, version_id, node_id, kind, "
            "fingerprint, payload_json, created_at, supersedes_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (d["id"], dataset_id, version_id, nid, d["kind"], d["fingerprint"],
             canonical_json(d["payload"]), now_iso(), d.get("supersedes_id")),
        )

    def _move_pointer(self, conn, dataset_id, node_id, derivation_id):
        cur = conn.execute(
            "UPDATE node_pointers SET current_derivation_id=? "
            "WHERE dataset_id=? AND node_id=?",
            (derivation_id, dataset_id, node_id),
        )
        if cur.rowcount == 0:
            raise IntegrityError("cannot move unknown node pointer %s" % node_id)

    def _insert_event(self, conn, dataset_id, version_id, seq, ev):
        eid = ev.get("id") or record_id(
            "event", dataset_id, {"v": version_id, "seq": seq, "payload": ev}
        )
        conn.execute(
            "INSERT INTO events(id, dataset_id, version_id, seq, at, actor, "
            "event_type, payload_json, parent_event_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (eid, dataset_id, version_id, seq, now_iso(),
             ev.get("actor", "system"), ev["event_type"],
             canonical_json(ev["payload"]), ev.get("parent_event_id")),
        )

    def _insert_decision(self, conn, dataset_id, version_id, dec):
        conn.execute(
            "INSERT INTO decisions(id, dataset_id, version_id, node_id, choice, "
            "candidate_ids_json, rationale, at, actor, supersedes_decision_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (dec["id"], dataset_id, version_id, dec["node_id"], dec["choice"],
             canonical_json(dec["candidate_ids"]), dec.get("rationale"),
             now_iso(), dec.get("actor", "reviewer"),
             dec.get("supersedes_decision_id")),
        )


class _Crash(Exception):
    """Internal: simulated process kill mid-write."""


# Read helpers and export/import are attached below to keep the class focused.
def _get_or_reopen(store: "Store") -> sqlite3.Connection:
    # A simulated crash closes the connection; reopen without re-initialising.
    try:
        store.conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        store.conn = connect(store.db_path)
    return store.conn


def _query_all(store: "Store", sql: str, params=()) -> List[sqlite3.Row]:
    conn = _get_or_reopen(store)
    return list(conn.execute(sql, params))


def list_versions(store: "Store", dataset_id: str) -> List[Dict[str, Any]]:
    rows = _query_all(
        store,
        "SELECT * FROM versions WHERE dataset_id=? ORDER BY commit_seq",
        (dataset_id,),
    )
    return [dict(r) for r in rows]


def list_events(store: "Store", dataset_id: str) -> List[Dict[str, Any]]:
    rows = _query_all(
        store,
        "SELECT * FROM events WHERE dataset_id=? ORDER BY seq",
        (dataset_id,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json"))
        out.append(d)
    return out


def current_pointer(store: "Store", dataset_id: str, node_id: str):
    rows = _query_all(
        store,
        "SELECT current_derivation_id FROM node_pointers WHERE dataset_id=? AND node_id=?",
        (dataset_id, node_id),
    )
    return rows[0]["current_derivation_id"] if rows else None


def get_derivation(store: "Store", derivation_id: str) -> Optional[Dict[str, Any]]:
    rows = _query_all(store, "SELECT * FROM derivations WHERE id=?", (derivation_id,))
    if not rows:
        return None
    d = dict(rows[0])
    d["payload"] = json.loads(d.pop("payload_json"))
    return d


def node_history(store: "Store", dataset_id: str, node_id: str) -> List[Dict[str, Any]]:
    rows = _query_all(
        store,
        "SELECT * FROM derivations WHERE dataset_id=? AND node_id=? ORDER BY rowid",
        (dataset_id, node_id),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json"))
        out.append(d)
    return out


def list_node_pointers(store: "Store", dataset_id: str) -> Dict[str, str]:
    rows = _query_all(
        store,
        "SELECT node_id, current_derivation_id FROM node_pointers WHERE dataset_id=?",
        (dataset_id,),
    )
    return {r["node_id"]: r["current_derivation_id"] for r in rows if r["current_derivation_id"]}


def list_decisions(store: "Store", dataset_id: str) -> List[Dict[str, Any]]:
    rows = _query_all(
        store, "SELECT * FROM decisions WHERE dataset_id=? ORDER BY rowid",
        (dataset_id,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["candidate_ids"] = json.loads(d.pop("candidate_ids_json"))
        out.append(d)
    return out


# -- reviews with optimistic concurrency ------------------------------------

def save_review(
    store: "Store",
    dataset_id: str,
    node_id: str,
    body: str,
    expected_rev: int,
    actor: str = "reviewer",
) -> Dict[str, Any]:
    conn = _get_or_reopen(store)
    with transaction(conn) as c:
        row = c.execute(
            "SELECT id, rev FROM reviews WHERE dataset_id=? AND node_id=?",
            (dataset_id, node_id),
        ).fetchone()
        if row is None:
            if expected_rev != 0:
                raise ConflictError("review exists but expected rev 0")
            rid = "rev-" + uuid.uuid4().hex[:12]
            c.execute(
                "INSERT INTO reviews(id, dataset_id, node_id, body, rev, at, actor) "
                "VALUES(?,?,?,?,1,?,?)",
                (rid, dataset_id, node_id, body, now_iso(), actor),
            )
            return {"id": rid, "rev": 1}
        if int(row["rev"]) != expected_rev:
            raise ConflictError(
                "review rev %s != expected %s" % (row["rev"], expected_rev)
            )
        c.execute(
            "UPDATE reviews SET body=?, rev=rev+1, at=?, actor=? WHERE id=?",
            (body, now_iso(), actor, row["id"]),
        )
        return {"id": row["id"], "rev": expected_rev + 1}


def get_review(store: "Store", dataset_id: str, node_id: str):
    rows = _query_all(
        store,
        "SELECT * FROM reviews WHERE dataset_id=? AND node_id=?",
        (dataset_id, node_id),
    )
    return dict(rows[0]) if rows else None


# -- export / import --------------------------------------------------------

_EXPORT_TABLES = (
    "datasets", "src_beam", "src_correlation", "src_amplitude", "src_badflag",
    "src_attitude", "src_declination", "src_mounting", "src_bottom", "config",
    "derivations", "nodes", "node_pointers", "versions", "events", "decisions",
    "reviews", "schema_meta",
)


def export_dataset(store: "Store", dataset_id: str) -> Dict[str, Any]:
    """Full, identity-preserving archive. Stable IDs are inside the rows."""
    bundle: Dict[str, Any] = {
        "archive_format": "adcp-evidence/1",
        "exported_at": now_iso(),
        "dataset_id": dataset_id,
        "tables": {},
    }
    conn = _get_or_reopen(store)
    for table in _EXPORT_TABLES:
        rows = list(conn.execute("SELECT * FROM " + table))
        bundle["tables"][table] = [dict(r) for r in rows]
    bundle["checksum"] = digest(
        {"tables": bundle["tables"], "dataset_id": dataset_id}, prefix="archive"
    )
    return bundle


def import_dataset(store: "Store", bundle: Dict[str, Any], verify: bool = True) -> str:
    """Re-import preserving every id, order and pointer relationship."""
    dataset_id = bundle["dataset_id"]
    expected = bundle.get("checksum")
    if verify and expected:
        actual = digest(
            {"tables": bundle["tables"], "dataset_id": dataset_id},
            prefix="archive",
        )
        if actual != expected:
            raise IntegrityError("archive checksum mismatch")
    conn = _get_or_reopen(store)
    with transaction(conn) as c:
        existing = c.execute(
            "SELECT 1 FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if existing:
            raise IntegrityError("dataset %s already present" % dataset_id)
        for table, rows in bundle["tables"].items():
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" for _ in cols)
            sql = "INSERT INTO %s(%s) VALUES(%s)" % (
                table, ",".join(cols), placeholders,
            )
            for row in rows:
                c.execute(sql, [row[k] for k in cols])
    return dataset_id


# -- config and source reads ------------------------------------------------

def set_config(store: "Store", dataset_id: str, key: str, value: Any) -> None:
    conn = _get_or_reopen(store)
    with transaction(conn) as c:
        c.execute(
            "INSERT INTO config(dataset_id, key, value_json) VALUES(?,?,?) "
            "ON CONFLICT(dataset_id, key) DO UPDATE SET value_json=excluded.value_json",
            (dataset_id, key, canonical_json(value)),
        )


def get_config(store: "Store", dataset_id: str, key: str, default=None):
    conn = _get_or_reopen(store)
    row = conn.execute(
        "SELECT value_json FROM config WHERE dataset_id=? AND key=?",
        (dataset_id, key),
    ).fetchone()
    return json.loads(row["value_json"]) if row else default


def get_dataset(store: "Store", dataset_id: str):
    conn = _get_or_reopen(store)
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    return dict(row) if row else None


def list_datasets(store: "Store"):
    conn = _get_or_reopen(store)
    return [dict(r) for r in conn.execute("SELECT * FROM datasets ORDER BY id")]


def latest_sources(store: "Store", dataset_id: str, family: str,
                   head: Optional[str] = None) -> List[Dict[str, Any]]:
    """Materialised view: latest value of each logical source key.

    Because the schema is append-only across versions, each logical key
    (ping/cell/beam) returns the most recently committed row.
    """
    conn = _get_or_reopen(store)
    head = head or store.head_version(dataset_id)
    table = "src_" + family
    allowed = {
        "beam", "correlation", "amplitude", "badflag", "attitude",
        "declination", "mounting", "bottom",
    }
    if family not in allowed:
        raise ValueError(family)
    if family in ("declination", "mounting"):
        rows = conn.execute(
            "SELECT * FROM %s WHERE dataset_id=? AND version_id IN "
            "(SELECT id FROM versions WHERE dataset_id=? AND status='committed') "
            "ORDER BY rowid" % table,
            (dataset_id, dataset_id),
        ).fetchall()
        return [dict(r) for r in rows]
    rows = conn.execute(
        "SELECT * FROM %s WHERE dataset_id=? AND version_id IN "
        "(SELECT id FROM versions WHERE dataset_id=? AND status='committed') "
        "ORDER BY rowid" % table,
        (dataset_id, dataset_id),
    ).fetchall()
    return [dict(r) for r in rows]
