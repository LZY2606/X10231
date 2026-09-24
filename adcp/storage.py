"""SQLite storage: separated sources, content-versioned derivations, events.

Durability model
-----------------
* Beam velocity, correlation, amplitude, bad-beam flag, heading/pitch/roll,
  magnetic declination and mounting angle are independent source records.
* A derivation batch computes every affected node in one SQLite transaction;
  the per-node current pointer only switches when the batch fully commits.
  A process killed mid-write therefore leaves either the old batch or the
  complete new batch visible, never a mix.
* Every batch, event and node payload is retained; history is never overwritten.
* Two editors starting from the same base batch create resolvable conflicts on
  save (see :meth:`Store.save_editing_batch`).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable

SCHEMA_VERSION = 2

SOURCE_KINDS = (
    "beam_velocity",
    "correlation",
    "amplitude",
    "bad_beam",
    "heading",
    "pitch",
    "roll",
    "declination",
    "mount",
    "bottom_track",
    "transducer",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
  uid TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at REAL NOT NULL,
  root_batch TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
  uid TEXT PRIMARY KEY,
  project_uid TEXT NOT NULL,
  parent_uid TEXT,
  base_uid TEXT,
  label TEXT NOT NULL,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,                 -- committed | editing | merged
  crash_tag TEXT
);
CREATE TABLE IF NOT EXISTS source_records (
  uid TEXT NOT NULL,
  batch_uid TEXT NOT NULL,
  project_uid TEXT NOT NULL,
  kind TEXT NOT NULL,
  ping_idx INTEGER,
  beam INTEGER,
  cell INTEGER,
  raw_repr TEXT NOT NULL,              -- original textual representation
  raw_value REAL,
  unit TEXT NOT NULL,
  ref TEXT,
  seq INTEGER NOT NULL,                -- stable ordering
  PRIMARY KEY (uid, batch_uid)
);
CREATE TABLE IF NOT EXISTS nodes (
  uid TEXT NOT NULL,
  batch_uid TEXT NOT NULL,
  project_uid TEXT NOT NULL,
  kind TEXT NOT NULL,
  ping_idx INTEGER,
  cell INTEGER,
  status TEXT NOT NULL,                -- valid | unresolved | excluded
  fingerprint TEXT NOT NULL,
  includes TEXT NOT NULL,              -- JSON of dependency uids + versions
  payload TEXT NOT NULL,
  seq INTEGER NOT NULL,
  PRIMARY KEY (uid, batch_uid)
);
CREATE TABLE IF NOT EXISTS candidates (
  uid TEXT NOT NULL,
  batch_uid TEXT NOT NULL,
  node_uid TEXT NOT NULL,
  ctype TEXT NOT NULL,                 -- four-beam | three-beam
  beam_combo TEXT NOT NULL,            -- JSON tuple of beam indices
  fingerprint TEXT NOT NULL,
  payload TEXT NOT NULL,
  accepted INTEGER NOT NULL DEFAULT 0,
  seq INTEGER NOT NULL,
  PRIMARY KEY (uid, batch_uid)
);
CREATE TABLE IF NOT EXISTS events (
  uid TEXT PRIMARY KEY,
  project_uid TEXT NOT NULL,
  batch_uid TEXT NOT NULL,
  seq INTEGER NOT NULL,
  at REAL NOT NULL,
  actor TEXT NOT NULL,
  etype TEXT NOT NULL,
  subject TEXT,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS current_nodes (
  project_uid TEXT NOT NULL,
  batch_uid TEXT NOT NULL,
  node_uid TEXT NOT NULL,
  PRIMARY KEY (project_uid, batch_uid, node_uid)
);
CREATE TABLE IF NOT EXISTS edit_sessions (
  uid TEXT PRIMARY KEY,
  project_uid TEXT NOT NULL,
  base_batch TEXT NOT NULL,
  editor TEXT NOT NULL,
  started_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_src_kind ON source_records(kind, batch_uid);
CREATE INDEX IF NOT EXISTS idx_node_batch ON nodes(batch_uid);
CREATE INDEX IF NOT EXISTS idx_cand_batch ON candidates(batch_uid);
CREATE INDEX IF NOT EXISTS idx_event_proj ON events(project_uid, seq);
"""


class ConflictError(Exception):
    """Raised when saving a batch whose base is no longer current."""

    def __init__(self, message: str, *, diverged: str, current: str):
        super().__init__(message)
        self.diverged = diverged
        self.current = current


class CrashHook:
    """Test hook: if set, raise before commit to simulate a write interruption."""

    def __init__(self, env_var: str = "ADCP_CRASH_BATCH"):
        self.env_var = env_var

    def should_crash(self, batch_uid: str) -> bool:
        want = os.environ.get(self.env_var)
        return bool(want) and (want == "1" or want == batch_uid)


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def init_db(conn: sqlite3.Connection) -> int:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        return SCHEMA_VERSION
    return int(row["value"])


def _uid(prefix: str, body: str) -> str:
    import hashlib

    return f"{prefix}_{hashlib.sha1(body.encode('utf-8')).hexdigest()[:12]}"


def _loads(text: str) -> Any:
    return json.loads(text)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        init_db(conn)
        self.crash_hook = CrashHook()

    # ------------------------------------------------------------------ utils
    def _now(self) -> float:
        return time.time()

    def get_project(self, project_uid: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM projects WHERE uid=?", (project_uid,)
        ).fetchone()

    def list_projects(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM projects ORDER BY created_at"))

    # --------------------------------------------------------------- projects
    def create_project(self, name: str, uid: str | None = None) -> str:
        puid = uid or _uid("proj", f"{name}:{self._now()}")
        root = _uid("batch", f"{puid}:root:{self._now()}")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO projects(uid, name, created_at, root_batch) "
                "VALUES(?,?,?,?)",
                (puid, name, self._now(), root),
            )
            self.conn.execute(
                "INSERT INTO batches(uid, project_uid, parent_uid, base_uid, "
                "label, created_at, status) VALUES(?,?,?,?,?,?,?)",
                (root, puid, None, None, "root", self._now(), "committed"),
            )
            self._set_current_batch(puid, root)
        return puid

    def _set_current_batch(self, project_uid: str, batch_uid: str) -> None:
        self.conn.execute(
            "DELETE FROM current_nodes WHERE project_uid=?", (project_uid,)
        )
        self.conn.execute(
            "INSERT INTO current_nodes(project_uid, batch_uid, node_uid) "
            "SELECT ?, ?, uid FROM nodes WHERE batch_uid=?",
            (project_uid, batch_uid, batch_uid),
        )

    def current_batch(self, project_uid: str) -> str:
        row = self.conn.execute(
            "SELECT batch_uid FROM current_nodes WHERE project_uid=? LIMIT 1",
            (project_uid,),
        ).fetchone()
        if row is not None:
            return row["batch_uid"]
        return self.get_project(project_uid)["root_batch"]

    def get_batch(self, batch_uid: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM batches WHERE uid=?", (batch_uid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown batch {batch_uid}")
        return row

    def chain(self, project_uid: str) -> list[sqlite3.Row]:
        cur = self.current_batch(project_uid)
        out: list[sqlite3.Row] = []
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            b = self.get_batch(cur)
            out.append(b)
            cur = b["parent_uid"]
        return list(reversed(out))

    # ---------------------------------------------------------------- sources
    def put_sources(
        self,
        batch_uid: str,
        project_uid: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        n = 0
        for rec in records:
            kind = rec["kind"]
            if kind not in SOURCE_KINDS:
                raise ValueError(f"unknown source kind {kind!r}")
            raw = rec["raw"]
            body = _dumps(
                {
                    "b": batch_uid,
                    "k": kind,
                    "p": rec.get("ping_idx"),
                    "be": rec.get("beam"),
                    "c": rec.get("cell"),
                    "r": raw,
                    "u": rec.get("unit"),
                    "rf": rec.get("ref"),
                    "s": rec["seq"],
                }
            )
            uid = rec.get("uid") or _uid("src", body)
            self.conn.execute(
                "INSERT OR REPLACE INTO source_records(uid, batch_uid, "
                "project_uid, kind, ping_idx, beam, cell, raw_repr, raw_value, "
                "unit, ref, seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    uid,
                    batch_uid,
                    project_uid,
                    kind,
                    rec.get("ping_idx"),
                    rec.get("beam"),
                    rec.get("cell"),
                    str(raw),
                    float(raw) if _numeric(raw) else None,
                    rec.get("unit"),
                    rec.get("ref"),
                    rec["seq"],
                ),
            )
            n += 1
        return n

    def batch_sources(self, batch_uid: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM source_records WHERE batch_uid=? ORDER BY seq",
                (batch_uid,),
            )
        )

    def effective_sources(self, project_uid: str, batch_uid: str) -> list[sqlite3.Row]:
        """Walk parents and return the latest version of every stable source key.

        The stable key is (kind, ping_idx, beam, cell); a child batch only
        contains changed records and inherits the rest from its ancestors.
        """
        latest: dict[tuple, sqlite3.Row] = {}
        order: list[tuple] = []
        cur = batch_uid
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            for row in self.batch_sources(cur):
                key = (row["kind"], row["ping_idx"], row["beam"], row["cell"])
                if key not in latest:
                    latest[key] = row
                    order.append(key)
            b = self.get_batch(cur)
            cur = b["parent_uid"]
        return [latest[k] for k in order]

    # -------------------------------------------------------------- derivations
    def put_nodes(
        self,
        batch_uid: str,
        project_uid: str,
        nodes: Iterable[dict[str, Any]],
    ) -> None:
        for nd in nodes:
            self.conn.execute(
                "INSERT OR REPLACE INTO nodes(uid, batch_uid, project_uid, "
                "kind, ping_idx, cell, status, fingerprint, includes, payload, "
                "seq) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    nd["uid"],
                    batch_uid,
                    project_uid,
                    nd["kind"],
                    nd.get("ping_idx"),
                    nd.get("cell"),
                    nd["status"],
                    nd["fingerprint"],
                    _dumps(nd["includes"]),
                    _dumps(nd["payload"]),
                    nd["seq"],
                ),
            )

    def put_candidates(
        self,
        batch_uid: str,
        nodes: Iterable[dict[str, Any]],
    ) -> None:
        for cd in nodes:
            self.conn.execute(
                "INSERT OR REPLACE INTO candidates(uid, batch_uid, node_uid, "
                "ctype, beam_combo, fingerprint, payload, accepted, seq) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    cd["uid"],
                    batch_uid,
                    cd["node_uid"],
                    cd["ctype"],
                    _dumps(cd["beam_combo"]),
                    cd["fingerprint"],
                    _dumps(cd["payload"]),
                    1 if cd.get("accepted") else 0,
                    cd["seq"],
                ),
            )

    def batch_nodes(self, batch_uid: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM nodes WHERE batch_uid=? ORDER BY seq", (batch_uid,)
            )
        )

    def node(self, batch_uid: str, node_uid: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE batch_uid=? AND uid=?",
            (batch_uid, node_uid),
        ).fetchone()
        if row is None:
            raise KeyError(f"node {node_uid} not in batch {batch_uid}")
        return row

    def batch_candidates(self, batch_uid: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM candidates WHERE batch_uid=? ORDER BY seq",
                (batch_uid,),
            )
        )

    # ------------------------------------------------------------- batch write
    def commit_batch(
        self,
        project_uid: str,
        label: str,
        *,
        sources: Iterable[dict[str, Any]] = (),
        nodes: Iterable[dict[str, Any]] = (),
        candidates: Iterable[dict[str, Any]] = (),
        parent_uid: str | None = None,
        base_uid: str | None = None,
        status: str = "committed",
        batch_uid: str | None = None,
        switch_pointer: bool = True,
        crash_tag: str | None = None,
    ) -> str:
        """Insert sources + full derivation atomically; then switch pointer."""
        parent_uid = parent_uid or self.current_batch(project_uid)
        buid = batch_uid or _uid("batch", f"{project_uid}:{parent_uid}:{label}:{self._now()}")
        sources = list(sources)
        nodes = list(nodes)
        candidates = list(candidates)
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO batches(uid, project_uid, parent_uid, base_uid, "
                "label, created_at, status, crash_tag) VALUES(?,?,?,?,?,?,?,?)",
                (
                    buid,
                    project_uid,
                    parent_uid,
                    base_uid,
                    label,
                    self._now(),
                    "committed" if status == "committed" else status,
                    crash_tag,
                ),
            )
            self.put_sources(buid, project_uid, sources)
            self.put_nodes(buid, project_uid, nodes)
            self.put_candidates(buid, candidates)
            if status == "committed" and switch_pointer:
                # Crash point: rows exist, pointer still targets the old batch.
                if self.crash_hook.should_crash(buid):
                    raise _SimulatedCrash(buid)
                self._set_current_batch(project_uid, buid)
        return buid

    def mark_pointer(self, project_uid: str, batch_uid: str) -> None:
        with transaction(self.conn):
            self._set_current_batch(project_uid, batch_uid)

    # ------------------------------------------------------- editing/conflict
    def start_edit_session(self, project_uid: str, editor: str) -> tuple[str, str]:
        base = self.current_batch(project_uid)
        suid = _uid("sess", f"{project_uid}:{base}:{editor}:{self._now()}")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO edit_sessions(uid, project_uid, base_batch, "
                "editor, started_at) VALUES(?,?,?,?,?)",
                (suid, project_uid, base, editor, self._now()),
            )
        return suid, base

    def save_editing_batch(
        self,
        session_uid: str,
        *,
        label: str,
        sources: Iterable[dict[str, Any]] = (),
        nodes: Iterable[dict[str, Any]] = (),
        candidates: Iterable[dict[str, Any]] = (),
        accept_candidates: Iterable[str] = (),
    ) -> tuple[str, bool]:
        """Commit an editor's batch.

        Returns ``(batch_uid, conflict)``.  When another editor has advanced
        the current batch since this session started, the batch is still
        stored (status ``editing``) but the pointer is NOT switched; the
        caller must resolve via :meth:`merge_batch`.
        """
        sess = self.conn.execute(
            "SELECT * FROM edit_sessions WHERE uid=?", (session_uid,)
        ).fetchone()
        if sess is None:
            raise KeyError(f"unknown session {session_uid}")
        puid = sess["project_uid"]
        base = sess["base_batch"]
        cur = self.current_batch(puid)
        buid = _uid("batch", f"{puid}:{base}:{label}:{self._now()}")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO batches(uid, project_uid, parent_uid, base_uid, "
                "label, created_at, status) VALUES(?,?,?,?,?,?,?)",
                (buid, puid, base, base, label, self._now(), "editing"),
            )
            self.put_sources(buid, puid, sources)
            self.put_nodes(buid, puid, nodes)
            self.put_candidates(
                buid,
                (
                    {**cd, "accepted": cd.get("accepted") or cd["uid"] in set(accept_candidates)}
                    for cd in candidates
                ),
            )
            if cur == base:
                self._set_current_batch(puid, buid)
                self.conn.execute(
                    "UPDATE batches SET status='committed' WHERE uid=?", (buid,)
                )
                return buid, False
            # Diverged: keep the work, raise a resolvable conflict.
        return buid, True

    def merge_batch(
        self,
        project_uid: str,
        editing_batch: str,
        *,
        label: str,
        sources: Iterable[dict[str, Any]],
        nodes: Iterable[dict[str, Any]],
        candidates: Iterable[dict[str, Any]] = (),
        decision: str = "manual-resolve",
    ) -> str:
        """Attach a merged child on top of the divergent current batch."""
        cur = self.current_batch(project_uid)
        mb = self.commit_batch(
            project_uid,
            label,
            sources=sources,
            nodes=nodes,
            candidates=candidates,
            parent_uid=cur,
            base_uid=editing_batch,
            status="merged",
        )
        self.log_event(
            project_uid,
            "conflict_resolved",
            subject=editing_batch,
            payload={"editing": editing_batch, "onto": cur, "decision": decision, "merged": mb},
        )
        return mb

    # ----------------------------------------------------------------- events
    def log_event(
        self,
        project_uid: str,
        etype: str,
        *,
        actor: str = "system",
        subject: str | None = None,
        payload: dict[str, Any] | None = None,
        batch_uid: str | None = None,
    ) -> str:
        batch_uid = batch_uid or self.current_batch(project_uid)
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM events "
            "WHERE project_uid=?",
            (project_uid,),
        ).fetchone()
        seq = int(row["next"])
        euid = _uid("evt", f"{project_uid}:{seq}:{etype}:{subject or ''}:{self._now()}")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO events(uid, project_uid, batch_uid, seq, at, "
                "actor, etype, subject, payload) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    euid,
                    project_uid,
                    batch_uid,
                    seq,
                    self._now(),
                    actor,
                    etype,
                    subject,
                    _dumps(payload or {}),
                ),
            )
        return euid

    def events(
        self, project_uid: str, *, up_to_seq: int | None = None
    ) -> list[sqlite3.Row]:
        if up_to_seq is None:
            return list(
                self.conn.execute(
                    "SELECT * FROM events WHERE project_uid=? ORDER BY seq",
                    (project_uid,),
                )
            )
        return list(
            self.conn.execute(
                "SELECT * FROM events WHERE project_uid=? AND seq<=? ORDER BY seq",
                (project_uid, up_to_seq),
            )
        )

    def batch_of_event(self, event_uid: str) -> str | None:
        row = self.conn.execute(
            "SELECT batch_uid FROM events WHERE uid=?", (event_uid,)
        ).fetchone()
        return row["batch_uid"] if row else None

    # ----------------------------------------------------------- export/import
    def export_bundle(self, project_uid: str) -> dict[str, Any]:
        """Serialize every batch, source, node, candidate and event.

        Uids, seq ordering and parent/base relationships are preserved
        verbatim so a delete + import restores the evidence graph exactly.
        """
        proj = self.get_project(project_uid)
        batches = list(
            self.conn.execute(
                "SELECT * FROM batches WHERE project_uid=? ORDER BY created_at",
                (project_uid,),
            )
        )
        sources = list(
            self.conn.execute(
                "SELECT * FROM source_records WHERE project_uid=? ORDER BY seq",
                (project_uid,),
            )
        )
        nodes = list(
            self.conn.execute(
                "SELECT * FROM nodes WHERE project_uid=? ORDER BY seq",
                (project_uid,),
            )
        )
        cands = list(
            self.conn.execute(
                "SELECT * FROM candidates c WHERE EXISTS (SELECT 1 FROM nodes n "
                "WHERE n.uid=c.node_uid AND n.project_uid=?) ORDER BY c.seq",
                (project_uid,),
            )
        )
        events = list(
            self.conn.execute(
                "SELECT * FROM events WHERE project_uid=? ORDER BY seq",
                (project_uid,),
            )
        )

        def rows(rs: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
            return [dict(r) for r in rs]

        return {
            "format": "adcp-evidence-bundle",
            "format_version": 2,
            "exported_at": self._now(),
            "project": dict(proj),
            "current_batch": self.current_batch(project_uid),
            "batches": rows(batches),
            "sources": rows(sources),
            "nodes": rows(nodes),
            "candidates": rows(cands),
            "events": rows(events),
        }

    def import_bundle(self, bundle: dict[str, Any]) -> str:
        """Restore a bundle into an empty project slot; uids are preserved."""
        proj = bundle["project"]
        puid = proj["uid"]
        if self.get_project(puid) is not None:
            raise ValueError(f"project {puid} already present; refusing overwrite")
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO projects(uid, name, created_at, root_batch) "
                "VALUES(?,?,?,?)",
                (puid, proj["name"], proj["created_at"], proj["root_batch"]),
            )
            for b in bundle["batches"]:
                self.conn.execute(
                    "INSERT INTO batches(uid, project_uid, parent_uid, base_uid, "
                    "label, created_at, status, crash_tag) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        b["uid"], puid, b["parent_uid"], b["base_uid"],
                        b["label"], b["created_at"], b["status"], b.get("crash_tag"),
                    ),
                )
            for r in bundle["sources"]:
                self.conn.execute(
                    "INSERT INTO source_records(uid, batch_uid, project_uid, "
                    "kind, ping_idx, beam, cell, raw_repr, raw_value, unit, ref, "
                    "seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        r["uid"], r["batch_uid"], puid, r["kind"], r["ping_idx"],
                        r["beam"], r["cell"], r["raw_repr"], r["raw_value"],
                        r["unit"], r["ref"], r["seq"],
                    ),
                )
            for n in bundle["nodes"]:
                self.conn.execute(
                    "INSERT INTO nodes(uid, batch_uid, project_uid, kind, "
                    "ping_idx, cell, status, fingerprint, includes, payload, seq) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        n["uid"], n["batch_uid"], puid, n["kind"], n["ping_idx"],
                        n["cell"], n["status"], n["fingerprint"], n["includes"],
                        n["payload"], n["seq"],
                    ),
                )
            for c in bundle["candidates"]:
                self.conn.execute(
                    "INSERT INTO candidates(uid, batch_uid, node_uid, ctype, "
                    "beam_combo, fingerprint, payload, accepted, seq) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        c["uid"], c["batch_uid"], c["node_uid"], c["ctype"],
                        c["beam_combo"], c["fingerprint"], c["payload"],
                        c["accepted"], c["seq"],
                    ),
                )
            for e in bundle["events"]:
                self.conn.execute(
                    "INSERT INTO events(uid, project_uid, batch_uid, seq, at, "
                    "actor, etype, subject, payload) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        e["uid"], puid, e["batch_uid"], e["seq"], e["at"],
                        e["actor"], e["etype"], e["subject"], e["payload"],
                    ),
                )
            self._set_current_batch(puid, bundle["current_batch"])
        return puid


class _SimulatedCrash(RuntimeError):
    """Raised at the crash point inside commit_batch (tests only)."""

    def __init__(self, batch_uid: str):
        super().__init__(f"simulated write interruption for {batch_uid}")
        self.batch_uid = batch_uid


# --------------------------------------------------------------------- v1 -> v2
# The shipped historical schema (v1) lacked candidates.accepted and events; a
# real team database may be handed over in that shape.  migrate_v1 upgrades it
# in place without losing old batches.
V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS projects (
  uid TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL,
  root_batch TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
  uid TEXT PRIMARY KEY, project_uid TEXT NOT NULL, parent_uid TEXT,
  base_uid TEXT, label TEXT NOT NULL, created_at REAL NOT NULL,
  status TEXT NOT NULL, crash_tag TEXT);
CREATE TABLE IF NOT EXISTS source_records (
  uid TEXT NOT NULL, batch_uid TEXT NOT NULL, project_uid TEXT NOT NULL,
  kind TEXT NOT NULL, ping_idx INTEGER, beam INTEGER, cell INTEGER,
  raw_repr TEXT NOT NULL, raw_value REAL, unit TEXT NOT NULL, ref TEXT,
  seq INTEGER NOT NULL, PRIMARY KEY (uid, batch_uid));
CREATE TABLE IF NOT EXISTS nodes (
  uid TEXT NOT NULL, batch_uid TEXT NOT NULL, project_uid TEXT NOT NULL,
  kind TEXT NOT NULL, ping_idx INTEGER, cell INTEGER, status TEXT NOT NULL,
  fingerprint TEXT NOT NULL, includes TEXT NOT NULL, payload TEXT NOT NULL,
  seq INTEGER NOT NULL, PRIMARY KEY (uid, batch_uid));
CREATE TABLE IF NOT EXISTS candidates (
  uid TEXT NOT NULL, batch_uid TEXT NOT NULL, node_uid TEXT NOT NULL,
  ctype TEXT NOT NULL, beam_combo TEXT NOT NULL, fingerprint TEXT NOT NULL,
  payload TEXT NOT NULL, seq INTEGER NOT NULL, PRIMARY KEY (uid, batch_uid));
CREATE TABLE IF NOT EXISTS current_nodes (
  project_uid TEXT NOT NULL, batch_uid TEXT NOT NULL, node_uid TEXT NOT NULL,
  PRIMARY KEY (project_uid, batch_uid, node_uid));
CREATE TABLE IF NOT EXISTS edit_sessions (
  uid TEXT PRIMARY KEY, project_uid TEXT NOT NULL, base_batch TEXT NOT NULL,
  editor TEXT NOT NULL, started_at REAL NOT NULL);
"""


def create_v1_database(path: str) -> None:
    """Create a legacy v1 database (used by the migration test fixture)."""
    conn = connect(path)
    conn.executescript(V1_SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','1')")
    conn.commit()
    conn.close()


def needs_migration(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    return row is not None and int(row["value"]) < SCHEMA_VERSION


def migrate(conn: sqlite3.Connection) -> int:
    """Upgrade a v1 database to v2. Idempotent and transactional."""
    if not needs_migration(conn):
        return int(
            conn.execute("SELECT value FROM meta WHERE key='schema_version'")
            .fetchone()["value"]
        )
    with transaction(conn):
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(candidates)")}
        if "accepted" not in cols:
            conn.execute("ALTER TABLE candidates ADD COLUMN accepted INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (str(SCHEMA_VERSION),),
        )
    return SCHEMA_VERSION


def _numeric(raw: Any) -> bool:
    if isinstance(raw, bool):
        return False
    try:
        float(raw)
    except (TypeError, ValueError):
        return False
    return True


# Convenience: decode a row into plain python.
def node_view(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["includes"] = _loads(out["includes"])
    out["payload"] = _loads(out["payload"])
    return out


def candidate_view(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["beam_combo"] = _loads(out["beam_combo"])
    out["payload"] = _loads(out["payload"])
    return out


def event_view(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["payload"] = _loads(out["payload"])
    return out
