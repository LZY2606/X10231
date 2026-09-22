"""Versioned evidence store (SQLite).

Semantics:
* beam velocity, correlation, amplitude, bad-beam flags, attitude,
  magnetic declination and mounting angle are stored as *separate* records;
* every derivation runs in its own version; the current pointer flips only
  after the version completes, in the same transaction -> after a crash a
  restarted process sees either the old version or a complete new one;
* two users editing from the same base version produce a resolvable
  conflict; the unchosen branch is kept, never overwritten;
* export/import preserves identities, ordering and provenance.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    label TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('building','complete','abandoned')),
    fingerprint TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS current_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    prefix TEXT PRIMARY KEY,
    n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    version_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    ping INTEGER, beam INTEGER, cell INTEGER,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_key ON records(version_id, kind, ping, beam, cell);
CREATE TABLE IF NOT EXISTS derived (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    version_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    ping INTEGER NOT NULL,
    cell INTEGER NOT NULL,
    status TEXT NOT NULL,
    value_json TEXT,
    candidates_json TEXT,
    fingerprint TEXT NOT NULL,
    explanation TEXT NOT NULL,
    diagnostic TEXT
);
CREATE INDEX IF NOT EXISTS idx_derived_key ON derived(version_id, stage, ping, cell);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    version_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    base_version INTEGER NOT NULL,
    branch_a_version INTEGER NOT NULL,
    branch_b_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open','resolved')),
    resolution TEXT,
    created_at REAL NOT NULL
);
"""

# v1 (legacy) schema: one merged measurement row per ping/beam/cell.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ping INTEGER NOT NULL, beam INTEGER NOT NULL, cell INTEGER NOT NULL,
    velocity REAL, correlation REAL, amplitude REAL, bad_flag INTEGER
);
"""


class ConflictError(Exception):
    def __init__(self, conflict_id: str):
        super().__init__(f"concurrent edit conflict: {conflict_id}")
        self.conflict_id = conflict_id


class Store:
    def __init__(self, path: str):
        self.path = path
        first = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if first:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
        else:
            self._migrate()

    # -- migration ---------------------------------------------------------
    def _migrate(self) -> None:
        v = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if v >= SCHEMA_VERSION:
            return
        if v == 1:
            with self.conn:
                self.conn.executescript(_SCHEMA)
                now = time.time()
                self.conn.execute(
                    "INSERT INTO versions(id,parent_id,label,state,created_at) "
                    "VALUES (1,NULL,'migrated-from-v1','complete',?)", (now,))
                self.conn.execute(
                    "INSERT INTO current_version(id,version_id) VALUES (1,1)")
                seq = 0
                for row in self.conn.execute(
                        "SELECT * FROM measurements ORDER BY id"):
                    seq += 1
                    for kind, col in (("beam_velocity", "velocity"),
                                      ("correlation", "correlation"),
                                      ("amplitude", "amplitude"),
                                      ("bad_flag", "bad_flag")):
                        rid = f"rec-mig-{row['id']:06d}-{kind}"
                        payload = {"value": row[col], "unit": "unknown",
                                   "datum": "legacy-v1", "raw": str(row[col]),
                                   "diagnostic": "migrated-from-merged-v1-row"}
                        self.conn.execute(
                            "INSERT INTO records(id,seq,version_id,kind,ping,beam,cell,payload)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (rid, seq, 1, kind, row["ping"], row["beam"],
                             row["cell"], json.dumps(payload)))
                self.conn.execute("DROP TABLE measurements")
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- id allocation -----------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        row = self.conn.execute("SELECT n FROM counters WHERE prefix=?",
                                (prefix,)).fetchone()
        n = 1 if row is None else row["n"] + 1
        self.conn.execute(
            "INSERT INTO counters(prefix,n) VALUES (?,?) "
            "ON CONFLICT(prefix) DO UPDATE SET n=excluded.n", (prefix, n))
        return f"{prefix}-{n:06d}"

    # -- versions ----------------------------------------------------------
    def begin_version(self, parent_id: Optional[int], label: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO versions(parent_id,label,state,created_at)"
                " VALUES (?,?, 'building', ?)",
                (parent_id, label, time.time()))
            return int(cur.lastrowid)

    def current_version(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT version_id FROM current_version WHERE id=1").fetchone()
        return None if row is None else int(row["version_id"])

    def commit_version(self, version_id: int, base_version: Optional[int] = None,
                       fingerprint: Optional[str] = None) -> Optional[str]:
        """Atomically mark complete and flip the current pointer.

        If ``base_version`` is given and the current pointer has moved on,
        the version is still completed (kept as a branch) but the pointer is
        NOT moved; a resolvable conflict is recorded and its id returned.
        """
        with self.conn:
            cur = self.current_version()
            conflict_id = None
            if base_version is not None and cur is not None and cur != base_version:
                conflict_id = self._next_id("cfl")
                self.conn.execute(
                    "INSERT INTO conflicts(id,base_version,branch_a_version,"
                    "branch_b_version,status,created_at) VALUES (?,?,?,?, 'open', ?)",
                    (conflict_id, base_version, cur, version_id, time.time()))
            else:
                self.conn.execute(
                    "INSERT INTO current_version(id,version_id) VALUES (1,?) "
                    "ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id",
                    (version_id,))
            self.conn.execute(
                "UPDATE versions SET state='complete', fingerprint=? WHERE id=?",
                (fingerprint, version_id))
            return conflict_id

    def abort_version(self, version_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE versions SET state='abandoned' WHERE id=?",
                              (version_id,))

    def resolve_conflict(self, conflict_id: str, winner: str,
                         note: str = "") -> None:
        """winner: 'a' (keep current) or 'b' (adopt the later branch)."""
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
            if row is None:
                raise KeyError(conflict_id)
            if row["status"] != "open":
                raise ValueError("conflict already resolved")
            chosen = row["branch_a_version"] if winner == "a" else row["branch_b_version"]
            self.conn.execute(
                "INSERT INTO current_version(id,version_id) VALUES (1,?) "
                "ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id",
                (chosen,))
            self.conn.execute(
                "UPDATE conflicts SET status='resolved', resolution=? WHERE id=?",
                (json.dumps({"winner": winner, "chosen_version": chosen,
                             "note": note}), conflict_id))
            self.add_event(chosen, "conflict-resolved", {
                "conflict_id": conflict_id, "winner": winner,
                "chosen_version": chosen, "note": note,
                "kept_unchosen": (row["branch_b_version"] if winner == "a"
                                  else row["branch_a_version"]),
            })

    # -- records / derived / events ---------------------------------------
    def add_record(self, version_id: int, kind: str, payload: dict,
                   ping: Optional[int] = None, beam: Optional[int] = None,
                   cell: Optional[int] = None, record_id: Optional[str] = None) -> str:
        rid = record_id or self._next_id("rec")
        seq = self._seq("records")
        self.conn.execute(
            "INSERT INTO records(id,seq,version_id,kind,ping,beam,cell,payload)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (rid, seq, version_id, kind, ping, beam, cell,
             json.dumps(payload, sort_keys=True)))
        return rid

    def add_derived(self, version_id: int, stage: str, ping: int, cell: int,
                    status: str, fingerprint: str, explanation: str,
                    value: Optional[dict] = None,
                    candidates: Optional[list] = None,
                    diagnostic: Optional[str] = None,
                    derived_id: Optional[str] = None) -> str:
        did = derived_id or self._next_id("der")
        seq = self._seq("derived")
        self.conn.execute(
            "INSERT INTO derived(id,seq,version_id,stage,ping,cell,status,"
            "value_json,candidates_json,fingerprint,explanation,diagnostic)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, seq, version_id, stage, ping, cell, status,
             json.dumps(value, sort_keys=True) if value is not None else None,
             json.dumps(candidates, sort_keys=True) if candidates is not None else None,
             fingerprint, explanation, diagnostic))
        return did

    def add_event(self, version_id: int, type_: str, payload: dict,
                  event_id: Optional[str] = None) -> str:
        eid = event_id or self._next_id("evt")
        seq = self._seq("events")
        self.conn.execute(
            "INSERT INTO events(id,seq,version_id,ts,type,payload)"
            " VALUES (?,?,?,?,?,?)",
            (eid, seq, version_id, time.time(), type_,
             json.dumps(payload, sort_keys=True)))
        return eid

    def _seq(self, table: str) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(seq),0)+1 AS n FROM {table}").fetchone()
        return int(row["n"])

    # -- queries -----------------------------------------------------------
    def records(self, version_id: int, kind: Optional[str] = None) -> List[dict]:
        q = "SELECT * FROM records WHERE version_id=?"
        args: list = [version_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args)]

    def derived(self, version_id: int, stage: Optional[str] = None,
                ping: Optional[int] = None) -> List[dict]:
        q = "SELECT * FROM derived WHERE version_id=?"
        args: list = [version_id]
        if stage:
            q += " AND stage=?"
            args.append(stage)
        if ping is not None:
            q += " AND ping=?"
            args.append(ping)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args)]

    def events(self, version_id: Optional[int] = None) -> List[dict]:
        if version_id is None:
            rows = self.conn.execute("SELECT * FROM events ORDER BY seq")
        else:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE version_id=? ORDER BY seq",
                (version_id,))
        return [dict(r) for r in rows]

    def versions(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM versions ORDER BY id")]

    def conflicts(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM conflicts ORDER BY created_at")]

    # -- export / import ---------------------------------------------------
    def export_archive(self) -> dict:
        """Full archive; identities, seq order and provenance are preserved."""
        return {
            "format": "adcp-evidence-archive",
            "schema_version": SCHEMA_VERSION,
            "versions": self.versions(),
            "current_version": self.current_version(),
            "counters": [dict(r) for r in self.conn.execute(
                "SELECT * FROM counters ORDER BY prefix")],
            "records": [dict(r) for r in self.conn.execute(
                "SELECT * FROM records ORDER BY seq")],
            "derived": [dict(r) for r in self.conn.execute(
                "SELECT * FROM derived ORDER BY seq")],
            "events": [dict(r) for r in self.conn.execute(
                "SELECT * FROM events ORDER BY seq")],
            "conflicts": self.conflicts(),
        }

    @staticmethod
    def import_archive(path: str, archive: dict) -> "Store":
        if archive.get("format") != "adcp-evidence-archive":
            raise ValueError("not an adcp evidence archive")
        store = Store(path)
        with store.conn:
            for v in archive["versions"]:
                store.conn.execute(
                    "INSERT INTO versions(id,parent_id,label,state,fingerprint,created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (v["id"], v["parent_id"], v["label"], v["state"],
                     v.get("fingerprint"), v["created_at"]))
            for c in archive["counters"]:
                store.conn.execute(
                    "INSERT INTO counters(prefix,n) VALUES (?,?)",
                    (c["prefix"], c["n"]))
            for r in archive["records"]:
                store.conn.execute(
                    "INSERT INTO records(id,seq,version_id,kind,ping,beam,cell,payload)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (r["id"], r["seq"], r["version_id"], r["kind"], r["ping"],
                     r["beam"], r["cell"], r["payload"]))
            for d in archive["derived"]:
                store.conn.execute(
                    "INSERT INTO derived(id,seq,version_id,stage,ping,cell,status,"
                    "value_json,candidates_json,fingerprint,explanation,diagnostic)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d["id"], d["seq"], d["version_id"], d["stage"], d["ping"],
                     d["cell"], d["status"], d["value_json"], d["candidates_json"],
                     d["fingerprint"], d["explanation"], d["diagnostic"]))
            for e in archive["events"]:
                store.conn.execute(
                    "INSERT INTO events(id,seq,version_id,ts,type,payload)"
                    " VALUES (?,?,?,?,?,?)",
                    (e["id"], e["seq"], e["version_id"], e["ts"], e["type"],
                     e["payload"]))
            for c in archive["conflicts"]:
                store.conn.execute(
                    "INSERT INTO conflicts(id,base_version,branch_a_version,"
                    "branch_b_version,status,resolution,created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (c["id"], c["base_version"], c["branch_a_version"],
                     c["branch_b_version"], c["status"], c.get("resolution"),
                     c["created_at"]))
            if archive["current_version"] is not None:
                store.conn.execute(
                    "INSERT INTO current_version(id,version_id) VALUES (1,?) "
                    "ON CONFLICT(id) DO UPDATE SET version_id=excluded.version_id",
                    (archive["current_version"],))
        return store

    def close(self) -> None:
        self.conn.close()
