"""Versioned, evidence-preserving repository.

Guarantees required by the case:

* Every derivation is an independent append-only revision; the current pointer
  switches only after the whole transaction commits. A crash mid-write leaves
  the previous version intact (WAL + single commit).
* Two editors starting from the same event seq get a resolvable
  :class:`ConflictError` on save.
* Events are machine readable and append-only; any historical pointer state is
  reachable by event seq.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from .schema import SCHEMA_VERSION, initialize

_EXPORT_TABLES = [
    "source_batch", "ping", "cell_meta",
    "beam_velocity", "correlation", "amplitude", "bad_beam_flag",
    "heading", "pitch", "roll",
    "declination", "install_angle", "transducer_config", "bottom_track",
    "derivation", "current_pointer", "pointer_log", "event", "meta",
]


class ConflictError(Exception):
    """Raised on optimistic-lock conflict (edits forked from same seq)."""


class Repository:
    def __init__(self, path: str = "adcp.db"):
        self.path = path
        first = not os.path.exists(path) or os.path.getsize(path) == 0
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if first:
            initialize(self.conn)
        self._ensure_ready()

    def _ensure_ready(self) -> None:
        ver = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if ver == 0:
            initialize(self.conn)
        elif ver < SCHEMA_VERSION:
            from .migrations import migrate
            migrate(self.conn, ver)

    # -- transactions -------------------------------------------------------
    @contextmanager
    def transaction(self):
        conn = self.conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def head_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0) AS s FROM event").fetchone()
        return int(row["s"])

    # -- events -------------------------------------------------------------
    def log_event(self, conn, event_type: str, scope: Optional[str], payload: dict) -> int:
        cur = conn.execute(
            "INSERT INTO event(event_type,scope,payload_json,created_at_s) VALUES(?,?,?,?)",
            (event_type, scope, json.dumps(payload, sort_keys=True, separators=(",", ":")),
             time.time()),
        )
        return int(cur.lastrowid)

    # -- source import ------------------------------------------------------
    def import_snapshot(self, snap: dict, *, label: str = "import") -> int:
        """Insert an immutable batch of direct source records in one tx."""
        with self.transaction() as conn:
            seq0 = self.head_seq_locked(conn)
            bid = snap.get("batch_id") or f"batch_{seq0+1}"
            conn.execute(
                "INSERT INTO source_batch(batch_id,label,created_at_s,seq) VALUES(?,?,?,?)",
                (bid, label, time.time(), seq0 + 1),
            )
            for p in snap["pings"]:
                conn.execute(
                    "INSERT INTO ping(ping_id,batch_id,seq,time_s,n_cells) VALUES(?,?,?,?,?)",
                    (p["ping_id"], bid, p["seq"], p["time_s"], p["n_cells"]),
                )
            self._insert_cells(conn, snap)
            for table, rows in (
                ("beam_velocity", snap.get("beam_velocity", [])),
                ("correlation", snap.get("correlation", [])),
                ("amplitude", snap.get("amplitude", [])),
                ("bad_beam_flag", snap.get("bad_beam_flag", [])),
                ("heading", snap.get("heading", [])),
                ("pitch", snap.get("pitch", [])),
                ("roll", snap.get("roll", [])),
                ("bottom_track", snap.get("bottom_track", [])),
            ):
                self._bulk(conn, table, rows)
            for d in snap.get("declination", []):
                self._bulk(conn, "declination", [d])
            for ia in snap.get("install_angle", []):
                self._bulk(conn, "install_angle", [ia])
            for tc in snap.get("transducer_config", []):
                conn.execute(
                    "INSERT INTO transducer_config(version_id,batch_id,config_json,fingerprint)"
                    " VALUES(?,?,?,?)",
                    (tc["version_id"], bid, tc["config_json"], tc["fingerprint"]),
                )
            eseq = self.log_event(conn, "source_imported", bid,
                                  {"batch_id": bid, "label": label})
            return eseq

    @staticmethod
    def head_seq_locked(conn) -> int:
        return int(conn.execute("SELECT COALESCE(MAX(seq),0) s FROM event").fetchone()["s"])

    def _insert_cells(self, conn, snap) -> None:
        for c in snap.get("cell_meta", []):
            conn.execute(
                "INSERT INTO cell_meta(ping_id,cell_index,depth_unit,depth_raw,depth_m,"
                "valid_cell) VALUES(?,?,?,?,?,?)",
                (c["ping_id"], c["cell_index"], c["depth_unit"], c["depth_raw"],
                 c.get("depth_m"), 1 if c["valid_cell"] else 0),
            )

    @staticmethod
    def _bulk(conn, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT INTO {table}({','.join(cols)}) VALUES({placeholders})"
        conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])

    # -- derivations and pointers ------------------------------------------
    def write_derivations(
        self,
        records: list[dict],
        *,
        event_type: str = "derived",
        scope: Optional[str] = None,
        payload: Optional[dict] = None,
        base_seq: Optional[int] = None,
    ) -> int:
        """Append new derivation revisions and switch pointers atomically.

        Either the whole batch (rows + pointer moves + event) is visible or
        none of it is. ``base_seq`` enforces optimistic concurrency.
        """
        if base_seq is not None and base_seq != self.head_seq():
            raise ConflictError(
                f"base_seq {base_seq} != head {self.head_seq()}; reload and re-decide"
            )
        with self.transaction() as conn:
            eseq = self.log_event(conn, event_type, scope, payload or {"count": len(records)})
            for r in records:
                rev = self._next_revision(conn, r["derivation_id"])
                conn.execute(
                    "INSERT INTO derivation(derivation_id,scope_type,ping_id,cell_index,"
                    "revision,kind,beams,assumption,instrument,ship,earth,error_velocity,"
                    "diagnostics,fingerprint,rule_geom,rule_decl_version,rule_install_version,"
                    "rule_order,created_event_seq) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        r["derivation_id"], r["scope_type"], r["ping_id"],
                        r.get("cell_index"), rev, r["kind"], json.dumps(r["beams"]),
                        r["assumption"], json.dumps(r["instrument"]), json.dumps(r["ship"]),
                        json.dumps(r["earth"]) if r.get("earth") is not None else None,
                        r.get("error_velocity"), json.dumps(r.get("diagnostics", [])),
                        r["fingerprint"], r["rule_geom"], r["rule_decl_version"],
                        r["rule_install_version"], r["rule_order"], eseq,
                    ),
                )
                self._move_pointer(conn, r, eseq)
            return eseq

    @staticmethod
    def _next_revision(conn, derivation_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(revision),0)+1 r FROM derivation WHERE derivation_id=?",
            (derivation_id,),
        ).fetchone()
        return int(row["r"])

    @staticmethod
    def _move_pointer(conn, r: dict, eseq: int) -> None:
        scope_type, ping_id = r["scope_type"], r["ping_id"]
        cell_index = r.get("cell_index")
        conn.execute(
            "INSERT INTO current_pointer(scope_type,ping_id,cell_index,derivation_id,"
            "revision,updated_event_seq) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(scope_type,ping_id,cell_index) DO UPDATE SET "
            "derivation_id=excluded.derivation_id,revision=excluded.revision,"
            "updated_event_seq=excluded.updated_event_seq",
            (scope_type, ping_id, cell_index, r["derivation_id"],
               r.get("revision") or None, eseq),
        )
        # revision is set at insert; for pointer log read actual revision
        rev = conn.execute(
            "SELECT revision FROM derivation WHERE derivation_id=? ORDER BY revision DESC LIMIT 1",
            (r["derivation_id"],),
        ).fetchone()["revision"]
        conn.execute(
            "INSERT INTO pointer_log(scope_type,ping_id,cell_index,derivation_id,revision,"
            "event_seq) VALUES(?,?,?,?,?,?)",
            (scope_type, ping_id, cell_index, r["derivation_id"], rev, eseq),
        )

    def record_decision(
        self,
        event_type: str,
        scope: str,
        payload: dict,
        *,
        pointer: Optional[dict] = None,
        base_seq: Optional[int] = None,
    ) -> int:
        """Accept / reject / parameter-adjust decision -> new event + optional pointer."""
        if base_seq is not None and base_seq != self.head_seq():
            raise ConflictError(
                f"decision base_seq {base_seq} != head {self.head_seq()}"
            )
        with self.transaction() as conn:
            eseq = self.log_event(conn, event_type, scope, payload)
            if pointer is not None:
                conn.execute(
                    "INSERT INTO current_pointer(scope_type,ping_id,cell_index,derivation_id,"
                    "revision,updated_event_seq) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(scope_type,ping_id,cell_index) DO UPDATE SET "
                    "derivation_id=excluded.derivation_id,revision=excluded.revision,"
                    "updated_event_seq=excluded.updated_event_seq",
                    (pointer["scope_type"], pointer["ping_id"], pointer.get("cell_index"),
                       pointer["derivation_id"], pointer["revision"], eseq),
                )
                conn.execute(
                    "INSERT INTO pointer_log(scope_type,ping_id,cell_index,derivation_id,"
                    "revision,event_seq) VALUES(?,?,?,?,?,?)",
                    (pointer["scope_type"], pointer["ping_id"], pointer.get("cell_index"),
                       pointer["derivation_id"], pointer["revision"], eseq),
                )
            return eseq

    # -- reads --------------------------------------------------------------
    def list_pings(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ping ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    def sources_for_ping(self, ping_id: str) -> dict:
        out: dict[str, Any] = {"ping_id": ping_id}
        for table in ("beam_velocity", "correlation", "amplitude", "bad_beam_flag"):
            out[table] = [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} WHERE ping_id=? ORDER BY cell_index,beam", (ping_id,))]
        for table in ("heading", "pitch", "roll"):
            out[table] = [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} WHERE ping_id=? ORDER BY time_s", (ping_id,))]
        out["cell_meta"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM cell_meta WHERE ping_id=? ORDER BY cell_index", (ping_id,))]
        out["bottom_track"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM bottom_track WHERE ping_id=? ORDER BY beam", (ping_id,))]
        return out

    def all_attitude(self) -> dict:
        return {
            t: [dict(r) for r in self.conn.execute(f"SELECT * FROM {t} ORDER BY time_s")]
            for t in ("heading", "pitch", "roll")
        }

    def versions(self) -> dict:
        return {
            "declination": [dict(r) for r in self.conn.execute(
                "SELECT * FROM declination ORDER BY valid_from_s")],
            "install_angle": [dict(r) for r in self.conn.execute(
                "SELECT * FROM install_angle ORDER BY valid_from_s")],
            "transducer_config": [dict(r) for r in self.conn.execute(
                "SELECT * FROM transducer_config")],
        }

    def current_derivation(self, scope_type: str, ping_id: str, cell_index: Optional[int]):
        row = self.conn.execute(
            "SELECT * FROM current_pointer WHERE scope_type=? AND ping_id=? AND "
            "COALESCE(cell_index,-1)=COALESCE(?,-1)",
            (scope_type, ping_id, cell_index),
        ).fetchone()
        if not row:
            return None
        d = self.conn.execute(
            "SELECT * FROM derivation WHERE derivation_id=? AND revision=?",
            (row["derivation_id"], row["revision"]),
        ).fetchone()
        return dict(d) if d else None

    def candidates_for_scope(self, scope_type: str, ping_id: str,
                             cell_index: Optional[int]) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM derivation WHERE scope_type=? AND ping_id=? AND "
            "COALESCE(cell_index,-1)=COALESCE(?,-1) ORDER BY revision,kind,beams",
            (scope_type, ping_id, cell_index),
        ).fetchall()
        return [dict(r) for r in rows]

    def current_results(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT d.* FROM current_pointer c JOIN derivation d "
            "ON d.derivation_id=c.derivation_id AND d.revision=c.revision "
            "ORDER BY d.ping_id,d.cell_index"
        ).fetchall()
        return [dict(r) for r in rows]

    def events(self, upto_seq: Optional[int] = None) -> list[dict]:
        if upto_seq is None:
            rows = self.conn.execute("SELECT * FROM event ORDER BY seq").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM event WHERE seq<=? ORDER BY seq", (upto_seq,)).fetchall()
        return [{"seq": r["seq"], "event_type": r["event_type"], "scope": r["scope"],
                 "payload": json.loads(r["payload_json"]),
                 "created_at_s": r["created_at_s"]} for r in rows]

    def pointer_at_seq(self, scope_type: str, ping_id: str, cell_index: Optional[int],
                       seq: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM pointer_log WHERE scope_type=? AND ping_id=? AND "
            "COALESCE(cell_index,-1)=COALESCE(?,-1) AND event_seq<=? "
            "ORDER BY event_seq DESC,id DESC LIMIT 1",
            (scope_type, ping_id, cell_index, seq),
        ).fetchone()
        return dict(row) if row else None

    # -- mutation of a single direct source --------------------------------
    def update_source_row(self, table: str, identity: dict, changes: dict,
                          *, base_seq: Optional[int] = None) -> int:
        """Change ONE direct source record; only downstream scopes recompute.

        ``identity`` is {"record_id": ...} for per-beam/attitude tables or
        {"ping_id","cell_index"} for cell_meta.
        """
        if base_seq is not None and base_seq != self.head_seq():
            raise ConflictError(f"edit base_seq {base_seq} != head {self.head_seq()}")
        allowed = {"beam_velocity", "correlation", "amplitude", "bad_beam_flag",
                   "heading", "pitch", "roll", "cell_meta"}
        if table not in allowed:
            raise ValueError(f"table {table} is not a single-source editable table")
        with self.transaction() as conn:
            if table == "cell_meta":
                where = "ping_id=? AND cell_index=?"
                where_args = [identity["ping_id"], identity["cell_index"]]
                target = f"{identity['ping_id']}#cell{identity['cell_index']}"
            else:
                where = "record_id=?"
                where_args = [identity["record_id"]]
                target = identity["record_id"]
            sets, vals = [], []
            for k, v in changes.items():
                sets.append(f"{k}=?")
                vals.append(v)
            sql = f"UPDATE {table} SET {','.join(sets)} WHERE {where}"
            conn.execute(sql, vals + where_args)
            return self.log_event(conn, "source_changed", target,
                                  {"table": table, "identity": identity, "changes": changes})

    def add_declination_version(self, version: dict, *, superseded: Optional[str] = None,
                                base_seq: Optional[int] = None) -> int:
        if base_seq is not None and base_seq != self.head_seq():
            raise ConflictError(f"decl base_seq {base_seq} != head {self.head_seq()}")
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO declination(version_id,batch_id,valid_from_s,unit,raw,value,"
                "normalized,diagnostic,superseded_by) VALUES(?,?,?,?,?,?,?,?,?)",
                (version["version_id"], version["batch_id"], version["valid_from_s"],
                 version["unit"], version["raw"], version.get("value"),
                 version.get("normalized"), version.get("diagnostic"), superseded),
            )
            if superseded:
                conn.execute("UPDATE declination SET superseded_by=? WHERE version_id=?",
                             (version["version_id"], superseded))
            return self.log_event(conn, "declination_revised", version["version_id"],
                                  {"new": version["version_id"], "superseded": superseded})

    # -- export / import (identity, order and provenance survive) ----------
    def export_json(self) -> dict:
        """Full logical dump. Event seq, record ids and pointer order preserved."""
        dump: dict[str, Any] = {"_format": "adcp-evidence", "_schema_version": SCHEMA_VERSION}
        for table in _EXPORT_TABLES:
            dump[table] = [dict(r) for r in self.conn.execute(f"SELECT * FROM {table}")]
        return dump

    @classmethod
    def import_json(cls, path_db: str, dump: dict) -> "Repository":
        """Load an export into a fresh db, preserving ids/seq/order exactly."""
        if os.path.exists(path_db):
            os.remove(path_db)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(path_db + suffix):
                os.remove(path_db + suffix)
        repo = cls(path_db)
        with repo.transaction() as conn:
            # order matters for FKs; event/pointer_log rows carry explicit ids
            ordered = [t for t in _EXPORT_TABLES]
            for table in ordered:
                rows = dump.get(table, [])
                if not rows:
                    continue
                cols = list(rows[0].keys())
                ph = ",".join("?" for _ in cols)
                conn.executemany(
                    f"INSERT INTO {table}({','.join(cols)}) VALUES({ph})",
                    [tuple(row.get(c) for c in cols) for row in rows]
                )
        return repo

    def close(self) -> None:
        self.conn.close()

