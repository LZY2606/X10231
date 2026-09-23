"""Versioned SQLite storage with atomic pointer switching.

Invariants
----------
- Seven source kinds are stored in their own tables (beam velocity,
  correlation, amplitude, bad flag, heading/pitch/roll, declination,
  installation angles), plus transducer config, depth cells and bottom track.
- Each derivation produces an independent *version*. The current pointer only
  moves after the whole version is committed (``status='ready'``). A process
  interrupted while writing leaves a ``building`` row that recovery discards;
  readers therefore always see the old version or the complete new one.
- Versions carry the parent version and an edit base. Two users editing the
  same base version produce a conflict that must be explicitly resolved.
- Import/export preserves stable ids, ordering and provenance.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fingerprint import fingerprint

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at_s REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS version (
    version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    parent_version_id TEXT,
    edit_base_version_id TEXT,
    label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building','ready','abandoned')),
    author TEXT NOT NULL,
    created_at_s REAL NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS current_pointer (
    dataset_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_beam (
    version_id TEXT NOT NULL,
    ping_id TEXT NOT NULL,
    cell_index INTEGER,
    beam_index INTEGER NOT NULL,
    velocity_raw TEXT, correlation_raw TEXT, amplitude_raw TEXT,
    bad_flag INTEGER NOT NULL,
    ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_attitude (
    version_id TEXT NOT NULL, time_s REAL NOT NULL,
    heading_raw TEXT, pitch_raw TEXT, roll_raw TEXT, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_declination (
    version_id TEXT NOT NULL, declination_version TEXT NOT NULL,
    valid_from_s REAL NOT NULL, valid_to_s REAL NOT NULL,
    declination_raw TEXT, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_mounting (
    version_id TEXT NOT NULL, mounting_version TEXT NOT NULL,
    valid_from_s REAL NOT NULL, valid_to_s REAL NOT NULL,
    pitch_raw TEXT, roll_raw TEXT, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_transducer (
    version_id TEXT PRIMARY KEY, config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_cell (
    version_id TEXT NOT NULL, cell_index INTEGER NOT NULL,
    bin_distance_raw TEXT, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_bottom_track (
    version_id TEXT NOT NULL, ping_id TEXT NOT NULL,
    range_raw_json TEXT, velocity_raw_json TEXT, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS source_ping_meta (
    version_id TEXT NOT NULL, ping_id TEXT NOT NULL,
    time_s REAL NOT NULL, ord INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS derivation_node (
    version_id TEXT NOT NULL,
    stable_id TEXT NOT NULL,
    node_kind TEXT NOT NULL,
    parent_stable_ids_json TEXT NOT NULL,
    rule TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_fingerprint TEXT NOT NULL,
    ord INTEGER NOT NULL,
    PRIMARY KEY (version_id, stable_id)
);
CREATE TABLE IF NOT EXISTS event (
    version_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    affected_stable_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_s REAL NOT NULL,
    PRIMARY KEY (version_id, event_id)
);
CREATE TABLE IF NOT EXISTS decision (
    version_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    scope_stable_id TEXT NOT NULL,
    chosen_candidate_id TEXT,
    action TEXT NOT NULL,
    rationale TEXT NOT NULL,
    rejected_candidates_json TEXT NOT NULL,
    created_at_s REAL NOT NULL,
    PRIMARY KEY (version_id, decision_id)
);
"""


class EditConflict(Exception):
    """Raised when a commit's edit base is no longer the current version."""


class VersionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._recover_pending()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def _recover_pending(self) -> None:
        """Building versions from an interrupted process are abandoned.

        They are never visible through the current pointer, so recovery is
        atomic with respect to readers: old version, or complete new version.
        """
        with self._tx() as conn:
            conn.execute(
                "UPDATE version SET status='abandoned' WHERE status='building'"
            )

    # ---------------------------------------------------------------- dataset
    def ensure_dataset(self, dataset_id: str, name: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dataset(dataset_id,name,created_at_s) VALUES(?,?,?)",
                (dataset_id, name, time.time()),
            )

    def list_datasets(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM dataset ORDER BY created_at_s").fetchall()
        return [dict(row) for row in rows]

    def current_version_id(self, dataset_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT version_id FROM current_pointer WHERE dataset_id=?", (dataset_id,)
        ).fetchone()
        return row["version_id"] if row else None

    def next_seq(self, dataset_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS next_seq FROM version WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        return int(row["next_seq"])

    # ----------------------------------------------------------------- commit
    def commit_version(
        self,
        dataset_id: str,
        label: str,
        author: str,
        sources: Dict[str, Any],
        nodes: Sequence[Dict[str, Any]],
        events: Sequence[Dict[str, Any]],
        parent_version_id: Optional[str] = None,
        edit_base_version_id: Optional[str] = None,
        expect_base_version_id: Optional[str] = None,
        decisions: Sequence[Dict[str, Any]] = (),
        force: bool = False,
    ) -> str:
        """Atomically write one complete version, then move the pointer.

        ``expect_base_version_id`` implements optimistic concurrency: if two
        users edit the same base and the pointer has moved, the second commit
        raises :class:`EditConflict` unless a resolution is explicitly forced.
        """
        current = self.current_version_id(dataset_id)
        if (
            not force
            and expect_base_version_id is not None
            and current is not None
            and expect_base_version_id != current
        ):
            raise EditConflict(
                f"base {expect_base_version_id} is not current ({current}); "
                "resolve the divergence before committing"
            )

        version_id = "ver_" + uuid.uuid4().hex[:12]
        seq = self.next_seq(dataset_id)
        manifest = {
            "label": label,
            "author": author,
            "sources": _source_manifest(sources),
            "nodes": len(nodes),
            "events": [event["event_type"] for event in events],
        }
        manifest_json = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
        manifest_fpr = fingerprint(manifest_json)

        with self._tx() as conn:
            conn.execute(
                "INSERT INTO version(version_id,dataset_id,seq,parent_version_id,"
                "edit_base_version_id,label,status,author,created_at_s,"
                "manifest_json,manifest_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version_id, dataset_id, seq, parent_version_id,
                    edit_base_version_id, label, "building", author, time.time(),
                    manifest_json, manifest_fpr,
                ),
            )
            _write_sources(conn, version_id, sources)
            for ordinal, node in enumerate(nodes):
                conn.execute(
                    "INSERT INTO derivation_node(version_id,stable_id,node_kind,"
                    "parent_stable_ids_json,rule,value_json,value_fingerprint,ord)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (
                        version_id, node["stable_id"], node["node_kind"],
                        json.dumps(node.get("parents", []), ensure_ascii=False),
                        node["rule"],
                        json.dumps(node["value"], ensure_ascii=False, default=str),
                        node.get("value_fingerprint")
                        or fingerprint(node["value"]),
                        ordinal,
                    ),
                )
            for ordinal, event in enumerate(events):
                conn.execute(
                    "INSERT INTO event(version_id,event_id,seq,event_type,scope,"
                    "affected_stable_ids_json,status,payload_json,created_at_s)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        version_id, event["event_id"], ordinal,
                        event["event_type"], event["scope"],
                        json.dumps(event.get("affected", []), ensure_ascii=False),
                        event["status"],
                        json.dumps(event.get("payload", {}), ensure_ascii=False, default=str),
                        time.time(),
                    ),
                )
            for decision in decisions:
                conn.execute(
                    "INSERT INTO decision(version_id,decision_id,scope_stable_id,"
                    "chosen_candidate_id,action,rationale,rejected_candidates_json,"
                    "created_at_s) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        version_id, decision["decision_id"], decision["scope_stable_id"],
                        decision.get("chosen_candidate_id"), decision["action"],
                        decision.get("rationale", ""),
                        json.dumps(decision.get("rejected_candidates", []), ensure_ascii=False),
                        time.time(),
                    ),
                )
            # Point-of-no-return: flip status and current pointer together.
            conn.execute(
                "UPDATE version SET status='ready' WHERE version_id=?", (version_id,)
            )
            conn.execute(
                "INSERT INTO current_pointer(dataset_id,version_id) VALUES(?,?) "
                "ON CONFLICT(dataset_id) DO UPDATE SET version_id=excluded.version_id",
                (dataset_id, version_id),
            )
        return version_id


_SOURCE_TABLES = (
    "source_beam",
    "source_attitude",
    "source_declination",
    "source_mounting",
    "source_transducer",
    "source_cell",
    "source_bottom_track",
    "source_ping_meta",
)


def _source_manifest(sources: Dict[str, Any]) -> Dict[str, int]:
    return {
        key: len(value) if isinstance(value, list) else 1
        for key, value in sources.items()
        if key in {"pings", "attitudes", "declinations", "mountings", "transducer", "cells", "bottom_tracks"}
    }


def _write_sources(conn: sqlite3.Connection, version_id: str, sources: Dict[str, Any]) -> None:
    ordinal = 0
    for ping in sources.get("pings", []):
        for cell_index, cell_beams in (ping.get("cell_beams") or {}).items():
            for beam in cell_beams:
                conn.execute(
                    "INSERT INTO source_beam(version_id,ping_id,cell_index,beam_index,"
                    "velocity_raw,correlation_raw,amplitude_raw,bad_flag,ord)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        version_id, ping["ping_id"], int(cell_index),
                        int(beam.get("beam_index", ordinal)), beam.get("velocity"),
                        beam.get("correlation"), beam.get("amplitude"),
                        1 if beam.get("bad_flag") else 0, ordinal,
                    ),
                )
                ordinal += 1
        if "beams" in ping:
            for beam in ping["beams"]:
                conn.execute(
                    "INSERT INTO source_beam(version_id,ping_id,cell_index,beam_index,"
                    "velocity_raw,correlation_raw,amplitude_raw,bad_flag,ord)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        version_id, ping["ping_id"], None,
                        int(beam.get("beam_index", ordinal)), beam.get("velocity"),
                        beam.get("correlation"), beam.get("amplitude"),
                        1 if beam.get("bad_flag") else 0, ordinal,
                    ),
                )
                ordinal += 1
    for index, ping in enumerate(sources.get("pings", [])):
        conn.execute(
            "INSERT INTO source_ping_meta(version_id,ping_id,time_s,ord)"
            " VALUES(?,?,?,?)",
            (version_id, str(ping["ping_id"]), float(ping["time_s"]), index),
        )
    for index, attitude in enumerate(sources.get("attitudes", [])):
        conn.execute(
            "INSERT INTO source_attitude(version_id,time_s,heading_raw,pitch_raw,roll_raw,ord)"
            " VALUES(?,?,?,?,?,?)",
            (
                version_id, float(attitude["time_s"]), attitude.get("heading_deg"),
                attitude.get("pitch_deg"), attitude.get("roll_deg"), index,
            ),
        )
    for index, declination in enumerate(sources.get("declinations", [])):
        conn.execute(
            "INSERT INTO source_declination(version_id,declination_version,valid_from_s,"
            "valid_to_s,declination_raw,ord) VALUES(?,?,?,?,?,?)",
            (
                version_id, declination["declination_version"],
                float(declination["valid_from_s"]), float(declination["valid_to_s"]),
                declination.get("declination_deg"), index,
            ),
        )
    for index, mounting in enumerate(sources.get("mountings", [])):
        conn.execute(
            "INSERT INTO source_mounting(version_id,mounting_version,valid_from_s,"
            "valid_to_s,pitch_raw,roll_raw,ord) VALUES(?,?,?,?,?,?,?)",
            (
                version_id, mounting["mounting_version"],
                float(mounting["valid_from_s"]), float(mounting["valid_to_s"]),
                mounting.get("pitch_deg"), mounting.get("roll_deg"), index,
            ),
        )
    if sources.get("transducer") is not None:
        conn.execute(
            "INSERT INTO source_transducer(version_id,config_json) VALUES(?,?)",
            (version_id, json.dumps(sources["transducer"], sort_keys=True)),
        )
    for index, cell in enumerate(sources.get("cells", [])):
        conn.execute(
            "INSERT INTO source_cell(version_id,cell_index,bin_distance_raw,ord)"
            " VALUES(?,?,?,?)",
            (version_id, int(cell["cell_index"]), cell.get("bin_distance_m"), index),
        )
    for index, bottom in enumerate(sources.get("bottom_tracks", [])):
        conn.execute(
            "INSERT INTO source_bottom_track(version_id,ping_id,range_raw_json,"
            "velocity_raw_json,ord) VALUES(?,?,?,?,?)",
            (
                version_id, bottom["ping_id"],
                json.dumps(bottom.get("range_m", [])),
                json.dumps(bottom.get("velocity", [])),
                index,
            ),
        )

    # ------------------------------------------------------------- read views
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM version WHERE version_id=? AND status='ready'", (version_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT version_id,seq,parent_version_id,edit_base_version_id,label,"
            "author,created_at_s,status,manifest_fingerprint FROM version "
            "WHERE dataset_id=? ORDER BY seq",
            (dataset_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_sources(self, version_id: str) -> Dict[str, Any]:
        trow = self._conn.execute(
            "SELECT config_json FROM source_transducer WHERE version_id=?", (version_id,)
        ).fetchone()
        transducer = json.loads(trow["config_json"]) if trow else None

        cells = [
            {"cell_index": row["cell_index"], "bin_distance_m": row["bin_distance_raw"]}
            for row in self._conn.execute(
                "SELECT cell_index,bin_distance_raw FROM source_cell WHERE version_id=? ORDER BY ord",
                (version_id,),
            )
        ]
        attitudes = [
            {
                "time_s": row["time_s"],
                "heading_deg": row["heading_raw"],
                "pitch_deg": row["pitch_raw"],
                "roll_deg": row["roll_raw"],
            }
            for row in self._conn.execute(
                "SELECT * FROM source_attitude WHERE version_id=? ORDER BY ord", (version_id,)
            )
        ]
        declinations = [dict(row) for row in self._conn.execute(
            "SELECT * FROM source_declination WHERE version_id=? ORDER BY ord", (version_id,))]
        for item in declinations:
            item["declination_deg"] = item.pop("declination_raw")
        mountings = [dict(row) for row in self._conn.execute(
            "SELECT * FROM source_mounting WHERE version_id=? ORDER BY ord", (version_id,))]
        for item in mountings:
            item["pitch_deg"] = item.pop("pitch_raw")
            item["roll_deg"] = item.pop("roll_raw")

        pings_map: Dict[str, Dict[str, Any]] = {}
        for row in self._conn.execute(
            "SELECT * FROM source_beam WHERE version_id=? ORDER BY ord", (version_id,)
        ):
            ping = pings_map.setdefault(row["ping_id"], {"ping_id": row["ping_id"], "beams": [], "cell_beams": {}})
            beam = {
                "beam_index": row["beam_index"],
                "velocity": row["velocity_raw"],
                "correlation": row["correlation_raw"],
                "amplitude": row["amplitude_raw"],
                "bad_flag": bool(row["bad_flag"]),
            }
            if row["cell_index"] is None:
                ping["beams"].append(beam)
            else:
                ping["cell_beams"].setdefault(str(row["cell_index"]), []).append(beam)

        time_rows = self._conn.execute(
            "SELECT ping_id,time_s FROM source_ping_meta WHERE version_id=? ORDER BY ord",
            (version_id,),
        ).fetchall()
        pings = []
        for trow in time_rows:
            ping = pings_map.pop(trow["ping_id"], {"ping_id": trow["ping_id"], "beams": [], "cell_beams": {}})
            ping["time_s"] = trow["time_s"]
            pings.append(ping)
        pings.extend(pings_map.values())
        pings.sort(key=lambda p: p.get("time_s", 0.0))

        bottom_tracks = []
        for row in self._conn.execute(
            "SELECT * FROM source_bottom_track WHERE version_id=? ORDER BY ord", (version_id,)
        ):
            bottom_tracks.append(
                {
                    "ping_id": row["ping_id"],
                    "range_m": json.loads(row["range_raw_json"]),
                    "velocity": json.loads(row["velocity_raw_json"]),
                }
            )

        return {
            "transducer": transducer,
            "cells": cells,
            "attitudes": attitudes,
            "declinations": declinations,
            "mountings": mountings,
            "pings": pings,
            "bottom_tracks": bottom_tracks,
        }

    def get_nodes(self, version_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM derivation_node WHERE version_id=? ORDER BY ord", (version_id,)
        ).fetchall()
        nodes = []
        for row in rows:
            nodes.append(
                {
                    "stable_id": row["stable_id"],
                    "node_kind": row["node_kind"],
                    "parents": json.loads(row["parent_stable_ids_json"]),
                    "rule": row["rule"],
                    "value": json.loads(row["value_json"]),
                    "value_fingerprint": row["value_fingerprint"],
                }
            )
        return nodes

    def get_events(self, version_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM event WHERE version_id=? ORDER BY seq", (version_id,)
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "scope": row["scope"],
                "affected": json.loads(row["affected_stable_ids_json"]),
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "created_at_s": row["created_at_s"],
            }
            for row in rows
        ]

    def get_decisions(self, version_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM decision WHERE version_id=? ORDER BY created_at_s", (version_id,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["rejected_candidates"] = json.loads(item.pop("rejected_candidates_json"))
            out.append(item)
        return out

    # --------------------------------------------------------- export/import
    def export_bundle(self, dataset_id: str) -> Dict[str, Any]:
        """Deterministic JSON bundle of the whole dataset lineage."""
        versions = []
        for meta in self.list_versions(dataset_id):
            version_id = meta["version_id"]
            version = self.get_version(version_id)
            if version is None:
                continue
            versions.append(
                {
                    "version_id": version_id,
                    "seq": version["seq"],
                    "parent_version_id": version["parent_version_id"],
                    "edit_base_version_id": version["edit_base_version_id"],
                    "label": version["label"],
                    "author": version["author"],
                    "created_at_s": version["created_at_s"],
                    "manifest": json.loads(version["manifest_json"]),
                    "manifest_fingerprint": version["manifest_fingerprint"],
                    "sources": self.get_sources(version_id),
                    "nodes": self.get_nodes(version_id),
                    "events": self.get_events(version_id),
                    "decisions": self.get_decisions(version_id),
                }
            )
        dataset_rows = self._conn.execute(
            "SELECT * FROM dataset WHERE dataset_id=?", (dataset_id,)
        ).fetchall()
        return {
            "bundle_format": "adcp-evidence/1",
            "dataset": dict(dataset_rows[0]) if dataset_rows else {"dataset_id": dataset_id},
            "current_version_id": self.current_version_id(dataset_id),
            "versions": versions,
        }

    def import_bundle(self, bundle: Dict[str, Any], verify: bool = True) -> str:
        """Re-import a bundle; stable ids, order and provenance are preserved."""
        dataset = bundle["dataset"]
        dataset_id = dataset["dataset_id"]
        self.ensure_dataset(dataset_id, dataset.get("name", dataset_id))

        # Clean-slate import into this database: bundles carry their own ids.
        with self._tx() as conn:
            for table in list(_SOURCE_TABLES) + [
                "derivation_node", "event", "decision", "version", "current_pointer",
            ]:
                conn.execute("DELETE FROM " + table)
            conn.execute("DELETE FROM dataset WHERE dataset_id=?", (dataset_id,))
            conn.execute(
                "INSERT INTO dataset(dataset_id,name,created_at_s) VALUES(?,?,?)",
                (dataset_id, dataset.get("name", dataset_id), dataset.get("created_at_s", time.time())),
            )
            for version in sorted(bundle["versions"], key=lambda v: v["seq"]):
                version_id = version["version_id"]
                manifest_json = json.dumps(version["manifest"], sort_keys=True, ensure_ascii=False)
                expected = version["manifest_fingerprint"]
                if verify and fingerprint(manifest_json) != expected:
                    raise ValueError(f"manifest fingerprint mismatch for {version_id}")
                conn.execute(
                    "INSERT INTO version(version_id,dataset_id,seq,parent_version_id,"
                    "edit_base_version_id,label,status,author,created_at_s,"
                    "manifest_json,manifest_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        version_id, dataset_id, version["seq"],
                        version["parent_version_id"], version["edit_base_version_id"],
                        version["label"], "ready", version["author"],
                        version["created_at_s"], manifest_json, expected,
                    ),
                )
                _write_sources(conn, version_id, version["sources"])
                for ordinal, node in enumerate(version["nodes"]):
                    conn.execute(
                        "INSERT INTO derivation_node(version_id,stable_id,node_kind,"
                        "parent_stable_ids_json,rule,value_json,value_fingerprint,ord)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (
                            version_id, node["stable_id"], node["node_kind"],
                            json.dumps(node.get("parents", []), ensure_ascii=False),
                            node["rule"], json.dumps(node["value"], ensure_ascii=False, default=str),
                            node["value_fingerprint"], ordinal,
                        ),
                    )
                for ordinal, event in enumerate(version["events"]):
                    conn.execute(
                        "INSERT INTO event(version_id,event_id,seq,event_type,scope,"
                        "affected_stable_ids_json,status,payload_json,created_at_s)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            version_id, event["event_id"], ordinal, event["event_type"],
                            event["scope"], json.dumps(event.get("affected", []), ensure_ascii=False),
                            event["status"], json.dumps(event.get("payload", {}), ensure_ascii=False, default=str),
                            event["created_at_s"],
                        ),
                    )
                for decision in version.get("decisions", []):
                    conn.execute(
                        "INSERT INTO decision(version_id,decision_id,scope_stable_id,"
                        "chosen_candidate_id,action,rationale,rejected_candidates_json,"
                        "created_at_s) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            version_id, decision["decision_id"], decision["scope_stable_id"],
                            decision.get("chosen_candidate_id"), decision["action"],
                            decision.get("rationale", ""),
                            json.dumps(decision.get("rejected_candidates", []), ensure_ascii=False),
                            decision["created_at_s"],
                        ),
                    )
            if bundle.get("current_version_id"):
                conn.execute(
                    "INSERT INTO current_pointer(dataset_id,version_id) VALUES(?,?)",
                    (dataset_id, bundle["current_version_id"]),
                )
        return self.current_version_id(dataset_id)  # type: ignore[return-value]

    # --------------------------------------------------------------- failures
    def crash_during_commit(
        self, dataset_id: str, label: str, sources: Dict[str, Any], author: str = "chaos"
    ) -> str:
        """Simulate a process killed mid-write, *before* pointer switching.

        Leaves rows in the database but no current pointer move; on reopen
        ``_recover_pending`` marks the partial version abandoned.
        """
        version_id = "ver_pending_" + uuid.uuid4().hex[:8]
        seq = self.next_seq(dataset_id)
        manifest = {"label": label, "partial": True}
        conn = self._conn
        conn.execute(
            "INSERT INTO version(version_id,dataset_id,seq,parent_version_id,"
            "edit_base_version_id,label,status,author,created_at_s,"
            "manifest_json,manifest_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                version_id, dataset_id, seq, None, None, label, "building", author,
                time.time(), json.dumps(manifest), fingerprint(manifest),
            ),
        )
        _write_sources(conn, version_id, sources)
        # Deliberately do NOT commit the status flip / pointer move.
        conn.commit()
        return version_id

    def reopen(self) -> "VersionStore":
        self.close()
        return VersionStore(self.db_path)


# ------------------------------------------------------------- legacy migrate
def migrate_legacy_database(path: str) -> None:
    """Upgrade a legacy (schema v1, flat per-ping) database to separated tables.

    The legacy layout stored a single ``legacy_ping`` row per ping with beam
    velocity, correlation, amplitude and flags packed into one JSON blob. The
    migration creates the separated tables, records a ``legacy_migration``
    machine-readable event, and keeps the old blob alongside for traceability.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is not None and int(row["value"]) >= SCHEMA_VERSION:
        conn.close()
        return

    legacy_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='legacy_ping'"
    ).fetchone()
    migrated = False
    if legacy_exists:
        legacy_rows = conn.execute("SELECT * FROM legacy_ping ORDER BY ping_time_s").fetchall()
        if legacy_rows:
            dataset_id = "legacy"
            version_id = "ver_legacy_migrated"
            conn.execute("INSERT OR IGNORE INTO dataset(dataset_id,name,created_at_s) VALUES(?,?,?)",
                         (dataset_id, "legacy imported dataset", time.time()))
            sources = {"pings": [], "attitudes": [], "transducer": {
                "preset": "rdi-janus", "beam_angle_deg": 20.0,
                "min_valid_distance_m": 0.0, "max_attitude_gap_s": 1.0,
                "transform_order": "RDI_HPR",
            }, "cells": [], "declinations": [], "mountings": [], "bottom_tracks": []}
            for grow in legacy_rows:
                blob = json.loads(grow["data_json"])
                sources["pings"].append({
                    "ping_id": str(grow["ping_id"]),
                    "time_s": float(grow["ping_time_s"]),
                    "beams": blob["beams"],
                })
                if "attitude" in blob:
                    att = blob["attitude"]
                    sources["attitudes"].append({
                        "time_s": float(grow["ping_time_s"]),
                        "heading_deg": att.get("heading_deg"),
                        "pitch_deg": att.get("pitch_deg"),
                        "roll_deg": att.get("roll_deg"),
                    })
            manifest = {"label": "legacy migration", "migrated_rows": len(legacy_rows)}
            conn.execute(
                "INSERT INTO version(version_id,dataset_id,seq,parent_version_id,"
                "edit_base_version_id,label,status,author,created_at_s,"
                "manifest_json,manifest_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    version_id, dataset_id, 1, None, None, "legacy migration",
                    "ready", "migrator", time.time(), json.dumps(manifest, sort_keys=True),
                    fingerprint(manifest),
                ),
            )
            _write_sources(conn, version_id, sources)
            conn.execute(
                "INSERT INTO event(version_id,event_id,seq,event_type,scope,"
                "affected_stable_ids_json,status,payload_json,created_at_s)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    version_id, "evt_legacy_migration", 0, "legacy_migration",
                    "dataset", json.dumps(["*"]), "applied",
                    json.dumps({"from_schema": 1, "to_schema": 2, "rows": len(legacy_rows)}),
                    time.time(),
                ),
            )
            conn.execute(
                "INSERT INTO current_pointer(dataset_id,version_id) VALUES(?,?)",
                (dataset_id, version_id),
            )
            migrated = True
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    conn.close()
    return migrated
