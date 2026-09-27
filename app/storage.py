"""SQLite storage: separate source records, immutable versioned derivations.

Guarantees
----------
* Beam velocity / correlation / amplitude / bad-beam / attitude /
  declination / installation are SEPARATE tables. Beam observations are
  keyed by (ping, channel ['water'|'bottom'], cell_idx, beam).
* A derivation is written in two phases. Phase A commits rows with
  status='pending'; phase B (separate transaction) marks it 'complete' and
  moves the current pointer. A crash between the phases leaves the pointer on
  the old complete version; stale pending derivations are reaped on restart.
  A reader therefore sees either the old version or a complete new one.
* Every identity is content/key based (app/ident.py), never a rowid, so
  export -> delete -> import preserves ids, order and provenance.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from .ident import fingerprint

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS pings (
  ping_id TEXT PRIMARY KEY, ord INTEGER NOT NULL, t REAL NOT NULL,
  case_id TEXT, raw_ref TEXT);
CREATE TABLE IF NOT EXISTS beam_vel (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL,
  channel TEXT NOT NULL, cell_idx INTEGER, beam INTEGER NOT NULL,
  raw_value TEXT, unit TEXT, geom_id TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS correlation (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL,
  channel TEXT NOT NULL, cell_idx INTEGER, beam INTEGER NOT NULL,
  raw_value TEXT, unit TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS amplitude (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL,
  channel TEXT NOT NULL, cell_idx INTEGER, beam INTEGER NOT NULL,
  raw_value TEXT, unit TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS badbeam (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL,
  channel TEXT NOT NULL, cell_idx INTEGER, beam INTEGER NOT NULL,
  flag INTEGER NOT NULL, raw_value TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS attitude (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL,
  heading TEXT, pitch TEXT, roll TEXT,
  h_unit TEXT, p_unit TEXT, r_unit TEXT,
  heading_basis TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS declination (
  id TEXT, sver TEXT NOT NULL, ord INTEGER NOT NULL,
  value TEXT, unit TEXT, version TEXT NOT NULL, label TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS installation (
  id TEXT, sver TEXT NOT NULL, ord INTEGER NOT NULL,
  heading_mount TEXT, pitch_offset TEXT, roll_offset TEXT,
  unit TEXT, version TEXT NOT NULL, label TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS geom (
  id TEXT, sver TEXT NOT NULL, ord INTEGER NOT NULL,
  beam_angle TEXT, beam_angle_unit TEXT, n_beams INTEGER,
  version TEXT NOT NULL, convention TEXT, label TEXT, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS cells (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL, cell_idx INTEGER NOT NULL,
  depth_raw TEXT, depth_unit TEXT, configured INTEGER, PRIMARY KEY (id));
CREATE TABLE IF NOT EXISTS bottom_range (
  id TEXT, sver TEXT NOT NULL, ping_id TEXT NOT NULL, beam INTEGER NOT NULL,
  raw_value TEXT, unit TEXT, PRIMARY KEY (id));

CREATE TABLE IF NOT EXISTS derivations (
  drv_id TEXT PRIMARY KEY, case_id TEXT, sver TEXT NOT NULL,
  rule_version TEXT, geom_version TEXT, decl_version TEXT, install_version TEXT,
  transform_order TEXT, beam_geometry TEXT, fp TEXT NOT NULL,
  status TEXT NOT NULL, parent_drv TEXT, base_event_id TEXT,
  created_seq INTEGER, transform_order_alt INTEGER);
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT, drv_id TEXT NOT NULL, ntype TEXT NOT NULL,
  ping_id TEXT, channel TEXT, cell_idx INTEGER, frame TEXT,
  status TEXT NOT NULL, fp TEXT, reused_from TEXT, payload_json TEXT,
  PRIMARY KEY (node_id));
CREATE TABLE IF NOT EXISTS edges (
  drv_id TEXT NOT NULL, from_node TEXT NOT NULL, to_node TEXT NOT NULL,
  kind TEXT, PRIMARY KEY (drv_id, from_node, to_node, kind));
CREATE TABLE IF NOT EXISTS dvalues (
  val_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, name TEXT NOT NULL,
  value_json TEXT, datum_json TEXT, raw_repr TEXT, diagnostics_json TEXT);
CREATE TABLE IF NOT EXISTS candidates (
  cand_id TEXT PRIMARY KEY, drv_id TEXT NOT NULL, ping_id TEXT NOT NULL,
  channel TEXT NOT NULL, cell_idx INTEGER, kind TEXT NOT NULL, method TEXT NOT NULL,
  beams TEXT, instrument_json TEXT, ship_json TEXT, earth_json TEXT,
  error_velocity REAL, resolution REAL, selected INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL, diagnostics_json TEXT);
CREATE TABLE IF NOT EXISTS branches (
  branch_id TEXT PRIMARY KEY, label TEXT, head_drv TEXT, head_event TEXT,
  parent_branch TEXT);
CREATE TABLE IF NOT EXISTS events (
  evt_id TEXT PRIMARY KEY, case_id TEXT, branch_id TEXT, seq INTEGER NOT NULL,
  code TEXT NOT NULL, kind TEXT NOT NULL, ts REAL,
  drv_id TEXT, parent_event_id TEXT, payload_json TEXT, basis_json TEXT);
CREATE TABLE IF NOT EXISTS pointers (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS locks (
  branch_id TEXT PRIMARY KEY, editor TEXT NOT NULL, base_drv TEXT NOT NULL,
  ts REAL NOT NULL);
"""

SOURCE_TABLES = ["beam_vel", "correlation", "amplitude", "badbeam",
                 "attitude", "declination", "installation", "geom",
                 "cells", "bottom_range"]


def connect(path: str) -> sqlite3.Connection:
    fresh = not os.path.exists(path)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    if fresh:
        con.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        con.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
        version = SCHEMA_VERSION
    _migrate(con, version)
    con.commit()


def _migrate(con: sqlite3.Connection, version: int) -> None:
    if version < 2:
        con.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)


def get_meta(con, k, default=None):
    r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def set_meta(con, k, v) -> None:
    con.execute("INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def get_pointer(con, k, default=None):
    r = con.execute("SELECT v FROM pointers WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def set_pointer(con, k, v) -> None:
    con.execute("INSERT INTO pointers(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def reap_pending(con) -> int:
    rows = con.execute(
        "SELECT drv_id FROM derivations WHERE status!='complete'").fetchall()
    n = 0
    for r in rows:
        drv = r[0]
        hit = con.execute("SELECT 1 FROM branches WHERE head_drv=?", (drv,)).fetchone()
        if hit:
            continue
        con.execute("DELETE FROM nodes WHERE drv_id=?", (drv,))
        con.execute("DELETE FROM edges WHERE drv_id=?", (drv,))
        con.execute("DELETE FROM candidates WHERE drv_id=?", (drv,))
        con.execute("DELETE FROM derivations WHERE drv_id=?", (drv,))
        con.execute("DELETE FROM locks WHERE base_drv=?", (drv,))
        n += 1
    con.commit()
    return n


def next_seq(con) -> int:
    return int(con.execute(
        "SELECT COALESCE(MAX(seq),0)+1 FROM events").fetchone()[0])


def insert_event(con, *, evt_id, case_id, branch_id, seq, code, kind,
                 drv_id=None, parent_event_id=None, payload=None, basis=None,
                 ts=None) -> None:
    con.execute(
        "INSERT INTO events(evt_id,case_id,branch_id,seq,code,kind,ts,drv_id,"
        "parent_event_id,payload_json,basis_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (evt_id, case_id, branch_id, seq, code, kind, ts or time.time(),
         drv_id, parent_event_id,
         json.dumps(payload or {}, ensure_ascii=False),
         json.dumps(basis or {}, ensure_ascii=False)))


def all_events(con, branch_id=None):
    q = "SELECT * FROM events"
    args: list[Any] = []
    if branch_id:
        q += " WHERE branch_id=?"
        args.append(branch_id)
    q += " ORDER BY seq"
    return [dict(r) for r in con.execute(q, args)]


EXPORT_TABLES = ["meta", "pings"] + SOURCE_TABLES + [
    "derivations", "nodes", "edges", "dvalues", "candidates",
    "branches", "events", "pointers", "locks"]


def export_db(con) -> dict:
    payload: dict[str, list[dict]] = {}
    for t in EXPORT_TABLES:
        payload[t] = [dict(r) for r in con.execute("SELECT * FROM " + t)]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {"format": "adcp-evidence", "schema_version": SCHEMA_VERSION,
            "checksum": fingerprint([blob]), "data": payload}


def import_db(con, bundle: dict, *, verify: bool = True) -> None:
    data = bundle["data"]
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if verify and fingerprint([blob]) != bundle["checksum"]:
        raise ValueError("checksum_mismatch")
    if con.execute("SELECT COUNT(*) FROM derivations").fetchone()[0]:
        raise ValueError("import_target_not_empty")
    con.execute("PRAGMA foreign_keys=OFF")
    for t in EXPORT_TABLES:
        rows = data.get(t, [])
        if not rows:
            continue
        cols = list(rows[0].keys())
        existing = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        cols = [c for c in cols if c in existing]
        ph = ",".join("?" for _ in cols)
        con.executemany(
            f"INSERT INTO {t}({','.join(cols)}) VALUES({ph})",
            [[r[c] for c in cols] for r in rows])
    con.execute("PRAGMA user_version=%d" % bundle.get("schema_version", SCHEMA_VERSION))
    con.commit()
    _migrate(con, con.execute("PRAGMA user_version").fetchone()[0])
    con.commit()
