"""Versioned evidence store on SQLite.

Source facts (beam velocity, correlation, amplitude, bad-beam flags,
heading/pitch/roll, magnetic declination, mounting angles, geometry/order
configuration) are recorded SEPARATELY. Each derivation run writes a new
immutable revision containing a full snapshot; the current pointer moves only
after every row of the revision committed, so a process killed mid-write
leaves either the old revision or a complete new one.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
  rev INTEGER NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  sid TEXT NOT NULL,
  source_ref TEXT,
  raw_unit TEXT,
  raw_repr TEXT,
  value TEXT NOT NULL,
  version INTEGER NOT NULL,
  replaces_sid TEXT,
  PRIMARY KEY (rev, kind, key)
);
CREATE TABLE IF NOT EXISTS objects (
  rev INTEGER NOT NULL,
  otype TEXT NOT NULL,
  key TEXT NOT NULL,
  sid TEXT NOT NULL,
  family TEXT,
  status TEXT NOT NULL,
  payload TEXT NOT NULL,
  fingerprint TEXT,
  adopted INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (rev, otype, key)
);
CREATE TABLE IF NOT EXISTS edges (
  rev INTEGER NOT NULL,
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  PRIMARY KEY (rev, src, dst)
);
CREATE TABLE IF NOT EXISTS events (
  rev INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  summary TEXT NOT NULL,
  data TEXT NOT NULL,
  PRIMARY KEY (rev, seq)
);
CREATE TABLE IF NOT EXISTS conflicts (
  rev INTEGER NOT NULL,
  cid TEXT NOT NULL,
  base_rev INTEGER NOT NULL,
  branch_sid TEXT NOT NULL,
  incoming_sid TEXT NOT NULL,
  key TEXT NOT NULL,
  resolved_rev INTEGER,
  resolution TEXT,
  PRIMARY KEY (rev, cid)
);
CREATE TABLE IF NOT EXISTS revisions (
  rev INTEGER PRIMARY KEY,
  parent_rev INTEGER,
  created_at REAL NOT NULL,
  label TEXT,
  change_set TEXT NOT NULL,
  complete INTEGER NOT NULL DEFAULT 0
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version','1')")
    migrate(conn)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key,value) VALUES('current_rev','-1')")
    # Crash recovery: a revision left incomplete (pointer never moved) is
    # removed, so readers only ever see a fully written revision.
    cur = conn.execute("SELECT value FROM meta WHERE key='current_rev'")
    current = int(cur.fetchone()[0])
    conn.execute("DELETE FROM revisions WHERE complete=0 AND rev>?", (current,))
    for table in ("facts", "objects", "edges", "events", "conflicts"):
        conn.execute(
            f"DELETE FROM {table} WHERE rev NOT IN (SELECT rev FROM revisions)")
    conn.commit()


def migrate(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    version = int(row[0]) if row else 1
    if version < 2:
        # v2 adds explicit family/adopted columns and the revisions ledger.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(objects)")}
        if "family" not in cols:
            conn.execute("ALTER TABLE objects ADD COLUMN family TEXT")
        if "adopted" not in cols:
            conn.execute("ALTER TABLE objects ADD COLUMN adopted INTEGER DEFAULT 0")
        conn.execute(
            "UPDATE meta SET value='2' WHERE key='schema_version'")
        version = 2
    conn.commit()
    return version


def current_rev(conn: sqlite3.Connection) -> int:
    return int(conn.execute(
        "SELECT value FROM meta WHERE key='current_rev'").fetchone()[0])


def all_revisions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT rev,parent_rev,created_at,label,change_set,complete "
        "FROM revisions ORDER BY rev").fetchall()
    return [dict(r) for r in rows]


def events(conn: sqlite3.Connection, rev: int | None = None) -> list[dict]:
    rev = current_rev(conn) if rev is None else rev
    rows = conn.execute(
        "SELECT seq,id,event_type,actor,summary,data FROM events "
        "WHERE rev=? ORDER BY seq", (rev,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d["data"])
        out.append(d)
    return out


def conflicts(conn: sqlite3.Connection, rev: int | None = None) -> list[dict]:
    rev = current_rev(conn) if rev is None else rev
    return [dict(r) for r in conn.execute(
        "SELECT cid,base_rev,branch_sid,incoming_sid,key,resolved_rev,"
        "resolution FROM conflicts WHERE rev=? ORDER BY cid", (rev,))]


def snapshot(conn: sqlite3.Connection, rev: int | None = None) -> dict:
    """Full materialized state of one revision."""
    rev = current_rev(conn) if rev is None else rev
    facts: dict[str, dict[str, dict]] = {}
    for r in conn.execute(
            "SELECT kind,key,sid,source_ref,raw_unit,raw_repr,value,version,"
            "replaces_sid FROM facts WHERE rev=?", (rev,)):
        d = dict(r)
        d["value"] = json.loads(d["value"])
        facts.setdefault(d.pop("kind"), {})[d.pop("key")] = d
    objects: dict[str, dict[str, dict]] = {}
    for r in conn.execute(
            "SELECT otype,key,sid,family,status,payload,fingerprint,adopted "
            "FROM objects WHERE rev=?", (rev,)):
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["adopted"] = bool(d["adopted"])
        objects.setdefault(d.pop("otype"), {})[d.pop("key")] = d
    edge_rows = conn.execute(
        "SELECT src,dst FROM edges WHERE rev=?", (rev,)).fetchall()
    return {
        "rev": rev,
        "facts": facts,
        "objects": objects,
        "edges": [(r["src"], r["dst"]) for r in edge_rows],
        "events": events(conn, rev),
        "conflicts": conflicts(conn, rev),
    }


def get_fact(conn: sqlite3.Connection, kind: str, key: str,
             rev: int | None = None) -> dict | None:
    rev = current_rev(conn) if rev is None else rev
    r = conn.execute(
        "SELECT kind,key,sid,source_ref,raw_unit,raw_repr,value,version,"
        "replaces_sid FROM facts WHERE rev=? AND kind=? AND key=?",
        (rev, kind, key)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["value"] = json.loads(d["value"])
    return d


def list_facts(conn: sqlite3.Connection, kind: str,
               rev: int | None = None) -> list[dict]:
    rev = current_rev(conn) if rev is None else rev
    rows = conn.execute(
        "SELECT kind,key,sid,source_ref,raw_unit,raw_repr,value,version,"
        "replaces_sid FROM facts WHERE rev=? AND kind=? ORDER BY key",
        (rev, kind)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["value"] = json.loads(d["value"])
        out.append(d)
    return out


def get_object(conn: sqlite3.Connection, otype: str, key: str,
               rev: int | None = None) -> dict | None:
    rev = current_rev(conn) if rev is None else rev
    r = conn.execute(
        "SELECT otype,key,sid,family,status,payload,fingerprint,adopted "
        "FROM objects WHERE rev=? AND otype=? AND key=?",
        (rev, otype, key)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["payload"] = json.loads(d["payload"])
    d["adopted"] = bool(d["adopted"])
    return d


class ConcurrentEdit(Exception):
    """Raised when a revision is committed against a stale parent revision."""


def _dump_state(conn: sqlite3.Connection, rev: int, state: dict) -> None:
    for kind, items in state.get("facts", {}).items():
        for key, f in items.items():
            conn.execute(
                "INSERT INTO facts(rev,kind,key,sid,source_ref,raw_unit,"
                "raw_repr,value,version,replaces_sid) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (rev, kind, key, f["sid"], f.get("source_ref"),
                 f.get("raw_unit"), f.get("raw_repr"),
                 json.dumps(f["value"], ensure_ascii=False),
                 int(f.get("version", 1)), f.get("replaces_sid")))
    for otype, items in state.get("objects", {}).items():
        for key, o in items.items():
            conn.execute(
                "INSERT INTO objects(rev,otype,key,sid,family,status,payload,"
                "fingerprint,adopted) VALUES(?,?,?,?,?,?,?,?,?)",
                (rev, otype, key, o["sid"], o.get("family"), o["status"],
                 json.dumps(o["payload"], ensure_ascii=False),
                 o.get("fingerprint"), 1 if o.get("adopted") else 0))
    for src, dst in state.get("edges", []):
        conn.execute("INSERT INTO edges(rev,src,dst) VALUES(?,?,?)",
                     (rev, src, dst))
    for i, ev in enumerate(state.get("events", [])):
        conn.execute(
            "INSERT INTO events(rev,seq,id,event_type,actor,summary,data)"
            " VALUES(?,?,?,?,?,?,?)",
            (rev, i, ev["id"], ev["event_type"], ev.get("actor", "system"),
             ev["summary"], json.dumps(ev.get("data", {}), ensure_ascii=False)))
    for c in state.get("conflicts", []):
        conn.execute(
            "INSERT INTO conflicts(rev,cid,base_rev,branch_sid,incoming_sid,"
            "key,resolved_rev,resolution) VALUES(?,?,?,?,?,?,?,?)",
            (rev, c["cid"], c["base_rev"], c["branch_sid"], c["incoming_sid"],
             c["key"], c.get("resolved_rev"), c.get("resolution")))


def commit_revision(conn: sqlite3.Connection, label: str, state: dict,
                    change_set: dict | None = None,
                    base_rev: int | None = None) -> int:
    """Atomically write a complete new revision and move the current pointer.

    `state` is the full snapshot after the edit (facts/objects/edges/events/
    conflicts). The pointer is advanced only after every row is inserted, in
    one transaction, so a crash exposes the previous revision or this complete
    one — never a partial mix.
    """
    parent = current_rev(conn)
    if base_rev is not None and base_rev != parent:
        raise ConcurrentEdit(
            f"edit based on rev {base_rev} but current is rev {parent}")
    next_rev = (parent + 1) if parent >= 0 else 0
    if conn.execute("SELECT 1 FROM revisions WHERE rev=?", (next_rev,)).fetchone():
        next_rev = int(conn.execute(
            "SELECT COALESCE(MAX(rev),-1)+1 FROM revisions").fetchone()[0])
    with conn:
        conn.execute(
            "INSERT INTO revisions(rev,parent_rev,created_at,label,change_set,"
            "complete) VALUES(?,?,?,?,?,0)",
            (next_rev, parent if parent >= 0 else None, time.time(), label,
             json.dumps(change_set or {}, ensure_ascii=False)))
        _dump_state(conn, next_rev, state)
        conn.execute("UPDATE revisions SET complete=1 WHERE rev=?", (next_rev,))
        conn.execute("UPDATE meta SET value=? WHERE key='current_rev'",
                     (str(next_rev),))
    return next_rev


def register_conflict(conn: sqlite3.Connection, cid: str, base_rev: int,
                      branch_sid: str, incoming_sid: str, key: str,
                      state: dict, label: str) -> int:
    """Persist a concurrent-edit conflict as a resolvable record.

    Both candidate values are retained (branch and incoming). The revision is
    written without silently choosing one.
    """
    state.setdefault("conflicts", []).append({
        "cid": cid, "base_rev": base_rev, "branch_sid": branch_sid,
        "incoming_sid": incoming_sid, "key": key,
        "resolved_rev": None, "resolution": None,
    })
    return commit_revision(conn, label, state,
                           {"conflict": cid, "key": key})


def resolve_conflict(conn: sqlite3.Connection, cid: str, resolution: str,
                     mutate_state, label: str) -> int:
    """Resolve a conflict. `mutate_state(state)` applies the chosen branch;
    the rejected candidate sid stays referenced in the conflict record."""
    state = snapshot(conn)
    found = None
    for c in state.get("conflicts", []):
        if c["cid"] == cid and c.get("resolved_rev") is None:
            found = c
    if found is None:
        raise KeyError(f"unresolved conflict {cid!r} not found")
    if resolution not in ("branch", "incoming"):
        raise ValueError("resolution must be 'branch' or 'incoming'")
    mutate_state(state, found, resolution)
    found["resolved_rev"] = current_rev(conn) + 1
    found["resolution"] = resolution
    return commit_revision(conn, label, state,
                           {"resolve_conflict": cid, "resolution": resolution})


def export_all(conn: sqlite3.Connection) -> dict:
    """Export every revision (full ledger). Stable sids ride along verbatim."""
    return {
        "format": "adcp-evidence-db",
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "current_rev": current_rev(conn),
        "revisions": [
            {
                "rev": r["rev"], "parent_rev": r["parent_rev"],
                "created_at": r["created_at"], "label": r["label"],
                "change_set": json.loads(r["change_set"]),
                "complete": bool(r["complete"]),
                "snapshot": snapshot(conn, r["rev"]),
            }
            for r in all_revisions(conn) if r["complete"]
        ],
    }


def import_all(conn: sqlite3.Connection, bundle: dict,
               replace: bool = True) -> int:
    """Import a ledger, replaying revisions in order. Sids, revision order and
    source/derived relationships are preserved exactly.

    With replace=True the database is wiped first (the documented
    export -> delete DB -> import round trip).
    """
    if bundle.get("format") != "adcp-evidence-db":
        raise ValueError("unsupported bundle format")
    with conn:
        if replace:
            for table in ("facts", "objects", "edges", "events",
                          "conflicts", "revisions", "meta"):
                conn.execute(f"DELETE FROM {table}")
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),))
        conn.execute("INSERT INTO meta(key,value) VALUES('current_rev','-1')")
        last = -1
        for r in bundle["revisions"]:
            rev = int(r["rev"])
            conn.execute(
                "INSERT INTO revisions(rev,parent_rev,created_at,label,"
                "change_set,complete) VALUES(?,?,?,?,?,1)",
                (rev, r.get("parent_rev"), r["created_at"], r.get("label"),
                 json.dumps(r.get("change_set", {}), ensure_ascii=False)))
            _dump_state(conn, rev, r["snapshot"])
            last = rev
        conn.execute("UPDATE meta SET value=? WHERE key='current_rev'",
                     (str(bundle.get("current_rev", last)),))
    return current_rev(conn)


def explain_edges(conn: sqlite3.Connection, rev: int | None = None,
                  dst_prefix: str | None = None) -> list[dict]:
    rev = current_rev(conn) if rev is None else rev
    q = "SELECT src,dst FROM edges WHERE rev=?"
    args: list[Any] = [rev]
    if dst_prefix:
        q += " AND dst LIKE ?"
        args.append(dst_prefix + "%")
    return [dict(r) for r in conn.execute(q + " ORDER BY dst,src", args)]
