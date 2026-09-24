"""Versioned SQLite storage with atomic pointer switches and an event journal.

Content model
-------------
Each *source* (a logical direct measurement such as one ping's beam
velocities or the magnetic declination) has a natural key. Editing it adds a
new *content version*; old content is never overwritten. Every derivation is
also an immutable content version. A ``current`` pointer names the accepted
content version per kind/scope. Batch writes are committed in one SQLite
transaction: either the whole batch and its pointer switch are visible, or
none of it is (a crash mid-write leaves the old pointer intact).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  uid TEXT NOT NULL,
  kind TEXT NOT NULL,
  natural_key TEXT NOT NULL,
  content_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  parent_content_id TEXT,
  payload TEXT NOT NULL,
  PRIMARY KEY (uid, content_id)
);
CREATE TABLE IF NOT EXISTS derivations (
  content_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  parent_content_id TEXT,
  status TEXT NOT NULL,
  rule_set TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_event_seq INTEGER
);
CREATE TABLE IF NOT EXISTS derivation_candidates (
  derivation_content_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  candidate_fingerprint TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (derivation_content_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS derivation_inputs (
  derivation_content_id TEXT NOT NULL,
  input_kind TEXT NOT NULL,
  input_natural_key TEXT NOT NULL,
  input_content_id TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  PRIMARY KEY (derivation_content_id, input_kind, input_natural_key)
);
CREATE TABLE IF NOT EXISTS diagnostics (
  owner_content_id TEXT NOT NULL,
  code TEXT NOT NULL,
  message TEXT NOT NULL,
  details TEXT NOT NULL,
  PRIMARY KEY (owner_content_id, code, message)
);
CREATE TABLE IF NOT EXISTS pointers (
  kind TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  branch_id TEXT NOT NULL DEFAULT 'main',
  content_id TEXT NOT NULL,
  PRIMARY KEY (kind, scope_key, branch_id)
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY,
  event_type TEXT NOT NULL,
  branch_id TEXT NOT NULL DEFAULT 'main',
  content_id TEXT,
  scope_key TEXT,
  payload TEXT NOT NULL,
  base_seq INTEGER,
  note TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_key ON sources(uid, seq);
CREATE INDEX IF NOT EXISTS idx_deriv_scope ON derivations(kind, scope_key);
CREATE INDEX IF NOT EXISTS idx_inputs_lookup
  ON derivation_inputs(input_kind, input_content_id);
CREATE INDEX IF NOT EXISTS idx_pointers_content ON pointers(content_id);
"""


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def content_hash(prefix: str, kind: str, scope_key: str,
                 payload: Any, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(prefix.encode())
    h.update(b"\0")
    h.update(canonical((kind, scope_key, payload, extra)).encode("utf-8"))
    return f"{prefix}_{h.hexdigest()[:16]}"


def source_uid(kind: str, natural_key: str) -> str:
    h = hashlib.sha256(canonical((kind, natural_key)).encode("utf-8")).hexdigest()[:12]
    return f"src_{h}"


def branch_id(name: str, seq: int) -> str:
    h = hashlib.sha256(f"{name}:{seq}".encode("utf-8")).hexdigest()[:10]
    return f"br_{h}"


class Store:
    def __init__(self, path: str = ":memory:", crash_hook: Optional[callable] = None):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.crash_hook = crash_hook
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- raw edit

    def head_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0) AS m FROM events").fetchone()
        return int(row["m"])

    def current_source_content(self, kind: str, natural_key: str,
                                branch_id: str = "main") -> Optional[str]:
        uid = source_uid(kind, natural_key)
        row = self.conn.execute(
            "SELECT content_id FROM pointers WHERE kind=? AND scope_key=? AND branch_id=?",
            (kind, uid, branch_id)).fetchone()
        if row:
            return row["content_id"]
        row = self.conn.execute(
            "SELECT content_id FROM sources WHERE uid=? ORDER BY seq DESC LIMIT 1",
            (uid,)).fetchone()
        return row["content_id"] if row else None

    def put_source(self, kind: str, natural_key: str, payload: dict,
                   note: str = "source edit", base_seq: Optional[int] = None,
                   expect_content_id: Optional[str] = None) -> dict:
        """Append a new source content version and switch its pointer atomically."""
        uid = source_uid(kind, natural_key)
        with self.conn:
            current = self.current_source_content(kind, natural_key)
            if expect_content_id is not None and current is not None \
                    and current != expect_content_id and base_seq is not None:
                conflict = self._record_conflict(kind, uid, note, base_seq, current)
                return {"status": "conflict", **conflict}
            seq_row = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS s FROM sources WHERE uid=?").fetchone()
            version_seq = int(seq_row["s"])
            cid = content_hash("srcver", kind, natural_key, payload)
            existing = self.conn.execute(
                "SELECT 1 FROM sources WHERE uid=? AND content_id=?",
                (uid, cid)).fetchone()
            if existing:
                return {"status": "noop", "content_id": cid, "uid": uid}
            self.conn.execute(
                "INSERT INTO sources(uid,kind,natural_key,content_id,seq,parent_content_id,payload)"
                " VALUES(?,?,?,?,?,?,?)",
                (uid, kind, natural_key, cid, version_seq, current, canonical(payload)))
            self.conn.execute(
                "INSERT INTO pointers(kind,scope_key,branch_id,content_id)"
                " VALUES(?,?,?,?) ON CONFLICT(kind,scope_key,branch_id)"
                " DO UPDATE SET content_id=excluded.content_id",
                (kind, uid, "main", cid))
            event_seq = self._append_event(
                "source_revised", "main", cid, uid,
                {"kind": kind, "natural_key": natural_key,
                 "parent_content_id": current}, base_seq, note)
        return {"status": "ok", "content_id": cid, "uid": uid, "event_seq": event_seq}

    def _record_conflict(self, kind: str, uid: str, note: str,
                         base_seq: int, current: str) -> dict:
        event_seq = self._append_event(
            "edit_conflict", "main", None, uid,
            {"kind": kind, "base_seq": base_seq,
             "observed_content_id": current}, None,
            f"conflict: {note}")
        return {"event_seq": event_seq, "kind": kind, "scope_key": uid,
                "observed_content_id": current}

    def resolve_conflict(self, kind: str, natural_key: str, payload: dict,
                         resolution: str, base_seq: int, note: str = "conflict resolved") -> dict:
        result = self.put_source(kind, natural_key, payload, note=note)
        if result.get("status") != "ok":
            return result
        self.append_event("conflict_resolved", "main",
                          result["content_id"], result["uid"],
                          {"resolution": resolution, "base_seq": base_seq},
                          note=f"{note} ({resolution})")
        return result

    # ----------------------------------------------------------- derivations

    def put_derivation(self, kind: str, scope_key: str, payload: dict,
                       status: str, rule_set: dict, fingerprint: str,
                       inputs: list[dict], candidates: list[dict],
                       diagnostics: list[dict], branch_id: str = "main",
                       parent_content_id: Optional[str] = None,
                       adopt: bool = True, note: str = "derivation") -> dict:
        cid = content_hash("der", kind, scope_key, payload,
                           canonical({"rule_set": rule_set,
                                      "fingerprint": fingerprint}))
        with self.conn:
            exists = self.conn.execute(
                "SELECT 1 FROM derivations WHERE content_id=?", (cid,)).fetchone()
            event_seq = None
            if not exists:
                if self.crash_hook is not None:
                    self.crash_hook("before_derivation_commit", cid)
                self.conn.execute(
                    "INSERT INTO derivations(content_id,kind,scope_key,branch_id,"
                    "parent_content_id,status,rule_set,fingerprint,payload)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, kind, scope_key, branch_id, parent_content_id, status,
                     canonical(rule_set), fingerprint, canonical(payload)))
                for item in inputs:
                    self.conn.execute(
                        "INSERT INTO derivation_inputs(derivation_content_id,"
                        "input_kind,input_natural_key,input_content_id,"
                        "input_fingerprint,rule_id) VALUES(?,?,?,?,?,?)",
                        (cid, item["kind"], item["natural_key"],
                         item["content_id"], item["fingerprint"], item["rule_id"]))
                for cand in candidates:
                    self.conn.execute(
                        "INSERT INTO derivation_candidates(derivation_content_id,"
                        "candidate_id,rule_id,candidate_fingerprint,payload)"
                        " VALUES(?,?,?,?,?)",
                        (cid, cand["candidate_id"], cand["rule_id"],
                         cand["fingerprint"], canonical(cand["payload"])))
                for diag in diagnostics:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO diagnostics(owner_content_id,code,message,details)"
                        " VALUES(?,?,?,?)",
                        (cid, diag["code"], diag["message"], canonical(diag.get("details", {}))))
                event_seq = self._append_event(
                    "derivation_created", branch_id, cid, scope_key,
                    {"kind": kind, "status": status, "adopt": adopt,
                     "candidate_count": len(candidates)}, note=note)
            if adopt:
                self.conn.execute(
                    "INSERT INTO pointers(kind,scope_key,branch_id,content_id)"
                    " VALUES(?,?,?,?) ON CONFLICT(kind,scope_key,branch_id)"
                    " DO UPDATE SET content_id=excluded.content_id",
                    (kind, scope_key, branch_id, cid))
                if event_seq is None:
                    event_seq = self._append_event(
                        "pointer_switched", branch_id, cid, scope_key,
                        {"kind": kind}, note=f"adopt {kind}")
            elif event_seq is None:
                event_seq = self.head_seq()
        return {"status": "ok", "content_id": cid, "event_seq": event_seq}

    # --------------------------------------------------------------- events

    def _append_event(self, event_type: str, branch_id: str,
                      content_id: Optional[str], scope_key: Optional[str],
                      payload: dict, base_seq: Optional[int] = None,
                      note: str = "") -> int:
        seq = self.head_seq() + 1
        self.conn.execute(
            "INSERT INTO events(seq,event_type,branch_id,content_id,scope_key,payload,base_seq,note)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (seq, event_type, branch_id, content_id, scope_key,
             canonical(payload), base_seq, note))
        return seq

    def append_event(self, event_type: str, branch_id: str,
                     content_id: Optional[str], scope_key: Optional[str],
                     payload: dict, base_seq: Optional[int] = None,
                     note: str = "") -> int:
        with self.conn:
            return self._append_event(event_type, branch_id, content_id,
                                      scope_key, payload, base_seq, note)

    def list_events(self, upto_seq: Optional[int] = None) -> list[dict]:
        sql = ("SELECT seq,event_type,branch_id,content_id,scope_key,payload,"
               "base_seq,note FROM events")
        params: tuple = ()
        if upto_seq is not None:
            sql += " WHERE seq<=?"
            params = (upto_seq,)
        sql += " ORDER BY seq"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._event_dict(r) for r in rows]

    def event(self, seq: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        return self._event_dict(row) if row else None

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict:
        return {"seq": row["seq"], "event_type": row["event_type"],
                "branch_id": row["branch_id"], "content_id": row["content_id"],
                "scope_key": row["scope_key"],
                "payload": json.loads(row["payload"]),
                "base_seq": row["base_seq"], "note": row["note"]}

    # ---------------------------------------------------------------- reads

    def get_source(self, content_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT s.* FROM sources s WHERE s.content_id=?",
            (content_id,)).fetchone()
        if not row:
            return None
        return {"uid": row["uid"], "kind": row["kind"],
                "natural_key": row["natural_key"], "content_id": row["content_id"],
                "seq": row["seq"], "parent_content_id": row["parent_content_id"],
                "payload": json.loads(row["payload"])}

    def get_derivation(self, content_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM derivations WHERE content_id=?",
            (content_id,)).fetchone()
        if not row:
            return None
        out = {"content_id": row["content_id"], "kind": row["kind"],
               "scope_key": row["scope_key"], "branch_id": row["branch_id"],
               "parent_content_id": row["parent_content_id"],
               "status": row["status"], "rule_set": json.loads(row["rule_set"]),
               "fingerprint": row["fingerprint"],
               "payload": json.loads(row["payload"]),
               "inputs": self.inputs_of(content_id),
               "candidates": self.candidates_of(content_id),
               "diagnostics": self.diagnostics_of(content_id)}
        return out

    def inputs_of(self, content_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM derivation_inputs WHERE derivation_content_id=?",
            (content_id,)).fetchall()
        return [{"kind": r["input_kind"], "natural_key": r["input_natural_key"],
                 "content_id": r["input_content_id"],
                 "fingerprint": r["input_fingerprint"],
                 "rule_id": r["rule_id"]} for r in rows]

    def candidates_of(self, content_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM derivation_candidates WHERE derivation_content_id=?",
            (content_id,)).fetchall()
        return [{"candidate_id": r["candidate_id"], "rule_id": r["rule_id"],
                 "fingerprint": r["candidate_fingerprint"],
                 "payload": json.loads(r["payload"])} for r in rows]

    def diagnostics_of(self, content_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM diagnostics WHERE owner_content_id=?",
            (content_id,)).fetchall()
        return [{"code": r["code"], "message": r["message"],
                 "details": json.loads(r["details"])} for r in rows]

    def pointer(self, kind: str, scope_key: str,
                branch_id: str = "main") -> Optional[str]:
        row = self.conn.execute(
            "SELECT content_id FROM pointers WHERE kind=? AND scope_key=? AND branch_id=?",
            (kind, scope_key, branch_id)).fetchone()
        return row["content_id"] if row else None

    def current_sources(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT s.* FROM pointers p JOIN sources s ON s.content_id=p.content_id"
            " WHERE p.branch_id='main' AND p.kind != 'derivation'"
        ).fetchall()
        # Pointers for sources store scope_key = uid.
        out = []
        rows = self.conn.execute(
            "SELECT p.scope_key AS uid, s.kind AS kind, s.natural_key AS natural_key,"
            " s.content_id AS content_id, s.payload AS payload"
            " FROM pointers p JOIN sources s ON s.content_id=p.content_id"
            " WHERE p.branch_id='main'").fetchall()
        for r in rows:
            out.append({"uid": r["uid"], "kind": r["kind"],
                        "natural_key": r["natural_key"],
                        "content_id": r["content_id"],
                        "payload": json.loads(r["payload"])})
        return out

    def dependents(self, input_content_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT derivation_content_id FROM derivation_inputs"
            " WHERE input_content_id=?", (input_content_id,)).fetchall()
        return [r["derivation_content_id"] for r in rows]

    def derivation_by_scope(self, kind: str, scope_key: str,
                            branch_id: str = "main") -> Optional[dict]:
        cid = self.pointer(kind, scope_key, branch_id)
        return self.get_derivation(cid) if cid else None

    def all_derivations(self) -> list[dict]:
        rows = self.conn.execute("SELECT content_id FROM derivations").fetchall()
        return [self.get_derivation(r["content_id"]) for r in rows]

    # ------------------------------------------------------------- as-of

    def state_at(self, seq: int, branch_id: str = "main") -> dict:
        """Replay events up to ``seq`` to recover pointers at that moment."""
        source_ptr: dict[tuple, str] = {}
        deriv_ptr: dict[tuple, str] = {}
        for event in self.list_events(upto_seq=seq):
            etype = event["event_type"]
            if etype == "source_revised":
                p = event["payload"]
                source_ptr[(p["kind"], p["natural_key"])] = event["content_id"]
            elif etype in ("derivation_created", "pointer_switched"):
                if event["branch_id"] != branch_id:
                    continue
                if etype == "derivation_created" and not event["payload"].get("adopt", True):
                    continue
                deriv_ptr[(event["payload"]["kind"], event["scope_key"])] = event["content_id"]
            elif etype in ("candidate_selected", "decision_accepted", "decision_rejected"):
                if event["branch_id"] != branch_id:
                    continue
                p = event["payload"]
                if event.get("content_id"):
                    deriv_ptr[(p["kind"], p["scope_key"])] = event["content_id"]
        return {"seq": seq, "source_pointers": source_ptr,
                "derivation_pointers": deriv_ptr}

    # ------------------------------------------------------------- decisions

    def select_candidate(self, kind: str, scope_key: str,
                         derivation_content_id: str, candidate_id: str,
                         decision: str = "accepted", note: str = "",
                         base_seq: Optional[int] = None) -> int:
        with self.conn:
            event_type = {"accepted": "decision_accepted",
                          "rejected": "decision_rejected"}[decision]
            return self._append_event(
                event_type, "main", derivation_content_id, scope_key,
                {"kind": kind, "scope_key": scope_key,
                 "candidate_id": candidate_id, "base_seq": base_seq},
                base_seq=base_seq, note=note or f"{decision} {candidate_id}")

    def create_branch(self, name: str, from_seq: int) -> str:
        bid = branch_id(name, from_seq)
        state = self.state_at(from_seq)
        with self.conn:
            for (kind, scope_key), cid in state["derivation_pointers"].items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO pointers(kind,scope_key,branch_id,content_id)"
                    " VALUES(?,?,?,?)", (kind, scope_key, bid, cid))
            self._append_event("branch_created", bid, None, None,
                               {"from_seq": from_seq, "name": name},
                               note=f"branch {name}")
        return bid

    # --------------------------------------------------------------- export

    def export_bundle(self) -> dict:
        sources = [dict(r) for r in self.conn.execute(
            "SELECT uid,kind,natural_key,content_id,seq,parent_content_id,payload"
            " FROM sources ORDER BY uid,seq").fetchall()]
        derivs = [dict(r) for r in self.conn.execute(
            "SELECT content_id,kind,scope_key,branch_id,parent_content_id,status,"
            "rule_set,fingerprint,payload,created_event_seq FROM derivations").fetchall()]
        cand = [dict(r) for r in self.conn.execute(
            "SELECT * FROM derivation_candidates").fetchall()]
        inputs = [dict(r) for r in self.conn.execute(
            "SELECT derivation_content_id,input_kind,input_natural_key,"
            "input_content_id,input_fingerprint,rule_id FROM derivation_inputs").fetchall()]
        diags = [dict(r) for r in self.conn.execute(
            "SELECT owner_content_id,code,message,details FROM diagnostics").fetchall()]
        ptrs = [dict(r) for r in self.conn.execute(
            "SELECT kind,scope_key,branch_id,content_id FROM pointers").fetchall()]
        events = [self._event_dict(r) for r in self.conn.execute(
            "SELECT * FROM events ORDER BY seq").fetchall()]
        return {"format": "adcp-evidence-bundle", "bundle_version": 2,
                "schema_version": SCHEMA_VERSION, "sources": sources,
                "derivations": derivs, "candidates": cand, "inputs": inputs,
                "diagnostics": diags, "pointers": ptrs, "events": events}

    def import_bundle(self, bundle: dict) -> dict:
        if bundle.get("format") != "adcp-evidence-bundle":
            raise ValueError("not an adcp evidence bundle")
        counts = {"sources": 0, "derivations": 0, "events": 0, "pointers": 0}
        with self.conn:
            for s in bundle["sources"]:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO sources(uid,kind,natural_key,content_id,seq,"
                    "parent_content_id,payload) VALUES(?,?,?,?,?,?,?)",
                    (s["uid"], s["kind"], s["natural_key"], s["content_id"],
                     s["seq"], s["parent_content_id"], s["payload"]))
                counts["sources"] += cur.rowcount
            for d in bundle["derivations"]:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO derivations(content_id,kind,scope_key,branch_id,"
                    "parent_content_id,status,rule_set,fingerprint,payload,created_event_seq)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (d["content_id"], d["kind"], d["scope_key"], d["branch_id"],
                     d.get("parent_content_id"), d["status"], d["rule_set"],
                     d["fingerprint"], d["payload"], d.get("created_event_seq")))
                counts["derivations"] += cur.rowcount
            for c in bundle["candidates"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO derivation_candidates(derivation_content_id,"
                    "candidate_id,rule_id,candidate_fingerprint,payload) VALUES(?,?,?,?,?)",
                    (c["derivation_content_id"], c["candidate_id"], c["rule_id"],
                     c["candidate_fingerprint"], c["payload"]))
            for i in bundle["inputs"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO derivation_inputs(derivation_content_id,input_kind,"
                    "input_natural_key,input_content_id,input_fingerprint,rule_id)"
                    " VALUES(?,?,?,?,?,?)",
                    (i["derivation_content_id"], i["input_kind"], i["input_natural_key"],
                     i["input_content_id"], i["input_fingerprint"], i["rule_id"]))
            for g in bundle["diagnostics"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO diagnostics(owner_content_id,code,message,details)"
                    " VALUES(?,?,?,?)",
                    (g["owner_content_id"], g["code"], g["message"], g["details"]))
            for p in bundle["pointers"]:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO pointers(kind,scope_key,branch_id,content_id)"
                    " VALUES(?,?,?,?)",
                    (p["kind"], p["scope_key"], p["branch_id"], p["content_id"]))
                counts["pointers"] += cur.rowcount
            for e in bundle["events"]:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO events(seq,event_type,branch_id,content_id,"
                    "scope_key,payload,base_seq,note) VALUES(?,?,?,?,?,?,?,?)",
                    (e["seq"], e["event_type"], e["branch_id"], e["content_id"],
                     e["scope_key"], canonical(e["payload"]), e["base_seq"], e["note"]))
                counts["events"] += cur.rowcount
        return counts


@contextmanager
def open_store(path: str, crash_hook: Optional[callable] = None):
    store = Store(path, crash_hook=crash_hook)
    try:
        yield store
    finally:
        store.close()
