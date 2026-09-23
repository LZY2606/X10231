"""Versioned evidence store on SQLite.

Semantics:
  * Sources (beam velocity, correlation, amplitude, bad-beam flags, attitude,
    magnetic declination, mounting angle, bottom track, transducer config)
    are recorded separately, each as immutable (id, version) rows keeping the
    raw representation.
  * Every derivation produces a new derived-row version; the branch pointer
    flips only after the whole transaction commits. A crash mid-write leaves
    either the old version or the complete new one.
  * Two users editing from the same base produce a resolvable conflict; the
    unchosen candidate is kept, never overwritten.
  * Export/import preserves identities, order (seq) and provenance.
"""

from __future__ import annotations

import json
import os
import sqlite3

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sources(
  id TEXT NOT NULL, version INTEGER NOT NULL, kind TEXT NOT NULL,
  payload TEXT NOT NULL, seq INTEGER NOT NULL,
  PRIMARY KEY(id, version));
CREATE TABLE IF NOT EXISTS derived(
  id TEXT NOT NULL, version INTEGER NOT NULL, stage TEXT NOT NULL,
  fingerprint TEXT NOT NULL, inputs TEXT NOT NULL, status TEXT NOT NULL,
  payload TEXT NOT NULL, seq INTEGER NOT NULL,
  PRIMARY KEY(id, version));
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, seq INTEGER NOT NULL, branch TEXT NOT NULL,
  parent TEXT, type TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branches(
  name TEXT PRIMARY KEY, head TEXT, base TEXT);
CREATE TABLE IF NOT EXISTS pointer(
  id INTEGER PRIMARY KEY CHECK(id=1), branch TEXT NOT NULL, head TEXT NOT NULL);
"""


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        existed = path != ":memory:" and os.path.exists(path)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        if existed:
            self._migrate_if_needed()
        else:
            self.db.executescript(_DDL)
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("seq", "0")
            self.db.execute(
                "INSERT INTO branches(name, head, base) VALUES('main', NULL, NULL)")
            self.db.commit()

    # -- schema migration -------------------------------------------------
    def _migrate_if_needed(self):
        version = int(self._get_meta("schema_version", "1"))
        if version == SCHEMA_VERSION:
            return
        if version == 1:
            # v1 -> v2: rows gained explicit seq ordering columns.
            for table in ("sources", "derived", "events"):
                cols = [r[1] for r in self.db.execute(f"PRAGMA table_info({table})")]
                if "seq" not in cols:
                    self.db.execute(
                        f"ALTER TABLE {table} ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self.db.commit()
        else:
            raise RuntimeError(f"unsupported schema version {version}")

    # -- meta -------------------------------------------------------------
    def _get_meta(self, key, default=None):
        try:
            row = self.db.execute(
                "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return default
        return row[0] if row else default

    def _set_meta(self, key, value):
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def _next_seq(self):
        seq = int(self._get_meta("seq", "0")) + 1
        self._set_meta("seq", str(seq))
        return seq

    # -- atomic commit ----------------------------------------------------
    def commit(self, branch, writes, events, crash_before_commit=False):
        """Apply writes + events and move the branch pointer atomically.

        writes: list of ("source"|"derived", row-dict)
        events: list of event dicts (id, type, payload)
        Returns the head event id of the branch after commit.
        """
        try:
            self.db.execute("BEGIN IMMEDIATE")
            seq = self._next_seq()
            head = None
            parent = self.head(branch)
            for table, row in writes:
                if table == "source":
                    self.db.execute(
                        "INSERT INTO sources(id, version, kind, payload, seq) "
                        "VALUES(?,?,?,?,?)",
                        (row["id"], row["version"], row["kind"],
                         json.dumps(row["payload"]), seq))
                else:
                    self.db.execute(
                        "INSERT INTO derived(id, version, stage, fingerprint, "
                        "inputs, status, payload, seq) VALUES(?,?,?,?,?,?,?,?)",
                        (row["id"], row["version"], row["stage"],
                         row["fingerprint"], json.dumps(row["inputs"]),
                         row["status"], json.dumps(row["payload"]), seq))
            for ev in events:
                self.db.execute(
                    "INSERT INTO events(id, seq, branch, parent, type, payload) "
                    "VALUES(?,?,?,?,?,?)",
                    (ev["id"], seq, branch, parent, ev["type"],
                     json.dumps(ev["payload"])))
                head = ev["id"]
            if crash_before_commit:
                raise RuntimeError("simulated crash before commit")
            if head is not None:
                self.db.execute(
                    "UPDATE branches SET head=? WHERE name=?", (head, branch))
                if self.current_branch() == branch:
                    self.db.execute(
                        "INSERT INTO pointer(id, branch, head) VALUES(1, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET branch=excluded.branch, "
                        "head=excluded.head", (branch, head))
            self.db.commit()
            return head
        except Exception:
            self.db.rollback()
            raise

    # -- reads ------------------------------------------------------------
    def current_branch(self):
        row = self.db.execute("SELECT branch FROM pointer WHERE id=1").fetchone()
        return row[0] if row else "main"

    def head(self, branch="main"):
        row = self.db.execute(
            "SELECT head FROM branches WHERE name=?", (branch,)).fetchone()
        return row[0] if row else None

    def head_seq(self, branch="main"):
        head = self.head(branch)
        if head is None:
            return 0
        row = self.db.execute(
            "SELECT seq FROM events WHERE id=?", (head,)).fetchone()
        return row[0] if row else 0

    def source(self, source_id, as_of=None):
        sql, args = "SELECT * FROM sources WHERE id=?", [source_id]
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " ORDER BY version DESC LIMIT 1"
        row = self.db.execute(sql, args).fetchone()
        return self._source_dict(row) if row else None

    def sources(self, kind=None, as_of=None):
        sql = ("SELECT s.* FROM sources s JOIN (SELECT id, MAX(version) v "
               "FROM sources WHERE 1=1")
        args = []
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " GROUP BY id) latest ON s.id=latest.id AND s.version=latest.v"
        if kind:
            sql += " WHERE s.kind=?"
            args.append(kind)
        sql += " ORDER BY s.id"
        return [self._source_dict(r) for r in self.db.execute(sql, args)]

    def derived(self, derived_id, as_of=None):
        sql, args = "SELECT * FROM derived WHERE id=?", [derived_id]
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " ORDER BY version DESC LIMIT 1"
        row = self.db.execute(sql, args).fetchone()
        return self._derived_dict(row) if row else None

    def derived_all(self, stage=None, as_of=None):
        sql = ("SELECT d.* FROM derived d JOIN (SELECT id, MAX(version) v "
               "FROM derived WHERE 1=1")
        args = []
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " GROUP BY id) latest ON d.id=latest.id AND d.version=latest.v"
        if stage:
            sql += " WHERE d.stage=?"
            args.append(stage)
        sql += " ORDER BY d.id"
        return [self._derived_dict(r) for r in self.db.execute(sql, args)]

    def events(self, branch=None, as_of=None):
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if branch:
            sql += " AND branch=?"
            args.append(branch)
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " ORDER BY seq, id"
        return [self._event_dict(r) for r in self.db.execute(sql, args)]

    def event(self, event_id):
        row = self.db.execute(
            "SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._event_dict(row) if row else None

    def branches(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM branches ORDER BY name")]

    # -- decisions & conflicts --------------------------------------------
    def record_decision(self, branch, base_head, obj, verdict, rationale,
                        event_id):
        """Record a human decision. If the branch moved since base_head, the
        decision goes to a new conflict branch and a conflict event is raised
        on the original branch; both candidates are kept."""
        current = self.head(branch)
        payload = {"object": obj, "verdict": verdict, "rationale": rationale}
        if current == base_head:
            self.commit(branch, [], [{"id": event_id, "type": f"decision.{verdict}",
                                      "payload": payload}])
            return {"status": "recorded", "branch": branch, "event": event_id}
        other = [e for e in self.events(branch)
                 if e["type"].startswith("decision.")
                 and e["payload"].get("object") == obj]
        alt = f"{branch}~{len(self.branches())}"
        self.db.execute(
            "INSERT INTO branches(name, head, base) VALUES(?, ?, ?)",
            (alt, base_head, base_head))
        self.db.commit()
        self.commit(alt, [], [{"id": event_id, "type": f"decision.{verdict}",
                               "payload": payload}])
        conflict_id = event_id + ":conflict"
        self.commit(branch, [], [{
            "id": conflict_id, "type": "conflict.detected",
            "payload": {"object": obj, "branches": [branch, alt],
                        "decisions": [e["id"] for e in other] + [event_id],
                        "resolvable": True}}])
        return {"status": "conflict", "branch": alt, "event": event_id,
                "conflict": conflict_id}

    def resolve_conflict(self, conflict_event_id, chosen_event_id, event_id):
        conflict = self.event(conflict_event_id)
        if not conflict or conflict["type"] != "conflict.detected":
            raise KeyError(f"no such conflict {conflict_event_id}")
        decisions = conflict["payload"]["decisions"]
        if chosen_event_id not in decisions:
            raise ValueError("chosen decision is not part of the conflict")
        branch = conflict["branch"]
        self.commit(branch, [], [{
            "id": event_id, "type": "conflict.resolved",
            "payload": {"conflict": conflict_event_id, "chosen": chosen_event_id,
                        "kept_unchosen": [d for d in decisions
                                          if d != chosen_event_id]}}])
        return {"status": "resolved", "chosen": chosen_event_id}

    # -- export / import ----------------------------------------------------
    def export(self):
        dump = {"schema_version": SCHEMA_VERSION,
                "seq": self._get_meta("seq", "0")}
        for table in ("sources", "derived", "events", "branches", "pointer"):
            dump[table] = [dict(r) for r in self.db.execute(
                f"SELECT * FROM {table}")]
        return dump

    @classmethod
    def import_dump(cls, path, dump):
        if path != ":memory:" and os.path.exists(path):
            os.remove(path)
        store = cls(path)
        with store.db:
            for row in dump["sources"]:
                store.db.execute(
                    "INSERT INTO sources(id, version, kind, payload, seq) "
                    "VALUES(?,?,?,?,?)",
                    (row["id"], row["version"], row["kind"], row["payload"],
                     row["seq"]))
            for row in dump["derived"]:
                store.db.execute(
                    "INSERT INTO derived(id, version, stage, fingerprint, "
                    "inputs, status, payload, seq) VALUES(?,?,?,?,?,?,?,?)",
                    (row["id"], row["version"], row["stage"], row["fingerprint"],
                     row["inputs"], row["status"], row["payload"], row["seq"]))
            for row in dump["events"]:
                store.db.execute(
                    "INSERT INTO events(id, seq, branch, parent, type, payload) "
                    "VALUES(?,?,?,?,?,?)",
                    (row["id"], row["seq"], row["branch"], row["parent"],
                     row["type"], row["payload"]))
            for row in dump["branches"]:
                store.db.execute(
                    "INSERT INTO branches(name, head, base) VALUES(?,?,?)",
                    (row["name"], row["head"], row["base"]))
            for row in dump["pointer"]:
                store.db.execute(
                    "INSERT INTO pointer(id, branch, head) VALUES(?,?,?)",
                    (row["id"], row["branch"], row["head"]))
            store._set_meta("seq", dump["seq"])
        return store

    # -- row helpers --------------------------------------------------------
    @staticmethod
    def _source_dict(row):
        return {"id": row["id"], "version": row["version"], "kind": row["kind"],
                "payload": json.loads(row["payload"]), "seq": row["seq"]}

    @staticmethod
    def _derived_dict(row):
        return {"id": row["id"], "version": row["version"], "stage": row["stage"],
                "fingerprint": row["fingerprint"],
                "inputs": json.loads(row["inputs"]), "status": row["status"],
                "payload": json.loads(row["payload"]), "seq": row["seq"]}

    @staticmethod
    def _event_dict(row):
        return {"id": row["id"], "seq": row["seq"], "branch": row["branch"],
                "parent": row["parent"], "type": row["type"],
                "payload": json.loads(row["payload"])}

    def close(self):
        self.db.close()
