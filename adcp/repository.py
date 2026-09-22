"""Versioned, append-only storage with atomic pointer switches.

A commit writes new source revisions and node revisions inside one SQLite
transaction and only then moves the branch pointer.  A crash mid-write rolls
the whole transaction back, so readers always see either the old version or
the complete new one - never a half-written state.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import sqlite3

from .db import connect, init_db
from .fingerprint import canonical, fingerprint


def _new_id(prefix: str) -> str:
    return "%s-%s" % (prefix, uuid.uuid4().hex[:12])


def _jload(text):
    return json.loads(text) if text is not None else None


class ConflictDetected(Exception):
    def __init__(self, conflict_id: str, keys_a: Sequence[str],
                 keys_b: Sequence[str]):
        super().__init__("concurrent edits overlap: %s" % conflict_id)
        self.conflict_id = conflict_id
        self.keys_a = list(keys_a)
        self.keys_b = list(keys_b)


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        init_db(conn)
        # test hook: invoked with (kind, payload) just before COMMIT.
        # raising here simulates a crash during the write.
        self.precommit_hook: Optional[callable] = None

    # ------------------------------------------------------------------ util
    def _next_seq(self, table: str) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM %s" % table).fetchone()
        return int(row[0])

    def head(self, branch: str = "main") -> Optional[str]:
        row = self.conn.execute(
            "SELECT head_commit FROM branches WHERE name=?", (branch,)).fetchone()
        return row[0] if row else None

    def commit(self, kind: str, label: str, branch: str,
               sources: Dict[str, dict], nodes: List[dict],
               events: List[dict], parent_commit: Optional[str] = None,
               actor: str = "system",
               edited_keys: Optional[Sequence[str]] = None,
               base_commit: Optional[str] = None,
               extra_parents: Optional[Sequence[str]] = None) -> dict:
        """Atomically create a commit and switch the branch pointer.

        sources : {source_id: {"kind","datum","raw"}} (full new values; the
                  previous revision is linked automatically).
        nodes   : list of node revisions created by this commit:
                  {"node_id","status","fingerprint","payload","scope_key",
                   "reason","prev_rev"}
        events  : list of {"kind","actor","payload"} (ids assigned here).
        """
        conn = self.conn
        # optimistic-concurrency check against a competing branch edit
        conflict = self._detect_conflict(kind, parent_commit, base_commit,
                                         edited_keys, branch)
        if conflict is not None:
            return conflict  # callers receive {"conflict": ...}
        try:
            conn.execute("SAVEPOINT sp_commit")
            commit_id = _new_id("c")
            seq = self._next_seq("commits")
            created = time.time()
            parents = []
            if parent_commit:
                parents.append(parent_commit)
            if base_commit and base_commit != parent_commit and base_commit not in parents:
                parents.append(base_commit)
            for ep in (extra_parents or []):
                if ep not in parents:
                    parents.append(ep)
            event_ids = []
            for ev in events:
                ev_id = ev.get("event_id") or _new_id("e")
                ev_seq = self._next_seq("events")
                ch = fingerprint("ev", ev.get("payload", {}))
                conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                    (ev_id, ev_seq, commit_id, ev.get("parent_event_id"),
                     ev["kind"], ev.get("actor", actor), created,
                     canonical(ev.get("payload", {})), ch))
                event_ids.append(ev_id)
            for sid, s in sources.items():
                prev = self._latest_source_rev_id(sid, parents)
                rev = self._next_source_rev(sid)
                conn.execute(
                    "INSERT INTO source_revs VALUES(?,?,?,?,?,?,?)",
                    (sid, rev, commit_id, s["kind"], s.get("datum"),
                     canonical(s.get("raw")),
                     prev))
            for n in nodes:
                prev = n.get("prev_rev")
                if prev is None:
                    prev = self._latest_node_rev_id(n["node_id"], parents)
                rev = self._next_node_rev(n["node_id"])
                conn.execute(
                    "INSERT INTO node_revs VALUES(?,?,?,?,?,?,?,?,?)",
                    (n["node_id"], rev, commit_id, n["status"],
                     n.get("fingerprint"), canonical(n.get("payload", {})),
                     n["scope_key"], n.get("reason"), prev))
            ch = fingerprint("commit", {
                "parents": parents, "kind": kind, "label": label,
                "sources": {s: _jload(canonical(v)) for s, v in sources.items()},
                "nodes": [(n["node_id"], n.get("fingerprint")) for n in nodes],
                "events": event_ids,
            })
            conn.execute(
                "INSERT INTO commits VALUES(?,?,?,?,?,?,?,?,?,?)",
                (commit_id, seq, canonical(parents), kind, label, created,
                 branch, event_ids[0] if event_ids else None,
                 canonical({"event_ids": event_ids}), ch))
            self._insert_closure(commit_id, parents)
            conn.execute(
                "INSERT INTO branches(name,head_commit) VALUES(?,?) "
                "ON CONFLICT(name) DO UPDATE SET head_commit=excluded.head_commit",
                (branch, commit_id))
            if self.precommit_hook is not None:
                self.precommit_hook(kind, {"commit_id": commit_id})
            conn.execute("RELEASE SAVEPOINT sp_commit")
            conn.commit()
            return {"commit_id": commit_id, "seq": seq,
                    "event_ids": event_ids}
        except Exception:
            try:
                conn.execute("ROLLBACK TO SAVEPOINT sp_commit")
                conn.execute("RELEASE SAVEPOINT sp_commit")
            except sqlite3.Error:
                pass
            raise

    # ------------------------------------------------------------- revisions
    def _next_source_rev(self, sid: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(rev),0)+1 FROM source_revs WHERE source_id=?",
            (sid,)).fetchone()
        return int(row[0])

    def _next_node_rev(self, nid: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(rev),0)+1 FROM node_revs WHERE node_id=?",
            (nid,)).fetchone()
        return int(row[0])

    def _latest_source_rev_id(self, sid: str, parents: Sequence[str]):
        if not parents:
            return None
        placeholders = ",".join("?" for _ in parents)
        row = self.conn.execute(
            "SELECT rev FROM source_revs WHERE source_id=? AND commit_id IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id IN (%s)) "
            "ORDER BY rev DESC LIMIT 1" % placeholders,
            [sid] + list(parents)).fetchone()
        return int(row[0]) if row else None

    def _latest_node_rev_id(self, nid: str, parents: Sequence[str]):
        if not parents:
            return None
        placeholders = ",".join("?" for _ in parents)
        row = self.conn.execute(
            "SELECT rev FROM node_revs WHERE node_id=? AND commit_id IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id IN (%s)) "
            "ORDER BY rev DESC LIMIT 1" % placeholders,
            [nid] + list(parents)).fetchone()
        return int(row[0]) if row else None

    def _insert_closure(self, commit_id: str, parents: Sequence[str]) -> None:
        conn = self.conn
        conn.execute("INSERT INTO commit_closure VALUES(?,?,0)",
                     (commit_id, commit_id))
        for p in parents:
            rows = conn.execute(
                "SELECT ancestor_id, depth FROM commit_closure WHERE descendant_id=?",
                (p,)).fetchall()
            for r in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO commit_closure VALUES(?,?,?)",
                    (r["ancestor_id"], commit_id, r["depth"] + 1))

    # ------------------------------------------------------------- conflicts
    def _detect_conflict(self, kind: str, parent_commit, base_commit,
                         edited_keys, branch: str):
        """Two users editing from the same version: overlapping keys clash.

        Non-overlapping edits merge automatically; overlapping source keys
        produce a resolvable conflict record and no pointer switch.
        """
        if kind != "edit" or not edited_keys or base_commit is None:
            return None
        head = self.head(branch)
        if head is None or head == base_commit:
            return None
        # what changed at head since the user's base?
        changed = self._sources_changed_between(base_commit, head)
        overlap = sorted(set(edited_keys) & set(changed))
        if not overlap:
            return None
        conflict_id = _new_id("cf")
        self.conn.execute(
            "INSERT INTO conflicts VALUES(?,?,?,?,?,?,?,NULL,NULL)",
            (conflict_id, base_commit, parent_commit, head,
             canonical(sorted(edited_keys)), canonical(overlap),
             time.time()))
        self.conn.commit()
        return {"conflict": conflict_id, "overlap": overlap,
                "head_commit": head}

    def _sources_changed_between(self, old_commit: str, new_commit: str):
        rows = self.conn.execute(
            "SELECT DISTINCT source_id FROM source_revs WHERE commit_id IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id=?) "
            "AND commit_id NOT IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id=?)",
            (new_commit, old_commit)).fetchall()
        return [r["source_id"] for r in rows]

    def resolve_conflict(self, conflict_id: str, resolution: str,
                         chosen_sources: Dict[str, dict], nodes: List[dict],
                         events: List[dict], actor: str = "reviewer",
                         label: str = "conflict resolved") -> dict:
        """Create a two-parent merge commit carrying the human resolution."""
        row = self.conn.execute(
            "SELECT * FROM conflicts WHERE conflict_id=?", (conflict_id,)).fetchone()
        if row is None:
            raise KeyError(conflict_id)
        if row["resolved_commit"]:
            raise ValueError("conflict already resolved")
        result = self.commit(
            kind="resolve_conflict", label=label, branch="main",
            sources=chosen_sources, nodes=nodes, events=events,
            parent_commit=row["commit_a"], actor=actor,
            extra_parents=[row["commit_b"]])
        # second parent = commit_b (head).  Rebuild as explicit merge:
        if "commit_id" in result:
            self.conn.execute(
                "UPDATE conflicts SET resolved_commit=?, resolution=? WHERE conflict_id=?",
                (result["commit_id"], resolution, conflict_id))
            self.conn.commit()
        return result

    # ---------------------------------------------------------------- reads
    def _visible_rev(self, table: str, id_col: str, item_id: str,
                     commit_id: str):
        placeholders = "?"
        return self.conn.execute(
            "SELECT * FROM %s WHERE %s=? AND commit_id IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id=?) "
            "ORDER BY rev DESC LIMIT 1" % (table, id_col),
            (item_id, commit_id)).fetchone()

    def source_at(self, source_id: str, commit_id: Optional[str] = None):
        commit_id = commit_id or self.head()
        row = self._visible_rev("source_revs", "source_id", source_id, commit_id)
        if row is None:
            return None
        return {"source_id": source_id, "rev": row["rev"],
                "commit_id": row["commit_id"], "kind": row["kind"],
                "datum": row["datum"], "raw": _jload(row["raw_json"]),
                "prev_rev": row["prev_rev"]}

    def node_at(self, node_id: str, commit_id: Optional[str] = None):
        commit_id = commit_id or self.head()
        row = self._visible_rev("node_revs", "node_id", node_id, commit_id)
        if row is None:
            return None
        return {"node_id": node_id, "rev": row["rev"],
                "commit_id": row["commit_id"], "status": row["status"],
                "fingerprint": row["fingerprint"],
                "payload": _jload(row["payload_json"]),
                "scope_key": row["scope_key"], "reason": row["reason"],
                "prev_rev": row["prev_rev"]}

    def sources_at(self, commit_id: Optional[str] = None) -> List[dict]:
        commit_id = commit_id or self.head()
        if commit_id is None:
            return []
        rows = self.conn.execute(
            "SELECT s.* FROM source_revs s JOIN ("
            " SELECT source_id, MAX(rev) m FROM source_revs WHERE commit_id IN "
            " (SELECT ancestor_id FROM commit_closure WHERE descendant_id=?) "
            " GROUP BY source_id) x ON s.source_id=x.source_id AND s.rev=x.m",
            (commit_id,)).fetchall()
        out = []
        for r in rows:
            out.append({"source_id": r["source_id"], "rev": r["rev"],
                        "kind": r["kind"], "datum": r["datum"],
                        "raw": _jload(r["raw_json"]),
                        "prev_rev": r["prev_rev"]})
        return out

    def nodes_at(self, commit_id: Optional[str] = None) -> List[dict]:
        commit_id = commit_id or self.head()
        if commit_id is None:
            return []
        rows = self.conn.execute(
            "SELECT n.* FROM node_revs n JOIN ("
            " SELECT node_id, MAX(rev) m FROM node_revs WHERE commit_id IN "
            " (SELECT ancestor_id FROM commit_closure WHERE descendant_id=?) "
            " GROUP BY node_id) x ON n.node_id=x.node_id AND n.rev=x.m",
            (commit_id,)).fetchall()
        return [{"node_id": r["node_id"], "rev": r["rev"],
                 "commit_id": r["commit_id"], "status": r["status"],
                 "fingerprint": r["fingerprint"],
                 "payload": _jload(r["payload_json"]),
                 "scope_key": r["scope_key"], "reason": r["reason"],
                 "prev_rev": r["prev_rev"]} for r in rows]

    def commit_row(self, commit_id: str):
        r = self.conn.execute("SELECT * FROM commits WHERE commit_id=?",
                              (commit_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["parent_ids"] = _jload(d["parent_ids"])
        d["payload_json"] = _jload(d["payload_json"])
        return d

    def commits(self, limit: int = 500) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM commits ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["parent_ids"] = _jload(d["parent_ids"])
            d["payload_json"] = _jload(d["payload_json"])
            out.append(d)
        return out

    def events_at(self, commit_id: Optional[str] = None) -> List[dict]:
        commit_id = commit_id or self.head()
        rows = self.conn.execute(
            "SELECT * FROM events WHERE commit_id IN "
            "(SELECT ancestor_id FROM commit_closure WHERE descendant_id=?) "
            "ORDER BY seq", (commit_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload_json"] = _jload(d["payload_json"])
            out.append(d)
        return out

    def all_events(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload_json"] = _jload(d["payload_json"])
            out.append(d)
        return out

    def conflicts(self, include_resolved: bool = True) -> List[dict]:
        sql = "SELECT * FROM conflicts"
        if not include_resolved:
            sql += " WHERE resolved_commit IS NULL"
        rows = self.conn.execute(sql).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["edited_a_json"] = _jload(d["edited_a_json"])
            d["edited_b_json"] = _jload(d["edited_b_json"])
            out.append(d)
        return out

    def ancestors(self, commit_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT ancestor_id FROM commit_closure WHERE descendant_id=? "
            "ORDER BY depth DESC", (commit_id,)).fetchall()
        return [r["anchor_id" if False else "ancestor_id"] for r in rows]

    # ------------------------------------------------------- export/import
    def export_bundle(self) -> dict:
        """Export the entire append-only history.

        All identities (commit, node, source revision, event) are preserved
        verbatim so deleting the local database and importing the bundle
        restores identities, ordering and provenance exactly.
        """
        tables = ("meta", "commits", "commit_closure", "branches",
                  "source_revs", "node_revs", "events", "conflicts")
        bundle = {"format": "adcp-evidence-bundle", "format_version": 2,
                  "tables": {}}
        for t in tables:
            cols = [r[1] for r in self.conn.execute(
                "PRAGMA table_info(%s)" % t).fetchall()]
            rows = [list(r) for r in self.conn.execute(
                "SELECT %s FROM %s" % (",".join(cols), t)).fetchall()]
            bundle["tables"][t] = {"columns": cols, "rows": rows}
        bundle["content_hash"] = fingerprint("bundle", bundle["tables"])
        return bundle

    def import_bundle(self, bundle: dict) -> None:
        if bundle.get("format") != "adcp-evidence-bundle":
            raise ValueError("not an ADCP evidence bundle")
        if fingerprint("bundle", bundle["tables"]) != bundle.get("content_hash"):
            raise ValueError("bundle content hash mismatch")
        conn = self.conn
        conn.execute("SAVEPOINT sp_import")
        try:
            for t, body in bundle["tables"].items():
                cols = body["columns"]
                ph = ",".join("?" for _ in cols)
                conn.executemany(
                    "INSERT OR REPLACE INTO %s(%s) VALUES(%s)"
                    % (t, ",".join(cols), ph), body["rows"])
            conn.execute("RELEASE SAVEPOINT sp_import")
            conn.commit()
        except Exception:
            try:
                conn.execute("ROLLBACK TO SAVEPOINT sp_import")
                conn.execute("RELEASE SAVEPOINT sp_import")
            except sqlite3.Error:
                pass
            raise

    def wipe(self) -> None:
        for t in ("events", "node_revs", "source_revs", "commit_closure",
                  "commits", "branches", "conflicts", "meta"):
            self.conn.execute("DELETE FROM %s" % t)
        self.conn.commit()
