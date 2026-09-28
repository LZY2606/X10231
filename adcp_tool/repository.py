"""Replayable repository: ingestion, event application, dependency propagation.

The repository materialises, for each (branch, event_seq), the complete set of
derived node pointers. Revisiting an old event therefore restores exactly the
dependencies and visible conclusions in force then; newer recomputation never
overwrites history. Source/derived rows are content-addressed and shared
across states/branches.
"""
from __future__ import annotations

import json
from typing import Optional

from . import db as dbmod
from .db import Conflict, connect, init_schema, transaction
from .geometry import BeamGeometry, fingerprint
from .units import quantity
from .attitude import AttitudeSeries, AttitudeSample
from .rotation import MountAngles, TransformConfig
from .quality import QualityPolicy
from .pipeline import derive_cell, derive_bottom_track


def _geom(doc: dict) -> BeamGeometry:
    g = doc["geometry"]
    return BeamGeometry(
        beam_angle_deg=g["beam_angle_deg"],
        azimuths_deg=tuple(g["azimuths_deg"]),
        convention=g.get("convention", "positive-away"),
        coord_frame=g.get("coord_frame", "x-forward,y-starboard,z-up"),
        id=g["id"])


def _qpolicy(doc: dict) -> QualityPolicy:
    q = doc["quality_policy"]
    return QualityPolicy(
        correlation_floor=q["correlation_floor"],
        correlation_hard=q["correlation_hard"],
        amplitude_min=q["amplitude_min"], amplitude_bad=q["amplitude_bad"],
        id=q["id"])


def _mount(rec: dict) -> MountAngles:
    return MountAngles(x_deg=rec["x_deg"], y_deg=rec["y_deg"],
                       heading_alignment_deg=rec["heading_alignment_deg"],
                       version=rec["id"])


class Repository:
    def __init__(self, path: str = ":memory:"):
        self.conn = connect(path)
        init_schema(self.conn)

    # ---------------- dataset + baseline ingestion -----------------
    def ingest(self, doc: dict) -> None:
        c = self.conn
        with transaction(c):
            c.execute("INSERT OR REPLACE INTO dataset VALUES(?,?,?,?)",
                      (doc["dataset_id"], doc["title"],
                       json.dumps(doc["geometry"]), json.dumps(doc)))
            c.execute("INSERT INTO branch(name,head_seq,parent_branch,fork_seq)"
                      " VALUES('main',0,NULL,NULL) "
                      "ON CONFLICT(name) DO NOTHING")
            self._ingest_streams(c, doc)
            self._ingest_params(c, doc)

    def _ingest_streams(self, c, doc: dict) -> None:
        ds = doc["dataset_id"]
        st = doc["streams"]
        c.execute("DELETE FROM source_slot WHERE dataset_id=?", (ds,))

        def slot(stream, ping, cell, beam, seq):
            sid = f"{ds}:{stream}:{ping}:{cell}:{beam}"
            c.execute("INSERT INTO source_slot VALUES(?,?,?,?,?,?,?)",
                      (sid, ds, stream, ping, cell, beam, seq))
            return sid

        for r in st["beam_velocity"]:
            sid = slot("beam_velocity", r["ping"], r["cell"], r["beam"],
                       r["ping"] * 1000 + r["cell"] * 10 + r["beam"])
            rid = fingerprint("src", {"slot": sid, "raw": r["raw"],
                                      "unit": r["unit"]})
            c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                      (rid, sid, 0, r["raw"], r["unit"], r.get("basis", ""),
                       json.dumps(r["raw"]), None))
        for stream in ("correlation", "amplitude", "bad_beam_flag"):
            for r in st[stream]:
                sid = slot(stream, r["ping"], r["cell"], r["beam"],
                           r["ping"] * 1000 + r["cell"] * 10 + r["beam"])
                rid = fingerprint("src", {"slot": sid, "value": r["value"]})
                c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                          (rid, sid, 0, str(r["value"]),
                           "count" if stream != "bad_beam_flag" else "flag",
                           "", json.dumps(r["value"]), None))
        for stream in ("heading", "pitch", "roll"):
            for r in st[stream]:
                sid = slot(stream, r["ping"], None, None, r["ping"])
                rid = fingerprint("src", {"slot": sid, "raw": r["raw"],
                                          "unit": r["unit"]})
                c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                          (rid, sid, 0, r["raw"], r["unit"],
                           r.get("basis", ""), json.dumps(r["raw"]), None))
        for r in st["bottom_track"]:
            sid = slot(f"bottom_track:{r['kind']}", r["ping"], None,
                       r["beam"], r["ping"] * 10 + r["beam"])
            rid = fingerprint("src", {"slot": sid, "raw": r.get("raw"),
                                      "value": r.get("value"),
                                      "kind": r["kind"]})
            c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                      (rid, sid, 0,
                       r["raw"] if r.get("raw") is not None else str(r["value"]),
                       r.get("unit", "count"), r.get("basis", ""),
                       json.dumps(r["raw"] if r.get("raw") is not None
                                  else r["value"]), None))
        # depth cell config
        for r in st["cell_config"]:
            sid = slot("cell_config", None, r["cell"], None, r["seq"])
            rid = fingerprint("src", {"slot": sid, "depth": r["depth_m"]})
            c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                      (rid, sid, 0, str(r["depth_m"]), "m", "depth",
                       json.dumps(r), None))

    def _ingest_params(self, c, doc: dict) -> None:
        ds = doc["dataset_id"]
        c.execute("DELETE FROM param_revision WHERE dataset_id=?", (ds,))
        c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                  (doc["declination"]["id"], ds, "declination", None, 0,
                   json.dumps(doc["declination"])))
        c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                  (doc["mount"]["id"], ds, "mount", 0, 0,
                   json.dumps(doc["mount"])))
        c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                  (doc["quality_policy"]["id"], ds, "quality_policy", None, 0,
                   json.dumps(doc["quality_policy"])))
        c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                  (doc["transform"]["id"]
                   if "id" in doc["transform"]
                   else "xform:roll-pitch:decl=1",
                   ds, "transform", None, 0,
                   json.dumps(doc["transform"])))

    # ---------------- state (re)construction -----------------------
    def doc(self) -> dict:
        row = self.conn.execute("SELECT doc_json FROM dataset").fetchone()
        return json.loads(row["doc_json"])

    def _active_stream(self, stream: str, seq: int) -> dict:
        """Latest revision per slot whose valid_from_event <= seq."""
        rows = self.conn.execute(
            """SELECT sr.* FROM source_revision sr
               JOIN source_slot ss ON ss.id=sr.slot_id
               WHERE ss.stream=? AND sr.valid_from_event<=?
                 AND sr.id=(SELECT sr2.id FROM source_revision sr2
                            JOIN source_slot ss2 ON ss2.id=sr2.slot_id
                            WHERE ss2.id=ss.id AND sr2.valid_from_event<=?
                            ORDER BY sr2.valid_from_event DESC LIMIT 1)""",
            (stream, seq, seq)).fetchall()
        out = {}
        for r in rows:
            slot = self.conn.execute(
                "SELECT ping,cell,beam,seq FROM source_slot WHERE id=?",
                (r["slot_id"],)).fetchone()
            out[(slot["ping"], slot["cell"], slot["beam"])] = {
                "raw": r["raw"], "unit": r["unit"], "basis": r["basis"],
                "value": json.loads(r["value_json"]),
                "revision": r["id"], "valid_from_event": r["valid_from_event"],
                "seq": slot["seq"]}
        return out

    def _active_params(self, seq: int, ping: Optional[int] = None) -> dict:
        out = {}
        for kind in ("declination", "quality_policy", "transform"):
            row = self.conn.execute(
                """SELECT * FROM param_revision WHERE kind=?
                   AND valid_from_event<=? ORDER BY valid_from_event DESC
                   LIMIT 1""", (kind, seq)).fetchone()
            out[kind] = json.loads(row["content_json"])
        # mount picks the revision applicable at the ping
        rows = self.conn.execute(
            """SELECT * FROM param_revision WHERE kind='mount'
               AND valid_from_event<=? ORDER BY valid_from_event""",
            (seq,)).fetchall()
        chosen = None
        for r in rows:
            vfp = r["valid_from_ping"] if "valid_from_ping" in r.keys() else None
            if ping is None or vfp is None or ping >= vfp:
                chosen = r
        out["mount"] = json.loads(chosen["content_json"])
        out["mount_row"] = dict(chosen) if chosen else None
        return out

    def build_state(self, branch: str, seq: int,
                    recompute_keys: Optional[set] = None) -> dict:
        """Materialise node pointers for (branch, seq).

        Copies the previous state forward and recomputes only affected keys
        (dependency propagation). Returns mapping key -> node dict plus the
        changed-key explanation.
        """
        doc = self.doc()
        ds = doc["dataset_id"]
        geom = _geom(doc)
        n_pings, n_cells = doc["n_pings"], doc["n_cells"]

        beam = self._active_stream("beam_velocity", seq)
        corr = self._active_stream("correlation", seq)
        amp = self._active_stream("amplitude", seq)
        bad = self._active_stream("bad_beam_flag", seq)
        head = self._active_stream("heading", seq)
        pitch = self._active_stream("pitch", seq)
        roll = self._active_stream("roll", seq)
        cellcfg = self._active_stream("cell_config", seq)
        bt_rows = {}
        for kind in ("beam_velocity", "correlation", "amplitude",
                     "bad_beam_flag"):
            for (bping, _bcell, bbeam), val2 in self._active_stream(
                    f"bottom_track:{kind}", seq).items():
                bt_rows[(kind, bping, bbeam)] = dict(val2)

        params = self._active_params(seq)
        qpolicy = QualityPolicy(**{k: params["quality_policy"][k] for k in
                                   ("correlation_floor", "correlation_hard",
                                    "amplitude_min", "amplitude_bad", "id")})
        tx = params["transform"]
        cfg = TransformConfig(tilt_order=tx.get("tilt_order", "roll-pitch"),
                              apply_declination=tx.get("apply_declination",
                                                      True))

        samples = []
        for (p, _, _), r in head.items():
            pi = pitch.get((p, None, None))
            ro = roll.get((p, None, None))
            if pi and ro:
                samples.append(AttitudeSample(
                    _ping_t(doc, p), float(r["raw"]), float(pi["raw"]),
                    float(ro["raw"]), source=r["revision"][:12]))
        attitude = AttitudeSeries(samples, doc["max_attitude_gap_s"])

        depths, min_depth, blank = {}, doc and 2.5, 0.0
        for (_, cell, _), r in cellcfg.items():
            depths[cell] = quantity(r["raw"], r["unit"], r["basis"])
            v = r["value"]
            if isinstance(v, dict):
                min_depth = v.get("min_valid_depth_m", min_depth)
                blank = v.get("blank_m", blank)

        # invalid cells from events
        invalid_cells = set()
        for evrow in self.conn.execute(
                "SELECT payload_json FROM event WHERE branch=? AND seq<=?",
                (branch, seq)):
            p = json.loads(evrow["payload_json"])
            if p.get("stream") == "cell_invalid":
                invalid_cells.add((p["ping"], p["cell"]))

        # deleted attitude samples (gap injection)
        gap_pings = set()
        samples = [s for i, s in enumerate(samples) if i not in gap_pings]
        attitude = AttitudeSeries(samples, doc["max_attitude_gap_s"])

        src_versions = {
            "beam": _rev_version(beam), "correlation": _rev_version(corr),
            "amplitude": _rev_version(amp), "bad_beam": _rev_version(bad),
            "attitude": fingerprint("att", {"h": _rev_version(head),
                                            "p": _rev_version(pitch),
                                            "r": _rev_version(roll),
                                            "gaps": sorted(gap_pings)})}

        nodes, changed = {}, set()
        prev = self._state_map(branch, seq - 1)
        for i in range(n_pings):
            t = _ping_t(doc, i)
            pid = f"p{i+1:02d}"
            mount_rec = self._mount_for_ping(seq, i)
            for j in range(n_cells):
                key = f"{pid}:c{j+1}"
                cell = {
                    "beams": [quantity(beam[(i, j, k)]["raw"],
                                       beam[(i, j, k)]["unit"],
                                       beam[(i, j, k)]["basis"])
                              for k in range(4)],
                    "correlation": [corr[(i, j, k)]["value"] for k in range(4)],
                    "amplitude": [amp[(i, j, k)]["value"] for k in range(4)],
                    "bad_beam": [bad[(i, j, k)]["value"] for k in range(4)],
                    "depth": depths.get(j), "min_depth_m": min_depth,
                    "blank_m": blank}
                prov = dict(src_versions)
                prov["mount"] = mount_rec["id"]
                prov["declination"] = params["declination"]["id"]
                node = derive_cell(pid, t, i, j, cell, geom,
                                   _mount(mount_rec), params["declination"],
                                   qpolicy, cfg, attitude, prov,
                                   invalid_cells)
                self._store_node(node, seq, key, branch,
                                 prev.get(key), nodes, changed,
                                 recompute_keys)

            # bottom track
            bt = {"beams": [quantity(bt_rows[("beam_velocity", i, k)]["raw"],
                                     bt_rows[("beam_velocity", i, k)]["unit"],
                                     bt_rows[("beam_velocity", i, k)]["basis"])
                            for k in range(4)],
                  "correlation": [bt_rows.get(("correlation", i, k),
                                              {}).get("value")
                                  for k in range(4)],
                  "amplitude": [bt_rows.get(("amplitude", i, k),
                                            {}).get("value")
                                for k in range(4)],
                  "bad_beam": [bt_rows.get(("bad_beam_flag", i, k),
                                           {}).get("value", 0)
                               for k in range(4)]}
            btnode = derive_bottom_track(pid, t, bt, geom,
                                         _mount(mount_rec),
                                         params["declination"], cfg, attitude,
                                         prov)
            key = f"{pid}:bt"
            self._store_node(btnode, seq, key, branch, prev.get(key),
                             nodes, changed, recompute_keys)
        return {"nodes": nodes, "changed": sorted(changed),
                "gap_pings": sorted(gap_pings)}

    # ---------------- node persistence / propagation --------------
    def _store_node(self, node: dict, seq: int, key: str, branch: str,
                    prev_fp: Optional[str], nodes: dict, changed: set,
                    force: Optional[set]) -> None:
        fp = node["candidates"][0]["fingerprint"] if node["candidates"] \
            else fingerprint("node", {"key": key,
                                      "empty": True,
                                      "prov": node.get("provenance"),
                                      "status": node.get("status")})
        # a node with multiple candidates is itself keyed by a node-level hash
        node_fp = fingerprint("node", {
            "key": key,
            "candidates": sorted(c["fingerprint"] for c in node["candidates"]),
            "status": node.get("status"),
            "verdicts": [(v["beam"], v["tier"]) for v in node["verdicts"]],
            "attitude": node["attitude"]["status"]})
        c = self.conn
        existing = c.execute("SELECT 1 FROM deriv_node WHERE fingerprint=?",
                             (node_fp,)).fetchone()
        if not existing:
            c.execute("INSERT INTO deriv_node VALUES(?,?,?,?,?,?,?,?)",
                      (node_fp, self.doc()["dataset_id"],
                       "bt" if key.endswith(":bt") else "cell",
                       node.get("ping_t") is not None and _ping_of(key),
                       node.get("cell_idx"), json.dumps(node),
                       "rules-2.1.0", seq))
            for cand in node["candidates"]:
                c.execute("INSERT OR IGNORE INTO node_candidate VALUES(?,?,?,?,?)",
                          (node_fp, cand["candidate_id"], cand["mode"],
                           json.dumps(cand["beams"]),
                           json.dumps(cand)))
        chosen = None
        if prev_fp:
            drow = c.execute(
                "SELECT chosen_candidate FROM node_state WHERE branch=? AND key=?"
                " ORDER BY event_seq DESC LIMIT 1", (branch, key)).fetchone()
            chosen = drow["chosen_candidate"] if drow else None
        c.execute("INSERT INTO node_state VALUES(?,?,?,?,?)",
                  (branch, seq, key, node_fp, chosen))
        nodes[key] = node
        if force is None or key in force or node_fp != prev_fp:
            changed.add(key)

    def _state_map(self, branch: str, seq: int) -> dict:
        rows = self.conn.execute(
            """SELECT ns.* FROM node_state ns JOIN (
                 SELECT key, MAX(event_seq) m FROM node_state
                 WHERE branch=? AND event_seq<=? GROUP BY key) x
               ON ns.key=x.key AND ns.event_seq=x.m AND ns.branch=?""",
            (branch, seq, branch)).fetchall()
        return {r["key"]: r["node_fingerprint"] for r in rows}

    def _mount_for_ping(self, seq: int, ping: int) -> dict:
        rows = self.conn.execute(
            "SELECT * FROM param_revision WHERE kind='mount' "
            "AND valid_from_event<=?", (seq,)).fetchall()
        chosen = None
        for r in rows:
            vfp = json.loads(r["content_json"]).get("valid_from_ping")
            rec = json.loads(r["content_json"])
            vfp = rec.get("valid_from_ping", vfp)
            if vfp is None or ping >= vfp:
                chosen = rec
        return chosen

    def _event_kind(self, *a):
        return None

    # ---------------- events / branches / decisions ----------------
    def head(self, branch: str) -> int:
        row = self.conn.execute("SELECT head_seq FROM branch WHERE name=?",
                                (branch,)).fetchone()
        return row["head_seq"] if row else 0

    def apply_event(self, kind: str, payload: dict,
                    branch: str = "main", expected_parent: Optional[int] = None,
                    crash_before_commit: bool = False) -> dict:
        c = self.conn
        if not c.execute("SELECT 1 FROM branch WHERE name=?",
                         (branch,)).fetchone():
            raise KeyError(f"unknown branch {branch}")
        parent = self.head(branch)
        if expected_parent is not None and expected_parent != parent:
            raise Conflict(expected_parent, parent,
                           {"kind": kind, "payload": payload})
        seq = parent + 1

        affected = self._affected_keys(payload, branch, seq)
        with transaction(c, crash_after_stage=crash_before_commit):
            self._mutate_source(c, payload, seq, branch)
            state = self.build_state(branch, seq, recompute_keys=affected)
            chosen = self._apply_decision_payload(c, payload, seq, branch)
            c.execute("INSERT INTO event VALUES(?,?,?,?,?,?)",
                      (branch, seq, kind, json.dumps(payload),
                       json.dumps({"affected": state["changed"],
                                   "gap_pings": state["gap_pings"],
                                   "decision": chosen}), parent))
            c.execute("UPDATE branch SET head_seq=? WHERE name=?",
                      (seq, branch))
            self._write_dep_edges(c, branch, seq, payload, state["changed"])
        return {"seq": seq, "affected": state["changed"],
                "gap_pings": state["gap_pings"]}

    def create_branch(self, name: str, from_branch: str = "main",
                      at_seq: Optional[int] = None,
                      overrides: Optional[dict] = None) -> int:
        at_seq = self.head(from_branch) if at_seq is None else at_seq
        c = self.conn
        with transaction(c):
            c.execute("INSERT INTO branch(name,head_seq,parent_branch,fork_seq)"
                      " VALUES(?,?,?,?)", (name, at_seq, from_branch, at_seq))
            # copy node_state snapshot so the branch starts from that point
            for key, v in self._state_rows(from_branch, at_seq).items():
                c.execute("INSERT OR REPLACE INTO node_state VALUES(?,?,?,?,?)",
                          (name, at_seq, key, v["fingerprint"], v["chosen"]))
        if overrides:
            self.apply_event("source_change", overrides, branch=name)
        return self.head(name)

    def _mutate_source(self, c, payload: dict, seq: int, branch: str) -> None:
        stream = payload.get("stream")
        if not stream or payload.get("reason", "").startswith("missing"):
            return
        ds = self.doc()["dataset_id"]
        ping, cell, beam = payload.get("ping"), payload.get("cell"), \
            payload.get("beam")
        if stream in ("correlation", "amplitude", "bad_beam_flag"):
            sid = f"{ds}:{stream}:{ping}:{cell}:{beam}"
            value = payload["value"]
            rid = fingerprint("src", {"slot": sid, "value": value,
                                      "event": seq})
            c.execute("INSERT INTO source_revision VALUES(?,?,?,?,?,?,?,?,1)",
                      (rid, sid, seq, str(value),
                       "count" if stream != "bad_beam_flag" else "flag",
                       "", json.dumps(value), None))
        elif stream == "mount":
            content = {"id": payload["id"], "x_deg": payload["x_deg"],
                       "y_deg": payload["y_deg"],
                       "heading_alignment_deg":
                       payload["heading_alignment_deg"],
                       "valid_from_ping": payload.get("valid_from_ping")}
            c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                      (payload["id"], ds, "mount",
                       payload.get("valid_from_ping"), seq,
                       json.dumps(content)))
        elif stream == "transform":
            content = {"tilt_order": payload["tilt_order"],
                       "apply_declination": payload.get("apply_declination",
                                                       True)}
            cid = f"xform:{content['tilt_order']}:decl=" \
                  f"{int(content['apply_declination'])}"
            c.execute("INSERT INTO param_revision VALUES(?,?,?,?,?,?)",
                      (cid, ds, "transform", None, seq,
                       json.dumps(content)))
        elif stream == "cell_invalid":
            return  # handled through event payload in build_state
        else:
            raise ValueError(f"unsupported source change stream {stream}")

    def _apply_decision_payload(self, c, payload, seq, branch):
        if payload.get("action") != "select-candidate":
            return None
        key = f"p{payload['ping']+1:02d}:c{payload['cell']+1}"
        node_fp = c.execute(
            "SELECT node_fingerprint FROM node_state WHERE branch=? AND key=?"
            " ORDER BY event_seq DESC LIMIT 1",
            (branch, key)).fetchone()["node_fingerprint"]
        valid = {r["candidate_id"] for r in c.execute(
            "SELECT candidate_id FROM node_candidate WHERE fingerprint=?",
            (node_fp,))}
        cand = payload["candidate"]
        if cand not in valid:
            raise ValueError(f"candidate {cand} not present; legal: {sorted(valid)}")
        c.execute("UPDATE node_state SET chosen_candidate=? WHERE branch=? "
                  "AND event_seq=? AND key=?",
                  (cand, branch, seq, key))
        c.execute("INSERT INTO decision VALUES(?,?,?,?,?,?,?)",
                  (branch, seq, payload["ping"], payload["cell"],
                   payload["action"], cand, json.dumps(payload)))
                   cand, cand, json.dumps(payload)))
        return cand

    def _affected_keys(self, payload: dict, branch: str, seq: int) -> set:
        """Compute which node keys a source change can affect (local edges)."""
        n_cells = self.doc()["n_cells"]
        stream = payload.get("stream")
        if stream in ("correlation", "amplitude", "bad_beam_flag",
                      "beam_velocity"):
            return {f"p{payload['ping']+1:02d}:c{payload['cell']+1}"}
        if stream == "attitude":
            # gap only suppresses interpolation use at the missing ping itself
            return {f"p{payload['ping']+1:02d}:c{j+1}"
                    for j in range(n_cells)} | {f"p{payload['ping']+1:02d}:bt"}
        if stream == "mount":
            vfp = payload.get("valid_from_ping") or 0
            n = self.doc()["n_pings"]
            keys = set()
            for i in range(vfp, n):
                keys |= {f"p{i+1:02d}:c{j+1}" for j in range(n_cells)}
                keys.add(f"p{i+1:02d}:bt")
            return keys
        if stream == "transform":
            n = self.doc()["n_pings"]
            return {f"p{i+1:02d}:c{j+1}" for i in range(n)
                    for j in range(n_cells)} | \
                   {f"p{i+1:02d}:bt" for i in range(n)}
        if stream == "cell_invalid":
            return {f"p{payload['ping']+1:02d}:c{payload['cell']+1}"}
        if payload.get("action") == "select-candidate":
            return set()
        return set()

    def _write_dep_edges(self, c, branch, seq, payload, changed):
        stream = payload.get("stream")
        if not stream:
            return
        for key in changed:
            c.execute("INSERT INTO dep_edge VALUES(?,?,?,?,?,?)",
                      (branch, key, "stream", stream, stream, seq))

    # ---------------- read views -----------------------------------
    def _state_rows(self, branch: str, seq: int) -> dict:
        rows = self.conn.execute(
            """SELECT ns.* FROM node_state ns JOIN (
                 SELECT key, MAX(event_seq) m FROM node_state
                 WHERE branch=? AND event_seq<=? GROUP BY key) x
               ON ns.key=x.key AND ns.event_seq=x.m AND ns.branch=?""",
            (branch, seq, branch)).fetchall()
        return {r["key"]: {"fingerprint": r["node_fingerprint"],
                           "chosen": r["chosen_candidate"]} for r in rows}

    def node_at(self, branch: str, seq: int, key: str) -> Optional[dict]:
        row = self.conn.execute(
            """SELECT ns.* FROM node_state ns JOIN (
                 SELECT key, MAX(event_seq) m FROM node_state
                 WHERE branch=? AND event_seq<=? AND key=? GROUP BY key) x
               ON ns.key=x.key AND ns.event_seq=x.m AND ns.branch=?""",
            (branch, seq, key, branch)).fetchone()
        if not row:
            return None
        content = json.loads(self.conn.execute(
            "SELECT content_json FROM deriv_node WHERE fingerprint=?",
            (row["node_fingerprint"],)).fetchone()["content_json"])
        content["_chosen"] = row["chosen_candidate"]
        content["_fingerprint"] = row["node_fingerprint"]
        return content

    def events(self, branch: str = "main") -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM event WHERE branch=? ORDER BY seq",
            (branch,)).fetchall()
        out = []
        for r in rows:
            out.append({"branch": branch, "seq": r["seq"], "kind": r["kind"],
                        "payload": json.loads(r["payload_json"]),
                        "result": json.loads(r["result_json"]),
                        "parent_seq": r["parent_seq"]})
        return out

    def branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branch ORDER BY name").fetchall()]

    def validity(self, branch: str, seq: int) -> list[dict]:
        """Explain, per key, whether it is still valid or must recompute."""
        cur = self._state_rows(branch, seq)
        prev = self._state_rows(branch, seq - 1)
        out = []
        for key in sorted(cur):
            out.append({"key": key,
                        "changed": cur[key]["fingerprint"] != prev.get(key,
                                                                      {}).get("fingerprint"),
                        "fingerprint": cur[key]["fingerprint"],
                        "chosen": cur[key]["chosen"]})
        return out

    def source_trace(self, branch: str, key: str, seq: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM dep_edge WHERE branch=? AND key=? AND event_seq<=?",
            (branch, key, seq)).fetchall()
        return [dict(r) for r in rows]


def _ping_t(doc: dict, ping: int) -> float:
    head = {r["ping"]: r for r in doc["streams"]["heading"]}
    return float(head[ping]["t"])


def _ping_of(key: str) -> int:
    # keys look like p06:c3 or p06:bt
    return int(key.split(":")[0][1:]) - 1


def _rev_version(stream_map: dict) -> str:
    return fingerprint("revs", {"slots": {f"{k[0]}/{k[1]}/{k[2]}":
                                          v["revision"]
                                          for k, v in sorted(stream_map.items())}})
