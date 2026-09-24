"""Dependency-graph derivation engine.

Pipeline (each stage a separate node kind, all versions immutable):

``source -> attitude-at-ping / bottom / quality / instrument / ship / earth /
profile / coverage``

The engine always derives a *new* version from a snapshot of sources + active
config versions. Node stable ids are::

    n_{kind}_{ping}[_c{cell}][_{candidate}]

Changed inputs change the *fingerprint* (same node slot, new fingerprint);
  a changed candidate set changes the node ids present in that version.

Fingerprint inputs (the semantic invariants):

* transform order version, geometry version, declination version, mount
  version, quality-threshold version;
* stable ids + canonical numeric values of each upstream source.

Propagation is read off the diff against the parent version:

* unchanged node slot + unchanged fingerprint -> "still-valid";
* unchanged slot + changed fingerprint -> "recomputed";
* disappeared slot -> removed; new slot -> added.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..attitudes import AttitudeSample, Gap, interpolate_attitude
from ..fingerprints import RULE_VERSIONS, fingerprint, stable_id
from ..math3d import (
    BeamGeometry,
    bottom_track_range_depth,
    instrument_to_ship,
    ship_to_earth,
    solve_instrument,
)
from ..quality import QualityThresholds, beam_candidates, cell_in_water, classify_beam
from ..quantities import declination_corrected_heading


def _fid(kind: str, ping: str, cell: int | None = None,
         cand: str | None = None) -> str:
    base = f"n_{kind}_{ping}"
    if cell is not None:
        base += f"_c{cell}"
    if cand:
        base += f"_{cand}"
    return base


def _vid(node: str, frame: str, comp: str) -> str:
    return f"val_{node}_{frame}_{comp}"


class _SimulatedInterrupt(RuntimeError):
    """Raised inside a version transaction to emulate a write crash."""


def _safe_float(token) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _vec3(arr, frame: str) -> dict | None:
    if arr is None:
        return None
    comps = {"instrument": ("x", "y", "z"),
             "ship": ("fore", "starboard", "down"),
             "earth": ("east", "north", "up")}[frame]
    return {"frame": frame, "unit": "m/s",
            "components": list(comps),
            "values": [float(x) for x in np.asarray(arr).reshape(3)]}


def _attitude_view(sources: dict) -> tuple[list[AttitudeSample], list[Gap]]:
    samples = []
    present_times = set()
    for row in sources.values():
        if row["kind"] == "att_time":
            present_times.add((row["numeric"], row["stream_key"]))
    time_rows = sorted(
        (r for r in sources.values() if r["kind"] == "att_time"),
        key=lambda r: r["numeric"])
    for tr in time_rows:
        att_id = tr["stream_key"].split(":", 1)[1]
        h = sources.get(stable_id("src", att_id, "heading"))
        p = sources.get(stable_id("src", att_id, "pitch"))
        r = sources.get(stable_id("src", att_id, "roll"))
        if not (h and p and r and h["numeric"] is not None
                and p["numeric"] is not None and r["numeric"] is not None):
            continue
        samples.append(AttitudeSample(
            t=tr["numeric"], heading=h["numeric"], pitch=p["numeric"],
            roll=r["numeric"]))
    gaps = [Gap(*json.loads(r["raw"]))
            for r in sources.values() if r["kind"] == "attitude_gap"]
    return samples, gaps


def _chosen_candidate(decisions: list[dict] | None) -> str | None:
    if not decisions:
        return None
    # Last explicit decision wins for rendering; both remain stored.
    return decisions[-1]["choice"]


def _source_digest(sources: dict) -> list:
    return sorted(
        [sid, r["kind"], r["raw"], r["unit"], r["datum"], r["numeric"],
         r["diagnostic"]]
        for sid, r in sources.items())


class Engine:
    # ------------------------------------------------------------ operations

    def update_source(self, source_id: str, *, raw: str,
                      event_label: str, code: str = "local-source-edit",
                      actor: str | None = None,
                      payload_extra: dict | None = None) -> str:
        """Change one direct source in place (its stable id never changes).

        Only nodes whose fingerprint material names that source can change;
        the graph diff reports exactly that scope.
        """
        existing = self.store.get_source(source_id)
        if existing is None:
            raise KeyError(source_id)
        conn = self.store.begin()
        try:
            base = self.store.current_version()
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending", code=code,
                label=event_label, actor=actor,
                payload={"source_id": source_id, "old_raw": existing["raw"],
                         "new_raw": raw, "parent": base,
                         "seq_marker": f"{source_id}:{raw}"})
            numeric = existing["numeric"]
            diagnostic = existing["diagnostic"]
            try:
                numeric = float(raw)
                if existing["unit"] == "cm/s":
                    numeric /= 100.0
                elif existing["unit"] == "mm/s":
                    numeric /= 1000.0
            except ValueError:
                numeric = None
                diagnostic = f"unparseable-token:{raw!r}"
            conn.execute(
                "UPDATE sources SET raw=?, numeric=?, diagnostic=? "
                "WHERE source_id=?",
                (raw, numeric, diagnostic, source_id))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label=event_label, code=code, actor=actor,
            event_label=event_label, parent=base,
            payload_extra={"changed_source": source_id,
                           **(payload_extra or {})})

    def insert_gap_source(self, *, gap_id: str, kind: str, t0: float,
                          t1: float, event_label: str,
                          code: str = "attitude-gap-refused",
                          actor: str | None = None) -> str:
        base = self.store.current_version()
        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending", code=code,
                label=event_label, actor=actor,
                payload={"gap_id": gap_id, "kind": kind,
                         "gap": [t0, t1], "parent": base,
                         "seq_marker": f"{gap_id}:{t0}:{t1}"})
            self.store.insert_source(
                conn, created_event=seq, source_id=gap_id, kind=kind,
                raw=json.dumps([t0, t1]), unit="s", stream_key=kind,
                numeric=t1 - t0)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label=event_label, code=code, actor=actor,
            event_label=event_label, parent=base,
            payload_extra={"gap_id": gap_id})

    def remove_source(self, source_id: str, *, event_label: str,
                      code: str, actor: str | None = None) -> str:
        base = self.store.current_version()
        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending", code=code,
                label=event_label, actor=actor,
                payload={"removed_source": source_id, "parent": base,
                         "seq_marker": f"rm:{source_id}"})
            conn.execute("DELETE FROM sources WHERE source_id=?",
                         (source_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label=event_label, code=code, actor=actor,
            event_label=event_label, parent=base,
            payload_extra={"removed_source": source_id})

    def revise_config(self, config_type: str, version_id: str,
                      payload: dict, *, event_label: str, code: str,
                      retire: bool = True,
                      actor: str | None = None) -> str:
        base = self.store.current_version()
        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending", code=code,
                label=event_label, actor=actor,
                payload={"config_type": config_type,
                         "new_version": version_id, "parent": base,
                         "seq_marker": f"{config_type}:{version_id}"})
            if retire:
                active = self.store.active_config(config_type)
                if active:
                    self.store.upsert_config(
                        conn, config_type, active[0],
                        active[1], introduced_event=seq,
                        retired_event=seq)
                    conn.execute(
                        "UPDATE config_versions SET retired_event=? "
                        "WHERE config_type=? AND version_id=?",
                        (seq, config_type, active[0]))
            self.store.upsert_config(conn, config_type, version_id, payload,
                                     introduced_event=seq)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label=event_label, code=code, actor=actor,
            event_label=event_label, parent=base,
            payload_extra={"config_type": config_type,
                           "new_version": version_id})

    def human_decision(self, *, ping_id: str, choice: str, actor: str,
                       event_label: str) -> dict:
        """Record a human choice of candidate.

        Returns ``{"version": ..., "status": "ok"|"conflict", ...}``. A
        conflicting decision is stored and derives a sibling version; the
        current pointer keeps the earlier winner until explicit resolution.
        """
        base = self.store.current_version()
        candidate_id = None
        for node in self.store.nodes(base, ping_id=ping_id, kind="earth"):
            if node["candidate_id"] == choice:
                candidate_id = node["candidate_id"]
        if candidate_id is None:
            raise ValueError(f"no candidate {choice} for {ping_id}")
        decision = {
            "ping_id": ping_id, "choice": choice, "actor": actor,
            "chosen_candidate_id": candidate_id}
        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending",
                code="decision-recorded", label=event_label, actor=actor,
                payload={"ping": ping_id, "choice": choice,
                         "based_on_version": base,
                         "seq_marker": f"{ping_id}:{actor}:{choice}"})
            decision_id = stable_id("dec", ping_id, actor, choice, base)
            status = self.store.add_decision(conn, {
                "decision_id": decision_id, "version_id": "ver-pending",
                "based_on_version": base, "ping_id": ping_id,
                "chosen_candidate_id": candidate_id, "actor": actor,
                "created_event": seq})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        version = self.derive(
            label=event_label,
            code="decision-conflict" if status == "conflict"
            else "decision-recorded",
            actor=actor, event_label=event_label, parent=base,
            kind="decision",
            decisions=[{"ping_id": ping_id, "choice": choice,
                        "actor": actor,
                        "chosen_candidate_id": candidate_id}],
            payload_extra={"ping": ping_id, "choice": choice,
                           "decision_status": status})
        return {"version": version, "status": status,
                "decision_id": decision_id, "base": base}

    def resolve(self, *, keep: str, event_label: str,
                actor: str = "reviewer-b") -> str:
        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id="ver-pending",
                code="conflict-resolved", label=event_label, actor=actor,
                payload={"keep": keep,
                         "seq_marker": f"resolve:{keep}:{seq}"})
            rows = self.store.list_decisions()
            for row in rows:
                if row["status"] in ("accepted", "conflict"):
                    conn.execute(
                        "UPDATE decisions SET status='resolved', "
                        "resolution=? WHERE decision_id=?",
                        (keep, row["decision_id"]))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label=event_label, code="conflict-resolved", actor=actor,
            event_label=event_label,
            parent=self.store.current_version(), kind="resolution",
            payload_extra={"kept": keep})

    def propagation_diff(self, new_version: str,
                         base_version: str | None = None) -> dict:
        """Explain why each node is still valid or had to recompute."""
        base_version = base_version or self.store.get_version(
            new_version)["parent_id"]
        old = {n["node_id"]: n for n in self.store.nodes(base_version)} \
            if base_version else {}
        new = {n["node_id"]: n for n in self.store.nodes(new_version)}
        still_valid, recomputed, added, removed = [], [], [], []
        for nid, node in new.items():
            if nid not in old:
                added.append(nid)
            elif old[nid]["fingerprint"] == node["fingerprint"]:
                still_valid.append(nid)
            else:
                recomputed.append(nid)
        for nid in old:
            if nid not in new:
                removed.append(nid)
        return {
            "base_version": base_version,
            "new_version": new_version,
            "still_valid": sorted(still_valid),
            "recomputed": sorted(recomputed),
            "added": sorted(added),
            "removed": sorted(removed),
            "recompute_reason": self._reasons(new_version, base_version),
        }

    def _reasons(self, new_version: str, base_version: str) -> dict:
        ev = self.store.list_events()
        trigger = next(
            (e for e in reversed(ev)
             if e["version_id"] == new_version), None)
        if not trigger:
            return {}
        p = trigger["payload"]
        reason = {}
        if "changed_source" in p:
            sid = p["changed_source"]
            for node in self.store.nodes(new_version):
                if sid in node["inputs"]:
                    reason[node["node_id"]] = f"upstream-source:{sid}"
        if p.get("config_type"):
            ctype = p["config_type"]
            key = {"mount": "mount_version", "geometry": "geometry_version",
                   "declination": "declination_version",
                   "transform_order": "order_version"}.get(ctype)
            if key:
                for node in self.store.nodes(new_version):
                    if key in node["evidence"] or any(
                        key in str(inp) for inp in node["inputs"]):
                        reason[node["node_id"]] = f"config:{ctype}"
        return reason
    def __sim(self):
        return None

    def __init__(self, store):
        self.store = store

    # ------------------------------------------------------------- importing

    def import_case(self, pack: dict, *, actor: str = "importer") -> str:
        """Import raw sources separately; build the baseline version."""
        conn = self.store.begin()
        try:
            version_id = "ver-00-import"
            seq, _ = self.store.add_event(
                conn, version_id=version_id, code="case-imported",
                label="导入原始案例（各直接来源分开记录）", actor=actor,
                payload={"case_id": pack["case_id"],
                         "seq_marker": version_id})
            self._write_pack_sources(conn, pack, seq)
            for ctype, payload, idkey in (
                ("geometry", pack["sources"]["geometry"], "geometry_id"),
                ("mount", pack["sources"]["mount_angle"], "version_id"),
                ("declination", pack["sources"]["declination"], "version_id"),
                ("quality", pack["sources"]["quality_thresholds"],
                 "threshold_id"),
                ("transform_order", pack["sources"]["transform_order"],
                 "version_id"),
            ):
                self.store.upsert_config(
                    conn, ctype, payload[idkey],
                    payload if isinstance(payload, dict) else
                    {"version_id": payload}, introduced_event=seq)
            self.store.create_version(
                conn, version_id=version_id, parent_id=None,
                kind="import", label="原始导入", created_event=seq)
            self.store.commit_version(conn, version_id=version_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.derive(
            label="基线完整推导", code="baseline-derived", actor=actor,
            event_label="基线：beam→instrument→ship→earth 全部推导",
            parent=version_id,
            kind="derivation")

    def _write_pack_sources(self, conn, pack: dict, event_seq: int) -> None:
        s = pack["sources"]
        cfg = pack["config"]
        for i, gap in enumerate(s.get("ping_gaps", [])):
            self.store.insert_source(
                conn, created_event=event_seq,
                source_id=gap.get("gap_id", f"pgap-{i}"),
                kind="ping_gap", raw=json.dumps(
                    [gap["t0_s"], gap["t1_s"]]), unit="s",
                stream_key="ping", numeric=gap["t1_s"] - gap["t0_s"])
        for i, gap in enumerate(s.get("attitude_gaps", [])):
            self.store.insert_source(
                conn, created_event=event_seq,
                source_id=gap.get("gap_id", f"agap-{i}"),
                kind="attitude_gap", raw=json.dumps(
                    [gap["t0_s"], gap["t1_s"]]), unit="s",
                stream_key="attitude", numeric=gap["t1_s"] - gap["t0_s"])
        for a in s["attitude"]:
            for kind, key, unit, datum in (
                ("heading", "heading_raw", a["heading_unit"], "magnetic"),
                ("pitch", "pitch_raw", a["pitch_unit"], None),
                ("roll", "roll_raw", a["roll_unit"], None),
            ):
                self.store.insert_source(
                    conn, created_event=event_seq,
                    source_id=stable_id("src", a["att_id"], kind),
                    kind=kind, raw=a[key], unit=unit, datum=datum,
                    numeric=_safe_float(a[key]),
                    stream_key=f"attitude:{a['att_id']}")
            self.store.insert_source(
                conn, created_event=event_seq,
                source_id=stable_id("src", a["att_id"], "time"),
                kind="att_time", raw=a["t_s"], unit="s",
                numeric=a["t_s"], stream_key=f"attitude:{a['att_id']}")
        for ping in s["pings"]:
            pid = ping["ping_id"]
            self.store.insert_source(
                conn, created_event=event_seq,
                source_id=stable_id("src", pid, "time"),
                kind="ping_time", raw=ping["t_s"], unit="s",
                numeric=ping["t_s"], stream_key="ping")
            for cell in ping["cells"]:
                cno = cell["cell_no"]
                self.store.insert_source(
                    conn, created_event=event_seq,
                    source_id=stable_id("src", pid, cno, "centre"),
                    kind="cell_centre", raw=cell["centre_m"], unit="m",
                    numeric=cell["centre_m"], ping_id=pid, cell_no=cno)
                for beam in range(1, 5):
                    b = beam - 1
                    self._insert_meas(
                        conn, event_seq, pid, cno, beam, "beam_velocity",
                        cell["beam_velocity"]["values"][b],
                        cell["beam_velocity"]["unit"])
                    self._insert_meas(
                        conn, event_seq, pid, cno, beam, "correlation",
                        cell["correlation"]["values"][b],
                        cell["correlation"].get("unit", "counts"))
                    self._insert_meas(
                        conn, event_seq, pid, cno, beam, "amplitude",
                        cell["amplitude"]["values"][b],
                        cell["amplitude"].get("unit", "dB"))
                    self._insert_meas(
                        conn, event_seq, pid, cno, beam, "bad_flag",
                        cell["bad_flag"]["values"][b], "flag")
            bt = ping["bottom_track"]
            for beam in range(1, 5):
                b = beam - 1
                self._insert_meas(
                    conn, event_seq, pid, None, beam, "bt_range",
                    bt["range"]["values"][b], bt["range"]["unit"])
                self._insert_meas(
                    conn, event_seq, pid, None, beam, "bt_range_bad",
                    bt["range_bad_flag"]["values"][b], "flag")
                self._insert_meas(
                    conn, event_seq, pid, None, beam, "bt_velocity",
                    bt["beam_velocity"]["values"][b],
                    bt["beam_velocity"]["unit"])
                self._insert_meas(
                    conn, event_seq, pid, None, beam, "bt_velocity_bad",
                    bt["velocity_bad_flag"]["values"][b], "flag")
        self.store.upsert_config(
            conn, "pipeline", "pipeline-base",
            {"cell_size_m": cfg["cell_size_m"],
             "blanking_m": cfg["blanking_m"],
             "n_cells": cfg["n_cells"],
             "gap_max_spacing_s": cfg["gap_max_spacing_s"]},
            introduced_event=event_seq)

    def _insert_meas(self, conn, event_seq, pid, cno, beam, kind, raw,
                     unit) -> None:
        source_id = stable_id("src", pid, cno if cno is not None else "bt",
                              kind, beam)
        numeric = None
        diagnostic = None
        if kind == "bad_flag" or kind.endswith("_bad"):
            try:
                numeric = float(int(str(raw)))
            except ValueError:
                diagnostic = f"unparseable-flag:{raw!r}"
        else:
            try:
                numeric = float(raw)
                if unit in ("m/s",):
                    pass
                elif unit == "cm/s":
                    numeric /= 100.0
                elif unit == "mm/s":
                    numeric /= 1000.0
                elif unit == "cm":
                    numeric /= 100.0
                # counts, dB, deg, m, s remain numerically as given
            except ValueError:
                diagnostic = f"unparseable-token:{raw!r}"
        self.store.insert_source(
            conn, created_event=event_seq, source_id=source_id, kind=kind,
            raw=raw, unit=unit, numeric=numeric, diagnostic=diagnostic,
            ping_id=pid, cell_no=cno, beam=beam)

    # ------------------------------------------------------------- snapshot

    def _snapshot(self, at_event: int | None = None) -> dict:
        s = self.store
        geo_id, geo = s.active_config("geometry", at_event)
        mount_id, mount = s.active_config("mount", at_event)
        decl_id, decl = s.active_config("declination", at_event)
        qc_id, qc = s.active_config("quality", at_event)
        order_id, order_cfg = s.active_config("transform_order", at_event)
        _, pipe = s.active_config("pipeline", at_event)
        src_rows = s.list_sources()
        sources = {r["source_id"]: r for r in src_rows}
        return {
            "geometry_id": geo_id, "geometry": geo,
            "mount_id": mount_id, "mount": mount,
            "decl_id": decl_id, "decl": decl,
            "qc_id": qc_id, "qc": qc,
            "order_id": order_id,
            "order": order_cfg.get("order", "hpr-v1"),
            "pipeline": pipe,
            "sources": sources,
        }

    # ------------------------------------------------------------ derivation

    def derive(
        self,
        *,
        label: str,
        code: str,
        actor: str | None = None,
        event_label: str | None = None,
        parent: str | None = None,
        kind: str = "derivation",
        decisions: list[dict] | None = None,
        fail_kind: str | None = None,
        payload_extra: dict | None = None,
    ) -> str:
        """Run the full graph over the current snapshot into a new version.

        The transaction either ends with the new version committed + current
        pointer moved, or it is rolled back leaving the previous current
        version intact (``fail_kind`` simulates that interruption for tests).
        """
        base = parent or self.store.current_version()
        # version id is deterministic in its inputs
        snap_for_id = self._snapshot()
        version_id = stable_id(
            "ver", code, base, actor, json.dumps(decisions, sort_keys=True),
            json.dumps(payload_extra, sort_keys=True, default=str),
            snap_for_id["geometry_id"], snap_for_id["mount_id"],
            snap_for_id["decl_id"], snap_for_id["qc_id"],
            snap_for_id["order_id"],
            _source_digest(snap_for_id["sources"]))

        conn = self.store.begin()
        try:
            seq, _ = self.store.add_event(
                conn, version_id=version_id, code=code, label=event_label or
                label, actor=actor,
                payload={"parent": base, "kind": kind,
                         "seq_marker": f"{version_id}:{code}",
                         **(payload_extra or {})})
            self.store.create_version(
                conn, version_id=version_id, parent_id=base, kind=kind,
                label=label, created_event=seq, based_on_version=base)
            snap = self._snapshot(seq)
            nodes, edges = self._build_graph(snap, version_id,
                                              decisions or [])
            if fail_kind:
                # Simulate a crash before the pointer switch.
                raise _SimulatedInterrupt(fail_kind)
            for node in nodes:
                self.store.insert_node(conn, node)
            for up, down in edges:
                self.store.insert_edge(conn, version_id, up, down)
            self.store.commit_version(conn, version_id=version_id)
            conn.commit()
        except Exception:
            conn.rollback()
            if fail_kind:
                return version_id
            raise
        return version_id

    def _build_graph(self, snap: dict, version_id: str,
                     decisions: list[dict]) -> tuple[list[dict], list[tuple]]:
        sources = snap["sources"]
        geom = BeamGeometry(snap["geometry_id"],
                            float(snap["geometry"]["beam_angle_deg"]))
        thresholds = QualityThresholds(
            snap["qc_id"],
            corr_min=int(snap["qc"]["corr_min"]),
            corr_marginal=int(snap["qc"]["corr_marginal"]),
            amp_min_db=float(snap["qc"]["amp_min_db"]))
        cfg = snap["pipeline"]
        cell_size = float(cfg["cell_size_m"])
        blanking = float(cfg["blanking_m"])
        max_gap = float(cfg["gap_max_spacing_s"])
        n_cells = int(cfg["n_cells"])

        att_samples, att_gaps = _attitude_view(sources)
        ping_gaps = [Gap(g[0], g[1])
                     for r in sources.values() if r["kind"] == "ping_gap"
                     for g in [json.loads(r["raw"])]]

        ping_times = sorted(
            (r["numeric"], r["ping_id"])
            for r in sources.values() if r["kind"] == "ping_time")

        decisions_by_ping = {}
        for d in decisions:
            decisions_by_ping.setdefault(d["ping_id"], []).append(d)

        nodes: list[dict] = []
        edges: list[tuple[str, str]] = []
        node_index: dict[str, dict] = {}

        def put(node: dict) -> None:
            nodes.append(node)
            node_index[node["node_id"]] = node

        # ----- coverage nodes: ping-stream gaps
        for _, pid in ping_times:
            t = sources[stable_id("src", pid, "time")]["numeric"]
            in_gap = next((g for g in ping_gaps if g.contains(t)), None)
            cov_id = _fid("coverage", pid)
            put({
                "node_id": cov_id, "version_id": version_id,
                "scope": "ping", "ping_id": pid, "cell_no": None,
                "kind": "coverage", "candidate_id": None,
                "status": "gap" if in_gap else "ok",
                "fingerprint": fingerprint(RULE_VERSIONS["quality"], {
                    "ping": pid, "t": t,
                    "gap": [in_gap.t0, in_gap.t1] if in_gap else None}),
                "value": {"in_ping_gap": bool(in_gap),
                          "gap": [in_gap.t0, in_gap.t1] if in_gap else None},
                "value_ids": [],
                "inputs": [stable_id("src", pid, "time")],
                "evidence": {"ping_gap_sources":
                             [r["source_id"] for r in sources.values()
                              if r["kind"] == "ping_gap"]},
                "diagnostics":
                    ([f"ping-gap[{in_gap.t0},{in_gap.t1}]"]
                     if in_gap else []),
            })

        # ----- one attitude node per ping
        att_node_by_ping: dict[str, str] = {}
        for _, pid in ping_times:
            t = sources[stable_id("src", pid, "time")]["numeric"]
            interp = interpolate_attitude(
                att_samples, att_gaps, t, max_spacing_s=max_gap)
            inputs = [stable_id("src", pid, "time")]
            diagnostic = interp["diagnostic"]
            h_mag = interp["heading"]
            h_true = (None if h_mag is None else
                      declination_corrected_heading(
                          h_mag, float(snap["decl"]["angle_deg"])))
            value = {
                "ok": interp["ok"], "heading_magnetic_deg": h_mag,
                "heading_true_deg": h_true,
                "pitch_deg": interp["pitch"], "roll_deg": interp["roll"],
                "declination_deg": float(snap["decl"]["angle_deg"]),
                "heading_datum": "true",
            }
            # bracket sources as evidence
            bracket = interp.get("bracket")
            if bracket is not None:
                for ai in bracket:
                    att_id = f"a{ai:02d}"
                    for k in ("heading", "pitch", "roll", "time"):
                        inputs.append(stable_id("src", att_id, k))
            inputs += [snap["decl_id"]]
            node_id = _fid("attitude", pid)
            att_node_by_ping[pid] = node_id
            put({
                "node_id": node_id, "version_id": version_id,
                "scope": "ping", "ping_id": pid, "cell_no": None,
                "kind": "attitude", "candidate_id": None,
                "status": "ok" if interp["ok"] else "refused",
                "fingerprint": fingerprint(
                    RULE_VERSIONS["attitude_interp"], {
                        "t": t, "bracket": bracket,
                        "attitude_gaps":
                            [[g.t0, g.t1] for g in att_gaps],
                        "max_spacing_s": max_gap,
                        "declination_version": snap["decl_id"],
                        "headings_mag": h_mag,
                        "declination_deg": snap["decl"]["angle_deg"]}),
                "value": value,
                "value_ids": ([
                    _vid(node_id, "true-heading", "deg"),
                    _vid(node_id, "pitch", "deg"),
                    _vid(node_id, "roll", "deg")] if interp["ok"] else []),
                "inputs": inputs,
                "evidence": {
                    "declination_version": snap["decl_id"],
                    "interpolation_rule":
                        RULE_VERSIONS["attitude_interp"],
                    "bracket_indices": bracket},
                "diagnostics": ([diagnostic] if diagnostic else []),
            })
            edges.append((_fid("coverage", pid), node_id))

        # ----- bottom nodes
        bottom_node_by_ping: dict[str, str] = {}
        for _, pid in ping_times:
            ranges, valid, inp, diags = [], [], [], []
            for beam in range(1, 5):
                rr = sources[stable_id("src", pid, "bt", "bt_range", beam)]
                bb = sources[stable_id("src", pid, "bt", "bt_range_bad", beam)]
                inp += [rr["source_id"], bb["source_id"]]
                if rr["numeric"] is None:
                    valid.append(False)
                    ranges.append(np.nan)
                    diags.append(rr["diagnostic"] or "range-missing")
                else:
                    valid.append(int(float(bb["raw"])) == 0
                                 if bb["numeric"] is not None else False)
                    ranges.append(rr["numeric"])
            depth = bottom_track_range_depth(
                np.array(ranges, dtype=float),
                np.array(valid, dtype=bool), geom)
            node_id = _fid("bottom", pid)
            bottom_node_by_ping[pid] = node_id
            put({
                "node_id": node_id, "version_id": version_id,
                "scope": "ping", "ping_id": pid, "cell_no": None,
                "kind": "bottom", "candidate_id": None,
                "status": "ok" if depth["depth_m"] is not None else "refused",
                "fingerprint": fingerprint(
                    RULE_VERSIONS["bottom_depth"], {
                        "ranges": ranges, "valid": valid,
                        "geometry_version": snap["geometry_id"],
                        "geometry_id2": geom.geometry_id}),
                "value": {"bottom_depth_m": depth["depth_m"],
                          "n_beams": depth["n_beams"],
                          "per_beam_vertical_m": depth["per_beam_m"],
                          "unit": "m"},
                "value_ids": ([_vid(node_id, "bottom-depth", "m")]
                              if depth["depth_m"] is not None else []),
                "inputs": inp + [snap["geometry_id"]],
                "evidence": {"rule": RULE_VERSIONS["bottom_depth"],
                             "geometry_version": snap["geometry_id"]},
                "diagnostics": diags,
            })
        nodes, edges = self._cells(
            snap, version_id, decisions_by_ping, ping_times, sources, geom,
            thresholds, cell_size, blanking, n_cells, nodes, edges,
            att_node_by_ping, bottom_node_by_ping, node_index)
        return nodes, edges

    def _cells(self, snap, version_id, decisions_by_ping, ping_times,
               sources, geom, thresholds, cell_size, blanking, n_cells,
               nodes, edges, att_node_by_ping, bottom_node_by_ping,
               node_index):
        for _, pid in ping_times:
            t_src = sources[stable_id("src", pid, "time")]
            att_node = node_index[att_node_by_ping[pid]]
            bottom_node = node_index[bottom_node_by_ping[pid]]
            att_ok = att_node["status"] == "ok"
            bottom_depth = bottom_node["value"]["bottom_depth_m"]

            chosen_for_ping = _chosen_candidate(
                decisions_by_ping.get(pid, []))

            for cno in range(1, n_cells + 1):
                beam_vals, corr_vals, amp_vals, flags = [], [], [], []
                classes = []
                source_ids: list[str] = []
                parse_diags: list[str] = []
                for beam in range(1, 5):
                    v = sources[stable_id(
                        "src", pid, cno, "beam_velocity", beam)]
                    c = sources[stable_id(
                        "src", pid, cno, "correlation", beam)]
                    a = sources[stable_id(
                        "src", pid, cno, "amplitude", beam)]
                    f = sources[stable_id(
                        "src", pid, cno, "bad_flag", beam)]
                    source_ids += [v["source_id"], c["source_id"],
                                   a["source_id"], f["source_id"]]
                    for row in (v, c, a):
                        if row["diagnostic"]:
                            parse_diags.append(
                                f"{row['kind']} b{beam}: "
                                f"{row['diagnostic']}")
                    beam_vals.append(v["numeric"])
                    corr_vals.append(
                        None if c["numeric"] is None else int(c["numeric"]))
                    amp_vals.append(a["numeric"])
                    flags.append(int(float(f["raw"])) if f["numeric"] is not None
                                 else True)
                    classes.append(classify_beam(
                        bad_flag=bool(flags[-1]),
                        correlation=corr_vals[-1],
                        amplitude_db=amp_vals[-1],
                        thresholds=thresholds))
                centre = sources[stable_id(
                    "src", pid, cno, "centre")]["numeric"]
                placement = cell_in_water(
                    cell_centre_m=centre, cell_size_m=cell_size,
                    blanking_m=blanking, bottom_depth_m=bottom_depth)

                quality_id = _fid("quality", pid, cno)
                cand_enum = beam_candidates(classes)
                nodes.append({
                    "node_id": quality_id, "version_id": version_id,
                    "scope": "cell", "ping_id": pid, "cell_no": cno,
                    "kind": "quality", "candidate_id": None,
                    "status": "ok" if cand_enum["subsets"] and
                    not parse_diags else "excluded",
                    "fingerprint": fingerprint(
                        RULE_VERSIONS["quality"], {
                            "beam_velocity": beam_vals,
                            "correlation": corr_vals,
                            "amplitude": amp_vals,
                            "bad_flags": flags,
                            "threshold_version": snap["qc_id"],
                            "thresholds": snap["qc"],
                            "placement": placement,
                            "bottom_depth_m": bottom_depth}),
                    "value": {
                        "classifications": classes,
                        "candidate_subsets":
                            [list(m) for m in cand_enum["subsets"]],
                        "hard_invalid_beams": cand_enum["hard_invalid"],
                        "suspect_beams": cand_enum["suspect"],
                        "placement": placement,
                        "correlation_unit": "counts",
                        "amplitude_unit": "dB",
                    },
                    "value_ids": [],
                    "inputs": source_ids + [
                        snap["qc_id"], bottom_node_by_ping[pid],
                        stable_id("src", pid, cno, "centre")],
                    "evidence": {
                        "threshold_version": snap["qc_id"],
                        "placement_rule":
                            RULE_VERSIONS["cell_placement"],
                        "bottom_node": bottom_node_by_ping[pid]},
                    "diagnostics": parse_diags + placement["reasons"],
                })
                edges.append((bottom_node_by_ping[pid], quality_id))

                excluded = (not placement["valid"]) or (
                    not cand_enum["subsets"])
                if excluded:
                    # Excluded cell: no velocity nodes, but the exclusion is
                    # itself an evidence-bearing node.
                    ex_id = _fid("excluded", pid, cno)
                    nodes.append({
                        "node_id": ex_id, "version_id": version_id,
                        "scope": "cell", "ping_id": pid, "cell_no": cno,
                        "kind": "excluded_cell", "candidate_id": None,
                        "status": "excluded",
                        "fingerprint": fingerprint(
                            RULE_VERSIONS["cell_placement"], {
                                "placement": placement,
                                "subsets":
                                    [list(m) for m in
                                     cand_enum["subsets"]],
                                "diags": parse_diags,
                                "bottom_depth_m": bottom_depth,
                                "threshold_version": snap["qc_id"]}),
                        "value": {"reasons": placement["reasons"]
                                  + parse_diags},
                        "value_ids": [],
                        "inputs": [quality_id],
                        "evidence": {"quality_node": quality_id},
                        "diagnostics": placement["reasons"] + parse_diags,
                    })
                    edges.append((quality_id, ex_id))
                    continue

                self._velocity_candidates(
                    snap, version_id, pid, cno, beam_vals, cand_enum,
                    classes, quality_id, att_node, att_ok, geom,
                    chosen_for_ping, nodes, edges, t_src)

            self._profile_node(
                snap, version_id, pid, n_cells, chosen_for_ping,
                att_node_by_ping[pid], bottom_node_by_ping[pid],
                decisions_by_ping.get(pid, []), nodes, edges)
        return nodes, edges

    # --------------------------------------------------- candidate velocities

    def _velocity_candidates(self, snap, version_id, pid, cno, beam_vals,
                             cand_enum, classes, quality_id, att_node,
                             att_ok, geom, chosen_for_ping, nodes, edges,
                             t_src):
        vb = np.array(
            [np.nan if x is None else x for x in beam_vals], dtype=float)
        for mask in cand_enum["subsets"]:
            n_beams = int(sum(mask))
            cand = f"b{n_beams}"
            if n_beams == 3:
                missing = next(i + 1 for i, m in enumerate(mask) if not m)
                cand += f"-drop{missing}"
            solved = solve_instrument(vb, geom, np.array(mask, dtype=bool))
            inputs = list(quality_id and [quality_id])
            inputs.append(snap["geometry_id"])
            value: dict[str, Any] = {
                "beam_mask": list(mask), "n_beams": n_beams,
                "rank": solved["rank"]}
            diagnostics: list[str] = []
            status = "candidate"
            if not att_ok:
                status = "blocked-attitude"
            instrument = ship = earth = None
            error_velocity = solved["error_velocity"]
            if solved["v"] is not None:
                instrument = solved["v"]
                ship = instrument_to_ship(
                    instrument, float(snap["mount"]["angle_deg"]))
                value["error_velocity_m_s"] = error_velocity
                value["residual_beam_m_s"] = (
                    None if solved["residual"] is None
                    else [float(x) for x in solved["residual"]])
            if att_ok and solved["v"] is not None:
                earth = ship_to_earth(
                    ship,
                    att_node["value"]["heading_true_deg"],
                    att_node["value"]["pitch_deg"],
                    att_node["value"]["roll_deg"],
                    order=snap["order"])
                inputs += [att_node["node_id"], snap["mount_id"],
                           snap["order_id"],
                           att_node["evidence"]["declination_version"]]
            else:
                diagnostics += att_node["diagnostics"]

            ambiguous = len(cand_enum["subsets"]) > 1
            if chosen_for_ping == cand:
                status = "accepted"
            elif chosen_for_ping is not None:
                status = "retained-not-chosen"
            elif not ambiguous:
                status = "unique"
            if not att_ok:
                status = "blocked-attitude"

            chain = {
                "instrument": _vec3(instrument, "instrument"),
                "ship": _vec3(ship, "ship"),
                "earth": _vec3(earth, "earth"),
            }
            value.update(chain)
            value["mount_angle_deg"] = float(snap["mount"]["angle_deg"])
            value["transform_order"] = snap["order"]

            fp_material = {
                "beam_velocity_m_s": [None if x is None else float(x)
                                      for x in beam_vals],
                "mask": list(mask),
                "geometry_version": snap["geometry_id"],
                "geometry": snap["geometry"],
                "mount_version": snap["mount_id"],
                "mount": snap["mount"],
                "declination_version": snap["decl_id"],
                "declination": snap["decl"],
                "order_version": snap["order_id"],
                "order": snap["order"],
                "attitude_fingerprint": att_node["fingerprint"],
                "rule_versions": {
                    "instrument": RULE_VERSIONS["instrument_solve"],
                    "ship": RULE_VERSIONS["ship_transform"],
                    "earth": (RULE_VERSIONS["earth_transform"]
                              if snap["order"] == "hpr-v1"
                              else RULE_VERSIONS["earth_transform_alt"])},
            }

            value_ids = []
            if solved["v"] is not None:
                for frame, vec in (("instrument", instrument),
                                   ("ship", ship), ("earth", earth)):
                    if vec is None:
                        continue
                    for comp, val in zip(("x", "y", "z"), vec):
                        value_ids.append(
                            f"val_{_fid(frame, pid, cno, cand)}_{comp}")
                value_ids.append(
                    f"val_{_fid('errv', pid, cno, cand)}_mps")
            for frame in ("instrument", "ship", "earth", "errv"):
                node_id = _fid(frame, pid, cno, cand)
                frame_value: dict[str, Any]
                if frame == "errv":
                    frame_value = {"error_velocity_m_s": error_velocity,
                                   "n_beams": n_beams}
                else:
                    frame_value = {
                        "vector": chain[frame],
                        "frame": frame,
                        "unit": "m/s",
                        "n_beams": n_beams,
                        "candidate": cand,
                        "ambiguous_with_others": ambiguous}
                nodes.append({
                    "node_id": node_id, "version_id": version_id,
                    "scope": "cell", "ping_id": pid, "cell_no": cno,
                    "kind": frame, "candidate_id": cand,
                    "status": status,
                    "fingerprint": fingerprint(
                        RULE_VERSIONS["earth_transform"]
                        if frame == "earth" else
                        (RULE_VERSIONS["ship_transform"]
                         if frame == "ship" else
                         (RULE_VERSIONS["instrument_solve"]
                          if frame == "instrument" else "error-velocity-v1")),
                        {"frame": frame, **fp_material}),
                    "value": frame_value,
                    "value_ids": [v for v in value_ids
                                  if (frame != "errv" and
                                      v.startswith(f"val_{node_id}"))
                                  or (frame == "errv" and
                                      v.startswith(f"val_{node_id}"))],
                    "inputs": inputs,
                    "evidence": {
                        "quality_node": quality_id,
                        "geometry_version": snap["geometry_id"],
                        "mount_version": snap["mount_id"],
                        "declination_version": snap["decl_id"],
                        "order_version": snap["order_id"],
                        "attitude_node": att_node["node_id"],
                        "candidate_mask": list(mask),
                        "beam_classifications": classes},
                    "diagnostics": diagnostics,
                })
                edges.append((quality_id, node_id))
                if frame == "instrument":
                    edges.append((node_id,
                                  _fid("ship", pid, cno, cand)))
                    edges.append((att_node["node_id"],
                                  _fid("earth", pid, cno, cand)))
                if frame == "ship":
                    edges.append((node_id, _fid("earth", pid, cno, cand)))

    def _profile_node(self, snap, version_id, pid, n_cells,
                      chosen_for_ping, att_node_id, bottom_node_id,
                      decisions, nodes, edges):
        # Aggregate the *accepted/unique* earth candidate of each valid cell.
        # Excluded cells are listed, never silently dropped.
        layers = []
        excluded = []
        candidates_seen = set()
        for node in nodes:
            if node["ping_id"] != pid:
                continue
            if node["kind"] == "excluded_cell":
                excluded.append({"cell_no": node["cell_no"],
                                 "reasons": node["diagnostics"]})
            if node["kind"] == "earth" and node["cell_no"] is not None:
                candidates_seen.add((node["cell_no"], node["candidate_id"]))
                if node["status"] in ("accepted", "unique"):
                    layers.append({
                        "cell_no": node["cell_no"],
                        "candidate": node["candidate_id"],
                        "earth": node["value"]["vector"],
                        "value_ids": node["value_ids"]})
        layers.sort(key=lambda d: d["cell_no"])
        ambiguity = [
            {"cell_no": c, "candidate": k}
            for (c, k) in sorted(candidates_seen)
            if sum(1 for (cc, kk) in candidates_seen if cc == c) > 1]
        status = "ok"
        diags: list[str] = []
        if ambiguity:
            status = "ambiguous"
            diags.append("multiple-legal-candidates-present")
        if excluded:
            diags.append(f"{len(excluded)}-cells-excluded")
        if any(n["ping_id"] == pid and n["kind"] == "attitude"
               and n["status"] == "refused" for n in nodes):
            status = "blocked-attitude"
        node_id = _fid("profile", pid)
        nodes.append({
            "node_id": node_id, "version_id": version_id,
            "scope": "ping", "ping_id": pid, "cell_no": None,
            "kind": "profile", "candidate_id": None,
            "status": status,
            "fingerprint": fingerprint("profile-assembly-v1", {
                "layers": layers, "excluded": excluded,
                "ambiguity": ambiguity,
                "chosen": chosen_for_ping,
                "order_version": snap["order_id"],
                "geometry_version": snap["geometry_id"],
                "declination_version": snap["decl_id"]}),
            "value": {
                "frame": "earth", "unit": "m/s",
                "layers": layers, "excluded_cells": excluded,
                "ambiguous_cells": ambiguity,
                "chosen_candidate": chosen_for_ping},
            "value_ids": sorted(
                v for layer in layers for v in layer["value_ids"]),
            "inputs": sorted(
                {n["node_id"] for n in nodes if n["ping_id"] == pid
                 and n["kind"] in ("earth", "excluded_cell")}),
            "evidence": {
                "attitude_node": att_node_id,
                "bottom_node": bottom_node_id,
                "decisions": [
                    {"actor": d["actor"], "choice": d["choice"]}
                    for d in decisions]},
            "diagnostics": diags,
        })
