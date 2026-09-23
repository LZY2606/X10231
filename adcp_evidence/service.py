"""Application services orchestrating storage, engine and replay actions."""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

from .engine import compute_state
from .fingerprint import canonical_json, content_id, event
from .model import load_case_pack
from .propagate import propagation_plan
from .store import ConflictError, Store
from . import RULE_VERSION


CASE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "cases", "stream_desk_v1"
)


def materials_to_sources(materials) -> dict[str, dict]:
    sources = {}
    for rid, payload in materials.record_items():
        if rid.startswith("doc:"):
            key = rid
        elif payload["record_kind"] == "ping":
            key = f"ping:{payload['ping_id']}"
        else:
            key = f"attitude:{payload['sample_id']}"
        sources[key] = {
            "version": content_id("src", payload),
            "record_kind": payload["record_kind"],
            "payload": payload,
        }
    return sources


def bootstrap_case(store: Store, case_dir: str = CASE_DIR,
                   *, mount_file="mount_v1.json",
                   decl_file="declination_v1.json",
                   order="v1_separated",
                   crash_after_rows: bool = False) -> int:
    materials = load_case_pack(case_dir, mount_file=mount_file,
                               decl_file=decl_file)
    sources = materials_to_sources(materials)
    state = compute_state(sources, order=order)
    changes = {
        rid: {"action": "upsert", "version": rec["version"],
              "payload": rec["payload"]}
        for rid, rec in sources.items()
    }
    events = [
        event("baseline_imported", "case:stream_desk_v1",
              principles=["P-raw-preserved"],
              detail={"source_records": len(sources),
                      "separate_kinds": [
                          "beam_velocity", "correlation", "amplitude",
                          "bad_beam_flag", "attitude", "declination",
                          "mount", "transducer", "bottom_track"]}),
        *state["events"],
    ]
    return store.commit(
        branch="main", author="importer", label="S0 baseline import",
        source_changes=changes, derivations=state["derivations"],
        slot_status=state["slot_status"], events=events,
        crash_after_rows=crash_after_rows,
    )


def _source_payloads(sources: dict) -> dict:
    return copy.deepcopy({rid: rec["payload"] for rid, rec in sources.items()})


class EvidenceService:
    def __init__(self, store: Store, order: str = "v1_separated"):
        self.store = store
        self.order = order

    # ---------- revisions ----------
    def revise_document(self, record_id: str, new_payload: dict, *,
                        author: str, label: str,
                        principles: list[str] | None = None,
                        branch: str = "main") -> int:
        sources = self.store.sources_at(self.store.head_seq(branch))
        if record_id not in sources:
            raise KeyError(record_id)
        version = content_id("src", new_payload)
        sources[record_id] = {
            "version": version,
            "record_kind": new_payload.get("record_kind", "unknown"),
            "payload": copy.deepcopy(new_payload),
        }
        state = compute_state(sources, order=self.order)
        events = [
            event("source_revision", record_id,
                  principles=principles or ["P-mount-versioned",
                                            "P-decl-versioned"],
                  detail={"new_version": version,
                          "new_payload": new_payload}),
            *state["events"],
        ]
        return self.store.commit(
            branch=branch, author=author, label=label,
            source_changes={
                record_id: {"action": "upsert", "version": version,
                            "payload": new_payload}
            },
            derivations=state["derivations"],
            slot_status=state["slot_status"], events=events,
        )

    def revise_ping_cell(self, ping_id: str, cell: int, mutate, *,
                         author: str, label: str,
                         branch: str = "main") -> tuple[int, dict]:
        rid = f"ping:{ping_id}"
        sources = self.store.sources_at(self.store.head_seq(branch))
        record = copy.deepcopy(sources[rid]["payload"])
        target = next(c for c in record["cells"] if c["index"] == cell)
        before = mutate(target)
        version = content_id("src", record)
        sources[rid] = {"version": version,
                        "record_kind": "ping", "payload": record}
        state = compute_state(sources, order=self.order)
        plan = propagation_plan(
            self._last_graph(), self.store.slots_at(
                self.store.head_seq(branch)),
            state,
            self.store.sources_at(self.store.head_seq(branch)), sources,
        )
        events = [
            event("source_correction_local", rid,
                  principles=["P-raw-preserved"],
                  detail={"cell": cell, "before": before,
                          "affected_slots": plan["affected_slots"]}),
            *state["events"],
        ]
        seq = self.store.commit(
            branch=branch, author=author, label=label,
            source_changes={rid: {"action": "upsert", "version": version,
                                  "payload": record}},
            derivations=state["derivations"],
            slot_status=state["slot_status"], events=events,
        )
        self._graph_cache = state["graph"]
        return seq, plan

    def _last_graph(self) -> dict:
        return getattr(self, "_graph_cache", {})

    # ---------- branches ----------
    def create_branch_with_order(self, name: str, order: str, *,
                                 author: str, label: str) -> int:
        sources = self.store.sources_at(self.store.head_seq("main"))
        state = compute_state(sources, order=order)
        events = [
            event("transform_order_branch", f"branch:{name}",
                  principles=["P-order-fingerprinted",
                              "P-geometry-fingerprinted"],
                  detail={"old_order": self.order, "new_order": order}),
            *state["events"],
        ]
        return self.store.commit(
            branch=name, author=author, label=label,
            source_changes={}, derivations=state["derivations"],
            slot_status=state["slot_status"], events=events,
            parent_branch="main",
        )

    # ---------- decisions ----------
    def decide(self, subject: str, verdict: str, *,
               rejected_candidates: list | None = None,
               note: str = "", author: str = "reviewer",
               branch: str = "main") -> int:
        """Record accept/reject/adjust without deleting rejected candidates."""
        sources = self.store.sources_at(self.store.head_seq(branch))
        state = compute_state(sources, order=self.order)
        decision_id = content_id(
            "dec",
            {"subject": subject, "verdict": verdict, "note": note,
             "rejected": rejected_candidates or [],
             "at": self.store.head_seq(branch)})
        events = [
            event(f"decision_{verdict}", subject,
                  principles=["P-history-immutable",
                              "P-multiple-candidates"],
                  detail={"decision_id": decision_id,
                          "rejected_candidates_kept": True,
                          "note": note}),
            *state["events"],
        ]
        return self.store.commit(
            branch=branch, author=author,
            label=f"decision {verdict}: {subject}",
            source_changes={}, derivations=state["derivations"],
            slot_status=state["slot_status"], events=events,
            decisions=[{
                "id": decision_id, "subject": subject, "verdict": verdict,
                "rejected_candidates": rejected_candidates or [],
                "note": note,
            }],
        )

    def replay_event(self, seq: int, branch: str = "main") -> dict:
        """Return the full materialized view at an old commit."""
        return {
            "seq": seq,
            "sources": self.store.sources_at(seq),
            "slots": self.store.slots_at(seq),
            "decisions": self.store.decisions_at(seq),
            "commit": dict(self.store.commit_row(seq)),
        }

    def explain_propagation(self, branch: str = "main") -> dict:
        seq = self.store.head_seq(branch)
        commit = self.store.commit_row(seq)
        if commit["parent_seq"] is None:
            return {"changed_records": {}, "affected_slots": [],
                    "unaffected_explanations": []}
        old_sources = self.store.sources_at(commit["parent_seq"])
        new_sources = self.store.sources_at(seq)
        # Rebuild graphs from stored slot fingerprints/deps is not persisted
        # separately, so recompute cheaply from both views.
        old_state = compute_state(old_sources, order=self.order)
        new_state = compute_state(new_sources, order=self.order)
        return propagation_plan(
            old_state["graph"], self.store.slots_at(commit["parent_seq"]),
            new_state, old_sources, new_sources)
