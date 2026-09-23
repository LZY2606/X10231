"""Deterministic replay of the stream_desk_v1 counter-example archive.

Every step changes at most one direct source or opens one branch. Both
adopted and un-adopted outputs remain addressable by fingerprint.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

from .fingerprint import content_id
from .model import load_case_pack
from .service import CASE_DIR, EvidenceService, bootstrap_case
from .store import Store


@dataclass
class ReplayReport:
    seqs: dict[str, int] = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)

    def record(self, step: str, name: str, ok: bool, detail: dict) -> None:
        self.findings.append({"step": step, "name": name, "ok": ok,
                              "detail": detail})


def run_replay(db_path: str = ":memory:") -> tuple[Store, ReplayReport]:
    store = Store(db_path)
    svc = EvidenceService(store)
    report = ReplayReport()

    seq = bootstrap_case(store)
    report.seqs["S0"] = seq

    def slot_status(commit_seq: int, slot: str) -> dict:
        return store.slots_at(commit_seq)[slot]

    # S1: single beam failure -> three beam, distinct from four beam.
    s = slot_status(seq, "ping:P02:cell:2")
    report.record("S1", "single beam failure is three-beam",
                  s["status"] == "three_beam", {"slot_status": s["status"]})

    # S2: heading wrap -> aligned without huge jump.
    state = store.slots_at(seq)
    t4 = state[f"attitude:t:{1006.0:.6g}"]["fingerprint"]
    t5 = state[f"attitude:t:{1008.0:.6g}"]["fingerprint"]
    h4 = store.derivation(t4)["payload"]["heading_true_deg"]
    h5 = store.derivation(t5)["payload"]["heading_true_deg"]
    wrap_ok = abs((h5 - h4 + 180) % 360 - 180) < 2.0
    report.record("S2", "359.4->0.6 wrap stays local (<2 deg step)",
                  wrap_ok, {"h_true_p03": h4, "h_true_p04": h5})

    # S3: time gap -> P07 blocked, no interpolation across.
    gap = slot_status(seq, "ping:P07:cell:0")
    report.record("S3", "attitude gap blocks earth transform",
                  gap["status"] == "attitude_unavailable",
                  {"slot_status": gap["status"]})

    # S4: multi-solution retained.
    amb = slot_status(seq, "ping:P06:cell:3")
    der = store.derivation(amb["fingerprint"])
    ncands = len(der["payload"].get("three_beam_candidates", []))
    has4 = der["payload"].get("four_beam_transform") is not None
    report.record("S4", "3- and 4-beam solutions both retained",
                  amb["status"] == "ambiguous_three_or_four"
                  and ncands >= 1 and has4,
                  {"status": amb["status"], "three_beam_candidates": ncands})

    # S5: mount revision.
    mount_v2 = load_case_pack(CASE_DIR, mount_file="mount_v2.json").mount
    seq5 = svc.revise_document(
        "doc:mount", mount_v2, author="surveyor",
        label="S5 mount revision v2",
        principles=["P-mount-versioned", "P-order-fingerprinted"])
    report.seqs["S5"] = seq5
    fp_before = slot_status(seq, "ping:P01:cell:0")["fingerprint"]
    fp_after = slot_status(seq5, "ping:P01:cell:0")["fingerprint"]
    report.record("S5", "mount revision changes downstream fingerprints",
                  fp_before != fp_after,
                  {"before": fp_before[:24], "after": fp_after[:24]})

    # S6: wrong transform order branch.
    seq6 = svc.create_branch_with_order(
        "alt-order", "v2_fused_yaw", author="reviewer",
        label="S6 fused-yaw order experiment")
    report.seqs["S6"] = seq6
    main_fp = slot_status(seq5, "ping:P01:cell:0")["fingerprint"]
    alt_fp = store.slots_at(seq6)["ping:P01:cell:0"]["fingerprint"]
    alt_order = store.derivation(alt_fp)["transform_order"]
    report.record("S6", "transform order enters fingerprint on a branch",
                  main_fp != alt_fp and alt_order == "v2_fused_yaw",
                  {"main_fp": main_fp[:24], "alt_fp": alt_fp[:24]})

    # S7: invalid depth cell.
    cell = slot_status(seq5, "ping:P09:cell:5")
    der = store.derivation(cell["fingerprint"])
    report.record("S7", "invalid depth cell excluded with reason",
                  cell["status"] == "cell_excluded" and
                  "cell_not_configured" in
                  der["payload"]["cell_verdict"]["reasons"],
                  {"reasons": der["payload"]["cell_verdict"]["reasons"]})

    # S8: declination revision shifts only heading-dependent results.
    decl_v2 = load_case_pack(CASE_DIR, decl_file="declination_v2.json").declination
    seq8 = svc.revise_document(
        "doc:declination", decl_v2, author="geophysicist",
        label="S8 declination v2",
        principles=["P-decl-versioned"])
    report.seqs["S8"] = seq8
    # Instrument solution must be identical; earth solution must shift.
    inst_a = store.derivation(slot_status(seq5, "ping:P01:cell:0")[
        "fingerprint"])["payload"]["transform"]["instrument_ms"]
    inst_b = store.derivation(slot_status(seq8, "ping:P01:cell:0")[
        "fingerprint"])["payload"]["transform"]["instrument_ms"]
    earth_a = store.derivation(slot_status(seq5, "ping:P01:cell:0")[
        "fingerprint"])["payload"]["transform"]["earth_ms"]
    earth_b = store.derivation(slot_status(seq8, "ping:P01:cell:0")[
        "fingerprint"])["payload"]["transform"]["earth_ms"]
    report.record(
        "S8", "declination changes earth but not instrument vectors",
        inst_a == inst_b and earth_a != earth_b,
        {"earth_before": earth_a, "earth_after": earth_b})

    # S9: local-only source correction on P11.
    def mutate(target):
        before = target["beam_velocity"][0]["raw"]
        target["beam_velocity"][0]["raw"] = "0.0012"
        target["beam_velocity"][0]["value"] = 0.0012
        target["beam_velocity"][0]["canonical"] = 0.0012
        target["beam_velocity"][0]["diagnostic"] = None
        return before

    seq9, plan = svc.revise_ping_cell(
        "P11", 0, mutate, author="qa",
        label="S9 local correction P11 b1")
    report.seqs["S9"] = seq9
    p01_fp = slot_status(seq8, "ping:P01:cell:0")["fingerprint"]
    p01_after = slot_status(seq9, "ping:P01:cell:0")["fingerprint"]
    local_ok = p01_fp == p01_after and all(
        "P11" in s for s in plan["affected_slots"] if s.startswith("ping:"))
    still_valid = any(
        e["action"] == "still_valid" and e["slot"] == "ping:P01:cell:0"
        for e in plan["unaffected_explanations"])
    report.record("S9", "only P11-local slots recompute",
                  local_ok and still_valid,
                  {"affected": plan["affected_slots"][:6],
                   "p01_still_valid": still_valid})

    # S10: conflicting human decisions are resolvable.
    seq10a = svc.decide(
        "ping:P06:cell:3", "accept_four_beam",
        rejected_candidates=["dropped_beam:3"],
        note="reviewer A keeps 4-beam least squares")
    base_before = store.sources_at(seq10a)["doc:mount"]["version"]
    mount_tweak = copy.deepcopy(
        store.sources_at(seq10a)["doc:mount"]["payload"])
    mount_tweak["gamma_deg"] = -1.05
    # Reviewer A commits a mount change first.
    seq10b = svc.revise_document(
        "doc:mount", mount_tweak, author="reviewer-B",
        label="S10 reviewer B mount tweak",
        principles=["P-mount-versioned"])
    # Reviewer B (second user) still holds the pre-change base version.
    store.stage_edit("doc:mount", mount_tweak,
                     base_record_version=base_before, allow_stale=True)
    conflicts = store.edit_conflicts()
    # Resolution: rebase onto current version, then the edit stages cleanly.
    resolved_base = store.active_source_version("doc:mount")
    store.discard_edits()
    store.stage_edit("doc:mount", mount_tweak,
                     base_record_version=resolved_base)
    assert not store.edit_conflicts()
    store.discard_edits()
    seq10 = svc.decide(
        "ping:P06:cell:3", "accept_three_beam",
        rejected_candidates=["four_beam"],
        note="reviewer B chooses beam-4-dropped after comparison",
        author="reviewer-B")
    report.seqs["S10"] = seq10
    decisions = store.decisions_at(seq10)
    verdicts = {d["verdict"] for d in decisions}
    rejected_kept = all(d["rejected_candidates"] for d in decisions)
    report.record(
        "S10", "conflicting decisions coexist; rejected candidates retained",
        len(conflicts) == 1 and conflicts[0]["resolvable"]
        and {"accept_four_beam", "accept_three_beam"} <= verdicts
        and rejected_kept,
        {"conflicts": conflicts,
         "decision_verdicts": sorted(verdicts)})

    # History immutability: replaying S5 restores the v1 declination.
    replay5 = svc.replay_event(seq5)
    decl_at_5 = replay5["sources"]["doc:declination"]["version"]
    report.record(
        "history", "old event restores old dependency versions",
        decl_at_5 != store.sources_at(seq10)["doc:declination"]["version"],
        {"decl_version_at_s5": decl_at_5[:20]})

    return store, report


def report_to_json(report: ReplayReport) -> str:
    return json.dumps(
        {"seqs": report.seqs, "findings": report.findings}, indent=2)
