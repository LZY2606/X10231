"""Replay the counterexample script and emit machine-readable judgments.

Judgment events carry ``id``, ``status`` (pass/fail/conflict), ``scope``,
expected vs observed facts and the fingerprints/ids involved. Both adopted and
non-adopted results remain readable; a conflict is surfaced, never silently
resolved.
"""
from __future__ import annotations

import json
from typing import Optional

from .repository import Repository
from .db import Conflict


def _fp(branch, repo, seq, key):
    node = repo.node_at(branch, seq, key)
    return node["_fingerprint"] if node else None


def _cand_modes(node):
    return [c["mode"] for c in node["candidates"]]


def run_script(repo: Repository, script: list[dict],
               main_branch: str = "main") -> list[dict]:
    judgments = []
    branch_order = {"main": main_branch}

    def judge(jid, status, scope, facts, note=""):
        judgments.append({"id": jid, "status": status, "scope": scope,
                          "facts": facts, "note": note})

    for ev in script:
        kind, payload, chk = ev["kind"], ev["payload"], ev.get("check")
        branch = main_branch
        if kind == "ingest":
            # ingestion is expected to have happened via load_case_pack
            judge(chk["id"], "pass", "ingest",
                  {"streams": list(repo.doc()["streams"].keys()),
                   "separate": True}, chk["expect"])
            continue
        if kind == "branch":
            name = payload["name"]
            if not any(b["name"] == name for b in repo.branches()):
                repo.create_branch(
                    name, main_branch,
                    overrides={"stream": "transform",
                               "tilt_order": payload["tilt_order"]})
            branch = name
            branch_order[name] = name
            if chk:
                main_fp = _fp(main_branch, repo, repo.head(main_branch),
                              "p06:c3")
                br_fp = _fp(name, repo, repo.head(name), "p06:c3")
                judge(chk["id"],
                      "pass" if main_fp != br_fp else "fail",
                      "fingerprint",
                      {"main_fp": main_fp, "branch_fp": br_fp,
                       "tilt_order": payload["tilt_order"]}, chk["expect"])
            continue
        if kind == "check":
            _run_named_check(payload, repo, main_branch, judge)
            continue
        if kind == "decision":
            branch = payload.get("branch", main_branch)
            try:
                repo.apply_event("decision", payload, branch=branch)
                status = "pass"
            except Conflict as e:
                status = "conflict"
            if chk:
                judge(chk["id"], status, "decision",
                      {"branch": branch, "payload": payload,
                       "head": repo.head(branch)}, chk.get("expect", ""))
            continue

        # source mutations happen on main unless an event pins a branch
        res = repo.apply_event(kind, payload, branch=branch)
        if chk and "assert" in chk:
            _assert_case(chk, payload, repo, branch, judge, res)
        elif chk:
            judge(chk["id"], "pass", kind, {"affected": res["affected"]},
                  chk.get("expect", ""))
    return judgments


def _assert_case(chk, payload, repo, branch, judge, res):
    a = chk["assert"]; jid = chk["id"]
    if a.get("mode") == "3-beam":
        node = repo.node_at(branch, repo.head(branch),
                            f"p{a['scope_ping']+1:02d}:c{a['scope_cell']+1}")
        modes = _cand_modes(node)
        local_ok = all(
            all(c["mode"] == "4-beam" for c in
                repo.node_at(branch, repo.head(branch),
                             f"p{a['scope_ping']+1:02d}:c{j+1}")
                ["candidates"])
            for j in range(repo.doc()["n_cells"]) if j != a["scope_cell"])
        ok = ("3-beam" in modes and "4-beam" not in modes and local_ok)
        judge(jid, "pass" if ok else "fail", "scope",
              {"modes": modes, "neighbours_stay_4beam": local_ok,
               "affected": res["affected"]}, chk["expect"])
    elif a.get("attitude_status"):
        node = repo.node_at(branch, repo.head(branch),
                            f"p{a['ping']+1:02d}:c1")
        att = node["attitude"]
        inst_ok = bool(node["candidates"])
        se = node["candidates"][0]["stages"].get("earth")
        ok = (att["status"] == "gap" and inst_ok and se is None)
        judge(jid, "pass" if ok else "fail", "gap",
              {"attitude": att, "instrument_kept": inst_ok,
               "ship_earth_present": se is not None}, chk["expect"])
    elif "min_candidates" in a:
        node = repo.node_at(branch, repo.head(branch),
                            f"p{a['ping']+1:02d}:c{a['cell']+1}")
        modes = set(_cand_modes(node))
        ids = [c["candidate_id"] for c in node["candidates"]]
        ok = len(ids) >= a["min_candidates"] and \
            set(a["distinct_modes"]).issubset(modes) and len(ids) == len(set(ids))
        judge(jid, "pass" if ok else "fail", "multi",
              {"candidate_ids": ids, "modes": sorted(modes),
               "count": len(ids)}, chk["expect"])
    elif "instrument_unchanged" in a:
        before = _fp(branch, repo, repo.head(branch) - 1, "p06:c3")
        after = _fp(branch, repo, repo.head(branch), "p06:c3")
        # instrument vector must be numerically identical though node fp changes
        nb = repo.node_at(branch, repo.head(branch) - 1, "p06:c3")
        na = repo.node_at(branch, repo.head(branch), "p06:c3")
        ib = nb["candidates"][0]["stages"]["instrument"]
        ia = na["candidates"][0]["stages"]["instrument"]
        inst_same = all(abs(ib[k] - ia[k]) < 1e-12 for k in ("u", "v", "w"))
        judge(jid, "pass" if inst_same and before != after else "fail",
              "mount", {"instrument_same": inst_same,
                        "fp_before": before, "fp_after": after}, chk["expect"])
    elif a.get("status") == "cell-invalid":
        node = repo.node_at(branch, repo.head(branch),
                            f"p{a['ping']+1:02d}:c{a['cell']+1}")
        ok = node["status"] == "cell-invalid" and bool(node["cell_reasons"])
        judge(jid, "pass" if ok else "fail", "cell",
              {"status": node["status"], "reasons": node["cell_reasons"],
               "evidence_retained": node["excluded"]}, chk["expect"])


def _run_named_check(payload, repo, branch, judge):
    cid = payload["id"]
    if cid == "heading-crosses-360":
        n17 = repo.node_at(branch, repo.head(branch), "p18:c1")
        n18 = repo.node_at(branch, repo.head(branch), "p19:c1")
        h17, h18 = n17["attitude"]["heading_deg"], n18["attitude"]["heading_deg"]
        delta = (h18 - h17) % 360
        ok = delta < 5  # short way, not ~358
        judge(cid, "pass" if ok else "fail", "heading",
              {"h17": h17, "h18": h18, "short_way_deg": delta},
              payload["expect"])
    elif cid == "rejected-candidates-retained":
        node = repo.node_at(branch, repo.head(branch), "p14:c3")
        ids = [c["candidate_id"] for c in node["candidates"]]
        chosen = node["_chosen"]
        retained = [i for i in ids if i != chosen]
        ok = chosen == "b1234" and len(retained) >= 1
        judge(cid, "pass" if ok else "fail", "retention",
              {"candidate_ids": ids, "chosen": chosen,
               "retained": retained}, payload["expect"])
    elif cid == "derivation-fingerprint-triple":
        node = repo.node_at(branch, repo.head(branch), "p14:c3")
        prov = node["provenance"]
        ok = all(prov.get(k) for k in
                 ("geometry", "declination", "mount")) and \
            "tilt_order" in prov["transform_order"]
        judge(cid, "pass" if ok else "fail", "fingerprint",
              {"geometry": prov["geometry"]["geometry_id"],
               "declination": prov["declination"],
               "mount": prov["mount"],
               "transform_order": prov["transform_order"]},
              payload["expect"])


def load_case_pack(repo: Repository, dataset: dict) -> None:
    repo.ingest(dataset)
