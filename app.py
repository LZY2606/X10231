"""FastAPI application: review UI + evidence API."""
from __future__ import annotations

import copy
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from adcp_evidence.replay import run_replay
from adcp_evidence.service import (
    CASE_DIR, EvidenceService, bootstrap_case, materials_to_sources,
)
from adcp_evidence.model import load_case_pack
from adcp_evidence.store import ConflictError, Store
from adcp_evidence.svg import beam_quality_svg, profile_svg, vector_chain_svg

DB_PATH = os.environ.get("ADCP_DB", "data/adcp.sqlite")

app = FastAPI(title="ADCP evidence workbench")
_service: EvidenceService | None = None


def get_service() -> EvidenceService:
    global _service
    if _service is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        store = Store(DB_PATH)
        if store.head_seq("main") is None:
            bootstrap_case(store)
        _service = EvidenceService(store)
    return _service


class DocRevision(BaseModel):
    record_id: str
    payload: dict
    author: str = "reviewer"
    label: str = "document revision"
    branch: str = "main"


class DecisionIn(BaseModel):
    subject: str
    verdict: str
    rejected_candidates: list = []
    note: str = ""
    author: str = "reviewer"
    branch: str = "main"


class BranchIn(BaseModel):
    name: str
    order: str = "v2_fused_yaw"


class EditIn(BaseModel):
    record_id: str
    payload: dict
    base_version: str
    allow_stale: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    from adcp_evidence.web import PAGE
    return PAGE


@app.get("/api/state")
def state(branch: str = "main", seq: int | None = None) -> dict:
    svc = get_service()
    if seq is None:
        seq = svc.store.head_seq(branch)
    commit = svc.store.commit_row(seq)
    if commit is None:
        raise HTTPException(404, "no such commit")
    sources = svc.store.sources_at(seq)
    slots = svc.store.slots_at(seq)
    pings = sorted(
        (rec for rid, rec in sources.items()
         if rec["record_kind"] == "ping"),
        key=lambda rec: rec["payload"]["t_s"],
    )
    return {
        "seq": seq, "branch": branch,
        "commit_label": commit["label"],
        "head": svc.store.head_seq(branch),
        "branches": svc.store.branches(),
        "timeline": svc.store.timeline(branch),
        "ping_ids": [rec["payload"]["ping_id"] for rec in pings],
        "decisions": svc.store.decisions_at(seq),
        "documents": {
            rid: {"version": rec["version"]}
            for rid, rec in sources.items() if rid.startswith("doc:")
        },
        "slot_status_count": _count_status(slots),
    }


def _count_status(slots: dict) -> dict:
    counts: dict[str, int] = {}
    for spec in slots.values():
        counts[spec["status"]] = counts.get(spec["status"], 0) + 1
    return counts


@app.get("/api/ping/{ping_id}")
def ping_detail(ping_id: str, branch: str = "main",
                seq: int | None = None) -> dict:
    svc = get_service()
    seq = seq or svc.store.head_seq(branch)
    sources = svc.store.sources_at(seq)
    rid = f"ping:{ping_id}"
    if rid not in sources:
        raise HTTPException(404, ping_id)
    record = sources[rid]["payload"]
    cells = []
    for cell in record["cells"]:
        slot = f"{rid}:cell:{cell['index']}"
        spec = svc.store.slots_at(seq).get(slot)
        entry = {"slot": slot, "cell": cell,
                 "source_version": sources[rid]["version"]}
        if spec and spec["fingerprint"]:
            der = svc.store.derivation(spec["fingerprint"])
            entry["status"] = spec["status"]
            entry["fingerprint"] = spec["fingerprint"]
            entry["derivation"] = der["payload"]
        cells.append(entry)
    return {
        "ping_id": ping_id, "t_s": record["t_s"],
        "source_version": sources[rid]["version"],
        "bottom_track": record.get("bottom_track"),
        "cells": cells,
    }


@app.get("/api/profile")
def profile(branch: str = "main", seq: int | None = None):
    svc = get_service()
    seq = seq or svc.store.head_seq(branch)
    spec = svc.store.slots_at(seq)["profile:earth_enu"]
    der = svc.store.derivation(spec["fingerprint"])
    return {"seq": seq, "fingerprint": spec["fingerprint"],
            "profile": der["payload"],
            "svg": profile_svg(der["payload"]["ensembles"])}


@app.get("/api/svg/ping/{ping_id}/cell/{cell}")
def cell_svg(ping_id: str, cell: int, kind: str = "vectors",
             branch: str = "main", seq: int | None = None) -> Response:
    svc = get_service()
    seq = seq or svc.store.head_seq(branch)
    slot = f"ping:{ping_id}:cell:{cell}"
    spec = svc.store.slots_at(seq).get(slot)
    if not spec or not spec["fingerprint"]:
        raise HTTPException(404)
    payload = svc.store.derivation(spec["fingerprint"])["payload"]
    if kind == "beams":
        svg = beam_quality_svg(payload["beam_quality"])
    elif kind == "vectors":
        chain = payload.get("transform") or payload.get(
            "four_beam_transform")
        svg = vector_chain_svg(chain) if chain else (
            "<svg xmlns='http://www.w3.org/2000/svg'/>")
    else:
        raise HTTPException(400, kind)
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/value/{fingerprint}")
def value(fingerprint: str) -> dict:
    svc = get_service()
    der = svc.store.derivation(fingerprint)
    if der is None:
        raise HTTPException(404, fingerprint)
    return der


@app.get("/api/timeline")
def timeline(branch: str = "main") -> dict:
    svc = get_service()
    return {"branch": branch, "timeline": svc.store.timeline(branch)}


@app.get("/api/explain")
def explain(branch: str = "main") -> dict:
    svc = get_service()
    return svc.explain_propagation(branch)


@app.post("/api/revise-document")
def revise_document(body: DocRevision) -> dict:
    svc = get_service()
    seq = svc.revise_document(body.record_id, body.payload,
                              author=body.author, label=body.label,
                              branch=body.branch)
    return {"seq": seq}


@app.post("/api/decision")
def decision(body: DecisionIn) -> dict:
    svc = get_service()
    seq = svc.decide(body.subject, body.verdict,
                     rejected_candidates=body.rejected_candidates,
                     note=body.note, author=body.author, branch=body.branch)
    return {"seq": seq}


@app.post("/api/branch")
def branch(body: BranchIn) -> dict:
    svc = get_service()
    seq = svc.create_branch_with_order(
        body.name, body.order, author="reviewer",
        label=f"branch {body.name} ({body.order})")
    return {"seq": seq, "branch": body.name}


@app.post("/api/stage")
def stage(body: EditIn) -> dict:
    svc = get_service()
    try:
        svc.store.stage_edit(body.record_id, body.payload,
                             body.base_version, allow_stale=body.allow_stale)
    except ConflictError as exc:
        raise HTTPException(409, str(exc))
    return {"conflicts": svc.store.edit_conflicts(),
            "staged": list(svc.store.staged_edits())}


@app.get("/api/work")
def work() -> dict:
    svc = get_service()
    return {"staged": svc.store.staged_edits(),
            "conflicts": svc.store.edit_conflicts()}


@app.post("/api/reset")
def reset() -> dict:
    global _service
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    store = Store(DB_PATH)
    bootstrap_case(store)
    _service = EvidenceService(store)
    return {"seq": store.head_seq("main")}


@app.post("/api/replay")
def replay() -> dict:
    global _service
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    store, report = run_replay(DB_PATH)
    _service = EvidenceService(store)
    return {"seq": store.head_seq("main"),
            "seqs": report.seqs,
            "findings": report.findings,
            "all_ok": all(f["ok"] for f in report.findings)}


@app.get("/api/export")
def export() -> JSONResponse:
    svc = get_service()
    return JSONResponse(svc.store.export_bundle())


@app.get("/api/manifest")
def manifest() -> dict:
    with open(os.path.join(CASE_DIR, "MANIFEST.json"), encoding="utf-8") as f:
        return json.load(f)
