"""FastAPI review application for the ADCP evidence archive.

Pages show, per ping: beam quality, the staged instrument->ship->earth vector
transforms, the vertical profile and excluded-cell evidence. Every rendered
value carries a stable node uid/fp or source sid; actions create new events
and commits, they never overwrite history.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adcp import replay
from adcp.storage import EditConflict, Store
from adcp.web import evidence_chain, excluded_view, ping_view

DB_PATH = os.environ.get("ADCP_DB", "adcp_evidence.sqlite3")

app = FastAPI(title="ADCP 流束换算台证据工具")
app.mount("/static", StaticFiles(directory="static"), name="static")

_state: Dict[str, Any] = {"store": None, "refs": {}, "base_commit": None}


def get_store() -> Store:
    if _state["store"] is None:
        _state["store"] = Store(DB_PATH)
        if _state["store"].branch_head("main") is None:
            base = replay.seed(_state["store"])
            _state["base_commit"] = base
            _state["refs"] = replay.play(_state["store"], base)
    return _state["store"]


class DecisionIn(BaseModel):
    ping: int
    cell: int
    candidate_id: str
    rationale: str = ""
    actor: str = "reviewer"
    base_commit: Optional[str] = None
    branch: str = "main"


class ParameterIn(BaseModel):
    key: str
    value: Dict[str, Any]
    actor: str = "hydrologist"
    note: str = ""
    base_commit: Optional[str] = None


class SourceIn(BaseModel):
    sid: str
    value: Any
    unit: str
    actor: str = "field-team"
    note: str = ""
    label: str = "manual-source-edit"


class ConflictIn(BaseModel):
    conflict_id: str
    resolution: str


def _nodes(cid: Optional[str]):
    store = get_store()
    cid = cid or store.branch_head("main")
    commit = store.get_commit(cid)
    if commit is None:
        raise HTTPException(404, f"unknown commit {cid}")
    return store, cid, commit, store.get_nodes(cid)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/state")
def state() -> Dict[str, Any]:
    store = get_store()
    return {"head": store.branch_head("main"),
            "base_commit": _state.get("base_commit"),
            "branches": store.list_branches(),
            "refs": _state.get("refs", {}),
            "event_count": len(store.list_events())}


@app.get("/api/events")
def events(branch: Optional[str] = None):
    store = get_store()
    out = []
    for e in store.list_events(branch):
        item = dict(e)
        item["payload"] = json.loads(e["payload_json"])
        out.append(item)
    return out


@app.get("/api/commits")
def commits(branch: Optional[str] = None):
    store = get_store()
    q = "SELECT * FROM commits WHERE complete=1"
    args: tuple = ()
    if branch:
        q += " AND branch=?"
        args = (branch,)
    q += " ORDER BY created_at"
    return [dict(r) for r in store.conn.execute(q, args)]


@app.get("/api/commit/{cid}")
def commit_detail(cid: str):
    store, _, commit, nodes = _nodes(cid)
    return {"commit": {k: commit[k] for k in
                       ("cid", "parent_cid", "branch", "label", "kind",
                        "note", "created_at")},
            "node_count": len(nodes), "uids": sorted(nodes)}


@app.get("/api/ping/{pno}")
def ping(pno: int, cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    return {"cid": cid0, "view": ping_view(nodes, pno)}


@app.get("/api/profile")
def profile(cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    p = nodes.get("profile")
    return {"cid": cid0, "profile": p["value"] if p else None}


@app.get("/api/excluded")
def excluded(cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    return {"cid": cid0, "excluded": excluded_view(nodes)}


@app.get("/api/node/{uid}")
def node(uid: str, cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    if uid not in nodes:
        raise HTTPException(404, uid)
    history = store.node_history(uid)
    return {"cid": cid0, "node": nodes[uid],
            "chain": evidence_chain(nodes, uid),
            "history": history}


@app.post("/api/decision")
def decide(d: DecisionIn):
    store = get_store()
    base = d.base_commit or store.branch_head(d.branch)
    decision = {"ping": d.ping, "cell": d.cell,
                "candidate_id": d.candidate_id, "rationale": d.rationale,
                "kind": "candidate_adoption"}
    from adcp.fixture import build_view
    from adcp.pipeline import run
    store.log_decision(decision, d.actor)
    result = run(build_view(store.active_pack(), [decision]))
    nodes = store.propagate(base, result["nodes"])
    try:
        out = store.commit(
            d.branch, nodes, "decision", "decision",
            f"人工裁定 ping{d.ping}/cell{d.cell} -> {d.candidate_id}",
            base, d.actor, "candidate_decided",
            {"ping": d.ping, "cell": d.cell,
             "candidate_id": d.candidate_id,
             "unadopted_kept": True,
             "propagation": store.propagation_report()},
            decisions=[decision])
    except EditConflict as exc:
        raise HTTPException(409, {"error": "edit_conflict",
                                  "conflict_id": exc.conflict_id})
    return out


@app.post("/api/source")
def edit_source(src: SourceIn, base_commit: Optional[str] = None):
    store = get_store()
    base = base_commit or store.branch_head("main")
    rev = store.add_source_revision(src.sid, src.value, src.unit, src.actor,
                                    src.note)
    from adcp.fixture import build_view
    from adcp.pipeline import run
    result = run(build_view(store.active_pack()))
    nodes = store.propagate(base, result["nodes"])
    try:
        out = store.commit("main", nodes, src.label, "source_change",
                           src.note or "直接来源修订", base, src.actor,
                           "local_source_change",
                           {"sid": src.sid, "rev": rev["rev"],
                            "propagation": store.propagation_report()})
    except EditConflict as exc:
        raise HTTPException(409, {"error": "edit_conflict",
                                  "conflict_id": exc.conflict_id})
    return out


@app.post("/api/parameter")
def edit_parameter(p: ParameterIn):
    store = get_store()
    base = p.base_commit or store.branch_head("main")
    eid = store.update_parameter(p.key, p.value, p.actor, p.note)
    from adcp.fixture import build_view
    from adcp.pipeline import run
    result = run(build_view(store.active_pack()))
    nodes = store.propagate(base, result["nodes"])
    try:
        out = store.commit("main", nodes, f"param-{p.key}", "parameter",
                           p.note or "参数修订", base, p.actor,
                           "parameter_revised",
                           {"key": p.key, "event_id": eid,
                            "propagation": store.propagation_report()})
    except EditConflict as exc:
        raise HTTPException(409, {"error": "edit_conflict",
                                  "conflict_id": exc.conflict_id})
    return out


@app.post("/api/conflict/resolve")
def resolve_conflict(c: ConflictIn):
    store = get_store()
    eid = store.resolve_conflict(c.conflict_id, c.resolution)
    return {"event_id": eid}


@app.get("/api/conflicts")
def conflicts():
    return get_store().list_conflicts()


@app.get("/api/replay")
def replay_state():
    get_store()
    return {"refs": _state.get("refs", {}),
            "base_commit": _state.get("base_commit")}


@app.get("/api/export")
def export_bundle():
    bundle = get_store().export_bundle()
    return JSONResponse(bundle,
                        headers={"Content-Disposition":
                                 "attachment; filename=adcp-bundle.json"})


@app.post("/api/import")
async def import_bundle(request: "starlette.requests.Request"):
    """Load an exported bundle into a fresh temp DB and swap it in."""
    from starlette.requests import Request  # noqa: F401
    bundle = await request.json()
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    os.unlink(path)
    new_store = Store(path)
    new_store.migrate_from(bundle)
    old = _state.get("store")
    _state["store"] = new_store
    if old is not None:
        old.close()
    _state["base_commit"] = None
    return {"imported": True,
            "head": new_store.branch_head("main"),
            "events": len(new_store.list_events())}


@app.get("/api/svg/quality/{pno}")
def svg_quality(pno: int, cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    view = ping_view(nodes, pno)
    from adcp.svg import quality_matrix
    return Response(quality_matrix(view), media_type="image/svg+xml")


@app.get("/api/svg/profile")
def svg_profile(cid: Optional[str] = None):
    store, cid0, _, nodes = _nodes(cid)
    p = nodes.get("profile")
    from adcp.svg import profile_plot
    return Response(profile_plot(p["value"] if p else {"cells": []}),
                    media_type="image/svg+xml")
