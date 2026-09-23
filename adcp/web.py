"""FastAPI 应用：按 ping 展示波束质量、逐级向量变换、垂向剖面与被排除 cell。

每个显示值带稳定身份（record/derivation/candidate id）。
SVG 剖面在服务端生成，保证导出后仍可复现同一张图。
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import replay
from .repo import ConflictError, Repository
from .service import Service

DEFAULT_DB = os.environ.get("ADCP_DB", os.path.join(os.getcwd(), "data", "adcp.db"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    app = FastAPI(title="ADCP 流束换算台")
    app.state.db_path = path

    def get_repo(self) -> Repository:  # 每次请求短连接，便于并发测试
        return Repository(app.state.db_path)
    app._repo_factory = lambda: Repository(app.state.db_path)

    def repo() -> Repository:
        return Repository(app.state.db_path)

    @app.post("/api/demo/reset")
    def reset_demo():
        result = replay.run_case(app.state.db_path, reset=True)
        return {"ok": True, "head_seq": result["head_seq"], "steps": result["steps"],
                "landmarks": result["landmarks"]}

    @app.get("/api/state")
    def state():
        with repo() as r:
            return {"head_seq": r.head_seq(), "state": r.current_state(),
                    "derivations": r.current_derivations(),
                    "events": r.list_events(), "commits": r.list_commits(),
                    "decisions": r.decisions()}

    @app.get("/api/ping/{ping_id}")
    def ping_detail(ping_id: str):
        with repo() as r:
            derivs = [d for d in r.current_derivations()
                      if d.get("ping_id") == ping_id]
            records = [x for x in r.current_state()["records"]
                       if x["payload"].get("ping_id") == ping_id or x["scope"] == "global"]
            decisions = [d for d in r.decisions() if d["subject"].startswith(ping_id)]
            return {"ping_id": ping_id, "records": records, "derivations": derivs,
                    "decisions": decisions}

    @app.get("/api/cell/{cell_id}/explain")
    def explain(cell_id: str):
        with repo() as r:
            return r.explain(cell_id)

    @app.get("/api/history/{seq}")
    def history(seq: int):
        with repo() as r:
            if seq > r.head_seq():
                raise HTTPException(404, "seq in future")
            return r.state_at(seq)

    @app.get("/api/compare/{derivation_id}")
    def compare(derivation_id: str, a: int, b: int):
        with repo() as r:
            return Service(r).branch_compare(a, b, derivation_id)

    class EditBody(BaseModel):
        stream_id: str
        payload: dict[str, Any]
        base_version: int
        actor: str = "reviewer"
        note: str | None = None
        force: bool = False

    @app.post("/api/edit")
    def edit(body: EditBody):
        with repo() as r:
            try:
                res = Service(r).edit(body.stream_id, body.payload,
                                      base_version=body.base_version,
                                      actor=body.actor, note=body.note, force=body.force)
            except ConflictError as exc:
                return JSONResponse(status_code=409,
                                    content={"error": "conflict", "stream_id": exc.stream_id,
                                             "base_version": exc.base_version,
                                             "current_version": exc.current_version})
            return {"ok": True, "result": {"seq": res["seq"]}}

    class DecisionBody(BaseModel):
        scope: str
        subject: str
        action: str
        chosen_candidate_id: str | None = None
        actor: str = "reviewer"
        rationale: str | None = None

    @app.post("/api/decision")
    def decision(body: DecisionBody):
        with repo() as r:
            res = Service(r).decide(body.scope, body.subject, body.action,
                                    chosen_candidate_id=body.chosen_candidate_id,
                                    actor=body.actor, rationale=body.rationale)
            return {"ok": True, "decision": res}

    @app.get("/api/export")
    def export():
        with repo() as r:
            return r.export_bundle()

    @app.get("/api/profile.svg")
    def profile_svg(ping_id: str, component: str = "north"):
        with repo() as r:
            derivs = [d for d in r.current_derivations()
                      if d.get("ping_id") == ping_id and d["kind"] == "cell_flow"]
            svg = render_profile_svg(ping_id, derivs, component)
            return Response(svg, media_type="image/svg+xml")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def render_profile_svg(ping_id: str, derivs: list[dict], component: str) -> str:
    idx = {"north": 0, "east": 1, "down": 2}[component]
    derivs = sorted(derivs, key=lambda d: d["payload"]["depth_m"])
    width, height = 360, 260
    pad = 46
    vals = []
    for d in derivs:
        chosen = d["payload"]["default_candidate_id"]
        cand = next((c for c in d["payload"]["candidates"] if c["candidate_id"] == chosen), None)
        if cand and cand.get("earth"):
            vals.append((d["payload"]["depth_m"], cand["earth"][idx], d["payload"]["excluded"]))
    if not vals:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' \
               f'<text x="20" y="40">no resolved earth velocity (gap or unresolved ambiguity)</text></svg>'
    vmin = min(v[1] for v in vals) - 0.02
    vmax = max(v[1] for v in vals) + 0.02
    dmin, dmax = min(v[0] for v in vals), max(v[0] for v in vals)

    def x(v):
        return pad + (v - vmin) / max(vmax - vmin, 1e-9) * (width - 2 * pad)

    def y(depth):
        return pad + (depth - dmin) / max(dmax - dmin, 1e-9) * (height - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#fff"/>',
             f'<text x="{pad}" y="22" font-size="13" font-weight="bold">{ping_id} earth {component} (m/s)</text>']
    for gv in (vmin, (vmin + vmax) / 2, vmax):
        parts.append(f'<line x1="{x(gv):.1f}" y1="{pad}" x2="{x(gv):.1f}" y2="{height-pad}" stroke="#eee"/>')
        parts.append(f'<text x="{x(gv):.1f}" y="{height-pad+16}" font-size="10" text-anchor="middle">{gv:.2f}</text>')
    prev = None
    for depth, v, excluded in vals:
        if prev:
            parts.append(f'<line x1="{x(prev[1]):.1f}" y1="{y(prev[0]):.1f}" x2="{x(v):.1f}" y2="{y(depth):.1f}"'
                         f' stroke="#2b6cb0" stroke-width="1.5"/>')
        color = "#c53030" if excluded else "#2b6cb0"
        parts.append(f'<circle cx="{x(v):.1f}" cy="{y(depth):.1f}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="8" y="{y(depth)+3:.1f}" font-size="9">{depth:.0f}m</text>')
        prev = (depth, v)
    parts.append("</svg>")
    return "".join(parts)


app = create_app()
