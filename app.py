"""FastAPI review UI for the ADCP transform-evidence toolkit.

Pages walk one continuous narrative: import -> verify separated records ->
trigger boundary combinations -> branch / decide -> review per-ping evidence
(beam quality, staged vector transforms, vertical profile, excluded cells).
Every displayed value carries a stable identity (version + node key).
"""
from __future__ import annotations

import html
import json
import os

from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from adcp.casepack import replay
from adcp.pipeline import Pipeline
from adcp.storage import Store

DB_PATH = os.environ.get("ADCP_DB", "adcp_evidence.db")
store = Store(DB_PATH)
app = FastAPI(title="ADCP 流束换算台 evidence review")


@app.on_event("startup")
def _bootstrap():
    if store.current_version() is None and not store.sources():
        replay(store)


def esc(x):
    return html.escape(str(x))


def fmt_vec(v):
    if v is None:
        return "<i>—</i>"
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def page(title, body):
    return HTMLResponse(f"""<!doctype html><html><head><meta charset=utf-8>
<title>{esc(title)}</title><style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;max-width:1200px}}
table{{border-collapse:collapse;margin:.5rem 0}}
td,th{{border:1px solid #ccc;padding:2px 8px;font-size:13px}}
.bad{{background:#ffd9d9}} .warn{{background:#fff3c4}} .ok{{background:#e4f5e4}}
.amb{{background:#e0d9ff}} code{{background:#f2f2f2;padding:0 3px}}
section{{border:1px solid #ddd;border-radius:8px;padding:.8rem;margin:.8rem 0}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin:.2rem 0}}
nav a{{margin-right:1rem}} .id{{color:#777;font-size:11px}}
</style></head><body>
<nav><a href="/">总览</a><a href="/events">事件流</a>
<a href="/versions">版本</a><a href="/api/export">导出 JSON</a></nav>
{body}</body></html>""")


# ------------------------------------------------------------------ home
@app.get("/", response_class=HTMLResponse)
def home():
    cur = store.current_version()
    versions = store.versions()
    pings = sorted({s["ping"] for s in store.sources("beam_velocity")})
    conflicts = store.conflicts()
    rows = "".join(
        f"<tr><td>v{v['id']}</td><td>{esc(v['label'])}</td>"
        f"<td>{esc(v['status'])}</td>"
        f"<td class=id>{(v['fingerprint'] or '')[:12]}</td>"
        f"<td>{'◀ current' if v['id']==cur else ''}</td></tr>"
        for v in versions)
    ping_links = " ".join(f"<a href='/ping/{p}'>p{p}</a>" for p in pings)
    conf_html = ""
    for c in conflicts:
        conf_html += (f"<li>conflict #{c['id']}: base v{c['base_version']} "
                      f"branchA v{c['branch_a']} vs branchB v{c['branch_b']} "
                      f"— {esc(c['status'])} "
                      f"<a href='/compare?a={c['branch_a']}&b={c['branch_b']}'>"
                      f"比较两分支</a></li>")
    body = f"""
<h1>流束换算台 · ADCP 波束速度 / 姿态旋转 / 质量层级</h1>
<section><h2>90 秒样例路径</h2><ol>
<li><a href="/ping/6">ping 6</a>：单 beam 失效 → 仅三波束解</li>
<li><a href="/ping/5">ping 5</a>：姿态跨越 360°（解缠插值）</li>
<li><a href="/ping/9">ping 9</a>：时间缺口 → 姿态不跨缺口插值，earth 变换挂起</li>
<li><a href="/ping/10">ping 10</a>：三波束多解 → 两个合法候选并列保留</li>
<li><a href="/versions">版本</a>：安装角修订 v3、变换顺序变体 v4、冲突 v5/v6</li>
<li><a href="/compare?a=5&b=6">比较冲突分支 v5 vs v6</a>，未选候选仍保留</li>
<li><a href="/api/export">导出</a> → 删除数据库 → 导入，身份/顺序/来源不变</li>
</ol></section>
<section><h2>版本线</h2><table>
<tr><th>版本</th><th>说明</th><th>状态</th><th>指纹</th><th></th></tr>
{rows}</table>
<form method=post action="/replay"><button>重新重放案例包</button></form>
</section>
<section><h2>按 ping 复核</h2>{ping_links}</section>
<section><h2>冲突</h2><ul>{conf_html or '<li>无</li>'}</ul></section>
"""
    return page("ADCP evidence", body)


@app.post("/replay")
def do_replay():
    out = replay(store)
    return RedirectResponse("/", status_code=303)


# -------------------------------------------------------------- ping page
def svg_profile(cells):
    """Vertical profile: u (blue) and v (green) water velocity vs depth."""
    W, H, pad = 460, 300, 40
    pts = [c for c in cells if c["included"] and c["water_velocity"]]
    exc = [c for c in cells if not c["included"]]
    if not pts:
        return "<p>无有效剖面（全部 cell 被排除或姿态缺失）</p>"
    us = [c["water_velocity"][0] for c in pts]
    vs = [c["water_velocity"][1] for c in pts]
    lo, hi = min(us + vs), max(us + vs)
    rng = (hi - lo) or 0.1
    lo -= rng * 0.15
    hi += rng * 0.15
    depths = [c["depth_m"] for c in cells]
    d0, d1 = min(depths) - 0.5, max(depths) + 0.5

    def X(v):
        return pad + (v - lo) / (hi - lo) * (W - 2 * pad)

    def Y(d):
        return pad + (d - d0) / (d1 - d0) * (H - 2 * pad)

    def line(key, color):
        p = " ".join(f"{X(c['water_velocity'][key]):.1f},{Y(c['depth_m']):.1f}"
                     for c in pts)
        dots = "".join(
            f"<circle cx={X(c['water_velocity'][key]):.1f} "
            f"cy={Y(c['depth_m']):.1f} r=3 fill={color}/>" for c in pts)
        return f"<polyline points='{p}' fill=none stroke={color} " \
               f"stroke-width=2/>{dots}"

    excluded = "".join(
        f"<line x1={pad} x2={W-pad} y1={Y(c['depth_m']):.1f} "
        f"y2={Y(c['depth_m']):.1f} stroke=#d00 stroke-dasharray=4 />"
        f"<text x={W-pad+4} y={Y(c['depth_m']):.1f} font-size=10 fill=#d00>"
        f"c{c['cell']} 排除</text>" for c in exc)
    zero = X(0)
    return f"""<svg width={W} height={H} style='border:1px solid #ccc'>
<line x1={zero:.1f} x2={zero:.1f} y1={pad} y2={H-pad} stroke=#999 />
{line(0,'#2266cc')}{line(1,'#22aa44')}{excluded}
<text x={pad} y=18 font-size=11 fill=#2266cc>u (m/s)</text>
<text x={pad+70} y=18 font-size=11 fill=#22aa44>v (m/s)</text>
<text x={pad} y={H-8} font-size=10>depth {d0:.1f}–{d1:.1f} m ↓</text>
</svg>"""


@app.get("/ping/{ping}", response_class=HTMLResponse)
def ping_page(ping: int, version: int = Query(None), cell: int = Query(3)):
    vid = version or store.current_version()
    qc_rows, instr_rows, ship_rows, earth_rows = [], [], [], []
    prof = store.derived(vid, f"profile/p{ping}")
    if not prof:
        return page(f"ping {ping}", f"<p>v{vid} 中没有 ping {ping}</p>")
    prof = prof[0]["payload"]
    att = store.derived(vid, f"attitude/p{ping}")[0]
    qc = {r["payload"]["cell"]: r for r in
          store.derived(vid) if r["node_key"].startswith(f"beam_qc/p{ping}/")}
    instr = {r["payload"]["cell"]: r for r in store.derived(vid)
             if r["node_key"].startswith(f"instr_vel/p{ping}/")}
    ship = {r["payload"]["cell"]: r for r in store.derived(vid)
            if r["node_key"].startswith(f"ship_vel/p{ping}/")}
    earth = {r["payload"]["cell"]: r for r in store.derived(vid)
             if r["node_key"].startswith(f"earth_vel/p{ping}/")}
    valid = {r["payload"]["cell"]: r for r in store.derived(vid)
             if r["node_key"].startswith(f"cell_validity/p{ping}/")}

    # 1) beam quality table
    for m in sorted(qc):
        for b in qc[m]["payload"]["beams"]:
            cls = "bad" if b["reasons"] else "ok"
            qc_rows.append(
                f"<tr class='{cls}'><td>{m}</td><td>{b['beam']}</td>"
                f"<td>{b['velocity'] if b['velocity'] is not None else '—'}</td>"
                f"<td>{b['correlation'] if b['correlation'] is not None else '—'}</td>"
                f"<td>{b['amplitude'] if b['amplitude'] is not None else '—'}</td>"
                f"<td>{b['bad_flag'] if b['bad_flag'] is not None else '—'}</td>"
                f"<td>{esc('; '.join(b['reasons']) or '通过')}</td>"
                f"<td class=id>src {','.join(map(str,b['source_ids']))}</td></tr>")

    # 2) staged transforms for the selected cell
    iv = instr[cell]["payload"]
    sv = ship[cell]["payload"]
    ev = earth[cell]["payload"]
    cand_rows = ""
    ship_map = {c["id"]: c["value"] for c in sv["candidates"]}
    earth_map = {c["id"]: c for c in ev["candidates"]}
    for c in iv["candidates"]:
        adopted = "✔ 采纳" if c.get("adopted") else (
            "待裁定" if c.get("adopted") is None else "")
        e = earth_map.get(c["id"], {})
        cand_rows += f"""<tr class={'amb' if iv['status']=='ambiguous' else ''}>
<td>{esc(c['kind'])}</td><td>{fmt_vec(c['value'])}</td>
<td>{fmt_vec(ship_map.get(c['id']))}</td>
<td>{fmt_vec(e.get('earth'))}</td><td>{fmt_vec(e.get('water'))}</td>
<td>{adopted} <a href='/decide?version={vid}&node={
esc(instr[cell]['node_key'])}&candidate={c['id']}'>选择此候选</a></td>
<td class=id>v{vid}·{esc(instr[cell]['node_key'])}#{esc(c['id'])}</td></tr>"""

    cell_tabs = " ".join(
        f"<a href='/ping/{ping}?version={vid}&cell={m}'>cell {m}</a>"
        for m in sorted(qc))
    att_line = ("无姿态 — " + esc(att["payload"]["diagnostic"])) \
        if att["payload"]["attitude"] is None else json.dumps(
            att["payload"]["attitude"], ensure_ascii=False)
    excluded_rows = "".join(
        f"<li>cell {c['cell']} ({c['depth_m']}m): {esc('; '.join(c['reasons']))} "
        f"· <code>{esc(c['node'])}@v{vid}</code></li>"
        for c in prof["cells"] if not c["included"])
    no_att_cells = "".join(
        f"<li>cell {c['cell']}: {esc(c['status'])} — 姿态缺失，"
        f"值保留在 instrument/ship 帧</li>"
        for c in prof["cells"] if c.get("status") == "no_attitude")
    body = f"""
<h1>Ping {ping} （t={prof['t']}s，查看版本 v{vid}）
<a href='/ping/{ping}?cell={cell}'>当前</a></h1>
<section><h2>① 波束质量</h2><p class=id>v{vid}·beam_qc/p{ping}/c*
（速度 m/s、correlation/amplitude count，原始单位随源数据保留）</p>
<table><tr><th>cell</th><th>beam</th><th>vel</th><th>corr</th>
<th>amp</th><th>坏标志</th><th>判定</th><th>来源身份</th></tr>
{''.join(qc_rows)}</table></section>

<section><h2>② 逐级向量变换 · {cell_tabs}</h2>
<p>姿态：{esc(att_line)} <code>attitude/p{ping}@v{vid}</code><br>
安装角 {sv['mounting_angle_deg']}°(v{sv['mounting_version']}) ·
磁偏角 {ev['declination_deg']}°(v{ev['declination_version']}) ·
顺序 <code>{esc(ev['order'])}</code>(v{ev['order_version']})</p>
<table><tr><th>候选</th><th>beam→instrument</th><th>instrument→ship</th>
<th>ship→earth</th><th>对地水流(减 bottom track)</th><th>裁定</th>
<th>稳定身份</th></tr>{cand_rows}</table></section>

<section><h2>③ 垂向剖面（被采纳候选，m/s）</h2>{svg_profile(prof['cells'])}
<p class=id>v{vid}·profile/p{ping} 指纹 {esc(instr[cell]['fingerprint'][:12])}</p></section>

<section><h2>④ 被排除 / 无姿态 cell 的证据</h2>
<ul>{excluded_rows or '<li>无排除 cell</li>'}{no_att_cells}</ul></section>
"""
    return page(f"ping {ping}", body)


@app.get("/decide")
def decide_form(version: int, node: str, candidate: str):
    target = f"/api/decision?version={version}&node={node}&candidate={candidate}"
    body = f"""<h1>记录人工决定</h1>
<form method=post action='/decision'>
<input type=hidden name=version value={version}>
<input type=hidden name=node value='{esc(node)}'>
<input type=hidden name=candidate value='{esc(candidate)}'>
<p>基于版本 v{version}，节点 <code>{esc(node)}</code>，采纳候选
<code>{esc(candidate)}</code>。若主线已离开 v{version}，将产生可解决冲突，
未选分支会保留。</p>
<p>说明：<input name=note size=60> <button>提交决定</button></p></form>"""
    return page("decision", body)


@app.post("/decision")
def do_decision(version: int = Form(...), node: str = Form(...),
                candidate: str = Form(...), note: str = Form("")):
    vid, conflict_id = Pipeline(store).decide(
        version, node, candidate, note=note)
    if conflict_id:
        return RedirectResponse(
            f"/compare?a={version}&b={vid}&conflict={conflict_id}", 303)
    ping = node.split("/")[1]
    return RedirectResponse(f"/ping/{ping}", 303)


# ---------------------------------------------------------------- events
@app.get("/events", response_class=HTMLResponse)
def events_page():
    rows = ""
    for e in store.events():
        rows += (f"<tr><td>{e['id']}</td><td>v{e['version_id']}</td>"
                 f"<td>{esc(e['kind'])}</td>"
                 f"<td><a href='/event/{e['id']}'>{esc(json.dumps(e['payload'], ensure_ascii=False))[:150]}</a>"
                 f"</td></tr>")
    body = f"""<h1>机器可读事件流</h1>
<p><a href="/api/events">/api/events</a> 提供 JSON。回到任一旧事件可恢复当时
依赖版本与可见结论。</p>
<table><tr><th>#</th><th>版本</th><th>类型</th><th>载荷</th></tr>{rows}</table>"""
    return page("events", body)


@app.get("/event/{event_id}", response_class=HTMLResponse)
def event_view(event_id: int):
    ev = next((e for e in store.events() if e["id"] == event_id), None)
    if not ev:
        return page("event", "未找到")
    body = f"""<h1>事件 #{ev['id']} @ v{ev['version_id']}</h1>
<p>类型：{esc(ev['kind'])}</p>
<pre>{esc(json.dumps(ev['payload'], ensure_ascii=False, indent=2))}</pre>
<p><a href="/versions">该版本结论按当时依赖版本固定，不受后续重算覆盖。</a></p>"""
    return page("event", body)


# ------------------------------------------------------------- versions
@app.get("/versions", response_class=HTMLResponse)
def versions_page():
    cur = store.current_version()
    rows = ""
    for v in store.versions():
        links = " ".join(
            f"<a href='/ping/{p}?version={v['id']}'>p{p}</a>"
            for p in range(1, 13))
        rows += f"<tr><td>v{v['id']}</td><td>{esc(v['label'])}</td>" \
                f"<td>{esc(v['status'])}</td>" \
                f"<td class=id>{(v['fingerprint'] or '')[:12]}</td>" \
                f"<td>{links}</td></tr>"
    body = f"""<h1>版本线（当前 v{cur}）</h1><table>
<tr><th>版本</th><th>说明</th><th>状态</th><th>指纹</th>
<th>按 ping 查看该版本</th></tr>{rows}</table>"""
    return page("versions", body)


# --------------------------------------------------------------- compare
@app.get("/compare", response_class=HTMLResponse)
def compare(a: int, b: int, conflict: int = Query(None)):
    ra = {r["node_key"]: r for r in store.derived(a)}
    rb = {r["node_key"]: r for r in store.derived(b)}
    keys = sorted(set(ra) | set(rb))
    diffs = ""
    for k in keys:
        fa = ra[k]["fingerprint"] if k in ra else "—"
        fb = rb[k]["fingerprint"] if k in rb else "—"
        cls = "bad" if fa != fb else "ok"
        pa = json.dumps(ra[k]["payload"], ensure_ascii=False) if k in ra else ""
        pb = json.dumps(rb[k]["payload"], ensure_ascii=False) if k in rb else ""
        short_a = pa[:180]
        short_b = pb[:180]
        if fa != fb:
            diffs += f"<tr class='{cls}'><td><code>{esc(k)}</code></td>" \
                     f"<td class=id>{fa[:10]}</td><td>{esc(short_a)}</td>" \
                     f"<td class=id>{fb[:10]}</td><td>{esc(short_b)}</td></tr>"
    resolve = ""
    if conflict:
        resolve = f"""<form method=post action='/resolve'>
<input type=hidden name=conflict value={conflict}>
解决冲突（保留未选分支）：
<button name=keep value=a>保留 v{a}（主线 A）</button>
<button name=keep value=b>保留 v{b}（分支 B）</button></form>"""
    body = f"""<h1>分支比较 v{a} vs v{b}</h1>{resolve}
<p>仅显示指纹不同的节点（依赖图传播后实际受影响项）。</p>
<table><tr><th>节点</th><th>v{a} 指纹</th><th>v{a} 值</th>
<th>v{b} 指纹</th><th>v{b} 值</th></tr>{diffs or
'<tr><td colspan=5>两版本完全一致</td></tr>'}</table>"""
    return page("compare", body)


@app.post("/resolve")
def resolve(conflict: int = Form(...), keep: str = Form(...)):
    store.resolve_conflict(conflict, keep)
    return RedirectResponse("/", 303)


# ------------------------------------------------------------------- api
@app.get("/api/events")
def api_events():
    return JSONResponse(store.events())


@app.get("/api/versions")
def api_versions():
    return JSONResponse({"current": store.current_version(),
                         "versions": store.versions(),
                         "conflicts": store.conflicts()})


@app.get("/api/ping/{ping}")
def api_ping(ping: int, version: int = Query(None)):
    vid = version or store.current_version()
    out = {"version": vid, "ping": ping, "nodes": {}}
    for r in store.derived(vid):
        if f"/p{ping}/" in r["node_key"] or r["node_key"] == f"profile/p{ping}":
            out["nodes"][r["node_key"]] = {
                "stable_id": f"{r['node_key']}@v{vid}",
                "fingerprint": r["fingerprint"], "status": r["status"],
                "inputs": r["inputs"], "reason": r["reason"],
                "payload": r["payload"]}
    return JSONResponse(out)


@app.get("/api/source_chain")
def source_chain(node: str, version: int = Query(None)):
    vid = version or store.current_version()
    rows = {r["node_key"]: r for r in store.derived(vid)}
    chain = []

    def walk(key, depth=0):
        r = rows.get(key)
        if not r:
            return
        chain.append({"depth": depth, "node": key,
                      "fingerprint": r["fingerprint"],
                      "reason": r["reason"], "inputs": r["inputs"]})
        for dep in r["inputs"]:
            if dep[0] == "upstream":
                # upstream descriptor is ["upstream", label, fp]; labels use
                # canonical node paths
                label = dep[1]
                if "/" in label and label in rows:
                    walk(label, depth + 1)

    walk(node)
    src_ids = []
    for step in chain:
        r = rows.get(step["node"])
        if r:
            src_ids += [d[1] for d in r["inputs"] if d and d[0] == "src"]
    sources = [s for s in store.sources() if s["id"] in src_ids]
    return JSONResponse({"version": vid, "node": node, "chain": chain,
                         "sources": sources})


@app.get("/api/export")
def api_export():
    return JSONResponse(store.export())


@app.post("/api/import_dump")
def api_import_dump(data: dict):
    store.import_dump(data)
    return JSONResponse({"ok": True, "current": store.current_version()})
