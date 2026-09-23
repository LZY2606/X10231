"""Single-page review UI (vanilla JS + server SVG)."""
from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>ADCP 流速证据工作台</title>
<style>
body{font:13px/1.5 system-ui,"PingFang SC",sans-serif;margin:0;color:#0f172a}
header{background:#0f172a;color:#fff;padding:10px 16px;display:flex;gap:12px;
  align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600}
.wrap{display:grid;grid-template-columns:240px 1fr 360px;gap:0;min-height:100vh}
nav{border-right:1px solid #e2e8f0;padding:10px;background:#f8fafc}
main{padding:12px 16px;overflow:auto}
aside{border-left:1px solid #e2e8f0;padding:10px;background:#f8fafc}
button{border:1px solid #cbd5e1;background:#fff;border-radius:6px;padding:4px 9px;
  cursor:pointer;font-size:12px}
button:hover{background:#eef2ff}
.commit{border:1px solid #e2e8f0;border-radius:6px;padding:6px 8px;margin:6px 0;
  cursor:pointer;background:#fff}
.commit.active{border-color:#4f46e5;box-shadow:0 0 0 2px #c7d2fe}
.pill{display:inline-block;border-radius:10px;padding:0 8px;font-size:11px;margin:1px}
.ok{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}
.warn{background:#fef3c7;color:#92400e}.info{background:#dbeafe;color:#1e40af}
.card{border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin:8px 0;background:#fff}
.fp{font:10px ui-monospace,monospace;color:#64748b;word-break:break-all}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eef2f7;
  padding:3px 6px;text-align:left;font-size:12px}
.mono{font-family:ui-monospace,monospace}
.tabbtns button{margin-right:6px}
#tour{position:fixed;right:12px;bottom:12px;width:330px;background:#111827;color:#f9fafb;
  border-radius:10px;padding:12px;display:none;z-index:20;box-shadow:0 8px 30px #0006}
#tour button{background:#4f46e5;color:#fff;border:0}
svg{max-width:100%}
details{margin:4px 0}
.small{font-size:11px;color:#475569}
</style></head><body>
<header>
  <h1>流束换算台 · ADCP 波束 / 姿态 / 坐标变换证据</h1>
  <select id="branch"></select>
  <button onclick="loadState()">刷新</button>
  <button onclick="runReplay()">一键重放反例档案</button>
  <button onclick="resetCase()">重置案例</button>
  <button onclick="openTour()">90 秒导览</button>
  <a href="/api/export" target="_blank"><button>导出 JSON</button></a>
</header>
<div class="wrap">
<nav>
  <b>提交时间线</b>
  <div id="timeline" class="small"></div>
  <hr><b>工作区 / 冲突</b>
  <div id="work" class="small"></div>
</nav>
<main>
  <div id="summary" class="card"></div>
  <div class="card">
    <b>ping 列表（波束质量 → 逐级变换 → 剖面 → 排除 cell）</b>
    <div id="pingtabs" class="tabbtns" style="margin-top:6px"></div>
  </div>
  <div id="pingview"></div>
  <div class="card"><b>垂向剖面（3-beam 与 4-beam 不混合）</b>
    <div id="profile"></div></div>
  <div class="card"><b>依赖图传播说明（上一提交 → 当前提交）</b>
    <div id="explain" class="small"></div></div>
</main>
<aside>
  <b>来源链 / 裁定</b>
  <div id="side" class="small"></div>
</aside>
</div>
<div id="tour">
  <b>90 秒样例路径</b>
  <p id="tourtext" class="small"></p>
  <button onclick="tourNext()">下一步</button>
  <button onclick="document.getElementById('tour').style.display='none'">结束</button>
</div>
<script>
let state=null, pingId=null, seqSel=null, branch='main';
const api=(u,o)=>fetch(u,o).then(r=>r.json());
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function loadState(){
  state = await api(`/api/state?branch=${branch}`);
  document.getElementById('branch').innerHTML = state.branches
    .map(b=>`<option ${b===branch?'selected':''}>${b}</option>`).join('');
  const tl = state.timeline.map(c=>
    `<div class="commit ${c.seq===state.seq?'active':''}" onclick="viewCommit(${c.seq})">
      #${c.seq} ${esc(c.label)}<div class="small">${esc(c.author)}</div>
      <span class="pill info">${c.events.length} 事件</span></div>`).join('');
  document.getElementById('timeline').innerHTML = tl;
  const counts = Object.entries(state.slot_status_count).map(([k,n])=>
    `<span class="pill ${k.includes('excluded')||k.includes('insufficient')?'bad':
      k.includes('ambig')||k.includes('three')||k.includes('gap')||k.includes('blocked')?'warn':'ok'}">
      ${k}: ${n}</span>`).join('');
  document.getElementById('summary').innerHTML =
    `<b>${esc(state.commit_label)}</b> @ seq ${state.seq}/${state.head}<br>
     <span class="small">分支 ${branch} · 独立记录的材料版本：</span><br>${counts}
     <div class="small">${Object.entries(state.documents).map(([k,v])=>
       `${esc(k)} <span class="fp">${esc(v.version.slice(0,20))}…</span>`).join('<br>')}</div>`;
  document.getElementById('pingtabs').innerHTML = state.ping_ids.map(p=>
    `<button onclick="selectPing('${p}')">${p}</button>`).join('');
  if(!pingId && state.ping_ids.length) selectPing(state.ping_ids[0]);
  else if(pingId) selectPing(pingId);
  loadProfile(); loadExplain(); loadSide(); loadWork();
}

function viewCommit(seq){
  seqSel = seq;
  location.hash=`#${branch}@${seq}`;
  loadStateAt(seq);
}
async function loadStateAt(seq){
  state = await api(`/api/state?branch=${branch}&seq=${seq}`);
  if(pingId) selectPingAt(pingId,seq);
  loadProfileAt(seq); loadSideAt(seq);
}

document.getElementById('branch').onchange = (e)=>{
  branch=e.target.value; seqSel=null; pingId=null; loadState();
};

async function selectPing(p){ pingId=p; const s=seqSel||state.head; selectPingAt(p,s); }
async function selectPingAt(p,s){
  const d = await api(`/api/ping/${p}?branch=${branch}&seq=${s}`);
  const cells = d.cells.map(c=>{
    const der=c.derivation||{};
    const statusCls = c.status==='four_beam'?'ok':
      (c.status==='three_beam'||(c.status||'').includes('ambig'))?'warn':'bad';
    let body='';
    if(der.transform||der.four_beam_transform){
      const chain = der.transform || der.four_beam_transform;
      body = `<img src="/api/svg/ping/${p}/cell/${c.cell.index}?kind=vectors&branch=${branch}&seq=${s}">`;
      const t=der.transform||{};
      if(t.earth_ms) body += `<table><tr><th>层</th><th>向量 (m/s)</th></tr>
        <tr><td>instrument</td><td class="mono">[${(t.instrument_ms||[]).map(x=>x.toFixed(4))}]</td></tr>
        <tr><td>ship fsu</td><td class="mono">[${(t.ship_fsu_ms||[]).map(x=>x.toFixed(4))}]</td></tr>
        <tr><td>earth ENU</td><td class="mono">[${(t.earth_ms||[]).map(x=>x.toFixed(4))}]</td></tr>
        <tr><td>error v</td><td class="mono">${(t.beam_solution?.error_velocity_ms??'—')}</td></tr>
        </table>`;
      if(der.three_beam_candidates?.length){
        body += `<details><summary>未采纳候选（${der.three_beam_candidates.length} 个三波束解）</summary>`+
          der.three_beam_candidates.map(cd=>
            `<div class="card">丢弃 beam ${cd.dropped_beam+1} · earth ENU
             <span class="mono">[${cd.chain.earth_ms.map(x=>x.toFixed(4))}]</span>
             <div class="fp">候选指纹来源 slot fp</div></div>`).join('')+`</details>`;
      }
    }
    const reasons = der.cell_verdict?.reasons ||
      der.attitude_diagnostics || [];
    return `<div class="card">
      <span class="pill ${statusCls}">${esc(c.status||'—')}</span>
      cell ${c.cell.index} · center ${c.cell.center_m?.canonical ?? 'NA'} m
      <img src="/api/svg/ping/${p}/cell/${c.cell.index}?kind=beams&branch=${branch}&seq=${s}">
      <div class="small">排除/阻塞依据: ${reasons.length?reasons.map(esc).join(', '):'—'}</div>
      ${body}
      <div class="fp">稳定身份 slot=${esc(c.slot)}<br>fingerprint=${esc(c.fingerprint||'—')}<br>
      源版本=${esc(c.source_version)}</div>
      <button onclick="decide('${c.slot}','accept')">接受</button>
      <button onclick="decide('${c.slot}','reject')">否定</button>
      <button onclick="compareBranch('${c.slot}')">比较 alt-order 分支</button>
      </div>`;
  }).join('');
  const bt = d.bottom_track ? `bottom track depth
    <b>${d.bottom_track.depth_m.canonical ?? d.bottom_track.depth_m.raw} m</b>
    (raw "${esc(d.bottom_track.depth_m.raw)}" ${esc(d.bottom_track.depth_m.unit)})` : '';
  document.getElementById('pingview').innerHTML =
    `<div class="card"><b>${d.ping_id}</b> t=${d.t_s}s · ${bt}
     <div class="fp">ping 源版本 ${esc(d.source_version)}</div></div>${cells}`;
}

async function loadProfile(){ loadProfileAt(seqSel||state.head); }
async function loadProfileAt(s){
  const d = await api(`/api/profile?branch=${branch}&seq=${s}`);
  document.getElementById('profile').innerHTML =
    `<img src="${d.svg}"><div class="fp">profile fingerprint ${esc(d.fingerprint)}</div>
     <table><tr><th>cell</th><th>解类型</th><th>n</th><th>ENU 均值</th><th>成员 ping</th></tr>
     ${d.profile.ensembles.map(e=>`<tr><td>${e.cell}</td><td>${e.solution_kind}</td><td>${e.n}</td>
       <td class="mono">${e.earth_mean_ms?e.earth_mean_ms.map(x=>x.toFixed(4)):'多解保留，不平均'}</td>
       <td>${e.members.join(',')}</td></tr>`).join('')}</table>`;
}

async function loadExplain(){
  const d = await api(`/api/explain?branch=${branch}`);
  const changed = Object.entries(d.changed_records||{}).map(([k,v])=>
    `<span class="pill warn">${v} ${esc(k)}</span>`).join(' ');
  const unaffected=(d.unaffected_explanations||[]).slice(0,5).map(e=>
    `<div>✓ ${esc(e.slot)}: ${esc(e.reason)}</div>`).join('');
  document.getElementById('explain').innerHTML =
    `改动记录: ${changed||'（基线）'}<br>受影响槽位 ${(d.affected_slots||[]).length} 个
     <details><summary>仍有效的槽位（前5）</summary>${unaffected}</details>`;
}

async function loadSide(){ loadSideAt(seqSel||state.head); }
async function loadSideAt(s){
  const st = await api(`/api/state?branch=${branch}&seq=${s}`);
  document.getElementById('side').innerHTML = st.decisions.map(d=>
    `<div class="card"><b>${esc(d.verdict)}</b><div>${esc(d.subject)}</div>
     <div class="small">${esc(d.note)}</div>
     <div>未采纳候选保留: <span class="mono">${esc(JSON.stringify(d.rejected_candidates))}</span></div>
     <div class="fp">decision ${esc(d.id)}</div></div>`).join('') ||
     '<div class="small">尚无人工裁定。从某个 cell 点“接受/否定”。</div>';
}

async function decide(subject,verdict){
  const note = prompt(`${verdict} 说明（会生成新事件，不覆盖历史）:`, verdict) ?? '';
  await api('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({subject,verdict,note,branch})});
  seqSel=null; loadState();
}

async function compareBranch(slot){
  try{
    await api('/api/branch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:'alt-order',order:'v2_fused_yaw'})});
  }catch(e){}
  const main = await api(`/api/value?x=1`);
  alert('已在 alt-order 分支保留错误顺序 v2_fused_yaw 的结果；切换顶部分支比较指纹差异。');
}

async function loadWork(){
  const d = await api('/api/work');
  document.getElementById('work').innerHTML =
    (d.conflicts||[]).map(c=>`<div class="pill bad">冲突（可解决）</div>
      ${esc(c.record_id)}<br>base ${esc(c.base.slice(0,12))} → current ${esc(c.current.slice(0,12))}`)
      .join('<hr>') +
    `<div class="small">已暂存: ${Object.keys(d.staged||{}).length} 条</div>`;
}

async function runReplay(){
  const d = await fetch('/api/replay',{method:'POST'}).then(r=>r.json());
  seqSel=null; pingId=null; await loadState();
  alert('重放完成: '+(d.all_ok?'全部断言通过':'存在失败')+'\n'+
    d.findings.map(f=>(f.ok?'✓':'✗')+' '+f.step+' '+f.name).join('\n'));
}
async function resetCase(){ await fetch('/api/reset',{method:'POST'});
  seqSel=null; pingId=null; loadState(); }

// ---------- 90 second guided tour ----------
const TOUR=[
  ['第1步', '时间线即反例档案：点“一键重放反例档案”，系统实际经过单 beam 失效、姿态跨 360、时间缺口、三波束多解、安装角修订、错误坐标顺序与无效深度 cell。'],
  ['第2步', '点 P02：beam 2 坏标志+低相关 → 三波束解，和四波束是不同身份，绝不混成一个值。'],
  ['第3步', '点 P03/P04：heading 从 359.4° 到 0.6°，插值步长仍小于 2°。点 P07：时间缺口处 earth 变换被阻塞，姿态插值不跨缺口。'],
  ['第4步', '点 P06 cell3：四波束和“丢弃弱 beam4”的三波束候选同时保留为多解；接受其中一个不会删除其他候选。'],
  ['第5步', '时间线 S5/S8：安装角与磁偏角各自版本化；S8 只改 earth 向量，instrument 向量指纹不变。S6 切到 alt-order 分支比较错误顺序。'],
  ['第6步', '点 P09：被排除 cell 带原因（未配置/旁瓣/低于底跟踪），原始表示仍保留。'],
  ['第7步', '剖面区：三波束与四波束分别成组平均；导出后删除本地数据库再导入，指纹、顺序与来源关系保持。'],
];
let tourIdx=0;
function openTour(){ tourIdx=0; showTour(); document.getElementById('tour').style.display='block'; }
function showTour(){
  const t=TOUR[tourIdx];
  document.getElementById('tourtext').innerHTML=`<b>${t[0]}/${TOUR.length}</b><br>${t[1]}`;
}
function tourNext(){ if(tourIdx<TOUR.length-1){tourIdx++;showTour();}
  else document.getElementById('tour').style.display='none'; }

(function(){
  if(location.hash){ const m=location.hash.slice(1).split('@'); branch=m[0]; seqSel=m[1]?+m[1]:null; }
  loadState();
})();
</script></body></html>
"""
