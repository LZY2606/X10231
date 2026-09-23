let STATE = null, SEL_PING = null;
const $ = (id) => document.getElementById(id);
const toast = (m) => { const t = $('toast'); t.textContent = m; t.style.display = 'block'; setTimeout(() => t.style.display = 'none', 2500); };
const f3 = (v) => (v === null || v === undefined) ? '—' : Number(v).toFixed(3);

async function resetDemo() {
  const r = await fetch('/api/demo/reset', { method: 'POST' });
  STATE = await r.json();
  await loadState();
  toast('已重放 7 类边界情形，head seq=' + STATE.head_seq);
}

async function loadState() {
  const r = await fetch('/api/state');
  STATE = await r.json();
  $('head').textContent = `head seq ${STATE.head_seq} · 派生 ${STATE.derivations.length} · 事件 ${STATE.events.length}`;
  renderSteps(); renderPings(); renderEvents();
  if (!SEL_PING) SEL_PING = 'ping-20';
  await showPing(SEL_PING);
}

function renderSteps() {
  const verdicts = {};
  STATE.events.filter(e => e.event_type.startsWith('verdict:')).forEach(e => {
    verdicts[e.event_type.split(':')[1]] = e.detail;
  });
  const names = ['single_beam_failure','heading_wrap_360','time_gap','three_beam_multiplicity',
                 'mount_revision','transform_order','invalid_depth_cell'];
  const labels = {single_beam_failure:'① 单 beam 失效', heading_wrap_360:'② 姿态跨 360°',
    time_gap:'③ 时间缺口不插值', three_beam_multiplicity:'④ 三波束多解不合并',
    mount_revision:'⑤ 安装角修订', transform_order:'⑥ 坐标变换顺序', invalid_depth_cell:'⑦ 无效深度 cell'};
  const jump = {single_beam_failure:'ping-20', heading_wrap_360:'ping-45', time_gap:'ping-40',
    three_beam_multiplicity:'ping-30', mount_revision:'ping-00', transform_order:'ping-00',
    invalid_depth_cell:'ping-15'};
  $('steps').innerHTML = names.map(n => {
    const v = verdicts[n];
    const cls = v ? v.status : '';
    return `<div class="step" onclick="showPing('${jump[n]}')">${labels[n]} ` +
           (v ? `<span class="${cls}">${v.status}@${v.seq}</span>` : '<span class="muted">未运行</span>') + '</div>';
  }).join('');
}

function renderPings() {
  const prof = {};
  STATE.derivations.filter(d => d.kind === 'profile').forEach(d => {
    prof[d.ping_id] = d.payload.attitude_status;
  });
  const pings = Object.keys(prof).sort();
  $('pings').innerHTML = pings.map(p => {
    const gap = prof[p] !== 'ok';
    return `<div class="ping ${p === SEL_PING ? 'sel' : ''}" onclick="showPing('${p}')">${p}` +
           (gap ? '<span class="tag t-warn">att:' + prof[p] + '</span>' : '') + '</div>';
  }).join('');
}

function renderEvents() {
  const evs = [...STATE.events].reverse().slice(0, 60);
  $('events').innerHTML = evs.map(e => {
    const v = e.event_type.startsWith('verdict:');
    const detail = v ? `<b class="${e.detail.status}">${e.detail.status}</b> → ${(e.detail.impacted||[]).slice(0,3).join(', ')}${(e.detail.impacted||[]).length>3?'…':''}`
                     : JSON.stringify(e.detail).slice(0, 110);
    return `<div class="ev mono">#${e.seq} ${e.event_type} <span class="muted">${e.subject}</span><br/>${detail}</div>`;
  }).join('') || '<span class="muted">无事件</span>';
}

async function showPing(ping) {
  SEL_PING = ping; renderPings();
  const r = await fetch('/api/ping/' + ping);
  const D = await r.json();
  const cells = D.derivations.filter(d => d.kind === 'cell_flow')
    .sort((a, b) => a.payload.depth_m - b.payload.depth_m);
  const bt = D.derivations.find(d => d.kind === 'bottom_track_flow');
  const prof = D.derivations.find(d => d.kind === 'profile');

  let html = `<div class="card"><h3>逐级向量变换与波束质量 · ${ping}</h3>`;
  if (prof) html += `<div class="mono">姿态：${prof.payload.attitude.status}` +
    (prof.payload.attitude.heading_deg !== null ? ` · heading ${f3(prof.payload.attitude.heading_deg)}° · bracket ${(prof.payload.attitude.bracketed_by||[]).join(' / ')}` : '') +
    ` · 段 ${prof.payload.attitude.segment_id || '—'}</div>`;
  html += qualityTable(cells) + '</div>';

  html += `<div class="card"><h3>垂向剖面（earth，按当前选定候选）</h3>
    <img src="/api/profile.svg?ping_id=${ping}&component=north" style="vertical-align:top"/>
    <img src="/api/profile.svg?ping_id=${ping}&component=east" style="vertical-align:top"/></div>`;

  html += cells.map(cellCard).join('');
  if (bt) html += btCard(bt);
  html += '<div class="card"><h3>配置依据（进入指纹）</h3>' +
    (cells[0] ? `<pre class="mono">${JSON.stringify(cells[0].basis, null, 1).slice(0, 900)}</pre>` : '') + '</div>';
  $('center').innerHTML = html;
}

function qualityTable(cells) {
  let rows = cells.map(c => {
    const p = c.payload;
    const obs = p.observations.map(o => {
      const bad = !o.quality_ok;
      const title = (o.quality_reasons.concat(o.amplitude_diagnostic ? ['amp:' + o.amplitude_diagnostic] : [])).join('; ');
      return `<td title="${title}" style="color:${bad ? '#c53030' : '#22543d'}">${bad ? '✗' : '✓'}${o.correlation_pct !== null ? ' ' + Math.round(o.correlation_pct) + '%' : ''}</td>`;
    }).join('');
    return `<tr class="${p.excluded ? 'excl' : ''} ${p.ambiguous ? 'ambig' : ''}">
      <td>${p.depth_m}m</td>${obs}
      <td>${p.excluded ? p.exclude_reasons.join(',') : (p.ambiguous ? '<b>多解未裁定</b>' : 'ok')}</td></tr>`;
  }).join('');
  return `<table><tr><th>cell(稳定身份)</th><th>b0</th><th>b1</th><th>b2</th><th>b3</th><th>状态/排除原因</th></tr>${rows}</table>`;
}

function cellCard(c) {
  const p = c.payload;
  const decs = (window.DEC||{});
  const cands = p.candidates.map(can => {
    const ids = `inst(${f3(can.instrument?.[0])},${f3(can.instrument?.[1])},${f3(can.instrument?.[2])})` +
      ` → ship(${f3(can.ship?.[0])},${f3(can.ship?.[1])},${f3(can.ship?.[2])})` +
      ` → earth(${can.earth ? can.earth.map(f3).join(',') : 'blocked:' + can.earth_reason})`;
    const errv = can.error_velocity_mps !== null ? ` · e=${f3(can.error_velocity_mps)}` : '';
    return `<div class="cand ${can.kind}"><b>${can.candidate_id}</b> <span class="mono">[${can.kind}${can.missing_beams.length ? ' 缺 beam'+can.missing_beams : ''}${errv}]</b></span><br/>
      <span class="mono">${ids}</span>
      <button class="small" onclick="decide('${p.cell_id}','${can.candidate_id}')">采用此候选</button></div>`;
  }).join('') || '<span class="muted">无合法候选（少于 3 个好 beam）</span>';
  const raw = p.observations.map((o, i) => `b${i}[${o.correlation_pct ?? '—'}% amp=${o.amplitude_counts ?? '诊断:' + (o.amplitude_diagnostic||'')}]`).join(' ');
  return `<div class="card ${p.excluded ? 'excl' : ''} ${p.ambiguous ? 'ambig' : ''}">
    <h3>cell ${p.cell_id} @ ${p.depth_m}m
      <button class="small" onclick="explain('${p.cell_id}')">来源链/为何重算</button></h3>
    <div class="mono">波束质量：${raw}</div>
    ${p.excluded ? `<div>⛔ 被排除：<b>${p.exclude_reasons.join(', ')}</b>（原始材料保留，不参与绘图）</div>` : ''}
    ${cands}
    <div class="mono">候选数 ${p.candidates.length} · 默认 ${p.default_candidate_id || '未裁定（不吞多解）'} · fp ${c.fingerprint.slice(0,14)}</div>
    <div id="ex-${p.cell_id}"></div></div>`;
}

function btCard(bt) {
  const can = bt.payload.candidates[0];
  return `<div class="card"><h3>Bottom Track · ${bt.payload.bt_id}（range ${bt.payload.range_m}m）</h3>
    <div class="mono">earth 船速：${can && can.earth ? can.earth.map(f3).join(', ') : 'blocked'} m/s · fp ${bt.fingerprint.slice(0,14)}</div></div>`;
}

async function explain(cellId) {
  const r = await fetch('/api/cell/' + cellId + '/explain');
  const d = await r.json();
  const rows = d.basis_checks.map(b =>
    `<tr><td>${b.input}</td><td class="mono">${b.ref}</td><td class="${b.state === 'stale' ? 'propagated' : 'contained'}">${b.state}</td></tr>`).join('');
  $('ex-' + cellId).innerHTML = `<table><tr><th>依赖</th><th>版本引用</th><th>状态</th></tr>${rows}</table>
    <div class="mono">needs_recompute=${d.needs_recompute} · version v${d.version}</div>`;
}

async function decide(subject, candidateId) {
  const actor = prompt('审阅者（reviewer-A / reviewer-B / lead）：', 'reviewer-A');
  if (!actor) return;
  const rationale = prompt('裁定理由：', 'continue selected branch; other candidates retained');
  await fetch('/api/decision', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ scope: 'cell_flow', subject, action: 'accept',
      chosen_candidate_id: candidateId, actor, rationale }) });
  toast('决定已 append（历史可回溯，候选未删除）');
  await loadState(); await showPing(SEL_PING);
}

async function showHistory() {
  const seq = parseInt($('histSeq').value, 10);
  const r = await fetch('/api/history/' + seq);
  if (!r.ok) { toast('该 seq 不存在'); return; }
  const h = await r.json();
  const count = { cell_flow: 0, profile: 0, bottom_track_flow: 0 };
  h.derivations.forEach(d => count[d.kind]++);
  $('histInfo').innerHTML = `投影 seq ${seq}：stream ${h.records.length}，派生 ${h.derivations.length}` +
    `<br/><span class="muted">这是只读历史视图：展示当时依赖版本与结论，不用最新计算覆盖</span>`;
  renderHistoryPing(h, SEL_PING);
}

function renderHistoryPing(h, ping) {
  const cells = h.derivations.filter(d => d.kind === 'cell_flow' && d.payload.ping_id === ping)
    .sort((a, b) => a.payload.depth_m - b.payload.depth_m);
  if (!cells.length) { $('center').innerHTML = `<div class="card">seq ${$('histSeq').value} 时 ${ping} 尚不存在</div>`; return; }
  $('center').innerHTML = `<div class="card"><h3>历史视图 · ${ping} @ seq ${$('histSeq').value}</h3>` +
    cells.map(c => {
      const p = c.payload;
      return `<div class="card"><b>${p.cell_id} v${c.version}</b> fp ${c.fingerprint.slice(0,12)}<br/>` +
        p.candidates.map(can => `<span class="mono">${can.candidate_id} earth=${can.earth ? can.earth.map(f3).join(',') : can.earth_reason}</span><br/>`).join('') +
        `</div>`;
    }).join('') + '</div>';
}

loadState();
JSEOF'
node -e "1" 2>/dev/null; .venv/bin/python -c "
import json,re
s=open('static/app.js').read()
print('js bytes', len(s))
"