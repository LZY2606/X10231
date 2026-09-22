"use strict";
let state = { ping: 0, cid: null, commits: [], view: null, tab: "quality",
              currentHead: null };

const $ = (s) => document.querySelector(s);
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.style.opacity = 1;
  setTimeout(() => (t.style.opacity = 0), 2600);
}
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    if (r.status === 409 && data.detail && data.detail.conflict_id) {
      toast("并发冲突已记录：" + data.detail.conflict_id + "，可在左侧解决");
      await loadState();
      throw data;
    }
    toast("请求失败 " + r.status);
    throw data;
  }
  return data;
}
function pill(status) {
  const map = { four_beam_ok: "four", three_beam: "three",
                rejected: "rejected", ambiguous_multi: "ambiguous_multi",
                four: "four", three: "three" };
  return `<span class="pill ${status}">${status}</span>`;
}
function f3(a) {
  return Array.isArray(a)
    ? "[" + a.map((x) => (x === null ? "–" : x.toFixed(3))).join(", ") + "]"
    : a;
}

async function loadState() {
  const st = await jget("/api/state");
  state.currentHead = st.head;
  $("#head-meta").textContent =
    `main HEAD ${st.head ? st.head.slice(0, 12) : "–"} · 事件 ${
      st.event_count}`;
  const commits = await jget("/api/commits");
  state.commits = commits;
  $("#cidselect").innerHTML = commits.map((c) =>
    `<option value="${c.cid}" ${c.cid === state.cid ||
      (!state.cid && c.cid === st.head) ? "selected" : ""}>` +
    `${c.label} (${c.branch}) ${c.cid.slice(0, 8)}</option>`).join("");
  if (!state.cid) state.cid = st.head;
  $("#cidselect").value = state.cid;
  renderEvents(await jget("/api/events"));
  renderBranches(st.branches);
  renderGuide(st);
}

function renderEvents(events) {
  $("#events").innerHTML = events.slice(-14).reverse().map((e) => {
    const bad = /conflict|excluded|gap|failure|ambiguous/i.test(e.type);
    const warn = /decision|revised|change/i.test(e.type);
    const note = (e.payload && (e.payload.note || e.payload.rationale ||
      e.payload.resolution || "")) || "";
    return `<div class="evline ${bad ? "bad" : warn ? "warn" : ""}">
      <div><b>${e.type}</b> <span class="id">${e.eid}</span></div>
      <div class="muted">${e.actor} · ${
        e.commit_cid ? e.commit_cid.slice(0, 8) : "–"}</div>
      <div>${note}</div></div>`;
  }).join("");
}

function renderBranches(branches) {
  $("#branches").innerHTML = branches.map((b) =>
    `<div class="row"><span>${b.name} <span class="id">${
      (b.current_cid || "–").slice(0, 10)}</span></span>` +
    (b.name !== "main"
      ? `<button onclick="continueBranch('${b.name}','${b.current_cid}')">继续此版</button>`
      : "") + `</div>`).join("");
}

function renderGuide(st) {
  const r = st.refs || {};
  $("#guide").innerHTML = `
  <b>90 秒路径：</b>
  <ol style="margin:6px 0 0 16px;padding:0">
   <li>ping <b>2</b>：单 beam 失效（三波束唯一解）。</li>
   <li>ping <b>4→5</b>：heading 跨 360°；ping <b>7</b>：姿态缺口不插值。</li>
   <li>ping <b>9</b>：四/三波束多解并存，可裁定候选。</li>
   <li>cell <b>5</b>：无效深度 cell 全 ping 排除。</li>
   <li>“垂向剖面”看图；事件时间线回看任意版本。</li>
  </ol>
  <div class="small muted">分支：变换顺序 ${
    (r.transform_order_alternative || "–").slice(0, 8)} · 磁偏角 ${
    (r.declination_alternative || "–").slice(0, 8)}</div>`;
}

function renderPingButtons() {
  $("#pings").innerHTML = Array.from({ length: 12 }, (_, i) =>
    `<button class="${i === state.ping ? "active" : ""}"
      onclick="setPing(${i})">${i}</button>`).join("");
}
function setPing(i) { state.ping = i; renderPingButtons(); loadPing(); }

async function loadPing() {
  state.cid = $("#cidselect").value;
  renderPingButtons();
  const data = await jget(`/api/ping/${state.ping}?cid=${state.cid}`);
  state.view = data.view;
  if (state.tab === "quality") renderQuality();
  if (state.tab === "transform") renderTransform();
}
