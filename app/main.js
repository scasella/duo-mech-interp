import { Model, setHubBase } from "../engine/model.js";
import { Tokenizer } from "../engine/tokenizer.js";
import { Explorer } from "../engine/analysis.js";
import { watch } from "./layout.js";
import { createMap } from "./map.js";
import { CORPUS } from "./corpus.js";

const Q = new URLSearchParams(location.search);
const REPO = Q.get("model") || "Qwen/Qwen3.5-0.8B";
const HUB = Q.get("hub") || "https://huggingface.co";  // tests point this at a local mirror
if (Q.get("hub")) setHubBase(HUB);
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const ADD_STEPS = [...Array.from({ length: 30 }, (_, i) => +((i + 1) * 0.05).toFixed(2)), 1.75, 2, 2.5, 3, 4, 5];
const MUL_STEPS = [...Array.from({ length: 9 }, (_, i) => i * 0.5), 5, 6, 8, 12, 16];

const store = {
  get(k, d) { try { const v = localStorage.getItem("duo." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("duo." + k, JSON.stringify(v)); } catch {} },
};

const state = {
  focus: null,
  history: [],
  hpos: -1,
  chat: { mode: "add", dir: "writes", dose: { add: 0.8, mul: 2 }, rows: [[], [], []] },
  names: store.get("names", {}),
  recent: store.get("recent", []),
  run: null,
  runPos: -1,
  pane: "browse",
};
let ex = null, model = null, tok = null, map = null;
let job = null;  // {cancel} of the running steer

// ---------- layout ----------

let layoutMode = "narrow";
watch(m => {
  layoutMode = m.mode;
  // the composer sits under the neuron in side-by-side and narrow layouts, and on the deck (bottom pane) when stacked
  const composer = $("#composer");
  if (m.mode === "tall" || m.mode === "laptop") $("#browse").insertBefore(composer, $("#btabs"));
  else $("#neuron").append(composer);
  setPane(state.pane);
  map?.redraw();
});

function setPane(p) {
  state.pane = p;
  $$("#panes button").forEach(b => b.classList.toggle("on", b.dataset.p === p));
  $("#browse").classList.toggle("active", p === "browse");
  $("#neuron").classList.toggle("active", p === "neuron");
  if (p === "browse") map?.redraw();
}
$$("#panes button").forEach(b => (b.onclick = () => setPane(b.dataset.p)));

function setBTab(t) {
  $$("#btabs button").forEach(b => b.classList.toggle("on", b.dataset.bt === t));
  $$(".bpane").forEach(p => p.classList.toggle("on", p.id === "bt-" + t));
  if (t === "map") map?.redraw();
  if (t === "list") renderList();
}
$$("#btabs button").forEach(b => (b.onclick = () => setBTab(b.dataset.bt)));

function setNTab(t) {
  $$("#ntabs button").forEach(b => b.classList.toggle("on", b.dataset.nt === t));
  $$(".npane").forEach(p => p.classList.toggle("on", p.id === "nt-" + t));
  $("#composer").style.display = t === "chat" ? "" : "none";
  if (t === "lens") renderLens();
  if (t === "examples") renderExamples();
}
$$("#ntabs button").forEach(b => (b.onclick = () => setNTab(b.dataset.nt)));

// ---------- status ----------

let statusText = "";
function status(s) { statusText = s; paintStatus(); }
function paintStatus() {
  const mem = model ? ` · ${(model.gpu.bytes / 1e6).toFixed(0)} MB` : "";
  $("#status").textContent = (statusText || (model ? "ready" : "")) + mem;
}

// ---------- boot ----------

async function boot() {
  const facts = [];
  const fail = msg => { $("#boot-err").hidden = false; $("#boot-err").textContent = msg; };
  if (!navigator.gpu) {
    fail("This browser has no WebGPU. On iPhone it needs iOS 26 or later (Safari 26+). On a desktop, use a recent Chrome, Edge or Safari.");
    $("#boot-go").disabled = true;
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { fail("WebGPU is present but no GPU adapter is available."); $("#boot-go").disabled = true; return; }
  const lim = adapter.limits, info = adapter.info || {};
  facts.push(`GPU: ${[info.vendor, info.architecture].filter(Boolean).join(" ") || "available"}; max buffer ${(lim.maxStorageBufferBindingSize / 2 ** 20) | 0} MiB`);
  facts.push("Model: Qwen3.5-0.8B (Apache-2.0), 24 layers: 18 Gated DeltaNet + 6 gated attention, SwiGLU MLP, 3,584 neurons per layer");
  facts.push("Weights: int8 on the GPU, about 0.8 GB. First load streams 1.5 GB from Hugging Face; later loads read the copy kept in this browser");
  if (lim.maxStorageBufferBindingSize < 255 * 2 ** 20) fail("This GPU's storage buffers are too small for the embedding table (needs 243 MiB).");
  const cached = await hasCache();
  if (cached) { facts.push("Weights are already stored in this browser."); $("#boot-go").textContent = "Open"; $("#boot-clear").hidden = false; }
  $("#boot-facts").innerHTML = facts.map(f => `<li>${esc(f)}</li>`).join("");
  $("#boot-go").onclick = () => load().catch(e => { console.error(e); fail(String(e.message || e)); $("#boot-go").disabled = false; });
  $("#boot-clear").onclick = async () => { await Model.clearCache(REPO); await caches.delete(calName()); location.reload(); };
  if (cached || Q.has("autoload")) $("#boot-go").click();
}

async function hasCache() {
  try { return !!(await (await caches.open(`duo-mech-interp:${REPO}:q8v1`)).match("/w/complete")); } catch { return false; }
}

async function load() {
  $("#boot-go").disabled = true;
  $("#boot-progress").hidden = false;
  const bar = $(".bar-in"), msg = $("#boot-msg");
  const t0 = performance.now();
  const tokP = Tokenizer.fromUrl(`${HUB}/${REPO}/resolve/main/tokenizer.json`);
  model = await Model.load(REPO, {
    cache: !Q.has("nocache"),
    onProgress: p => {
      bar.style.width = `${(p.frac * 100).toFixed(1)}%`;
      const secs = ((performance.now() - t0) / 1000).toFixed(0);
      if (p.phase === "download") msg.textContent = `streaming weights ${(p.bytes / 1e9).toFixed(2)} / ${(p.total / 1e9).toFixed(2)} GB · ${secs}s`;
      else if (p.phase === "cache") msg.textContent = `reading stored weights ${(p.frac * 100).toFixed(0)}%`;
      else if (p.phase === "caching") msg.textContent = `storing weights in this browser for next time ${(p.frac * 100).toFixed(0)}%`;
    },
  });
  msg.textContent = "tokenizer…";
  tok = await tokP;
  model.alloc(3, 1024, 128);
  ex = new Explorer(model, tok);
  msg.textContent = "measuring dose units…";
  await ex.init();
  await loadCalibration();
  model.gpu.device.lost.then(i => status("GPU lost: " + (i.message || i.reason) + " — reload"));
  $("#boot").hidden = true;
  start();
}

// ---------- main ----------

function start() {
  map = createMap($("#map"), {
    L: model.L, I: model.I, types: model.types,
    onOpen: (l, n) => openNeuron(l, n),
    onPreview: previewNeuron,
    onLayer: l => ($("#map-legend").textContent = l == null ? "tap a layer to see its sheet, then drag across neurons to preview" : "drag across the sheet to preview, release to open · tap the title to go back"),
  });
  map.setMarks(markSet());
  paintStatus();
  const h = parseHash();
  const f = h || store.get("focus", null);
  if (f) openNeuron(f.layer, f.n, { pane: false });
  else { renderChatHeads(); $("#f-meta").textContent = "open one from the map, the list, or type L12 N1234 above"; }
  renderRunCal();
}

function parseHash() {
  const m = /L(\d+)\D*N?(\d+)/i.exec(location.hash);
  return m ? { layer: +m[1], n: +m[2] } : null;
}
addEventListener("hashchange", () => { const h = parseHash(); if (h && ex) openNeuron(h.layer, h.n); });

$("#jump").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const m = /L?\s*(\d+)\D+N?\s*(\d+)/i.exec(e.target.value);
  if (m) { openNeuron(+m[1], +m[2]); e.target.blur(); }
});

const key = (l, n) => `L${l}/N${n}`;
const markSet = () => new Set(Object.keys(state.names).filter(k => state.names[k]).map(k => { const m = /L(\d+)\/N(\d+)/.exec(k); return `${m[1]}:${m[2]}`; }));

// ---------- preview (map hover / scrub) ----------

let prevGen = 0;
async function previewNeuron(p) {
  const tip = $("#map-tip");
  const g = ++prevGen;
  if (!p) { tip.hidden = true; return; }
  const name = state.names[key(p.layer, p.n)];
  tip.hidden = false;
  tip.style.left = `${Math.min(p.x + 14, $("#map").clientWidth - 200)}px`;
  tip.style.top = `${Math.max(4, p.y - 60)}px`;
  tip.innerHTML = `<b>L${p.layer} N${p.n}</b>${name ? " · " + esc(name) : ""}`;
  await new Promise(r => setTimeout(r, 120));  // scrubbing: only look up the cell the finger rests on
  if (g !== prevGen) return;
  const lens = await ex.lens(p.layer, p.n, 6);
  if (g !== prevGen) return;
  tip.innerHTML += `<div class="up">+ ${lens.up.map(t => esc(show(t))).join("  ")}</div><div class="down">− ${lens.down.map(t => esc(show(t))).join("  ")}</div>`;
}

const show = t => t.replace(/\n/g, "⏎").replace(/^ +$/, m => "␣".repeat(m.length)).replace(/\t/g, "⇥");

// ---------- neuron ----------

async function openNeuron(layer, n, { pane = true, push = true } = {}) {
  if (!ex || layer < 0 || layer >= model.L || n < 0 || n >= model.I) return;
  stopJob();
  state.focus = { layer, n };
  store.set("focus", state.focus);
  if (push) {
    state.history = state.history.slice(0, state.hpos + 1).concat([{ layer, n }]);
    state.hpos = state.history.length - 1;
  }
  state.recent = [{ layer, n }, ...state.recent.filter(r => r.layer !== layer || r.n !== n)].slice(0, 40);
  store.set("recent", state.recent);
  history.replaceState(null, "", `#L${layer}N${n}`);
  $("#f-id").textContent = `L${layer} N${n}`;
  $("#f-name").value = state.names[key(layer, n)] || "";
  sizeName();
  $("#f-meta").textContent = `${model.types[layer] === "full_attention" ? "full attn" : "linear attn"} layer`;
  state.chat.rows = [[], [], []];
  renderThread();
  renderChatHeads();
  $("#c-reads").hidden = true;
  map.setFocus({ layer, n });
  map.setWeb(null);
  if (pane && layoutMode === "narrow") setPane("neuron");
  const cur = () => state.focus && state.focus.layer === layer && state.focus.n === n;
  const lens = await ex.lens(layer, n, 10);
  if (!cur()) return;
  $("#c-lens").innerHTML = `<span class="up">+</span><div class="chips">${chips(lens.up)}</div><span class="down">−</span><div class="chips">${chips(lens.down)}</div>`;
  const unit = await ex.doseValue(layer, n, 1, "add");
  if (!cur()) return;
  $("#f-meta").textContent = `${model.types[layer] === "full_attention" ? "full attn" : "linear attn"} · dose 1 = ${unit.toFixed(2)} act`;
  if ($("#ntabs .on").dataset.nt !== "chat") setNTab($("#ntabs .on").dataset.nt);
  const con = await ex.connections(layer, n, 12);
  if (!cur()) return;
  map.setWeb({ up: con.up.slice(0, 6), down: con.down.slice(0, 6) });
}

const chips = list => list.map(t => `<span class="chip">${esc(show(t))}</span>`).join("");

$("#f-prev").onclick = () => { if (state.hpos > 0) { const f = state.history[--state.hpos]; openNeuron(f.layer, f.n, { push: false }); } };
$("#f-next").onclick = () => { if (state.hpos < state.history.length - 1) { const f = state.history[++state.hpos]; openNeuron(f.layer, f.n, { push: false }); } };

function sizeName() { const t = $("#f-name"); t.style.height = "auto"; t.style.height = `${t.scrollHeight}px`; }
$("#f-name").addEventListener("input", e => {
  sizeName();
  if (!state.focus) return;
  const k = key(state.focus.layer, state.focus.n);
  if (e.target.value.trim()) state.names[k] = e.target.value; else delete state.names[k];
  store.set("names", state.names);
  map?.setMarks(markSet());
});

// ---------- chat: steer (writes) and read (reads you) ----------

function doseLabel(r) {
  const c = state.chat, d = c.dose[c.mode];
  if (c.mode === "mul") return r === 0 ? `▲ ×${d}` : r === 1 ? "×1" : `▼ ×−${d}`;
  return r === 0 ? `▲ +${d}` : r === 1 ? "0" : `▼ −${d}`;
}
function renderChatHeads() {
  $$("#c-thread .colhead").forEach((h, r) => (h.innerHTML = `<span>${doseLabel(r)}</span><span class="tps"></span>`));
}
function renderThread() {
  $$("#c-thread .col").forEach((col, r) => {
    const msgs = state.chat.rows[r];
    let html = "";
    for (let i = 0; i < msgs.length; i += 2) {
      const a = msgs[i + 1];
      html += `<div class="turn"><div class="q">${esc(msgs[i].content)}</div><div class="a${a?.live ? " live" : ""}">${esc(a?.content || "")}</div></div>`;
    }
    col.querySelector(".turns").innerHTML = html || `<div class="dim" style="font-size:12px">${r === 1 ? "unsteered" : r === 0 ? "neuron pushed up" : "neuron pushed down"}</div>`;
  });
}
function showDose() { const c = state.chat; $("#c-dose").value = c.mode === "mul" ? `×${c.dose.mul}` : c.dose.add; renderChatHeads(); }
$$("#c-mode button").forEach(b => (b.onclick = () => {
  state.chat.mode = b.dataset.m;
  $$("#c-mode button").forEach(x => x.classList.toggle("on", x === b));
  showDose();
}));
$$("#c-dir button").forEach(b => (b.onclick = () => {
  state.chat.dir = b.dataset.d;
  $$("#c-dir button").forEach(x => x.classList.toggle("on", x === b));
  const reads = state.chat.dir === "reads";
  $("#c-thread").hidden = reads; $("#c-dots").style.visibility = reads ? "hidden" : "";
  $("#c-reads").hidden = !reads;
  $("#c-input").placeholder = reads ? "Type text: see where this neuron fires" : "Who are you?";
  $("#c-go").textContent = reads ? "read" : "run";
}));
function step(dir) {
  const c = state.chat, steps = c.mode === "mul" ? MUL_STEPS : ADD_STEPS, cur = c.dose[c.mode];
  let i = steps.findIndex(s => s >= cur - 1e-9);
  if (i < 0) i = steps.length - 1;
  if (dir > 0) i = steps[i] > cur + 1e-9 ? i : Math.min(steps.length - 1, i + 1);
  else i = Math.max(0, steps[i] < cur - 1e-9 ? i : i - 1);
  c.dose[c.mode] = steps[i];
  showDose();
}
$("#c-minus").onclick = () => step(-1);
$("#c-plus").onclick = () => step(1);
$("#c-dose").addEventListener("change", e => {
  const v = parseFloat(String(e.target.value).replace("×", ""));
  if (isFinite(v) && v >= 0) state.chat.dose[state.chat.mode] = v;
  showDose();
});
$("#c-clear").onclick = () => { stopJob(); state.chat.rows = [[], [], []]; renderThread(); $("#c-reads").innerHTML = ""; };

function stopJob() { if (job) job.cancel.stop = true; }

$("#composer").addEventListener("submit", async e => {
  e.preventDefault();
  if (!ex || !state.focus) return;
  const text = $("#c-input").value.trim() || $("#c-input").placeholder;
  if (state.chat.dir === "reads") return runReads(text);
  if (job) { stopJob(); return; }
  const { layer, n } = state.focus, c = state.chat;
  $("#c-input").value = "";
  for (let r = 0; r < 3; r++) c.rows[r].push({ role: "user", content: text }, { role: "assistant", content: "", live: true });
  renderThread();
  const cancel = { stop: false };
  job = { cancel };
  $("#c-go").textContent = "stop";
  const t0 = performance.now();
  let ntok = 0;
  const turnEls = () => $$("#c-thread .col").map(col => [...col.querySelectorAll(".turn .a")].pop());
  let els = turnEls();
  try {
    await ex.steer({
      pushes: [{ layer, n, dose: c.dose[c.mode], mode: c.mode }],
      rowsMessages: c.rows.map(rows => rows.slice(0, -1).map(m => ({ role: m.role, content: m.content }))),
      maxNew: 200,
      cancel,
      onDelta: (r, d) => {
        const msg = c.rows[r][c.rows[r].length - 1];
        msg.content += d;
        if (els[r]) els[r].textContent = msg.content;
        ntok++;
        if (ntok % 6 === 0) status(`steering · ${(ntok / 3 / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s per column`);
        const box = els[r]?.closest(".turns");
        if (box && box.scrollHeight - box.scrollTop - box.clientHeight < 80) box.scrollTop = box.scrollHeight;
      },
    });
  } catch (err) {
    console.error(err);
    for (let r = 0; r < 3; r++) c.rows[r][c.rows[r].length - 1].content += `\n[${err.message || err}]`;
  }
  for (let r = 0; r < 3; r++) delete c.rows[r][c.rows[r].length - 1].live;
  renderThread();
  job = null;
  $("#c-go").textContent = "run";
  status("");
});

async function runReads(text) {
  const { layer, n } = state.focus;
  $("#c-reads").innerHTML = `<span class="dim">reading…</span>`;
  const res = await ex.read(layer, n, text, { chat: true });
  const max = Math.max(1e-6, ...res.acts.map(Math.abs));
  $("#c-reads").innerHTML = res.units.map(u => {
    const a = u.idx.reduce((s, i) => (Math.abs(res.acts[i]) > Math.abs(s) ? res.acts[i] : s), 0);
    return `<span title="${a.toFixed(3)}" data-a="${a.toFixed(3)}" style="background:${shade(a / max)}">${esc(show(u.text))}</span>`;
  }).join("") + `<div class="note" style="width:100%;padding:8px 0 0">max |act| ${max.toFixed(3)} · tap a token for its value</div>`;
  $$("#c-reads span[data-a]").forEach(s => (s.onclick = () => status(`act ${s.dataset.a}`)));
}

function shade(v) {
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(255,181,71,${a * 0.85})` : `rgba(255,79,154,${a * 0.85})`;
}

// narrow layout: page dots follow the column pager
$("#c-thread").addEventListener("scroll", e => {
  const el = e.target, i = Math.round(el.scrollLeft / Math.max(1, el.scrollWidth - el.clientWidth) * 2);
  $$("#c-dots i").forEach((d, k) => d.classList.toggle("on", k === i));
}, { passive: true });

// ---------- lens tab ----------

async function renderLens() {
  if (!state.focus) return;
  const { layer, n } = state.focus, el = $("#nt-lens");
  const f0 = state.focus;
  el.innerHTML = `<div class="sect"><h3>logit lens · what the down column writes</h3><div id="ll" class="dim">…</div></div>
    <div class="sect"><h3>steer lens · next-token change when pushed ±0.8</h3><div id="sl"><button class="btn" id="sl-go">compute (3 prompts × 3 rows)</button></div></div>
    <div class="sect"><h3>connections · direct path through the weights</h3><div id="cn" class="dim">…</div></div>`;
  const lens = await ex.lens(layer, n, 16);
  if (state.focus !== f0) return;
  $("#ll").innerHTML = `<div class="lensgrid"><span class="up">+</span><div class="chips">${chips(lens.up)}</div><span class="down">−</span><div class="chips">${chips(lens.down)}</div></div>`;
  $("#sl-go").onclick = async () => {
    $("#sl").innerHTML = `<span class="dim">pushing the neuron on three prompts…</span>`;
    const s = await ex.steerLens(layer, n, { dose: 0.8, k: 14 });
    if (state.focus !== f0) return;
    $("#sl").innerHTML = `<div class="lensgrid"><span class="up">+</span><div class="chips">${chips(s.up)}</div><span class="down">−</span><div class="chips">${chips(s.down)}</div></div>
      <div class="note" style="padding:6px 0 0">mean KL from baseline: up ${s.klUp.toFixed(3)} · down ${s.klDown.toFixed(3)}</div>`;
  };
  const con = await ex.connections(layer, n, 16);
  if (state.focus !== f0) return;
  const rows = (list, label) => `<div class="note" style="padding:6px 0 2px">${label}</div><ol class="rows">${list.map(c =>
    `<li data-l="${c.layer}" data-n="${c.n}"><span class="n ${c.sign > 0 ? "up" : "down"}">L${c.layer} N${c.n}</span><span class="z">${c.z.toFixed(1)}</span><span class="w">${esc(state.names[key(c.layer, c.n)] || "")}</span></li>`).join("")}</ol>`;
  $("#cn").innerHTML = rows(con.up, "fed by (earlier layers) · yellow pushes this neuron up, pink down") + rows(con.down, "feeds (later layers) · yellow: this neuron pushes it up");
  $$("#cn li").forEach(li => (li.onclick = () => openNeuron(+li.dataset.l, +li.dataset.n)));
  // fill lens words for the listed neurons lazily
  for (const li of $$("#cn li")) {
    const l = +li.dataset.l, nn = +li.dataset.n;
    const w = li.querySelector(".w");
    if (w.textContent) continue;
    const ls = await ex.lens(l, nn, 4);
    if (state.focus !== f0) return;
    w.innerHTML = `<b>${esc(ls.up.map(show).join(" "))}</b> · ${esc(ls.down.map(show).join(" "))}`;
  }
}

// ---------- examples tab (calibration contexts) ----------

function renderExamples() {
  const el = $("#nt-examples");
  if (!state.focus) return;
  const { layer, n } = state.focus;
  if (!ex.examples) {
    el.innerHTML = `<div class="sect"><p class="dim">Where does this neuron fire? Run the built-in corpus (${CORPUS.length} short passages: prose, code, chat, Chinese, French, law, medicine…) through the model once; every neuron's strongest contexts are kept on this device.</p><button class="btn" id="cal-go">run calibration</button> <span id="cal-msg" class="dim mono"></span></div>`;
    $("#cal-go").onclick = () => calibrate($("#cal-msg")).then(renderExamples);
    return;
  }
  const ctx = ex.contexts(layer, n, 28) || [];
  const chunks = ex.examples.chunks;
  const pos = ex.scale.pos[layer * model.I + n], neg = ex.scale.neg[layer * model.I + n];
  el.innerHTML = `<div class="note">range on the corpus: ${neg.toFixed(2)} … ${pos.toFixed(2)}. Each context ends a few tokens after the peak; shading is this neuron's activation on every token (rerun live).</div>` +
    ctx.map((c, i) => `<div class="ctx" data-i="${i}"><span class="act ${c.sign === "+" ? "up" : "down"}">${c.sign} ${c.act.toFixed(3)}</span>${esc(tok.decode(chunks[c.chunk].slice(c.s, c.e)))}</div>`).join("");
  // shade each context with the neuron's live activation over the whole chunk
  (async () => {
    for (const div of $$("#nt-examples .ctx")) {
      const c = ctx[+div.dataset.i];
      const ids = chunks[c.chunk];
      const r = await ex.readIds(layer, n, ids);
      if (!state.focus || state.focus.layer !== layer || state.focus.n !== n) return;
      const scale = Math.max(Math.abs(c.act), 1e-6);
      const units = tok.units(ids.slice(c.s, c.e));
      div.innerHTML = `<span class="act ${c.sign === "+" ? "up" : "down"}">${c.sign} ${c.act.toFixed(3)}</span>` + units.map(u => {
        const a = u.idx.reduce((s, j) => { const v = r[c.s + j]; return Math.abs(v) > Math.abs(s) ? v : s; }, 0);
        return `<span style="background:${shade(a / scale)}">${esc(u.text)}</span>`;
      }).join("");
    }
  })();
}

async function calibrate(msgEl) {
  const t0 = performance.now();
  await ex.calibrate(CORPUS, {
    onProgress: p => { if (msgEl) msgEl.textContent = `${p.chunk}/${p.chunks} · ${((performance.now() - t0) / 1000).toFixed(0)}s`; status(`calibrating ${p.chunk}/${p.chunks}`); },
  });
  status("");
  await saveCalibration();
  renderRunCal();
}

const calName = () => `duo-mech-interp:${REPO}:cal1`;
async function saveCalibration() {
  try {
    const c = await caches.open(calName());
    const { pos, neg } = ex.scale, { chunks, topP, topN } = ex.examples;
    const pack = list => { const out = new Float32Array(list.length * 12).fill(NaN); list.forEach((l, i) => (l || []).forEach((t, k) => out.set(t, i * 12 + k * 3))); return out; };
    await c.put("/pos", new Response(pos)); await c.put("/neg", new Response(neg));
    await c.put("/topP", new Response(pack(topP))); await c.put("/topN", new Response(pack(topN)));
    await c.put("/chunks", new Response(JSON.stringify(chunks)));
  } catch (e) { console.warn("could not store calibration", e); }
}
async function loadCalibration() {
  try {
    const c = await caches.open(calName());
    const get = async k => { const r = await c.match(k); return r ? r.arrayBuffer() : null; };
    const [pos, neg, tp, tn] = await Promise.all(["/pos", "/neg", "/topP", "/topN"].map(get));
    const ch = await c.match("/chunks");
    if (!pos || !neg || !tp || !tn || !ch) return;
    const unpack = buf => { const f = new Float32Array(buf), n = f.length / 12, out = new Array(n);
      for (let i = 0; i < n; i++) { const l = []; for (let k = 0; k < 4; k++) { const o = i * 12 + k * 3; if (!isNaN(f[o])) l.push([f[o], f[o + 1], f[o + 2]]); } out[i] = l; } return out; };
    ex.scale = { pos: new Float32Array(pos), neg: new Float32Array(neg) };
    ex.examples = { chunks: await ch.json(), topP: unpack(tp), topN: unpack(tn) };
  } catch (e) { console.warn("no stored calibration", e); }
}

// ---------- fires (run tab) ----------

function renderRunCal() {
  $("#run-cal").innerHTML = ex?.scale ? "scores: activation as a multiple of the neuron's range on the built-in corpus"
    : `scores are raw activations until you calibrate · <button class="btn" id="run-cal-go">calibrate</button> <span class="mono" id="run-cal-msg"></span>`;
  const b = $("#run-cal-go");
  if (b) b.onclick = () => calibrate($("#run-cal-msg"));
}

let runSign = "both";
$$("#run-sign button").forEach(b => (b.onclick = () => { runSign = b.dataset.s; $$("#run-sign button").forEach(x => x.classList.toggle("on", x === b)); renderRun(); }));
["#run-lo", "#run-hi"].forEach(s => $(s).addEventListener("change", renderRun));
$("#run-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (!ex) return;
  const text = $("#run-text").value.trim();
  if (!text) return;
  $("#run-go").disabled = true;
  status("running text…");
  try {
    state.run = await ex.activations(text, { maxTokens: 128 });
    state.runPos = -1;
    renderRunToks();
    renderRun();
  } finally { $("#run-go").disabled = false; status(""); }
});

function renderRunToks() {
  const r = state.run;
  $("#run-toks").innerHTML = r.units.map((u, i) => `<span data-i="${i}" class="${state.runPos === u.idx[u.idx.length - 1] ? "sel" : ""}">${esc(show(u.text))}</span>`).join("") +
    (r.ids.length >= 128 ? `<span class="dim"> (first 128 tokens)</span>` : "");
  $$("#run-toks span[data-i]").forEach(s => (s.onclick = () => {
    const u = r.units[+s.dataset.i], p = u.idx[u.idx.length - 1];
    state.runPos = state.runPos === p ? -1 : p;
    renderRunToks(); renderRun();
  }));
}

function renderRun() {
  const r = state.run;
  if (!r) return;
  const lo = +$("#run-lo").value || 0, hi = +$("#run-hi").value || 23;
  const top = ex.topAt(r, state.runPos, { sign: runSign, lo, hi, limit: 150 });
  const tokAt = i => show(tok.piece(r.ids[i]));
  $("#run-rows").innerHTML = top.map(t => `<li data-l="${t.layer}" data-n="${t.n}"><span class="n ${t.score > 0 ? "up" : "down"}">L${t.layer} N${t.n}</span><span class="z">${Math.abs(t.score).toFixed(1)}</span><span class="w">${esc(state.names[key(t.layer, t.n)] || "")} <b>${esc(tokAt(t.pos))}</b> · ${t.act.toFixed(2)}</span></li>`).join("");
  $$("#run-rows li").forEach(li => (li.onclick = () => openNeuron(+li.dataset.l, +li.dataset.n)));
  // map overlay: each neuron's score at the chosen token (or its strongest over the text), scaled so 1 = strong
  const ov = new Float32Array(model.L * model.I);
  const T = r.T, I = model.I;
  for (let l = 0; l < model.L; l++) for (let n = 0; n < I; n++) {
    let best = 0;
    const t0 = state.runPos >= 0 ? state.runPos : 1, t1 = state.runPos >= 0 ? state.runPos + 1 : T;
    for (let t = t0; t < t1; t++) { const s = ex.scoreOf(l, n, r.acts[(l * T + t) * I + n]); if (Math.abs(s) > Math.abs(best)) best = s; }
    ov[l * I + n] = ex.scale ? best : best / 4;
  }
  map.setOverlay(ov);
}

// ---------- list tab ----------

function renderList() {
  const named = Object.entries(state.names).filter(([, v]) => v).map(([k, v]) => { const m = /L(\d+)\/N(\d+)/.exec(k); return { layer: +m[1], n: +m[2], name: v }; });
  const rows = [...named.map(x => ({ ...x, kind: "named" })), ...state.recent.filter(r => !state.names[key(r.layer, r.n)]).map(x => ({ ...x, kind: "recent" }))];
  $("#list-rows").innerHTML = rows.map(r => `<li data-l="${r.layer}" data-n="${r.n}"><span class="n">L${r.layer} N${r.n}</span><span class="z">${r.kind === "named" ? "★" : ""}</span><span class="w"><b>${esc(r.name || "")}</b></span></li>`).join("") ||
    `<li><span></span><span></span><span class="w">nothing yet</span></li>`;
  $$("#list-rows li[data-l]").forEach(li => (li.onclick = () => openNeuron(+li.dataset.l, +li.dataset.n)));
}

boot().catch(e => { console.error(e); $("#boot-err").hidden = false; $("#boot-err").textContent = String(e.stack || e); });
