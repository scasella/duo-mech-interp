import { Model, setHubBase } from "../engine/model.js";
import { Tokenizer } from "../engine/tokenizer.js";
import { Explorer } from "../engine/analysis.js";
import { watch } from "./layout.js";
import { createMap } from "./map.js";
import { CORPUS } from "./corpus.js";
import { EXAMPLES } from "./examples.js";

// Two places: Steer (ask once, compare three answers) and Explore (Neuron · Find · Browse). On the inner display they
// face each other across the fold; on the outer display they are two tabs.

const Q = new URLSearchParams(location.search);
const REPO = Q.get("model") || "Qwen/Qwen3.5-0.8B";
const HUB = Q.get("hub") || "https://huggingface.co";  // tests point this at a local mirror
if (Q.get("hub")) setHubBase(HUB);
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = t => t.replace(/\n/g, "⏎").replace(/^ +$/, m => "␣".repeat(m.length)).replace(/\t/g, "⇥");
const SUGGESTIONS = ["Who are you?", "What do you love most?", "Describe the ocean.", "Tell me a short story."];
const RANGE = { add: { max: 2, step: 0.05 }, mul: { max: 8, step: 0.5 } };

const store = {
  get(k, d) { try { const v = localStorage.getItem("duo." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("duo." + k, JSON.stringify(v)); } catch {} },
};
const key = (l, n) => `L${l}/N${n}`;
const parseKey = k => { const m = /L(\d+)\/N(\d+)/.exec(k); return m ? { layer: +m[1], n: +m[2] } : null; };

const names0 = store.get("names", {});
const state = {
  focus: null,
  mode: "add",
  dose: { add: 0.8, mul: 2 },
  keep: false,
  rows: [[], [], []],
  lastQ: null,
  names: names0,
  saved: store.get("saved", Object.keys(names0).filter(k => names0[k])),
  recent: store.get("recent", []),
  pane: "steer",
  xtab: "find",
  find: null, findPos: -1, findSign: "both",
};
let ex = null, model = null, tok = null, map = null;
let job = null;            // the running steer: {cancel}
let calibration = null;    // promise of the background calibration
let calFrac = 0;

// ---------- layout ----------

let layoutMode = "narrow", lastMeasure = null;
watch(m => {
  layoutMode = m.mode;
  lastMeasure = m;
  placeComposer();
  setPane(state.pane);
  map?.redraw();
});

// The question box sits under the answers, except in portrait (stacked) layouts where it lives on the deck. It moves
// only when it has to and never while focused: moving a focused field blurs it, which closes the iOS keyboard.
function placeComposer() {
  const c = $("#composer");
  const stacked = layoutMode === "tall" || layoutMode === "laptop";
  const home = stacked ? $("#explore") : $("#steer");
  if (c.parentElement === home) return;
  if (c.contains(document.activeElement)) { c.addEventListener("focusout", () => setTimeout(placeComposer, 0), { once: true }); return; }
  if (stacked) home.insertBefore(c, home.firstElementChild); else home.append(c);
}

const narrow = () => layoutMode === "narrow";
function setPane(p) {
  state.pane = p;
  $$("#tabbar button").forEach(b => b.classList.toggle("on", b.dataset.p === p));
  $("#steer").classList.toggle("active", p === "steer");
  $("#explore").classList.toggle("active", p === "explore");
  if (p === "explore" && state.xtab === "browse") map?.redraw();
}
$$("#tabbar button").forEach(b => (b.onclick = () => setPane(b.dataset.p)));

function setXTab(t) {
  state.xtab = t;
  $$("#xtabs button").forEach(b => { b.classList.toggle("on", b.dataset.x === t); b.setAttribute("aria-selected", b.dataset.x === t); });
  $$(".xpane").forEach(p => p.classList.toggle("on", p.id === "x-" + t));
  if (t === "browse") { renderSaved(); map?.redraw(); }
}
$$("#xtabs button").forEach(b => (b.onclick = () => setXTab(b.dataset.x)));

// ---------- feedback ----------

let toastTimer;
function toast(msg, action) {
  const t = $("#toast");
  t.innerHTML = `<span>${esc(msg)}</span>` + (action ? `<button type="button">${esc(action.label)}</button>` : "");
  if (action) t.querySelector("button").onclick = () => { t.hidden = true; action.run(); };
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), action ? 9000 : 4000);
}
const fail = (what, err) => { console.error(err); toast(`${what}: ${err?.message || err}`); };

// ---------- welcome / loading ----------

async function boot() {
  const facts = [];
  const stop = msg => { $("#w-err").hidden = false; $("#w-err").textContent = msg; $("#w-go").disabled = true; };
  if (!navigator.gpu) return stop("This browser can't run the model: it has no WebGPU. On iPhone, use Safari on iOS 26 or later.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return stop("WebGPU is present, but no GPU is available to it.");
  const lim = adapter.limits, info = adapter.info || {};
  if (lim.maxStorageBufferBindingSize < 255 * 2 ** 20) return stop("This device's GPU buffers are too small for the model.");
  const lm = lastMeasure;
  facts.push("Model: Qwen3.5-0.8B (Apache-2.0), 24 layers × 3,584 MLP neurons; the text decoder only.");
  facts.push("Runs on this device with WebGPU (int8 weights, about 0.8 GB of GPU memory). Nothing you type leaves the phone.");
  facts.push(`GPU: ${[info.vendor, info.architecture].filter(Boolean).join(" ") || "available"}.`);
  facts.push(`Screen: ${lm.mode} layout` + (lm.segmented ? ", fold reported by Safari." : lm.mode === "narrow" ? "." :
    ". For the exact fold position, turn on Settings › Apps › Safari › Advanced › Feature Flags › Viewport Segments API."));
  facts.push("A port of lyramakesmusic/neuron-explorer.");
  $("#w-facts").innerHTML = facts.map(f => `<li>${esc(f)}</li>`).join("");
  const cached = await hasCache();
  $("#w-go").onclick = () => load(cached).catch(e => { console.error(e); $("#w-err").hidden = false; $("#w-err").textContent = String(e.message || e); $("#w-go").disabled = false; $("#w-progress").hidden = true; });
  $("#w-forget").onclick = async () => {
    await Model.clearCache(REPO); await caches.delete(calName());
    try { localStorage.removeItem(`duo.residP50:${REPO}:q8v1`); } catch {}
    location.reload();
  };
  if (cached) {
    $("#w-forget").hidden = false;
    $("#w-go").textContent = "Open";
    $("#w-note").textContent = "The model is already on this device.";
  }
  if (cached || Q.has("autoload")) $("#w-go").click();
}

async function hasCache() {  // caches.open would create the cache, so look before opening
  const name = `duo-mech-interp:${REPO}:q8v1`;
  try { return (await caches.has(name)) && !!(await (await caches.open(name)).match("/w/complete")); } catch { return false; }
}

async function load(cached) {
  $("#w-go").hidden = true;
  $("#w-note").hidden = true;
  $("#w-progress").hidden = false;
  const bar = $(".meter i"), msg = $("#w-msg");
  const t0 = performance.now();
  msg.textContent = cached ? "Opening…" : "Starting download…";
  const tokP = Tokenizer.fromUrl(`${HUB}/${REPO}/resolve/main/tokenizer.json`);
  model = await Model.load(REPO, {
    cache: !Q.has("nocache"),
    onProgress: p => {
      bar.style.width = `${(p.frac * 100).toFixed(1)}%`;
      if (p.phase === "download") {
        const el = (performance.now() - t0) / 1000, rate = p.bytes / Math.max(el, 1);
        const left = Math.max(0, (p.total - p.bytes) / Math.max(rate, 1));
        msg.textContent = `Downloading ${(p.bytes / 1e9).toFixed(2)} of ${(p.total / 1e9).toFixed(2)} GB` + (el > 5 ? ` · about ${left > 90 ? Math.round(left / 60) + " min" : Math.round(left) + " s"} left` : "");
      } else if (p.phase === "cache") msg.textContent = "Opening…";
      else if (p.phase === "caching") msg.textContent = "Saving to this device for next time…";
    },
  });
  tok = await tokP;
  model.alloc(3, 1024, 128);
  ex = new Explorer(model, tok);
  // dose units depend only on the weights: measured once, then remembered
  const unitsKey = `residP50:${REPO}:q8v1`;
  const known = store.get(unitsKey, null);
  if (Array.isArray(known) && known.length === model.L) ex.residP50 = known;
  else { msg.textContent = "Getting ready (first time only)…"; store.set(unitsKey, await ex.init()); }
  await loadCalibration();
  model.gpu.device.lost.then(() => toast("The GPU was reset (was the app in the background?).", { label: "Reload", run: () => location.reload() }));
  $("#welcome").hidden = true;
  start();
}

// ---------- start ----------

function start() {
  map = createMap($("#map"), {
    L: model.L, I: model.I, types: model.types,
    onOpen: (l, n) => openNeuron(l, n, { go: "steer" }),
    onPreview: previewNeuron,
    onLayer: l => {
      previewNeuron(null);
      $("#map-legend").textContent = l == null ? "Each band is one of the 24 layers. Tap a band, then slide across it."
        : "Slide across the sheet to preview a neuron; lift to open it. Tap the title to go back.";
    },
  });
  map.setMarks(new Set(state.saved.map(k => { const f = parseKey(k); return `${f.layer}:${f.n}`; })));
  showWeightMap();
  renderStrength();
  const h = parseHash();
  const f = h || store.get("focus", null);
  if (f) openNeuron(f.layer, f.n, { go: null });
  else renderEmpty();
  if (!ex.scale) setTimeout(() => (calibration ||= runCalibration()), 1500);
}

function parseHash() {
  const m = /L(\d+)\D*N?(\d+)/i.exec(location.hash);
  return m ? { layer: +m[1], n: +m[2] } : null;
}
addEventListener("hashchange", () => { const h = parseHash(); if (h && ex && (h.layer !== state.focus?.layer || h.n !== state.focus?.n)) openNeuron(h.layer, h.n); });

// ---------- steer ----------

const label = (l, n) => state.names[key(l, n)] || `Neuron ${n}`;

function renderHeader() {
  const f = state.focus;
  $("#who-name").textContent = f ? label(f.layer, f.n) : "Choose a neuron";
  $("#who-sub").textContent = f ? (state.names[key(f.layer, f.n)] ? `Neuron ${f.n} · layer ${f.layer}` : `Layer ${f.layer} of ${model.L}`) : "";
  $("#save").hidden = !f;
  const saved = f && state.saved.includes(key(f.layer, f.n));
  $("#save").textContent = saved ? "★" : "☆";
  $("#save").classList.toggle("on", !!saved);
  $("#save").setAttribute("aria-label", saved ? "Remove from your neurons" : "Save this neuron");
}

$("#who").onclick = () => { setXTab("about"); if (narrow()) setPane("explore"); };
$("#save").onclick = () => {
  const f = state.focus; if (!f) return;
  const k = key(f.layer, f.n);
  state.saved = state.saved.includes(k) ? state.saved.filter(x => x !== k) : [k, ...state.saved];
  store.set("saved", state.saved);
  renderHeader(); renderSaved();
  map?.setMarks(new Set(state.saved.map(x => { const p = parseKey(x); return `${p.layer}:${p.n}`; })));
  toast(state.saved.includes(k) ? "Saved to Your neurons (Explore › Browse)." : "Removed from Your neurons.");
};

function renderEmpty() {
  const f = state.focus, e = $("#empty");
  const hasAnswers = state.rows[0].length > 0;
  $("#lanes").hidden = !hasAnswers;
  e.hidden = hasAnswers;
  $("#composer").hidden = !f;
  renderSuggest();
  if (hasAnswers) return;
  if (!f) {
    e.innerHTML = `<h2>Choose a neuron to start</h2>
      <p>You ask a question once. The model answers three times: with the neuron <b class="up">boosted</b>, left <b>as is</b>, and <b class="down">suppressed</b>.</p>
      <button id="try-example" class="primary" type="button">Try an example</button>
      <button id="go-find" class="secondary" type="button">Find neurons in your own words</button>
      <button id="surprise" class="link" type="button">Surprise me</button>`;
  } else {
    e.innerHTML = `<h2>Ask ${esc(label(f.layer, f.n))} something</h2>
      <p>Pick a question below or type your own. You'll see its answer with this neuron <b class="up">boosted</b>, <b>as is</b>, and <b class="down">suppressed</b>.</p>` +
      (state.lastQ ? `<button id="ask-again" class="primary" type="button">Ask “${esc(state.lastQ)}”</button>` : "") +
      `<button id="surprise" class="link" type="button">Try another neuron</button>`;
  }
  $("#try-example")?.addEventListener("click", tryExample);
  $("#go-find")?.addEventListener("click", () => { setXTab("find"); if (narrow()) setPane("explore"); setTimeout(() => $("#find-text").focus(), 50); });
  $("#surprise")?.addEventListener("click", surprise);
  $("#ask-again")?.addEventListener("click", () => ask(state.lastQ));
}

function renderSuggest() {
  const show = state.focus && !job && state.rows[0].length === 0;
  $("#suggest").innerHTML = show ? SUGGESTIONS.map(s => `<button type="button">${esc(s)}</button>`).join("") : "";
  $("#suggest").hidden = !show;
}
$("#suggest").addEventListener("click", e => { const b = e.target.closest("button"); if (b) ask(b.textContent); });

let exampleIdx = 0;
function tryExample() {
  const x = EXAMPLES[exampleIdx++ % EXAMPLES.length];
  state.dose.add = x.dose ?? 0.8;
  renderStrength();
  openNeuron(x.layer, x.n, { go: "steer" });
  ask(x.prompt);
}

function surprise() {
  const l = 6 + Math.floor(Math.random() * 13), n = Math.floor(Math.random() * model.I);  // middle layers steer most visibly
  openNeuron(l, n, { go: "steer" });
  if (state.lastQ) ask(state.lastQ);
}

function doseText(r) {
  const d = state.dose[state.mode];
  if (state.mode === "mul") return r === 0 ? `×${d}` : `×${-d}`;
  return r === 0 ? `+${d}` : `−${d}`;
}
function renderStrength() {
  const s = $("#strength"), rg = RANGE[state.mode];
  s.max = rg.max; s.step = rg.step; s.value = state.dose[state.mode];
  $("#strength-v").textContent = state.mode === "mul" ? `×${state.dose.mul}` : state.dose.add;
  $$(".lane .dose").forEach((el, i) => (el.textContent = doseText(i === 0 ? 0 : 2)));
  $$("#mode button").forEach(b => b.classList.toggle("on", b.dataset.m === state.mode));
}
$("#strength").addEventListener("input", e => { state.dose[state.mode] = +(+e.target.value).toFixed(2); renderStrength(); });

async function openNeuron(layer, n, { go = "steer" } = {}) {
  if (!ex || !(layer >= 0 && layer < model.L && n >= 0 && n < model.I)) return;
  stopJob();
  state.focus = { layer, n };
  store.set("focus", state.focus);
  state.recent = [{ layer, n }, ...state.recent.filter(r => r.layer !== layer || r.n !== n)].slice(0, 30);
  store.set("recent", state.recent);
  history.replaceState(null, "", `#L${layer}N${n}`);
  state.rows = [[], [], []];
  renderHeader();
  renderLanes();
  renderEmpty();
  renderAbout();
  map?.setFocus({ layer, n });
  map?.setWeb(null);
  if (go && narrow()) setPane(go === "about" ? "explore" : "steer");
  if (go === "about") setXTab("about");
  try {
    const con = await ex.connections(layer, n, 12);
    if (state.focus?.layer === layer && state.focus?.n === n) map?.setWeb({ up: con.up.slice(0, 6), down: con.down.slice(0, 6) });
  } catch (err) { console.error(err); }
}

function stopJob() { if (job) job.cancel.stop = true; }

function renderLanes() {
  const rows = state.rows;
  const keep = state.keep;
  $("#asked").innerHTML = !keep && rows[0].length ? `You asked <b>${esc(rows[0][rows[0].length - 2].content)}</b>` : "";
  $$(".lane").forEach((lane, r) => {
    const msgs = rows[r], say = lane.querySelector(".say");
    let html = "";
    for (let i = keep ? 0 : Math.max(0, msgs.length - 2); i < msgs.length; i += 2) {
      if (keep) html += `<span class="q">${esc(msgs[i].content)}</span>`;
      html += esc(msgs[i + 1]?.content || "");
    }
    say.innerHTML = html;
    const last = msgs[msgs.length - 1];
    say.classList.toggle("live", !!last?.live);
    say.classList.toggle("wait", !!last?.live && !last.content);
    if (last?.live && !last.content) say.textContent = "thinking…";
  });
  renderStrength();
}

$("#composer").addEventListener("submit", e => {
  e.preventDefault();
  if (job) { stopJob(); return; }
  const text = $("#prompt").value.trim();
  if (text) ask(text);
});

async function ask(text) {
  if (!ex || !state.focus) return;
  if (job) { stopJob(); await job.done; }
  const { layer, n } = state.focus;
  state.lastQ = text;
  // each run writes into its own rows: opening another neuron replaces state.rows, and the run (which stops at its
  // next token) must not touch the new ones
  const rows = state.keep ? state.rows : (state.rows = [[], [], []]);
  const live = () => state.rows === rows;
  for (let r = 0; r < 3; r++) rows[r].push({ role: "user", content: text }, { role: "assistant", content: "", live: true });
  $("#prompt").value = "";
  $("#prompt").blur();
  renderEmpty();
  renderLanes();
  const cancel = { stop: false };
  let release;
  job = { cancel, done: new Promise(r => (release = r)) };
  $("#busy").hidden = false;
  $("#send").textContent = "■"; $("#send").classList.add("stop"); $("#send").setAttribute("aria-label", "Stop");
  renderSuggest();
  const says = $$(".lane .say");
  try {
    await ex.steer({
      pushes: [{ layer, n, dose: state.dose[state.mode], mode: state.mode }],
      rowsMessages: rows.map(rs => rs.slice(0, -1).map(m => ({ role: m.role, content: m.content }))),
      maxNew: 120,
      cancel,
      onDelta: (r, d) => {
        const msg = rows[r][rows[r].length - 1];
        msg.content += d;
        if (!live() || state.keep) { if (live()) renderLanes(); return; }
        const el = says[r];
        el.classList.remove("wait");
        el.textContent = msg.content;
        if (el.closest(".lane").scrollHeight > el.closest(".lane").clientHeight) el.closest(".lane").scrollTop = 1e9;
      },
    });
  } catch (err) {
    for (let r = 0; r < 3; r++) rows[r][rows[r].length - 1].content += ` [${err.message || err}]`;
    fail("Could not finish", err);
  } finally {
    for (let r = 0; r < 3; r++) delete rows[r][rows[r].length - 1].live;
    job = null;
    release();
    $("#busy").hidden = true;
    $("#send").textContent = "↑"; $("#send").classList.remove("stop"); $("#send").setAttribute("aria-label", "Ask");
    if (live()) renderLanes();
    renderSuggest();
  }
}

// ---------- explore: this neuron ----------

const words = list => list.map(t => `<span>${esc(show(t))}</span>`).join("");

async function renderAbout() {
  const f = state.focus;
  $("#about-none").hidden = !!f;
  $("#about").hidden = !f;
  if (!f) return;
  const { layer, n } = f;
  $("#a-title").textContent = label(layer, n);
  $("#a-sub").textContent = `Neuron ${n} in layer ${layer} of ${model.L} · one of ${model.I.toLocaleString()} in this layer`;
  $("#a-name").value = state.names[key(layer, n)] || "";
  $("#a-up").innerHTML = $("#a-down").innerHTML = "";
  $("#a-links").innerHTML = "";
  $("#a-next").innerHTML = `<p class="fine">Pushes this neuron up and down on three short prompts and lists the next words that gain the most.</p><button id="next-go" class="secondary small" type="button">Measure</button>`;
  $("#read-out").innerHTML = "";
  for (const d of ["d-fires", "d-links", "d-next"]) if ($("#" + d).open) loadDetail(d);
  renderTech();
  try {
    const lens = await ex.lens(layer, n, 8);
    if (!same(f)) return;
    $("#a-up").innerHTML = words(lens.up);
    $("#a-down").innerHTML = words(lens.down);
  } catch (err) { fail("Could not read the neuron", err); }
}
const same = f => state.focus === f;

$("#a-name").addEventListener("input", e => {
  const f = state.focus; if (!f) return;
  const k = key(f.layer, f.n), v = e.target.value.trim();
  if (v) { state.names[k] = e.target.value; if (!state.saved.includes(k)) { state.saved = [k, ...state.saved]; store.set("saved", state.saved); } }
  else delete state.names[k];
  store.set("names", state.names);
  renderHeader();
  $("#a-title").textContent = label(f.layer, f.n);
});

for (const d of ["d-fires", "d-links", "d-next"]) $("#" + d).addEventListener("toggle", e => { if (e.target.open) loadDetail(d); });

async function loadDetail(d) {
  const f = state.focus; if (!f) return;
  const { layer, n } = f;
  try {
    if (d === "d-fires") await renderFires(f);
    if (d === "d-links") {
      $("#a-links").innerHTML = `<li class="none">Tracing the weights…</li>`;
      const con = await ex.connections(layer, n, 8);
      if (!same(f)) return;
      const row = (c, dir) => `<li data-l="${c.layer}" data-n="${c.n}"><span class="t"><b>${esc(label(c.layer, c.n))}</b><span class="dim">layer ${c.layer} · ${dir}</span><span class="r ${c.sign > 0 ? "up" : "down"}">${c.sign > 0 ? "▲ excites" : "▼ inhibits"}</span></span><span class="w"></span></li>`;
      $("#a-links").innerHTML = `<li class="none">Earlier neurons that feed it</li>` + con.up.slice(0, 6).map(c => row(c, "feeds this")).join("") +
        `<li class="none">Later neurons it feeds</li>` + con.down.slice(0, 6).map(c => row(c, "fed by this")).join("");
      for (const li of $$("#a-links li[data-l]")) {
        const ls = await ex.lens(+li.dataset.l, +li.dataset.n, 4);
        if (!same(f)) return;
        li.querySelector(".w").textContent = "favors " + ls.up.map(show).join("  ");
      }
    }
  } catch (err) { fail("Could not load this section", err); }
}
$("#a-links").addEventListener("click", e => { const li = e.target.closest("li[data-l]"); if (li) openNeuron(+li.dataset.l, +li.dataset.n, { go: "about" }); });

$("#a-next").addEventListener("click", async e => {
  if (!e.target.closest("#next-go")) return;
  const f = state.focus; if (!f) return;
  $("#a-next").innerHTML = `<p class="fine">Measuring…</p>`;
  try {
    const s = await ex.steerLens(f.layer, f.n, { dose: 0.8, k: 8 });
    if (!same(f)) return;
    const size = kl => (kl < 0.05 ? "barely" : kl < 0.3 ? "a little" : kl < 1 ? "clearly" : "strongly");
    $("#a-next").innerHTML = `<div class="fact"><h4>Boosted, these next words gain the most</h4><div class="words">${words(s.up)}</div></div>
      <div class="fact"><h4>Suppressed, these gain the most</h4><div class="words">${words(s.down)}</div></div>
      <p class="fine" style="margin-top:10px">Boosting changes the next word ${size(s.klUp)} (KL ${s.klUp.toFixed(2)}); suppressing, ${size(s.klDown)} (KL ${s.klDown.toFixed(2)}).</p>`;
  } catch (err) { $("#a-next").innerHTML = ""; fail("Could not measure", err); }
});

$("#read-form").addEventListener("submit", async e => {
  e.preventDefault();
  const f = state.focus, text = $("#read-text").value.trim();
  if (!f || !text) return;
  $("#read-text").blur();
  $("#read-out").innerHTML = `<span class="cap">Reading…</span>`;
  try {
    const res = await ex.read(f.layer, f.n, text, { chat: false });
    if (!same(f)) return;
    const max = Math.max(1e-6, ...res.acts.map(Math.abs));
    $("#read-out").innerHTML = res.units.map(u => {
      const a = u.idx.reduce((s, i) => (Math.abs(res.acts[i]) > Math.abs(s) ? res.acts[i] : s), 0);
      return `<span title="${a.toFixed(3)}" style="background:${shade(a / max)}">${esc(show(u.text))}</span>`;
    }).join("") + `<span class="cap">Orange: the neuron fires; pink: it goes negative. Strongest: ${max.toFixed(2)}.</span>`;
  } catch (err) { $("#read-out").innerHTML = ""; fail("Could not read the text", err); }
});

function shade(v) {
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(255,181,71,${a * 0.85})` : `rgba(255,79,154,${a * 0.85})`;
}

async function renderFires(f) {
  const el = $("#a-fires");
  if (!ex.examples) {
    el.innerHTML = `<p class="fine">Scanning sample texts (one time)… ${Math.round(calFrac * 100)}%</p>`;
    calibration ||= runCalibration();
    await calibration;
    if (!same(f) || !ex.examples) return;
  }
  const { layer, n } = f;
  const ctx = (ex.contexts(layer, n, 24) || []).filter(c => c.sign === "+").slice(0, 4);
  if (!ctx.length) { el.innerHTML = `<p class="fine">It barely fires on the sample texts.</p>`; return; }
  const chunks = ex.examples.chunks;
  el.innerHTML = `<p class="fine">Its strongest moments in ${CORPUS.length} short sample texts (prose, code, chat, other languages).</p>` +
    ctx.map((c, i) => `<div class="ctx" data-i="${i}">${esc(tok.decode(chunks[c.chunk].slice(c.s, c.e)))}</div>`).join("");
  for (const div of $$("#a-fires .ctx")) {  // shade each with the neuron's activation, token by token
    const c = ctx[+div.dataset.i];
    const r = await ex.readIds(layer, n, chunks[c.chunk]);
    if (!same(f)) return;
    const scale = Math.max(Math.abs(c.act), 1e-6);
    div.innerHTML = tok.units(chunks[c.chunk].slice(c.s, c.e)).map(u => {
      const a = u.idx.reduce((s, j) => { const v = r[c.s + j]; return Math.abs(v) > Math.abs(s) ? v : s; }, 0);
      return `<span style="background:${shade(a / scale)}">${esc(u.text)}</span>`;
    }).join("");
  }
}

$$("#mode button").forEach(b => (b.onclick = () => { state.mode = b.dataset.m; renderStrength(); }));
$("#keep").addEventListener("change", e => { state.keep = e.target.checked; });
function renderTech() {
  const f = state.focus;
  if (!f || !model) return;
  const unit = ex.residP50 ? ex.residP50[f.layer] / (ex.colNorm.get(f.layer * 1e6 + f.n) || NaN) : NaN;
  $("#a-tech").textContent = `${model.types[f.layer] === "full_attention" ? "gated attention" : "Gated DeltaNet"} layer · ` +
    (isFinite(unit) ? `strength 1 adds ${unit.toFixed(2)} to the activation · ` : "") + `GPU memory ${(model.gpu.bytes / 1e6).toFixed(0)} MB`;
}

// ---------- explore: find ----------

function setFindNote() {
  const busy = calibration && !ex?.scale;
  $("#find-note").textContent = busy ? `Preparing neuron ranges in the background (one time)… ${Math.round(calFrac * 100)}%` : "";
}

$("#find-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (!ex) return;
  const text = $("#find-text").value.trim();
  if (!text) { $("#find-text").focus(); return; }
  $("#find-text").blur();
  $("#find-go").disabled = true;
  $("#find-rows").innerHTML = `<li class="none">Reading your text…</li>`;
  try {
    if (calibration && !ex.scale) { $("#find-rows").innerHTML = `<li class="none">Finishing a one-time preparation first…</li>`; await calibration; }
    state.find = await ex.activations(text, { maxTokens: 128 });
    state.find.at = Date.now();
    state.findPos = -1;
    findKey = "";
    $("#find-filters").hidden = false;
    renderFindToks();
    renderFind();
  } catch (err) { $("#find-rows").innerHTML = ""; fail("Could not search", err); }
  finally { $("#find-go").disabled = false; }
});

function renderFindToks() {
  const r = state.find;
  $("#find-toks").innerHTML = `<span class="cap">Tap a word to see the neurons for just that word.</span>` +
    r.units.map((u, i) => `<span data-i="${i}" class="${state.findPos === u.idx[u.idx.length - 1] ? "sel" : ""}">${esc(show(u.text))}</span>`).join("") +
    (r.ids.length >= 128 ? `<span class="cap">Only the first 128 tokens were read.</span>` : "");
}
$("#find-toks").addEventListener("click", e => {
  const s = e.target.closest("span[data-i]"); if (!s) return;
  const u = state.find.units[+s.dataset.i], p = u.idx[u.idx.length - 1];
  state.findPos = state.findPos === p ? -1 : p;
  renderFindToks(); renderFind();
});
$$("#find-sign button").forEach(b => (b.onclick = () => { state.findSign = b.dataset.s; $$("#find-sign button").forEach(x => x.classList.toggle("on", x === b)); renderFind(); }));
["#find-lo", "#find-hi"].forEach(s => $(s).addEventListener("change", renderFind));
// taps are handled on the list, so a re-render between touch-down and click cannot swallow them
$("#find-rows").addEventListener("click", e => {
  const li = e.target.closest("li[data-l]");
  if (li) { openNeuron(+li.dataset.l, +li.dataset.n, { go: "steer" }); if (narrow()) toast("Now ask it something."); }
});

let findKey = "";
async function renderFind() {
  const r = state.find;
  if (!r) return;
  const lo = Math.max(0, +$("#find-lo").value || 0), hi = Math.min(model.L - 1, +($("#find-hi").value || model.L - 1));
  // a layer field's "change" also fires when it loses focus to a tap on a result: nothing to redo then
  const k = [r.at, state.findPos, state.findSign, lo, hi, !!ex.scale].join(",");
  if (k === findKey) return;
  findKey = k;
  const top = ex.topAt(r, state.findPos, { sign: state.findSign, lo, hi, limit: 30 });
  const tokAt = i => show(tok.units(r.ids.slice(i, i + 1))[0]?.text ?? tok.piece(r.ids[i]));
  $("#find-rows").innerHTML = top.length ? top.map(t => `<li data-l="${t.layer}" data-n="${t.n}">
      <span class="t"><b>${esc(label(t.layer, t.n))}</b><span class="dim">layer ${t.layer}</span><span class="r ${t.score > 0 ? "up" : "down"}">${t.score > 0 ? "▲" : "▼"} on “${esc(tokAt(t.pos).trim() || tokAt(t.pos))}”</span></span>
      <span class="w"></span></li>`).join("") : `<li class="none">No neuron passes these filters.</li>`;
  overlayFind(top);
  const g = findKey;
  for (const li of $$("#find-rows li[data-l]").slice(0, 12)) {  // what each one favors, filled in as it comes
    try {
      const ls = await ex.lens(+li.dataset.l, +li.dataset.n, 5);
      if (g !== findKey) return;
      li.querySelector(".w").textContent = "favors " + ls.up.map(show).join("  ");
    } catch { return; }
  }
}

function overlayFind() {
  if (!map) return;
  const r = state.find, ov = new Float32Array(model.L * model.I), T = r.T, I = model.I;
  for (let l = 0; l < model.L; l++) for (let n = 0; n < I; n++) {
    let best = 0;
    const t0 = state.findPos >= 0 ? state.findPos : 1, t1 = state.findPos >= 0 ? state.findPos + 1 : T;
    for (let t = t0; t < t1; t++) { const s = ex.scoreOf(l, n, r.acts[(l * T + t) * I + n]); if (Math.abs(s) > Math.abs(best)) best = s; }
    ov[l * I + n] = ex.scale ? best : best / 4;
  }
  map.setOverlay(ov);
  $("#map-legend").textContent = "Colours: how strongly each neuron reacts to your Find text (orange up, pink down). Tap a band, then slide across it.";
}

// ---------- explore: browse ----------

function renderSaved() {
  const saved = state.saved.map(parseKey).filter(Boolean);
  const recent = state.recent.filter(r => !state.saved.includes(key(r.layer, r.n))).slice(0, 8);
  const row = (f, star) => `<li data-l="${f.layer}" data-n="${f.n}"><span class="t"><b>${esc(label(f.layer, f.n))}</b><span class="dim">layer ${f.layer}${state.names[key(f.layer, f.n)] ? ` · neuron ${f.n}` : ""}</span><span class="r">${star ? "★" : ""}</span></span></li>`;
  $("#saved-rows").innerHTML = (saved.length ? saved.map(f => row(f, true)).join("") : `<li class="none">Tap ☆ next to a neuron's name to keep it here.</li>`) +
    (recent.length ? `<li class="none">Recently opened</li>` + recent.map(f => row(f, false)).join("") : "");
}
$("#saved-rows").addEventListener("click", e => { const li = e.target.closest("li[data-l]"); if (li) openNeuron(+li.dataset.l, +li.dataset.n, { go: "steer" }); });

$("#goto").addEventListener("submit", e => {
  e.preventDefault();
  const l = parseInt($("#go-l").value, 10), n = parseInt($("#go-n").value, 10);
  if (l >= 0 && l < model.L && n >= 0 && n < model.I) { document.activeElement?.blur(); openNeuron(l, n, { go: "steer" }); }
  else toast(`Layer is 0–${model.L - 1}, neuron 0–${model.I - 1}.`);
});

// until Find runs, the map shows each neuron's write strength (|down column|) relative to its layer
async function showWeightMap() {
  try {
    const v = await ex.writeNorms();
    if (state.find) return;
    const ov = new Float32Array(v.length), I = model.I;
    for (let l = 0; l < model.L; l++) {
      const row = Array.from(v.subarray(l * I, (l + 1) * I)).sort((a, b) => a - b);
      const lo = row[Math.floor(I * 0.5)], hi = row[Math.floor(I * 0.999)];
      for (let n = 0; n < I; n++) ov[l * I + n] = Math.max(0, (v[l * I + n] - lo) / Math.max(1e-6, hi - lo));
    }
    map.setOverlay(ov, { neutral: true });
  } catch (err) { console.error(err); }
}

let prevGen = 0;
async function previewNeuron(p) {
  const tip = $("#map-tip");
  const g = ++prevGen;
  if (!p) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.style.left = `${Math.min(p.x + 14, $("#map").clientWidth - 200)}px`;
  tip.style.top = `${Math.max(4, p.y - 64)}px`;
  tip.innerHTML = `<b>${esc(label(p.layer, p.n))}</b> · layer ${p.layer}`;
  await new Promise(r => setTimeout(r, 120));  // sliding: only look up where the finger rests
  if (g !== prevGen) return;
  try {
    const lens = await ex.lens(p.layer, p.n, 5);
    if (g !== prevGen) return;
    tip.innerHTML += `<div class="up">favors ${esc(lens.up.map(show).join(" "))}</div>`;
  } catch {}
}

// ---------- calibration (background, one time) ----------

async function runCalibration() {
  try {
    await ex.calibrate(CORPUS, {
      hold: () => !!job,  // steering goes first
      onProgress: p => {
        calFrac = p.frac;
        setFindNote();
        const fi = $("#a-fires .fine"); if (fi && !ex.examples) fi.textContent = `Scanning sample texts (one time)… ${Math.round(p.frac * 100)}%`;
      },
    });
    await saveCalibration();
  } catch (err) { console.error(err); } finally { setFindNote(); }
}

const calName = () => `duo-mech-interp:${REPO}:cal1`;
async function saveCalibration() {
  if (!ex.scale) return;
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
    if (!(await caches.has(calName()))) return;
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

boot().catch(e => { console.error(e); $("#w-err").hidden = false; $("#w-err").textContent = String(e.stack || e); });
