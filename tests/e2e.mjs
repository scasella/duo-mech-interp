// node tests/e2e.mjs <site url> <out dir> [hub mirror url]
//
// End-user test of the deployed app in headless Chromium (WebGPU on SwiftShader), driven by taps and typing at
// iPhone Duo viewport sizes. A persistent browser profile carries Cache Storage and localStorage from the first visit
// (download from Hugging Face) to the second (load from the browser's copy). Every step prints PASS / FAIL; failing
// steps leave a screenshot in <out dir>. Without a hub mirror the weights stream from huggingface.co, as on a phone.
import { createRequire } from "node:module";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [site, out, hub] = process.argv.slice(2);
const q = hub ? `?hub=${encodeURIComponent(hub)}` : "";
const only = (process.env.STEPS || "").split(",").filter(Boolean);
const results = [];
const errors = [];
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + "s";

const profile = mkdtempSync(join(process.env.PROFILE_DIR || "/tmp", "duo-e2e-"));
const ctx = await chromium.launchPersistentContext(profile, {
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
  proxy: process.env.HTTPS_PROXY && !hub ? { server: process.env.HTTPS_PROXY } : undefined,
  viewport: { width: 951, height: 620 }, deviceScaleFactor: 2, hasTouch: true,
});
// SMALL_CORPUS=1 serves a two-passage calibration corpus (the real one is ~30 chunks: seconds on a phone, hours on a
// software GPU); the app code and its calibration path are unchanged.
if (process.env.SMALL_CORPUS) {
  await ctx.route(/\/app\/corpus\.js(\?.*)?$/, route => route.fulfill({ contentType: "application/javascript", body:
    "export const CORPUS = [\"The river had flooded twice that spring, and by May the lower fields were still soft.\", " +
    "\"def add(a, b):\\n    return a + b\\n\"];" }));
}
let page = ctx.pages()[0] || await ctx.newPage();
const watch = p => {
  p.on("pageerror", e => { errors.push(e.message); console.log(`${secs()}  [pageerror] ${e.message}`); });
  p.on("console", m => { if (m.type() === "error") { errors.push(m.text()); console.log(`${secs()}  [console.error] ${m.text()}`); } });
};
watch(page);

async function step(name, fn) {
  if (only.length && !only.some(s => name.startsWith(s))) return;
  const s0 = Date.now();
  try {
    const note = await fn();
    results.push([name, true]);
    console.log(`${secs()}  PASS  ${name}${note ? "  — " + note : ""}  (${((Date.now() - s0) / 1000).toFixed(1)}s)`);
  } catch (e) {
    results.push([name, false]);
    console.log(`${secs()}  FAIL  ${name}  — ${String(e.message || e).split("\n")[0]}`);
    await page.screenshot({ path: `${out}/fail-${results.length}.png` }).catch(() => {});
  }
}
const expect = (cond, msg) => { if (!cond) throw new Error(msg); };
const text = sel => page.$eval(sel, e => e.textContent.trim()).catch(() => "");
const until = (fn, arg, timeout = 0) => page.waitForFunction(fn, arg, { timeout, polling: 500 });
const tap = sel => page.tap(sel);
const type = async (sel, s) => { await page.tap(sel); await page.fill(sel, ""); await page.type(sel, s); };
const layout = () => page.evaluate(() => document.documentElement.dataset.layout);
const resize = async (w, h) => { await page.setViewportSize({ width: w, height: h }); await page.waitForTimeout(300); };

// ---------------- visit 1: first load on the inner display (landscape) ----------------

await page.goto(site + q);

await step("boot screen", async () => {
  await until(() => document.querySelector("#boot-facts li") || !document.getElementById("boot-err").hidden, null, 30000);
  const err = await page.$eval("#boot-err", e => (e.hidden ? "" : e.textContent));
  expect(!err, "boot error: " + err);
  const facts = await page.$$eval("#boot-facts li", l => l.map(x => x.textContent));
  expect(facts.some(f => f.startsWith("Layout: wide")), "layout fact missing: " + facts.join(" | "));
  expect((await text("#boot-go")) === "Load model", "button says " + await text("#boot-go"));
  expect(await page.$eval("#boot-clear", e => e.hidden), "forget button shown before anything is stored");
  return facts[0];
});

const phases = new Set();
await step("load model (stream from the hub, quantize, store)", async () => {
  await tap("#boot-go");
  const poll = setInterval(async () => {
    const m = await text("#boot-msg").catch(() => "");
    const ph = m.split(" ")[0];
    if (ph && !phases.has(ph)) { phases.add(ph); console.log(`${secs()}        loader: ${m}`); }
  }, 1000);
  try { await until(() => document.getElementById("boot").hidden || !document.getElementById("boot-err").hidden); }
  finally { clearInterval(poll); }
  const err = await page.$eval("#boot-err", e => (e.hidden ? "" : e.textContent));
  expect(!err, "load error: " + err);
  expect(phases.has("streaming"), "never showed download progress");
  expect(phases.has("storing"), "never stored the weights");
  const st = await text("#status");
  expect(/ready/.test(st) && /MB/.test(st), "status: " + st);
  return st;
});

await step("map shows write strength before any run", async () => {
  await until(() => /brightness/.test(document.getElementById("map-legend").textContent), null, 600000);
  expect(/open one from the map/.test(await text("#f-meta")), "neuron pane hint missing");
});

await step("map: tap a layer opens its sheet, drag previews, release opens the neuron", async () => {
  const box = await page.$eval("#map", e => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.mouse.click(box.x + box.w * 0.5, box.y + box.h * 0.5);  // middle of the stack: around layer 11-12
  await until(() => /drag across the sheet/.test(document.getElementById("map-legend").textContent), null, 10000);
  await page.mouse.move(box.x + box.w * 0.4, box.y + box.h * 0.45);
  await page.mouse.down();
  await page.mouse.move(box.x + box.w * 0.5, box.y + box.h * 0.5, { steps: 4 });
  await until(() => !document.getElementById("map-tip").hidden, null, 10000);
  await until(() => document.querySelector("#map-tip .up"), null, 600000);  // its logit lens arrives in the tooltip
  const tip = await text("#map-tip");
  await page.mouse.up();
  await until(() => document.getElementById("f-id").textContent.startsWith("L"), null, 10000);
  const id = await text("#f-id");
  expect(tip.startsWith(id.replace(" ", " ")), `tooltip "${tip.slice(0, 12)}" vs opened ${id}`);
  await page.mouse.click(box.x + 60, box.y + 18);  // "‹ layer N" title: back to the stack
  await until(() => !/drag across the sheet/.test(document.getElementById("map-legend").textContent), null, 10000);
  return `opened ${id}`;
});

await step("jump box opens a neuron", async () => {
  await type("#jump", "L12 N1234");
  await page.keyboard.press("Enter");
  await until(() => document.getElementById("f-id").textContent === "L12 N1234", null, 10000);
  expect((await page.evaluate(() => location.hash)) === "#L12N1234", "hash not updated");
  await until(() => document.querySelectorAll("#c-lens .chip").length >= 20, null, 600000);
  await until(() => /dose 1 =/.test(document.getElementById("f-meta").textContent), null, 600000);
  return await text("#f-meta");
});

await step("jump box ignores bad input", async () => {
  for (const s of ["L99 N5", "L0 N3584", "hello", ""]) { await type("#jump", s); await page.keyboard.press("Enter"); }
  await page.waitForTimeout(500);
  expect((await text("#f-id")) === "L12 N1234", "focus changed to " + await text("#f-id"));
});

await step("name a neuron; it shows in the list and survives", async () => {
  await type("#f-name", "self-identity?");
  const names = await page.evaluate(() => JSON.parse(localStorage.getItem("duo.names") || "{}"));
  expect(names["L12/N1234"] === "self-identity?", "not saved: " + JSON.stringify(names));
  await tap('#btabs button[data-bt="list"]');
  const rows = await page.$$eval("#list-rows li", l => l.map(x => x.textContent));
  expect(rows.some(r => r.includes("L12 N1234") && r.includes("★") && r.includes("self-identity?")), "list: " + rows.slice(0, 3).join(" | "));
  await tap('#btabs button[data-bt="map"]');
});

await step("lens tab: logit lens and connections; a connection opens; back / forward", async () => {
  await tap('#ntabs button[data-nt="lens"]');
  await until(() => document.querySelectorAll("#ll .chip").length >= 32, null, 600000);
  await until(() => document.querySelectorAll("#cn li").length >= 20, null, 600000);
  await until(() => document.querySelector("#cn li .w b"), null, 600000);
  const first = await page.$eval("#cn li", li => `L${li.dataset.l} N${li.dataset.n}`);
  await tap("#cn li");
  await until(f => document.getElementById("f-id").textContent === f, first, 10000);
  await until(() => document.querySelectorAll("#ll .chip").length >= 32, null, 600000);  // lens re-renders for it
  await tap("#f-prev");
  await until(() => document.getElementById("f-id").textContent === "L12 N1234", null, 10000);
  await tap("#f-next");
  await until(f => document.getElementById("f-id").textContent === f, first, 10000);
  await tap("#f-prev");
  await until(() => document.getElementById("f-id").textContent === "L12 N1234", null, 10000);
  expect((await page.$eval("#f-name", e => e.value)) === "self-identity?", "name not shown when coming back");
  return `first connection ${first}`;
});

await step("steer lens", async () => {
  await until(() => document.getElementById("sl-go"), null, 600000);
  await tap("#sl-go");
  await until(() => /mean KL/.test(document.getElementById("sl").textContent), null, 0);
  const kl = (await text("#sl .note"));
  expect(/up \d/.test(kl) && /down \d/.test(kl), kl);
  return kl;
});

await step("dose stepper and mode", async () => {
  await tap('#ntabs button[data-nt="chat"]');
  const dose = () => page.$eval("#c-dose", e => e.value);
  const heads = () => page.$$eval("#c-thread .colhead span:first-child", l => l.map(x => x.textContent).join(" | "));
  await tap("#c-plus"); expect((await dose()) === "0.85", "+ gave " + await dose());
  await tap("#c-minus"); await tap("#c-minus"); expect((await dose()) === "0.75", "− − gave " + await dose());
  await page.fill("#c-dose", "1.5"); await page.dispatchEvent("#c-dose", "change");
  expect((await heads()) === "▲ +1.5 | 0 | ▼ −1.5", "heads " + await heads());
  await tap('#c-mode button[data-m="mul"]');
  expect((await heads()) === "▲ ×2 | ×1 | ▼ ×−2", "mul heads " + await heads());
  await tap('#c-mode button[data-m="add"]');
  await page.fill("#c-dose", "0.8"); await page.dispatchEvent("#c-dose", "change");
  return await heads();
});

const answers = () => page.$$eval("#c-thread .col", cols => cols.map(c => [...c.querySelectorAll(".turn .a")].map(a => a.textContent)));
await step("steered chat: three columns stream, then stop", async () => {
  await type("#c-input", "Who are you?");
  await tap("#c-go");
  await until(() => document.getElementById("c-go").textContent === "stop", null, 10000);
  await until(() => [...document.querySelectorAll("#c-thread .col")].every(c => (c.querySelector(".turn .a")?.textContent || "").length >= 12), null, 0);
  const tps = await text("#status");
  await tap("#c-go");  // stop
  await until(() => document.getElementById("c-go").textContent === "run", null, 0);
  const a = await answers();
  expect(a.every(col => col.length === 1 && col[0].length > 0), "answers " + JSON.stringify(a));
  expect(!(await page.$(".turn .a.live")), "a column still shows the typing cursor");
  expect(/^I am/.test(a[1][0]), "unsteered column: " + JSON.stringify(a[1][0]));
  return `${tps} · ${a.map(c => JSON.stringify(c[0].slice(0, 40))).join(" / ")}`;
});

await step("multi-turn: a second question keeps each column's history", async () => {
  await type("#c-input", "Say one word.");
  await tap("#c-go");
  await until(() => document.querySelectorAll("#c-thread .col .turn").length === 6, null, 10000);
  await until(() => [...document.querySelectorAll("#c-thread .col")].every(c => ([...c.querySelectorAll(".turn .a")][1]?.textContent || "").length >= 3), null, 0);
  if ((await text("#c-go")) === "stop") await tap("#c-go");
  await until(() => document.getElementById("c-go").textContent === "run", null, 0);
  const a = await answers();
  expect(a.every(col => col.length === 2 && col[1].length > 0), "answers " + JSON.stringify(a));
  return a.map(c => JSON.stringify(c[1].slice(0, 30))).join(" / ");
});

await step("clear empties the columns", async () => {
  await tap("#c-clear");
  expect((await page.$$("#c-thread .turn")).length === 0, "turns left after clear");
});

await step("reads you: tokens shaded by the neuron, tap shows the value", async () => {
  await tap('#c-dir button[data-d="reads"]');
  expect((await text("#c-go")) === "read", "button: " + await text("#c-go"));
  expect(await page.$eval("#c-thread", e => e.hidden), "columns still visible");
  await type("#c-input", "The cat sat on the mat 🇺🇸");
  await tap("#c-go");
  await until(() => document.querySelectorAll("#c-reads span[data-a]").length >= 6, null, 0);
  const chips = await page.$$eval("#c-reads span[data-a]", l => l.map(x => x.textContent));
  expect(chips.includes(" 🇺🇸"), "flag split across chips: " + JSON.stringify(chips.slice(-3)));
  await tap("#c-reads span[data-a]");
  expect(/^act -?\d/.test(await text("#status")), "status " + await text("#status"));
  await tap('#c-dir button[data-d="writes"]');
  expect(!(await page.$eval("#c-thread", e => e.hidden)), "columns did not come back");
  return `${chips.length} chips`;
});

await step("fires: run text, rows, map overlay, tap token, filters, open a row", async () => {
  await tap('#btabs button[data-bt="run"]');
  await type("#run-text", "Paris is the capital of France. 2 + 2 = 4.");
  await tap("#run-go");
  await until(() => document.querySelectorAll("#run-rows li").length >= 50, null, 0);
  expect(/colour: each neuron/.test(await text("#map-legend")), "map legend: " + await text("#map-legend"));
  const toks = await page.$$eval("#run-toks span[data-i]", l => l.map(x => x.textContent));
  await page.tap("#run-toks span[data-i]:nth-child(2)");
  await until(() => document.querySelector("#run-toks span.sel"), null, 5000);
  const tokRows = await page.$$eval("#run-rows li .w b", l => [...new Set(l.map(x => x.textContent))]);
  expect(tokRows.length === 1 && tokRows[0] === toks[1], `rows for token ${JSON.stringify(toks[1])}: ${JSON.stringify(tokRows.slice(0, 3))}`);
  await page.tap('#run-sign button[data-s="+"]');
  expect((await page.$$eval("#run-rows li .n", l => l.every(x => x.classList.contains("up")))), "+ filter shows negatives");
  await page.fill("#run-lo", "10"); await page.fill("#run-hi", "12"); await page.dispatchEvent("#run-hi", "change");
  const layers = await page.$$eval("#run-rows li", l => l.map(x => +x.dataset.l));
  expect(layers.length && layers.every(l => l >= 10 && l <= 12), "layer filter: " + layers.slice(0, 5));
  const first = await page.$eval("#run-rows li", li => `L${li.dataset.l} N${li.dataset.n}`);
  await tap("#run-rows li");
  await until(f => document.getElementById("f-id").textContent === f, first, 10000);
  return `${toks.length} tokens, opened ${first}`;
});

await step("examples: calibration on the built-in corpus, contexts shaded", async () => {
  await tap('#ntabs button[data-nt="examples"]');
  await until(() => document.getElementById("cal-go"), null, 10000);
  await tap("#cal-go");
  let last = "";
  const poll = setInterval(async () => { const m = await text("#cal-msg"); if (m && m.split(" ")[0] !== last) { last = m.split(" ")[0]; console.log(`${secs()}        calibration ${m}`); } }, 15000);
  try { await until(() => document.querySelector("#nt-examples .ctx"), null, 0); } finally { clearInterval(poll); }
  await until(() => document.querySelector("#nt-examples .ctx span[style]"), null, 0);  // live shading arrived
  expect(/scores: activation as a multiple/.test(await text("#run-cal")), "fires note did not switch to calibrated scores");
  return `${(await page.$$("#nt-examples .ctx")).length} contexts`;
});

// ---------------- visit 2: reopen (weights from this browser), other screens ----------------

await step("reopen: loads from the stored copy, restores neuron, name and calibration", async () => {
  await page.goto(site + q + "#L12N1234");
  await until(() => document.getElementById("boot").hidden || !document.getElementById("boot-err").hidden, null, 0);
  const msgs = await page.$eval("#boot-msg", e => e.textContent);
  expect(!/streaming/.test(msgs), "downloaded again: " + msgs);
  expect((await text("#f-id")) === "L12 N1234", "focus " + await text("#f-id"));
  expect((await page.$eval("#f-name", e => e.value)) === "self-identity?", "name lost");
  await tap('#ntabs button[data-nt="examples"]');
  await until(() => document.querySelector("#nt-examples .ctx"), null, 60000);
});

await step("outer display: one pane at a time, column pager", async () => {
  await resize(382, 618);
  expect((await layout()) === "narrow", "layout " + await layout());
  await tap('#panes button[data-p="neuron"]');
  await tap('#ntabs button[data-nt="chat"]');
  expect(await page.$eval("#neuron", e => e.offsetWidth > 300) && await page.$eval("#browse", e => e.offsetWidth === 0), "panes");
  await page.$eval("#c-thread", e => e.scrollTo({ left: e.scrollWidth }));
  await page.waitForTimeout(400);
  const dot = await page.$$eval("#c-dots i", l => l.findIndex(x => x.classList.contains("on")));
  expect(dot === 2, "pager dot " + dot);
  await tap('#panes button[data-p="browse"]');
  expect(await page.$eval("#map", e => e.offsetWidth > 300), "map hidden");
});

await step("inner display portrait and laptop pose: neuron on top, deck below", async () => {
  await resize(669, 902);
  expect((await layout()) === "tall", "layout " + await layout());
  expect((await page.$eval("#composer", e => e.parentElement.id)) === "browse", "composer not on the deck");
  await page.evaluate(() => { window.__duoSim = "laptop"; dispatchEvent(new Event("resize")); });
  await page.waitForTimeout(300);
  expect((await layout()) === "laptop", "layout " + await layout());
  const [top, bottom] = await page.$$eval("#neuron, #browse", l => l.map(e => e.getBoundingClientRect().top));
  expect(top < bottom, "neuron not above the deck");
  await page.evaluate(() => { window.__duoSim = "book"; dispatchEvent(new Event("resize")); });
  await resize(951, 620);
  expect((await layout()) === "book", "layout " + await layout());
  await page.evaluate(() => { window.__duoSim = null; dispatchEvent(new Event("resize")); });
});

await step("GPU memory back to its resting size", async () => {
  const mb = +((await text("#status")).match(/(\d+) MB/) || [])[1];
  expect(mb > 0 && mb < 1100, "status " + await text("#status"));
  return `${mb} MB`;
});

await step("forget downloaded weights", async () => {
  const p2 = await ctx.newPage(); watch(p2);
  await page.close(); page = p2;
  await page.goto(site + q);
  await until(() => !document.getElementById("boot-clear").hidden, null, 30000);
  await tap("#boot-clear");
  await page.waitForLoadState("load");
  await until(() => document.querySelector("#boot-facts li"), null, 30000);
  expect((await text("#boot-go")) === "Load model", "button " + await text("#boot-go"));
  expect(!(await page.evaluate(async () => (await caches.keys()).length)), "caches left: " + await page.evaluate(async () => (await caches.keys()).join(",")));
});

await step("no page errors", async () => { expect(!errors.length, errors.slice(0, 3).join(" | ")); });

await ctx.close();
const failed = results.filter(r => !r[1]);
console.log(`\n${results.length - failed.length}/${results.length} passed${failed.length ? "; failed: " + failed.map(f => f[0]).join("; ") : ""}`);
process.exit(failed.length ? 1 : 0);
