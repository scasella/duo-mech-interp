// node tests/e2e.mjs <site url> <out dir> [hub mirror url]
//
// End-user test of the deployed app in headless Chromium (WebGPU on SwiftShader), driven by taps and typing at
// iPhone Duo viewport sizes, following the app's main flows: first visit (welcome, download), try an example, steer
// with the strength slider, ask your own question, learn about the neuron, find neurons from text, browse, then a
// second visit (opens from this device), the other layouts, and removing the model. A persistent browser profile
// carries Cache Storage and localStorage between visits. Every step prints PASS / FAIL; failures leave a screenshot.
// SMALL_CORPUS=1 serves a two-passage calibration corpus (the real one takes seconds on a phone, hours here).
import { createRequire } from "node:module";
import { mkdtempSync } from "node:fs";
import { join } from "node:path";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [site, out, hub] = process.argv.slice(2);
// probe / previews: how many neurons the topic search pushes and previews (a software GPU is slow; the app tests 36)
const q = `?probe=${process.env.PROBE || 3}&previews=${process.env.PREVIEWS || 2}` + (hub ? `&hub=${encodeURIComponent(hub)}` : "");
const only = (process.env.STEPS || "").split(",").filter(Boolean);
const results = [], errors = [];
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0).padStart(5) + "s";

const profile = mkdtempSync(join(process.env.PROFILE_DIR || "/tmp", "duo-e2e-"));
const ctx = await chromium.launchPersistentContext(profile, {
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
  proxy: process.env.HTTPS_PROXY && !hub ? { server: process.env.HTTPS_PROXY } : undefined,
  viewport: { width: 951, height: 620 }, deviceScaleFactor: 2, hasTouch: true,
});
if (process.env.SMALL_CORPUS) {
  await ctx.route(/\/app\/corpus\.js(\?.*)?$/, route => route.fulfill({ contentType: "application/javascript", body:
    "export const CORPUS = [\"The river had flooded twice that spring, and by May the lower fields were still soft.\", " +
    "\"Friday night! I can't wait for the weekend, we're going to the beach.\"];" }));
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
const visible = sel => page.$eval(sel, e => !!(e.offsetWidth || e.offsetHeight)).catch(() => false);
const until = (fn, arg, timeout = 0) => page.waitForFunction(fn, arg, { timeout, polling: 500 });
const tap = sel => page.tap(sel);
const type = async (sel, s) => { await page.tap(sel); await page.fill(sel, ""); await page.type(sel, s); };
const layout = () => page.evaluate(() => document.documentElement.dataset.layout);
const resize = async (w, h, sim = null) => {
  await page.evaluate(s => { window.__duoSim = s; }, sim);
  await page.setViewportSize({ width: w, height: h });
  await page.evaluate(() => dispatchEvent(new Event("resize")));
  await page.waitForTimeout(300);
};
const answers = () => page.$$eval(".lane .say", l => l.map(x => x.textContent));
const answered = (min = 8) => until(m => [...document.querySelectorAll(".lane .say")].every(s => s.textContent.length >= m && !s.classList.contains("wait")), min);
const stopRun = async () => { if (await page.$("#send.stop")) await tap("#send"); await until(() => !document.querySelector("#send.stop")); };
const xtab = async t => { if ((await layout()) === "narrow") await tap('#tabbar button[data-p="explore"]'); await tap(`#xtabs button[data-x="${t}"]`); };
const steerPane = async () => { if ((await layout()) === "narrow") await tap('#tabbar button[data-p="steer"]'); };

// ---------------- first visit, inner display (landscape) ----------------

await page.goto(site + q);

await step("welcome: one clear action, details tucked away", async () => {
  await until(() => document.querySelector("#w-facts li") || !document.getElementById("w-err").hidden, null, 30000);
  const err = await page.$eval("#w-err", e => (e.hidden ? "" : e.textContent));
  expect(!err, "error: " + err);
  expect((await text("#w-go")) === "Download the model", "button: " + await text("#w-go"));
  expect(/1\.5 GB/.test(await text("#w-note")), "size note: " + await text("#w-note"));
  expect(!(await page.$eval("#w-more", e => e.open)), "technical details open by default");
  expect(await page.$eval("#w-forget", e => e.hidden), "remove button shown before anything is stored");
  return (await page.$$eval(".steps li b", l => l.map(x => x.textContent))).join(" → ");
});

const phases = new Set();
await step("download: progress in plain words, then the app", async () => {
  await tap("#w-go");
  const poll = setInterval(async () => {
    const m = await text("#w-msg"); const ph = m.split(" ")[0];
    if (ph && !phases.has(ph)) { phases.add(ph); console.log(`${secs()}        ${m}`); }
  }, 1000);
  try { await until(() => document.getElementById("welcome").hidden || !document.getElementById("w-err").hidden); }
  finally { clearInterval(poll); }
  const err = await page.$eval("#w-err", e => (e.hidden ? "" : e.textContent));
  expect(!err, "load error: " + err);
  expect(phases.has("Downloading"), "no download progress shown: " + [...phases]);
  expect(phases.has("Saving"), "never saved to the device: " + [...phases]);
  return [...phases].join(" → ");
});

await step("first screen: choose a neuron, three ways in", async () => {
  expect(await visible("#try-example") && await visible("#go-find") && await visible("#surprise"), "start buttons missing");
  expect((await text("#who-name")) === "Neuron Explorer", "header: " + await text("#who-name"));
  expect(!(await visible("#composer")), "question box shown before a neuron is chosen");
  expect(await visible("#x-find"), "Find is not the first Explore tab");
});

await step("try an example: three answers stream, then stop", async () => {
  await tap("#try-example");
  await until(() => !document.getElementById("lanes").hidden, null, 10000);
  expect(/You asked/.test(await text("#asked")), "question not shown: " + await text("#asked"));
  expect(await page.$("#send.stop"), "send did not turn into stop");
  await answered(12);
  await stopRun();
  const a = await answers();
  expect(a.every(x => x.length >= 12), "answers: " + JSON.stringify(a));
  expect(!(await page.$(".say.live")), "a lane still shows the cursor");
  return `${await text("#who-name")} · ` + a.map(x => JSON.stringify(x.slice(0, 32))).join(" / ");
});

await step("strength slider relabels the lanes", async () => {
  await page.$eval("#strength", e => { e.value = "1.25"; e.dispatchEvent(new Event("input", { bubbles: true })); });
  const doses = await page.$$eval(".lane .dose", l => l.map(x => x.textContent));
  expect(doses[0] === "+1.25" && doses[1] === "−1.25", "lane labels " + doses);
  expect((await text("#strength-v")) === "1.25", "value " + await text("#strength-v"));
  await page.$eval("#strength", e => { e.value = "0.8"; e.dispatchEvent(new Event("input", { bubbles: true })); });
});

await step("ask your own question: replaces the answers", async () => {
  await type("#prompt", "Say one word.");
  await page.keyboard.press("Enter");
  await until(() => /Say one word/.test(document.getElementById("asked").textContent), null, 10000);
  await answered(3);
  await stopRun();
  const a = await answers();
  expect(a.every(x => x.length >= 3), "answers " + JSON.stringify(a));
  return a.map(x => JSON.stringify(x.slice(0, 24))).join(" / ");
});

await step("save the neuron: star, toast, listed in Browse", async () => {
  await tap("#save");
  expect((await text("#save")) === "★", "star not filled");
  await until(() => !document.getElementById("toast").hidden, null, 5000);
  await xtab("browse");
  const rows = await page.$$eval("#saved-rows li[data-l]", l => l.map(x => x.textContent));
  expect(rows.some(r => r.includes("★")), "not in Your neurons: " + rows.slice(0, 2));
});

await step("tap the neuron's name: what it favors, name it", async () => {
  await tap("#who");
  await until(() => document.querySelectorAll("#a-up span").length >= 5, null, 0);
  expect((await text("#a-title")) === (await text("#who-name")), "title mismatch");
  await type("#a-name", "my first neuron");
  expect((await text("#who-name")) === "my first neuron", "header did not follow the name: " + await text("#who-name"));
  return (await page.$$eval("#a-up span", l => l.map(x => x.textContent))).join(" ");
});

await step("details: test it on your text", async () => {
  await tap("#d-read summary");
  await type("#read-text", "The cat sat on the mat 🇺🇸");
  await page.keyboard.press("Enter");
  await until(() => document.querySelectorAll("#read-out span[title]").length >= 6, null, 0);
  const chips = await page.$$eval("#read-out span[title]", l => l.map(x => x.textContent));
  expect(chips.includes(" 🇺🇸"), "flag split: " + JSON.stringify(chips.slice(-3)));
  return `${chips.length} tokens shaded`;
});

await step("details: effect on the next word", async () => {
  await tap("#d-next summary");
  await tap("#next-go");
  await until(() => /Boosting changes the next word/.test(document.getElementById("a-next").textContent), null, 0);
  return (await text("#a-next .fine")).slice(0, 80);
});

await step("details: where it fires (one-time preparation, then examples)", async () => {
  await tap("#d-fires summary");
  await until(() => document.querySelector("#a-fires .ctx span[style]") || /barely fires/.test(document.getElementById("a-fires").textContent), null, 0);
  return (await page.$$("#a-fires .ctx")).length + " examples";
});

let linked = "";
await step("details: connected neurons, tap one to open it", async () => {
  await tap("#d-links summary");
  await until(() => document.querySelector("#a-links li[data-l] .w")?.textContent, null, 0);
  linked = await page.$eval("#a-links li[data-l]", li => `Neuron ${li.dataset.n}`);
  await tap("#a-links li[data-l]");
  await until(t => document.getElementById("a-title").textContent === t, linked, 10000);
  expect((await text("#who-name")) === linked, "header " + await text("#who-name"));
  return linked;
});

await step("new neuron: offers the last question again", async () => {
  await steerPane();
  await until(() => document.getElementById("ask-again"), null, 10000);
  expect(/Say one word/.test(await text("#ask-again")), "button: " + await text("#ask-again"));
  await tap("#ask-again");
  await answered(2);
  await stopRun();
});

await step("find › steer toward a topic: tested by pushing, story previews, open one", async () => {
  await xtab("find");
  expect(await visible("#f-goal"), "Steer toward is not the default Find mode");
  await type("#goal-text", "the ocean");
  await page.keyboard.press("Enter");
  await until(() => document.querySelectorAll("#goal-words span:not(.lab)").length >= 3, null, 0);
  const looking = await page.$$eval("#goal-words span:not(.lab)", l => l.map(x => x.textContent));
  expect(looking.some(w => /sea|ocean/i.test(w)), "topic words: " + looking);
  await until(() => /Tested \d+ neurons/.test(document.getElementById("goal-note").textContent) || /No neuron/.test(document.getElementById("goal-rows").textContent), null, 0);
  const rows = await page.$$eval("#goal-rows li[data-i]", l => l.map(x => x.innerText.replace(/\s+/g, " ")));
  expect(rows.length >= 1, "no results: " + await text("#goal-rows"));
  expect(await page.$("#goal-rows .pv:not(.wait)"), "no story preview");
  await page.screenshot({ path: `${out}/steer-toward.png` });
  const rec = await page.$eval("#goal-rows li[data-i] .r", e => e.textContent.split("·")[1].trim());
  await tap("#goal-rows li[data-i]");
  await until(() => /Steering toward “the ocean”/.test(document.getElementById("asked").textContent), null, 10000);
  expect((await text("#strength-v")) === rec, `strength ${await text("#strength-v")} vs recommended ${rec}`);
  await answered(8);
  await stopRun();
  return rows[0].slice(0, 150) + " | steered: " + JSON.stringify((await answers())[0].slice(0, 50));
});

await step("find › reacts to: neurons for your words, per word, filter, open one", async () => {
  await xtab("find");
  await tap('#find-mode button[data-f="text"]');
  await type("#find-text", "Paris is the capital of France.");
  await tap("#find-go");
  await until(() => document.querySelectorAll("#find-rows li[data-l]").length >= 10, null, 0);
  await until(() => document.querySelector("#find-rows li[data-l] .w")?.textContent, null, 0);
  const toks = await page.$$eval("#find-toks span[data-i]", l => l.map(x => x.textContent));
  await page.tap("#find-toks span[data-i]:nth-of-type(2)");
  await until(() => document.querySelector("#find-toks span.sel"), null, 5000);
  const on = await page.$$eval("#find-rows li .r", l => [...new Set(l.map(x => x.textContent.replace(/^[▲▼] on /, "")))]);
  expect(on.length === 1, "rows for several words: " + on.slice(0, 3));
  await tap("#find-filters summary");
  await page.tap('#find-sign button[data-s="+"]');
  expect(await page.$$eval("#find-rows li .r", l => l.every(x => x.textContent.startsWith("▲"))), "up filter shows down rows");
  const first = await page.$eval("#find-rows li[data-l]", li => [li.dataset.l, li.dataset.n]);
  await tap("#find-rows li[data-l]");
  await until(n => document.getElementById("who-sub").textContent.includes(`layer ${n[0]}`) || document.getElementById("who-name").textContent === `Neuron ${n[1]}`, first, 10000);
  return `${toks.length} tokens, opened layer ${first[0]} neuron ${first[1]}`;
});

await step("browse: map sheet, slide to preview, lift to open; open by number", async () => {
  await xtab("browse");
  await page.$eval("#map", e => e.scrollIntoView({ block: "center" }));
  await page.waitForTimeout(200);
  const box = await page.$eval("#map", e => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.mouse.click(box.x + box.w * 0.5, box.y + box.h * 0.5);
  await until(() => /Slide across the sheet/.test(document.getElementById("map-legend").textContent), null, 10000);
  await page.mouse.move(box.x + box.w * 0.4, box.y + box.h * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.w * 0.55, box.y + box.h * 0.6, { steps: 4 });
  await until(() => document.querySelector("#map-tip .up"), null, 0);
  await page.mouse.up();
  await page.mouse.click(box.x + 60, box.y + 18);  // back to all layers
  await page.$eval("#goto", e => e.scrollIntoView({ block: "center" }));
  await type("#go-l", "3"); await type("#go-n", "7"); await page.keyboard.press("Enter");
  await until(() => document.getElementById("who-name").textContent === "Neuron 7", null, 10000);
});

await step("advanced: multiply mode, keep the conversation", async () => {
  await xtab("about");
  await tap("#d-adv summary");
  await tap('#mode button[data-m="mul"]');
  expect((await text("#strength-v")) === "×2", "strength " + await text("#strength-v"));
  expect((await page.$$eval(".lane .dose", l => l.map(x => x.textContent))).join() === "×2,×-2", "labels");
  await tap('#mode button[data-m="add"]');
  await page.check("#keep");
  await steerPane();
  for (const qn of ["Hi!", "And?"]) { await type("#prompt", qn); await page.keyboard.press("Enter"); await answered(2); await stopRun(); }
  const qs = await page.$$eval(".lane[data-row='1'] .say .q", l => l.map(x => x.textContent));
  expect(qs.join("|") === "Hi!|And?", "conversation " + qs);
  await xtab("about"); await page.uncheck("#keep");
  return (await text("#a-tech")).slice(0, 90);
});

// ---------------- second visit: opens from this device; other screens ----------------

await step("reopen: no download, same neuron, names and saved list kept", async () => {
  await page.goto("about:blank");
  await page.goto(site + q);
  await until(() => document.querySelector("#w-facts li"), null, 30000);
  expect((await text("#w-go")) === "Open", "button " + await text("#w-go"));
  await until(() => document.getElementById("welcome").hidden, null, 0);
  expect((await text("#who-name")) === "Neuron 7", "neuron " + await text("#who-name"));
  await xtab("browse");
  const rows = await page.$$eval("#saved-rows li", l => l.map(x => x.textContent).join(" | "));
  expect(rows.includes("my first neuron"), "saved list " + rows.slice(0, 80));
});

await step("outer display: two tabs, stacked answers", async () => {
  await resize(382, 618);
  expect((await layout()) === "narrow", "layout " + await layout());
  expect(await visible("#tabbar"), "no tab bar");
  await tap('#tabbar button[data-p="steer"]');
  expect(await visible("#steer") && !(await visible("#explore")), "panes");
  await tap('#tabbar button[data-p="explore"]');
  expect(await visible("#explore") && !(await visible("#steer")), "panes 2");
});

await step("inner display portrait and laptop pose: answers on top, question on the deck", async () => {
  await resize(669, 902);
  expect((await layout()) === "tall", "layout " + await layout());
  expect((await page.$eval("#composer", e => e.parentElement.id)) === "explore", "question box not on the deck");
  await resize(669, 902, "laptop");
  expect((await layout()) === "laptop", "layout " + await layout());
  const top = await page.$eval("#steer", e => e.getBoundingClientRect().top), deck = await page.$eval("#explore", e => e.getBoundingClientRect().top);
  expect(top < deck, "steer not above the deck");
  await resize(951, 620, "book");
  expect((await layout()) === "book", "layout " + await layout());
  const l = await page.$eval("#explore", e => e.getBoundingClientRect().left), r = await page.$eval("#steer", e => e.getBoundingClientRect().left);
  expect(l < r, "explore not on the left page");
  await resize(951, 620);
});

await step("remove the downloaded model", async () => {
  const p2 = await ctx.newPage(); watch(p2);
  await page.close(); page = p2;
  await page.goto(site + q);
  await until(() => !document.getElementById("w-forget").hidden, null, 30000);
  await tap("#w-more summary");
  await Promise.all([page.waitForNavigation({ timeout: 60000 }), tap("#w-forget")]);
  await until(() => document.querySelector("#w-facts li"), null, 30000);
  expect((await text("#w-go")) === "Download the model", "button " + await text("#w-go"));
  expect(!(await page.evaluate(async () => (await caches.keys()).length)), "caches left: " + await page.evaluate(async () => (await caches.keys()).join(",")));
});

await step("no page errors", async () => { expect(!errors.length, errors.slice(0, 3).join(" | ")); });

await ctx.close();
const failed = results.filter(r => !r[1]);
console.log(`\n${results.length - failed.length}/${results.length} passed${failed.length ? "; failed: " + failed.map(f => f[0]).join("; ") : ""}`);
process.exit(failed.length ? 1 : 0);
