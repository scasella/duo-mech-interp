// node tests/scales.mjs <served base url> <hub mirror url> <out dir>
// Scale steering in the real app (real model, SwiftShader WebGPU): start screen → Scales list → Wind → ask a suggested
// question → tap another point on the scale (re-asks) → days of the week by link. Prints the three answers of each run
// and screenshots at iPhone Duo viewport sizes (outer display, inner display).
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [base, hub, out] = process.argv.slice(2);
const RUN_MS = +(process.env.RUN_MS || 60000);  // a software GPU is slow: stop each answer after this long
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const ctx = await browser.newContext({ viewport: { width: 382, height: 618 }, deviceScaleFactor: 2, hasTouch: true });
await ctx.route(/\/app\/corpus\.js(\?.*)?$/, route => route.fulfill({ contentType: "application/javascript", body:
  "export const CORPUS = [\"The river had flooded twice that spring, and by May the lower fields were still soft.\"];" }));
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", e => { errors.push(e.message); console.log("[pageerror]", e.message); });
page.on("console", m => { if (m.type() === "error") { errors.push(m.text()); console.log("[console.error]", m.text()); } });
const shot = async (name, w = 382, h = 618) => {
  await page.setViewportSize({ width: w, height: h });
  await page.evaluate(() => dispatchEvent(new Event("resize")));
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${out}/scales-${name}.png` });
  const sw = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
  console.log("shot", name, "layout", await page.evaluate(() => document.documentElement.dataset.layout), "scrollWidth", sw);
};
const answers = async (label) => {
  await page.waitForFunction(() => [...document.querySelectorAll(".lane .say")].every(s => s.textContent.length > 4 && !s.classList.contains("wait")), null, { timeout: 0, polling: 1000 });
  const t0 = Date.now();
  await page.waitForFunction(() => !document.querySelector("#send.stop"), null, { timeout: RUN_MS, polling: 1000 }).catch(() => {});
  if (await page.$("#send.stop")) await page.tap("#send");
  await page.waitForFunction(() => !document.querySelector("#send.stop"), null, { timeout: 0 });
  const r = await page.evaluate(() => [...document.querySelectorAll(".lane")].map(l => [l.querySelector("h3").textContent, l.querySelector(".say").textContent]));
  console.log(`\n${label} (${((Date.now() - t0) / 1000).toFixed(0)} s after first words)`);
  for (const [h, t] of r) console.log(`  ${h}: ${JSON.stringify(t)}`);
  return r;
};

await page.goto(`${base}/index.html?hub=${encodeURIComponent(hub)}&nocache&autoload`);
await page.waitForFunction(() => document.getElementById("welcome").hidden, null, { timeout: 0, polling: 1000 });
await page.waitForSelector("#go-scales", { state: "visible" });
await shot("start-outer");
await page.tap("#go-scales");
await page.waitForFunction(() => document.querySelectorAll("#scale-rows li[data-s]").length > 0, null, { timeout: 60000 });
console.log("scales listed:", await page.evaluate(() => [...document.querySelectorAll("#scale-rows li b")].map(b => b.textContent).join(", ")));
await shot("list-outer");
await page.tap('#scale-rows li[data-s="wind"]');
await page.waitForSelector("#scale-vals button.on", { state: "visible" });
console.log("header:", await page.evaluate(() => `${document.getElementById("who-name").textContent} | ${document.getElementById("who-sub").textContent}`),
  "| point:", await page.evaluate(() => document.querySelector("#scale-vals button.on").textContent));
await shot("wind-empty-outer");
await page.tap("#suggest button");
await answers("Wind, first suggested question");
await shot("wind-answers-outer");
await shot("wind-answers-inner", 951, 620);
await page.setViewportSize({ width: 382, height: 618 });
await page.evaluate(() => dispatchEvent(new Event("resize")));
const last = await page.evaluate(() => document.querySelectorAll("#scale-vals button").length - 1);
await page.tap(`#scale-vals button[data-i="${last}"]`);
await answers("Wind, last point (re-asked by tapping it)");
await page.evaluate(() => { location.hash = "#scale-weekday"; });  // a link to a scale while the app is open
await page.waitForFunction(() => document.getElementById("who-name").textContent === "Days of the week" && document.querySelector("#suggest button"), null, { timeout: 60000 });
console.log("by link:", await page.evaluate(() => `${document.getElementById("who-name").textContent} | point ${document.querySelector("#scale-vals button.on").textContent}`));
await page.tap("#suggest button");
await answers("Days of the week by link, first suggested question");
await shot("weekday-answers-inner", 951, 620);
console.log("\nerrors:", errors.length ? errors : "none");
await browser.close();
