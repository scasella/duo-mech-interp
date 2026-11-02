// node tests/screens.mjs <served base url> <hub mirror url> <out dir>
// Loads the real app (real model, SwiftShader WebGPU) and screenshots the main moments at iPhone Duo viewport sizes
// (Safari 27.1 figures from the WebKit blog): first screen after loading, an example run (real model output, stopped
// after a few tokens because a software GPU is slow), Explore's tabs, each layout.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [base, hub, out] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const ctx = await browser.newContext({ viewport: { width: 951, height: 620 }, deviceScaleFactor: 2, hasTouch: true });
// the full calibration corpus takes seconds on a phone but hours on a software GPU: serve a two-passage one
await ctx.route(/\/app\/corpus\.js(\?.*)?$/, route => route.fulfill({ contentType: "application/javascript", body:
  "export const CORPUS = [\"The river had flooded twice that spring, and by May the lower fields were still soft.\", " +
  "\"Friday night! I can't wait for the weekend, we're going to the beach.\"];" }));
const page = await ctx.newPage();
page.on("pageerror", e => console.log("[pageerror]", e.message));
page.on("console", m => { if (m.type() === "error") console.log("[console.error]", m.text()); });
const shot = async (name, w, h, sim = null) => {
  await page.evaluate(s => { window.__duoSim = s; }, sim);
  await page.setViewportSize({ width: w, height: h });
  await page.evaluate(() => dispatchEvent(new Event("resize")));
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${out}/v2-${name}.png` });
  console.log("shot", name, await page.evaluate(() => document.documentElement.dataset.layout));
};
await page.goto(`${base}/index.html?hub=${encodeURIComponent(hub)}&nocache&autoload`);
await page.waitForFunction(() => document.getElementById("welcome").hidden, null, { timeout: 0, polling: 1000 });
await shot("start-inner", 951, 620);
await shot("start-outer", 382, 618);
await page.setViewportSize({ width: 951, height: 620 });
await page.tap("#try-example");
await page.waitForFunction(() => [...document.querySelectorAll(".lane .say")].every(s => s.textContent.length > 8 && !s.classList.contains("wait")), null, { timeout: 0, polling: 1000 });
await page.waitForTimeout(+(process.env.RUN_MS || 30000));
if (await page.$("#send.stop")) await page.tap("#send");
await page.waitForFunction(() => !document.querySelector("#send.stop"), null, { timeout: 0 });
await shot("steer-inner", 951, 620);
await shot("steer-book", 951, 620, "book");
await shot("steer-portrait", 669, 902);
await shot("steer-laptop", 669, 902, "laptop");
await shot("steer-outer", 382, 618);
await shot("steer-split", 385, 609);
await page.tap('#tabbar button[data-p="explore"]');
await page.tap('#xtabs button[data-x="about"]');
await page.waitForFunction(() => document.querySelectorAll("#a-up span").length > 0, null, { timeout: 0 });
await shot("about-outer", 382, 618);
await page.tap("#d-links summary");
await page.waitForFunction(() => document.querySelector("#a-links .w")?.textContent, null, { timeout: 0 });
await shot("about-links-outer", 382, 618);
await page.tap('#xtabs button[data-x="browse"]');
await shot("browse-outer", 382, 618);
await page.tap('#xtabs button[data-x="find"]');
await page.fill("#find-text", "I can't wait for the weekend!");
await page.tap("#find-go");
await page.waitForFunction(() => document.querySelector("#find-rows li[data-l] .w")?.textContent, null, { timeout: 0, polling: 1000 });
await shot("find-outer", 382, 618);
await shot("find-inner", 951, 620);
await browser.close();
