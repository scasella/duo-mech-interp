// node tests/screens.mjs <served base url> <hub mirror url> <out dir>
// Loads the real app (real model, SwiftShader WebGPU) and screenshots it at iPhone Duo viewport sizes
// (Safari 27.1 figures from the WebKit blog) to check the layouts.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [base, hub, out] = process.argv.slice(2);
const sizes = [
  ["inner-landscape", 951, 620, ""],
  ["inner-landscape-book", 951, 620, "&duo=book"],
  ["inner-portrait", 669, 902, ""],
  ["inner-portrait-laptop", 669, 902, "&duo=laptop"],
  ["outer-portrait", 382, 618, ""],
  ["outer-landscape", 594, 406, ""],
  ["split-view", 385, 609, ""],
];
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const ctx = await browser.newContext({ viewport: { width: 951, height: 620 }, deviceScaleFactor: 2, hasTouch: true, isMobile: false });
const page = await ctx.newPage();
page.on("console", m => console.log("[page]", m.text()));
page.on("pageerror", e => console.log("[pageerror]", e.message));
const url = `${base}/index.html?hub=${encodeURIComponent(hub)}&nocache&autoload#L12N1234`;
await page.goto(url);
await page.waitForFunction(() => document.getElementById("boot").hidden, null, { timeout: 0, polling: 1000 });
await page.waitForFunction(() => document.querySelector("#c-lens .chip"), null, { timeout: 0, polling: 1000 });
// fill the three columns with sample text so the layout can be judged (screenshots only; the app never does this)
if (process.env.FAKE_CHAT) await page.evaluate(() => {
  const t = ["I am a curious traveller from the coast, here to help you find your way.", "I am Qwen3.5, a large language model developed by Tongyi Lab.", "I am Q3, or **Q3**, or **Q3**."];
  document.querySelectorAll("#c-thread .turns").forEach((el, i) => (el.innerHTML = `<div class="turn"><div class="q">Who are you?</div><div class="a">${t[i]}</div></div>`));
});
for (const [name, w, h, q] of sizes) {
  await page.evaluate(sim => { window.__duoSim = sim || null; }, q.replace("&duo=", ""));
  await page.setViewportSize({ width: w, height: h });
  await page.evaluate(() => dispatchEvent(new Event("resize")));
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log("shot", name, await page.evaluate(() => document.documentElement.dataset.layout));
}
// the neuron pane on a narrow screen, and the lens tab on the inner display
await page.evaluate(() => { window.__duoSim = null; });
await page.setViewportSize({ width: 382, height: 618 });
await page.evaluate(() => dispatchEvent(new Event("resize")));
await page.click('#panes button[data-p="neuron"]');
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/outer-portrait-neuron.png` });
await page.setViewportSize({ width: 951, height: 620 });
await page.evaluate(() => dispatchEvent(new Event("resize")));
await page.click('#ntabs button[data-nt="lens"]');
await page.waitForFunction(() => document.querySelector("#cn li .w b"), null, { timeout: 0, polling: 1000 });
await page.screenshot({ path: `${out}/inner-landscape-lens.png` });
await page.click('#bt-map canvas', { position: { x: 300, y: 300 } });
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/inner-landscape-sheet.png` });
console.log("extra shots done");
await browser.close();
