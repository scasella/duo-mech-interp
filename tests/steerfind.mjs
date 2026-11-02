// node tests/smoke.mjs <served base url> <hub mirror url>: runs tests/steerfind.html in headless Chromium (WebGPU).
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [base, hub] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", m => console.log("[page]", m.text()));
page.on("pageerror", e => console.log("[pageerror]", e.message));
await page.goto(`${base}/tests/steerfind.html?hub=${encodeURIComponent(hub)}${process.argv[4] ? "&" + process.argv[4] : ""}`);
await page.waitForFunction(() => window.__done, null, { timeout: 0, polling: 2000 });
const ok = await page.evaluate(() => window.__ok);
await browser.close();
process.exit(ok ? 0 : 1);
