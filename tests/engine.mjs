// node tests/engine.mjs <served base url> <expected.json url>
// Runs tests/engine.html in headless Chromium (SwiftShader WebGPU) and compares with tests/engine_ref.py output.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const [base, expected, hub, steps = "24"] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", m => console.log("[page]", m.text()));
page.on("pageerror", e => console.log("[pageerror]", e.message));
await page.goto(`${base}/tests/engine.html?hub=${encodeURIComponent(hub)}&expected=${encodeURIComponent(expected)}&steps=${steps}`);
await page.waitForFunction(() => window.__done, null, { timeout: 0, polling: 2000 });
const err = await page.evaluate(() => window.__error);
if (err) { console.log(err); process.exit(1); }
const got = await page.evaluate(() => window.__result);
const exp = await (await fetch(expected)).json();
const rel = (a, b) => { let num = 0, den = 0; a.forEach((x, i) => { num += (x - b[i]) ** 2; den += b[i] ** 2; }); return Math.sqrt(num / den); };
const corr = (a, b) => { const n = a.length, ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n;
  let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };
console.log("top-1 token:", got.logits_top[0], "vs", exp.logits_top[0]);
console.log("top-20 overlap:", got.logits_top.filter(x => exp.logits_top.includes(x)).length, "/ 20");
console.log("logit sample rel err:", rel(got.logits_sample, exp.logits_sample).toFixed(4), "corr", corr(got.logits_sample, exp.logits_sample).toFixed(5));
for (const l of Object.keys(exp.acts)) {
  const g = got.acts[l], e = exp.acts[l];
  console.log(`L${l} acts: last-token rel err ${rel(g.last, e.last).toFixed(4)}, n100 rel err ${rel(g.n100, e.n100).toFixed(4)}, absmax ${g.absmax.toFixed(2)} vs ${e.absmax.toFixed(2)}`);
}
console.log("vector row with no vector matches the plain greedy run:", got.greedy_vec_none_matches);
if (exp.vec_logits_top) {
  console.log(`with the residual vector (+, after L12): top-1 ${got.vec_logits_top[0]} vs ${exp.vec_logits_top[0]}, top-20 overlap ` +
    `${got.vec_logits_top.filter(x => exp.vec_logits_top.includes(x)).length} / 20, logit rel err ${rel(got.vec_logits_sample, exp.vec_logits_sample).toFixed(4)}, ` +
    `reference top-2 margin ${(exp.vec_logits_top_vals[0] - exp.vec_logits_top_vals[1]).toFixed(3)}`);
}
for (const k of ["greedy", "greedy_push_up", "greedy_push_down", "greedy_vec_up", "greedy_vec_down"]) {
  if (!exp[k]) { console.log(`${k}: no reference (rerun tests/engine_ref.py)`); continue; }
  let same = 0; while (same < Math.min(exp[k].length, got[k].length) && got[k][same] === exp[k][same]) same++;
  console.log(`${k}: first ${same} of ${got[k].length} decoded tokens match the reference`);
}
await browser.close();
