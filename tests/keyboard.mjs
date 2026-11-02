// node tests/keyboard.mjs <served base url>
// iOS dismisses the keyboard when the focused field is moved in the DOM or hidden. Simulates the keyboard opening
// (viewport height shrinking, resize events) while each text field has focus, at every iPhone Duo size, and checks the
// field keeps focus; then checks that a real screen change (rotate / unfold) still switches the layout.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const pw = require(process.env.PLAYWRIGHT || "playwright");
const chromium = pw[process.env.BROWSER || "chromium"];  // BROWSER=webkit runs the same checks in Safari's engine
const base = process.argv[2];
const sizes = [["outer", 382, 618, 300], ["split", 385, 609, 290], ["inner-landscape", 951, 620, 250], ["inner-portrait", 669, 902, 480]];
const fields = [["#c-input", null], ["#jump", null], ["#run-text", "run"], ["#f-name", null]];
const browser = await chromium.launch();
let bad = 0;
for (const [name, w, h, kh] of sizes) {
  const p = await browser.newPage({ viewport: { width: w, height: h } });
  p.on("pageerror", e => { console.log("[pageerror]", e.message); bad++; });
  await p.goto(`${base}/index.html`);
  await p.waitForTimeout(500);
  await p.evaluate(() => { document.getElementById("boot").hidden = true; });
  for (const [sel, tab] of fields) {
    await p.setViewportSize({ width: w, height: h });
    await p.evaluate(([sel, tab]) => {  // show the field the way tapping through the UI would
      if (tab) document.querySelector(`#btabs button[data-bt="${tab}"]`).click();
      const pane = document.querySelector(sel).closest(".pane");
      document.querySelector(`#panes button[data-p="${pane.id}"]`).click();
    }, [sel, tab]);
    await p.waitForTimeout(100);
    await p.focus(sel);
    for (const hh of [h - 40, kh, kh - 10]) {  // the keyboard animating in
      await p.setViewportSize({ width: w, height: hh });
      await p.evaluate(() => { dispatchEvent(new Event("resize")); window.visualViewport?.dispatchEvent(new Event("resize")); });
      await p.waitForTimeout(60);
    }
    const ok = await p.evaluate(sel => document.activeElement === document.querySelector(sel), sel);
    console.log(`${name.padEnd(16)} ${sel.padEnd(10)} ${ok ? "keeps focus" : "LOSES FOCUS"} (layout: ${await p.evaluate(() => document.documentElement.dataset.layout)})`);
    if (!ok) bad++;
    await p.evaluate(() => document.activeElement?.blur());
  }
  await p.close();
}
// rotating / unfolding (a width change) must still switch layouts
const p = await browser.newPage({ viewport: { width: 382, height: 618 } });
await p.goto(`${base}/index.html`);
await p.waitForTimeout(300);
for (const [w, h, want] of [[382, 618, "narrow"], [951, 620, "wide"], [669, 902, "tall"], [382, 618, "narrow"]]) {
  await p.setViewportSize({ width: w, height: h });
  await p.waitForTimeout(150);
  const got = await p.evaluate(() => document.documentElement.dataset.layout);
  console.log(`screen ${w}x${h}: layout ${got}${got === want ? "" : ` (expected ${want})`}`);
  if (got !== want) bad++;
}
await browser.close();
console.log(bad ? `${bad} failures` : "ok");
process.exit(bad ? 1 : 0);
