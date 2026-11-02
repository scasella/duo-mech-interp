// Screen model for iPhone Duo (and any foldable or plain screen).
//
// iPhone Duo in Safari 27.1 (WebKit blog, "WebKit Features for Safari 27.1"):
//   outer display  382 x 580-618 CSS px portrait, 594 x 368-406 landscape
//   inner display  669 x 867-902 portrait (hinge horizontal), 951 x 585-620 landscape (hinge vertical, book)
//   split view     385 x 571-609 per window
//   Device Posture API (navigator.devicePosture.type: "folded" | "continuous", "change" event) and Viewport Segments
//   API (window.viewport.segments: DOMRect[], env(viewport-segment-* x y)) ship behind Safari feature flags.
//
// Modes, written to <html data-layout>:
//   narrow  one pane at a time (outer display, split view, ordinary phones)
//   wide    two panes side by side, split at the middle (inner display, landscape: the fold runs down the middle)
//   book    same, split exactly at the reported fold (segments API, phone folded like a book)
//   tall    two panes stacked, split at the middle (inner display, portrait)
//   laptop  same, split exactly at the reported fold (segments API, phone folded like a laptop)
// CSS variables --split-a / --split-b give the end of the first pane and the start of the second (px along the split
// axis); the gap between them is the fold, kept clear of content.

// ?duo=book|laptop (or window.__duoSim) simulates two viewport segments on a desktop browser
const SIM0 = new URLSearchParams(location.search).get("duo");
const sim = () => window.__duoSim ?? SIM0;

function segments() {
  const s = window.viewport?.segments || window.visualViewport?.segments;
  if (s && s.length === 2) return [...s];
  const SIM = sim();
  if (SIM === "book") { const w = innerWidth, h = innerHeight; return [new DOMRect(0, 0, w / 2 - 6, h), new DOMRect(w / 2 + 6, 0, w / 2 - 6, h)]; }
  if (SIM === "laptop") { const w = innerWidth, h = innerHeight; return [new DOMRect(0, 0, w, h / 2 - 6), new DOMRect(0, h / 2 + 6, w, h / 2 - 6)]; }
  return null;
}

export function posture() {
  if (sim()) return "folded";
  return navigator.devicePosture?.type || null;
}

// The on-screen keyboard shrinks the viewport's height (and Safari fires resize events) while the screen stays the
// same. Layout decisions use the tallest height seen at the current width, so opening the keyboard never changes the
// layout (a layout change can hide or move the field being typed in, which dismisses the keyboard).
let stableW = 0, stableH = 0;
function stableHeight(w, h) {
  if (w !== stableW) { stableW = w; stableH = h; }
  else stableH = Math.max(stableH, h);
  return stableH;
}

export function measure() {
  const w = innerWidth, h = stableHeight(innerWidth, innerHeight);
  const segs = segments();
  let mode = "narrow", axis = null, a = 0, b = 0;
  if (segs) {
    const [s0, s1] = segs.sort((p, q) => p.left - q.left || p.top - q.top);
    if (Math.abs(s0.top - s1.top) < 2) { mode = "book"; axis = "x"; a = s0.right; b = s1.left; }
    else { mode = "laptop"; axis = "y"; a = s0.bottom; b = s1.top; }
  } else if (w >= 840 && w > h * 1.2) {
    mode = "wide"; axis = "x"; a = b = Math.round(w / 2);
  } else if (w >= 600 && h >= 760 && h > w * 1.15) {
    mode = "tall"; axis = "y"; a = b = Math.round(h / 2);
    // On the inner display Safari's tab bar is at the top, so the page ends at the screen's bottom edge and the fold
    // (the middle of the screen) sits screenLong/2 above it. Use that when the screen size is consistent.
    const long = screen.width === w ? screen.height : screen.height === w ? screen.width : 0;
    if (long > h && long - h < 200) a = b = Math.round(h - long / 2);
  }
  return { mode, axis, a, b, w, h, posture: posture(), segmented: !!segs };
}

export function watch(cb) {
  let last = "";
  const apply = () => {
    const m = measure();
    const key = JSON.stringify(m);
    if (key === last) return;
    last = key;
    const root = document.documentElement;
    root.dataset.layout = m.mode;
    root.dataset.posture = m.posture || "unknown";
    // keep at least a small gutter even when the fold has no width (flat), so nothing sits on the crease
    const gap = Math.max(m.b - m.a, m.mode === "narrow" ? 0 : 12);
    const mid = (m.a + m.b) / 2;
    root.style.setProperty("--split-a", `${Math.round(mid - gap / 2)}px`);
    root.style.setProperty("--split-b", `${Math.round(mid + gap / 2)}px`);
    root.style.setProperty("--fold", `${gap}px`);
    cb(m);
  };
  addEventListener("resize", apply);  // not visualViewport: that resizes for the keyboard and for pinch zoom
  navigator.devicePosture?.addEventListener?.("change", apply);
  screen.orientation?.addEventListener?.("change", () => setTimeout(apply, 50));
  apply();
  return apply;
}
