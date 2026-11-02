// The model as a stack of MLP sheets on a 2D canvas: one band per layer (layer 0 at the bottom), every neuron a
// column of pixels in its band, coloured by the current overlay (activation scores of the last run, or calibrated
// range). Tap a band to open that layer's sheet face-on (a 64 x 56 grid, one cell per neuron); drag across cells to
// preview, release to open. The focused neuron and its connection web are drawn on top: yellow links push the reader
// up, magenta push it down.

const UP = [255, 181, 71], DOWN = [255, 79, 154], LINK = [56, 200, 255];

export function createMap(canvas, { L, I, types, onOpen, onPreview, onLayer }) {
  const ctx = canvas.getContext("2d");
  const COLS = 64, ROWS = Math.ceil(I / COLS);
  let overlay = null;        // Float32Array [L * I] signed scores, or null
  let neutral = false;       // overlay is a weight statistic (unsigned, drawn in grey-blue), not an activation
  let focus = null;          // {layer, n}
  let web = null;            // {up: [{layer, n, sign}], down: [...]}
  let marks = new Set();     // "l:n" of named neurons
  let sheet = null;          // layer shown face-on, or null for the stack
  let hover = null;
  let dpr = 1, W = 0, H = 0;
  const pad = { l: 34, r: 8, t: 8, b: 8 };

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(devicePixelRatio || 1, 3);
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    draw();
  }

  const color = v => {
    const a = Math.min(1, Math.abs(v));
    if (neutral) return [40 + 90 * a, 60 + 110 * a, 70 + 130 * a];
    const c = v >= 0 ? UP : DOWN;
    return [c[0] * a, c[1] * a, c[2] * a];
  };

  // ---- stack view geometry ----
  const band = l => {
    const h = (H - pad.t - pad.b) / L;
    const y = pad.t + (L - 1 - l) * h;
    return { y, h };
  };
  const nx = n => pad.l + (n + 0.5) / I * (W - pad.l - pad.r);
  // ---- sheet view geometry ----
  function cellGeom() {
    const aw = W - pad.l - pad.r, ah = H - pad.t - pad.b - 20;
    const s = Math.max(2, Math.floor(Math.min(aw / COLS, ah / ROWS)));
    const ox = pad.l + Math.floor((aw - s * COLS) / 2), oy = pad.t + 20 + Math.floor((ah - s * ROWS) / 2);
    return { s, ox, oy };
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    sheet == null ? drawStack() : drawSheet();
  }

  function drawStack() {
    const iw = Math.max(1, Math.round((W - pad.l - pad.r) * dpr));
    for (let l = 0; l < L; l++) {
      const { y, h } = band(l);
      const bh = Math.max(1, Math.round((h - 2) * dpr));
      // per pixel column: the neuron with the largest |score| that falls in it
      const img = ctx.createImageData(iw, bh);
      const full = types[l] === "full_attention";
      for (let px = 0; px < iw; px++) {
        const n0 = Math.floor(px / iw * I), n1 = Math.max(n0 + 1, Math.floor((px + 1) / iw * I));
        let best = 0;
        if (overlay) for (let n = n0; n < n1; n++) { const v = overlay[l * I + n]; if (Math.abs(v) > Math.abs(best)) best = v; }
        let [r, g, b] = overlay ? color(best) : [0, 0, 0];
        const base = full ? 34 : 22;
        r = Math.max(r, base); g = Math.max(g, base + 6); b = Math.max(b, base + 10);
        for (let py = 0; py < bh; py++) {
          const o = (py * iw + px) * 4;
          img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
        }
      }
      ctx.putImageData(img, Math.round(pad.l * dpr), Math.round((y + 1) * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = l === focus?.layer ? "#ddd" : "#666";
      ctx.font = "10px ui-monospace, SF Mono, Menlo, monospace";
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      if (h >= 9 || l % 2 === 0) ctx.fillText(`L${l}`, pad.l - 6, y + h / 2);
      if (full) { ctx.fillStyle = "#3c5a66"; ctx.fillRect(pad.l - 3, y + 2, 2, h - 4); }
    }
    for (const k of marks) {
      const [l, n] = k.split(":").map(Number);
      const { y, h } = band(l);
      ctx.fillStyle = "#fff"; ctx.fillRect(nx(n) - 0.5, y + 1, 1, h - 2);
    }
    if (web && focus) {
      const fb = band(focus.layer), fx = nx(focus.n), fy = fb.y + fb.h / 2;
      ctx.lineWidth = 1;
      for (const c of [...web.up, ...web.down]) {
        const b = band(c.layer);
        const col = c.sign > 0 ? UP : DOWN;
        ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},0.55)`;
        ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(nx(c.n), b.y + b.h / 2); ctx.stroke();
        ctx.fillStyle = `rgb(${col})`; ctx.fillRect(nx(c.n) - 1.5, b.y + b.h / 2 - 1.5, 3, 3);
      }
    }
    if (focus) {
      const { y, h } = band(focus.layer);
      ctx.strokeStyle = `rgb(${LINK})`; ctx.lineWidth = 1.5;
      ctx.strokeRect(nx(focus.n) - 3, y - 1, 6, h + 2);
    }
  }

  function drawSheet() {
    const l = sheet, { s, ox, oy } = cellGeom();
    ctx.fillStyle = "#aaa"; ctx.font = "12px ui-monospace, SF Mono, Menlo, monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(`‹ layer ${l} · ${types[l] === "full_attention" ? "full attention" : "linear attention"} · ${I} neurons`, pad.l, pad.t);
    for (let n = 0; n < I; n++) {
      const x = ox + (n % COLS) * s, y = oy + Math.floor(n / COLS) * s;
      let [r, g, b] = overlay ? color(overlay[l * I + n]) : [0, 0, 0];
      r = Math.max(r, 26); g = Math.max(g, 32); b = Math.max(b, 36);
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect(x, y, s - (s > 3 ? 1 : 0), s - (s > 3 ? 1 : 0));
      if (marks.has(`${l}:${n}`)) { ctx.fillStyle = "#fff"; ctx.fillRect(x + s / 2 - 1, y + s / 2 - 1, 2, 2); }
    }
    if (web) for (const c of [...web.up, ...web.down]) if (c.layer === l) {
      const x = ox + (c.n % COLS) * s, y = oy + Math.floor(c.n / COLS) * s;
      ctx.strokeStyle = `rgb(${c.sign > 0 ? UP : DOWN})`; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, s, s);
    }
    for (const h of [focus?.layer === l ? focus.n : null, hover]) {
      if (h == null) continue;
      const x = ox + (h % COLS) * s, y = oy + Math.floor(h / COLS) * s;
      ctx.strokeStyle = h === hover ? "#fff" : `rgb(${LINK})`; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, s + 1, s + 1);
    }
  }

  // ---- input ----
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    if (sheet == null) {
      for (let l = 0; l < L; l++) {
        const b = band(l);
        if (y >= b.y && y < b.y + b.h) {
          const n = Math.max(0, Math.min(I - 1, Math.floor((x - pad.l) / (W - pad.l - pad.r) * I)));
          return { layer: l, n, x, y, inBand: x >= pad.l };
        }
      }
      return null;
    }
    if (y < pad.t + 20 && x < W * 0.6) return { back: true };
    const { s, ox, oy } = cellGeom();
    const c = Math.floor((x - ox) / s), rr = Math.floor((y - oy) / s);
    if (c < 0 || c >= COLS || rr < 0 || rr >= ROWS) return null;
    const n = rr * COLS + c;
    return n < I ? { layer: sheet, n, x, y } : null;
  }

  let down = null;
  canvas.addEventListener("pointerdown", ev => {
    down = { x: ev.clientX, y: ev.clientY, t: performance.now() };
    canvas.setPointerCapture(ev.pointerId);
    const p = pick(ev);
    if (sheet != null && p && !p.back) { hover = p.n; draw(); onPreview?.(p); }
  });
  canvas.addEventListener("pointermove", ev => {
    const p = pick(ev);
    if (sheet != null) {
      if (down && p && !p.back && p.n !== hover) { hover = p.n; draw(); onPreview?.(p); }
      return;
    }
    if (ev.pointerType === "mouse" && p?.inBand) onPreview?.(p);
  });
  canvas.addEventListener("pointerup", ev => {
    const p = pick(ev);
    const tap = down && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) < 10;
    down = null;
    if (sheet == null) {
      if (p && tap) {
        if (overlay && !neutral && p.inBand) {  // with activation scores shown, a tap opens the strongest neuron near it
          const span = Math.max(2, Math.round(I / (W - pad.l - pad.r) * 6));
          let best = p.n, bv = 0;
          for (let n = Math.max(0, p.n - span); n < Math.min(I, p.n + span); n++) {
            const v = Math.abs(overlay[p.layer * I + n]); if (v > bv) { bv = v; best = n; }
          }
          if (bv > 0.3) { onOpen?.(p.layer, best); return; }
        }
        setSheet(p.layer);
      }
      return;
    }
    if (p?.back && tap) { setSheet(null); return; }
    if (hover != null) { const n = hover; hover = null; draw(); onPreview?.(null); onOpen?.(sheet, n); }
  });
  canvas.addEventListener("pointercancel", () => { down = null; hover = null; draw(); onPreview?.(null); });
  canvas.addEventListener("pointerleave", ev => { if (ev.pointerType === "mouse") onPreview?.(null); });

  function setSheet(l) { sheet = l; hover = null; onLayer?.(l); draw(); }

  new ResizeObserver(resize).observe(canvas);
  resize();

  return {
    setOverlay(v, opts = {}) { overlay = v; neutral = !!opts.neutral; draw(); },
    setFocus(f) { focus = f; draw(); },
    setWeb(w) { web = w; draw(); },
    setMarks(m) { marks = m; draw(); },
    setSheet,
    get sheet() { return sheet; },
    redraw: resize,
  };
}
