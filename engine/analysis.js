// The explorer's operations on the live model: steered three-row chat, per-token neuron reads, logit lens and steer
// lens of a neuron, direct-path connections, which neurons fire on a text, and an on-device calibration pass over a
// small built-in corpus (each neuron's activation range and strongest contexts). Every operation takes the GPU lock,
// so the shared scratch buffers are never used by two at once.

import { F } from "./gpu.js";
import * as K from "./kernels.js";

// Same text the reference explorer measures its dose unit on (median residual norm leaving each layer).
const REFERENCE_TEXT =
  "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, " +
  "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. " +
  "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n" +
  "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. ";

const STEER_LENS_PROMPTS = ["Who are you?", "Tell me about your day.", "Write one sentence about the sea."];

export class Explorer {
  constructor(model, tok) {
    this.m = model;
    this.tok = tok;
    this.g = model.gpu;
    this._lock = Promise.resolve();
    this.residP50 = null;
    this.colNorm = new Map();
    this.lensCache = new Map();
    this.conCache = new Map();
    this.scale = null;   // {pos: Float32Array[L*I], neg: Float32Array[L*I]} from calibration
    this.examples = null;
    const g = this.g, H = model.H, I = model.I;
    this.b = {
      vec: g.buffer(Math.max(H, I) * 4, "vec"),
      vec2: g.buffer(Math.max(H, I) * 4, "vec2"),
      out: g.buffer(2 * I * 4 * model.L, "outvec"),
      ids: g.buffer(16, "ids1"),
      ext: g.buffer(model.L * I * 16, "extremes"),
    };
    this.eos = tok.id("<|im_end|>");
  }

  lock(fn) {
    const run = this._lock.then(fn, fn);
    this._lock = run.catch(() => {});
    return run;
  }

  get L() { return this.m.L; }
  get I() { return this.m.I; }

  // ---------- units ----------

  async init() {
    return this.lock(async () => {
      if (this.residP50) return this.residP50;
      const m = this.m, g = this.g;
      const ids = this.tok.encode(REFERENCE_TEXT);
      const T = Math.min(ids.length, m.maxT);
      const rb = g.buffer(m.L * T * m.H * 4, "resid");
      m.reset(0);
      m._forward(ids.slice(0, T), 0, 1, T, { resid: true, residBuf: rb, capT: T, capT0: 0 });
      const r = new Float32Array(await g.read(rb));
      g.free(rb);
      m.reset(0);
      this.residP50 = [];
      for (let l = 0; l < m.L; l++) {
        const norms = [];
        for (let t = 1; t < T; t++) {
          let s = 0;
          for (let i = 0; i < m.H; i++) { const v = r[(l * T + t) * m.H + i]; s += v * v; }
          norms.push(Math.sqrt(s));
        }
        norms.sort((a, b) => a - b);
        this.residP50.push(norms[norms.length >> 1]);
      }
      return this.residP50;
    });
  }

  async _downCol(layer, n) {  // (called under the lock) the neuron's down column into b.vec; returns it
    const m = this.m, d = m.layers[layer].down;
    this.g.run("column", K.column, [d.w, d.s, this.b.vec], Math.ceil(m.H / 64), [m.H, m.I, n, F(1)]);
    return new Float32Array(await this.g.read(this.b.vec, m.H * 4));
  }

  async colNormOf(layer, n) {
    const key = layer * 1e6 + n;
    if (!this.colNorm.has(key)) {
      const v = await this.lock(() => this._downCol(layer, n));
      this.colNorm.set(key, Math.hypot(...v));
    }
    return this.colNorm.get(key);
  }

  // add-mode dose -> activation units: dose x (median residual norm leaving the layer) / |down column|
  async doseValue(layer, n, dose, mode) {
    if (mode === "mul") return dose;
    await this.init();
    return dose * this.residP50[layer] / (await this.colNormOf(layer, n));
  }

  async pushes(list) {  // [{layer, n, dose, mode, rows: [+1, 0, -1]}] -> {layer: [{n, mode, vals}]}
    const out = {};
    for (const p of list) {
      const v = p.n == null ? p.dose : await this.doseValue(p.layer, p.n, p.dose, p.mode);
      const rows = p.rows || [1];
      const vals = rows.map(r => (p.mode === "mul" ? (r > 0 ? v : r === 0 ? 1 : -v) : r * v));
      (out[p.layer] ||= []).push({ n: p.n, mode: p.mode, vals });
    }
    return out;
  }

  // ---------- steering ----------

  // Greedy (or sampled) decode of three conversations at once: row 0 pushed up, row 1 untouched, row 2 pushed down.
  // rowsMessages[r] is row r's own message list. onDelta(row, textDelta) streams text; returns the three texts.
  // vec: {layer, rows: [Float32Array(H) | null] x 3} adds each row's vector to the residual stream after that layer.
  async steer({ pushes: list = [], vec = null, rowsMessages, maxNew = 160, temperature = 0, onDelta = () => {}, cancel = { stop: false } }) {
    const pushes = await this.pushes(list.map(p => ({ ...p, rows: [1, 0, -1] })));
    return this.lock(async () => {
      const m = this.m, tok = this.tok;
      const seqs = rowsMessages.map(msgs => tok.encode(tok.chat(msgs)));
      if (Math.max(...seqs.map(s => s.length)) + maxNew > m.maxLen) throw new Error("conversation too long for the cache");
      const fw = { pushes };
      if (vec) { m.setVectors(vec.rows); fw.vec = vec.layer; }
      for (let r = 0; r < 3; r++) { m.reset(r); m.prefill(r, seqs[r], fw); }
      const out = [[], [], []], sent = ["", "", ""], done = [false, false, false];
      for (let s = 0; s < maxNew && !cancel.stop; s++) {
        let nxt;
        if (temperature > 0) {
          const lg = await m.logitsAll();
          nxt = [0, 1, 2].map(r => sample(lg.subarray(r * m.V, (r + 1) * m.V), temperature));
        } else nxt = await m.argmaxAll();
        for (let r = 0; r < 3; r++) {
          if (done[r]) continue;
          if (nxt[r] === this.eos || nxt[r] === tok.id("<|endoftext|>")) { done[r] = true; continue; }
          out[r].push(nxt[r]);
          const text = tok.decode(out[r]);
          if (!text.endsWith("\ufffd")) { onDelta(r, text.slice(sent[r].length)); sent[r] = text; }
        }
        if (done.every(Boolean)) break;
        m.step(nxt, fw);
      }
      return sent;
    });
  }

  // ---------- reading ----------

  // Activation of neuron n at `layer` on every token of `text` (a user turn when chat), with optional pushes.
  async read(layer, n, text, { chat = true, pushes: list = [] } = {}) {
    const pushes = await this.pushes(list);
    return this.lock(async () => {
      const m = this.m, g = this.g;
      let ids = this.tok.encode(chat ? this.tok.chat([{ role: "user", content: text }]) : text);
      ids = ids.slice(0, m.maxLen);
      const T = ids.length;
      const cap = g.buffer(T * m.I * 4, "cap1");
      m.reset(0);
      m.prefill(0, ids, { capture: cap, capT: T, capLayers: [layer], pushes, stopAfter: layer });
      const a = new Float32Array(await g.read(cap));
      g.free(cap);
      m.reset(0);
      const acts = Array.from({ length: T }, (_, t) => a[t * m.I + n]);
      return { ids, acts, units: this.tok.units(ids) };
    });
  }

  // Activation of neuron n at `layer` on every token of a token-id list (no template), stopping after that layer.
  async readIds(layer, n, ids) {
    return this.lock(async () => {
      const m = this.m, g = this.g;
      ids = ids.slice(0, m.maxLen);
      const T = ids.length;
      const cap = g.buffer(T * m.I * 4, "cap1");
      m.reset(0);
      m.prefill(0, ids, { capture: cap, capT: T, capLayers: [layer], stopAfter: layer });
      const a = new Float32Array(await g.read(cap));
      g.free(cap);
      m.reset(0);
      return Array.from({ length: T }, (_, t) => a[t * m.I + n]);
    });
  }

  // Every neuron at every token: {ids, units, acts: Float32Array [L, T, I]}. Long texts are cut to maxTokens.
  async activations(text, { chat = false, maxTokens = 128 } = {}) {
    return this.lock(async () => {
      const m = this.m, g = this.g;
      let ids = this.tok.encode(chat ? this.tok.chat([{ role: "user", content: text }], { generation: false }) : text);
      ids = ids.slice(0, maxTokens);
      const T = ids.length;
      const cap = g.buffer(m.L * T * m.I * 4, "capall");
      m.reset(0);
      m.prefill(0, ids, { capture: cap, capT: T });
      const acts = new Float32Array(await g.read(cap));
      g.free(cap);
      m.reset(0);
      return { ids, units: this.tok.units(ids), acts, T };
    });
  }

  // Signed score of an activation: positive side as a multiple of the neuron's calibrated max, negative of its min.
  scoreOf(l, n, a) {
    if (!this.scale) return a;
    const i = l * this.I + n;
    const pos = Math.max(this.scale.pos[i], 0.1 * -this.scale.neg[i], 1e-3);
    const neg = Math.min(this.scale.neg[i], -0.1 * this.scale.pos[i], -1e-3);
    return a >= 0 ? a / pos : -(a / neg);
  }

  // Strongest neurons at token t (or over all tokens when t < 0) of an activations() result.
  topAt(res, t, { sign = "both", lo = 0, hi = 99, limit = 200 } = {}) {
    const { acts, T } = res, I = this.I, out = [];
    for (let l = Math.max(0, lo); l <= Math.min(this.L - 1, hi); l++) {
      for (let n = 0; n < I; n++) {
        let best = 0, at = t, raw = 0;
        const t0 = t >= 0 ? t : 1, t1 = t >= 0 ? t + 1 : T;
        for (let s = t0; s < t1; s++) {
          if (t < 0 && s === 0) continue;
          const a = acts[(l * T + s) * I + n];
          const sc = this.scoreOf(l, n, a);
          if (Math.abs(sc) > Math.abs(best)) { best = sc; at = s; raw = a; }
        }
        if ((sign === "+" && best <= 0) || (sign === "-" && best >= 0) || best === 0) continue;
        out.push({ layer: l, n, score: best, act: raw, pos: at });
      }
    }
    out.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    return out.slice(0, limit);
  }

  // ---------- lenses ----------

  // Logit lens of the neuron's down column: lm_head(final_norm(+d)), top k each way.
  async lens(layer, n, k = 12) {
    const key = `${layer}:${n}`;
    let res = this.lensCache.get(key);
    if (!res || res.up.length < k) {
      res = await this.lock(async () => {
        const m = this.m, g = this.g, st = m.st, d = m.layers[layer].down;
        g.run("column", K.column, [d.w, d.s, st.last], Math.ceil(m.H / 64), [m.H, m.I, n, F(1)]);
        const lg = await m.logitsRow(0);
        return topk(lg, Math.max(k, 16), this.tok);
      });
      if (this.lensCache.size > 4000) this.lensCache.clear();
      this.lensCache.set(key, res);
    }
    return { up: res.up.slice(0, k), down: res.down.slice(0, k), upIds: res.upIds.slice(0, k), downIds: res.downIds.slice(0, k) };
  }

  // Steer lens: next-token log-probability change when the neuron is pushed +dose / -dose on a few chat prompts
  // (averaged), and the mean KL of each push.
  async steerLens(layer, n, { dose = 0.8, k = 12 } = {}) {
    const pushes = await this.pushes([{ layer, n, dose, mode: "add", rows: [1, 0, -1] }]);
    return this.lock(async () => {
      const m = this.m, V = m.V;
      const up = new Float32Array(V), down = new Float32Array(V);
      let klUp = 0, klDown = 0;
      this._baseLP ||= new Map();  // the unsteered distribution of each prompt is the same for every neuron
      for (const p of STEER_LENS_PROMPTS) {
        const ids = this.tok.encode(this.tok.chat([{ role: "user", content: p }]));
        let base = this._baseLP.get(p);
        if (!base) { m.reset(1); m.prefill(1, ids); base = logSoftmax(await m.logitsRow(1)); this._baseLP.set(p, base); }
        m.reset(0); m.prefill(0, ids, { pushes });
        const lpUp = logSoftmax(await m.logitsRow(0));
        m.reset(2); m.prefill(2, ids, { pushes });
        const lpDown = logSoftmax(await m.logitsRow(2));
        for (let i = 0; i < V; i++) { up[i] += lpUp[i] - base[i]; down[i] += lpDown[i] - base[i]; }
        klUp += kl(lpUp, base); klDown += kl(lpDown, base);
      }
      const P = STEER_LENS_PROMPTS.length;
      const tu = topk(up, k, this.tok), td = topk(down, k, this.tok);
      return { up: tu.up, down: td.up, klUp: klUp / P, klDown: klDown / P };
    });
  }

  // ---------- connections ----------

  // Direct-path weights to neurons in other layers (everything in between ignored), as in the reference explorer.
  // Downstream B in layer j > layer: g = W_gate_j[B] . ((1 + w_j) * d), u = W_up_j[B] . (...), d = this down column.
  // Upstream A in layer i < layer: A's down column against this neuron's gate and up rows. Score |g * u| z-scored
  // within each layer; sign = sign(g * u).
  async connections(layer, n, k = 24) {
    const key = `${layer}:${n}:${k}`;
    if (this.conCache.has(key)) return this.conCache.get(key);
    const res = await this.lock(async () => {
      const m = this.m, g = this.g, b = this.b, I = m.I, H = m.H, L = m.L;
      const me = m.layers[layer];
      const down = [], up = [];
      // downstream
      g.run("column", K.column, [me.down.w, me.down.s, b.vec], Math.ceil(H / 64), [H, I, n, F(1)]);
      for (let j = layer + 1; j < L; j++) {
        const ly = m.layers[j];
        g.run("scale1", K.scaleBy(true), [b.vec, ly.postNormB, b.vec2], Math.ceil(H / 256), [H, 0, 0, 0]);
        m._mm(b.vec2, ly.gu, { buffer: b.out, offset: j * 2 * I * 4, size: 2 * I * 4 }, 1);
      }
      // upstream: this neuron's gate and up rows, scaled by its pre-MLP norm weight
      g.write(b.ids, new Uint32Array([n, I + n, 0, 0]));
      g.run("gather", K.gather, [b.ids, me.gu.w, me.gu.s, b.vec2], 2, [2, H, F(1), 0]);
      const rows = new Float32Array(await g.read(b.vec2, 2 * H * 4));
      const w = me.postNorm;
      for (let i = 0; i < 2 * H; i++) rows[i] *= 1 + w[i % H];
      for (let i = 0; i < layer; i++) {
        const D = m.layers[i].down;
        for (let side = 0; side < 2; side++) {
          g.write(b.vec, rows.subarray(side * H, (side + 1) * H));
          g.run("mvT", K.matvecT, [b.vec, D.w, D.s, { buffer: b.out, offset: (i * 2 + side) * I * 4, size: I * 4 }],
            Math.ceil(I / 4 / 64), [H, I, 0, 0]);
          g.submit();  // b.vec is rewritten for the next side
        }
      }
      const all = new Float32Array(await g.read(b.out, L * 2 * I * 4));
      for (let j = 0; j < L; j++) {
        if (j === layer) continue;
        const gv = all.subarray(j * 2 * I, j * 2 * I + I), uv = all.subarray(j * 2 * I + I, (j + 1) * 2 * I);
        const prod = new Float32Array(I);
        let mean = 0;
        for (let x = 0; x < I; x++) { prod[x] = gv[x] * uv[x]; mean += Math.abs(prod[x]); }
        mean /= I;
        let v = 0;
        for (let x = 0; x < I; x++) v += (Math.abs(prod[x]) - mean) ** 2;
        const sd = Math.sqrt(v / (I - 1)) + 1e-8;
        const idx = [...prod.keys()].sort((a, c) => Math.abs(prod[c]) - Math.abs(prod[a])).slice(0, k);
        const list = j > layer ? down : up;
        for (const x of idx) list.push({ layer: j, n: x, z: (Math.abs(prod[x]) - mean) / sd, sign: prod[x] > 0 ? 1 : -1 });
      }
      down.sort((a, c) => c.z - a.z); up.sort((a, c) => c.z - a.z);
      return { down: down.slice(0, k), up: up.slice(0, k) };
    });
    this.conCache.set(key, res);
    return res;
  }

  // |down column| of every neuron, [L * I]: how hard each neuron can write into the residual stream.
  async writeNorms() {
    if (this._wn) return this._wn;
    this._wn = await this.lock(async () => {
      const m = this.m, g = this.g, I = m.I;
      for (let l = 0; l < m.L; l++) {
        const d = m.layers[l].down;
        g.run("colnorms", K.colNorms, [d.w, d.s, { buffer: this.b.out, offset: l * I * 4, size: I * 4 }], Math.ceil(I / 4 / 64), [m.H, I, 0, 0]);
      }
      const v = new Float32Array(await g.read(this.b.out, m.L * I * 4));
      for (let n = 0; n < I * m.L; n++) this.colNorm.set(Math.floor(n / I) * 1e6 + (n % I), v[n]);
      return v;
    });
    return this._wn;
  }

  // ---------- finding neurons that steer toward a topic ----------
  //
  // Activation is the wrong signal for steering: a neuron that fires on "ocean" often does nothing to what the model
  // says when pushed. This search looks at the write side and then measures it:
  //   1. concept   the topic's words plus their nearest neighbours in the model's own embedding space (tied with
  //                the unembedding), whole words only
  //   2. shortlist every neuron's down column, read through the final norm and unembedding, scored by how much it
  //                favors those words over the average word (direction only: a push's size in residual units is
  //                fixed by the dose); z-scored per layer and taken per band of layers, both signs
  //   3. probe     each candidate actually pushed (0.8) in two short contexts that end where a topic word could
  //                come next; measured: probability mass gained by the topic words, their log-odds gain, and KL
  //                (how much everything else moves). Pushes with large KL derail into repetition, so each result
  //                gets a recommended strength that aims for a moderate KL (KL grows about with dose squared)
  //   4. preview   the best few write the start of a short story at their recommended strength

  async _vocabStats() {  // (under the lock) |E row| per token and the vocab-mean of E * (1 + final norm weight)
    if (this._vstats) return this._vstats;
    const m = this.m, g = this.g, E = m.embed, V = m.V, H = m.H;
    const norms = g.buffer(V * 4, "enorm");
    const wgs = Math.ceil(V / 64), sx = Math.min(wgs, 65535);
    g.run("rownorms", K.rowNorms, [E.w, E.s, norms], [sx, Math.ceil(wgs / sx)], [V, H, 0, 0]);
    const ones = g.buffer(V * 4, "invV");
    g.write(ones, new Float32Array(V).fill(1 / V));
    g.run("mvT", K.matvecT, [ones, E.w, E.s, this.b.vec], Math.ceil(H / 4 / 64), [V, H, 0, 0]);
    const n = new Float32Array(await g.read(norms));
    const mean = new Float32Array(await g.read(this.b.vec, H * 4));
    g.free(norms); g.free(ones);
    const gain = Float32Array.from(m.finalNorm, w => 1 + w);
    for (let i = 0; i < H; i++) mean[i] *= gain[i];
    return (this._vstats = { norms: n, mean, gain });
  }

  async _rows(ids) {  // (under the lock) dequantized embedding rows -> Float32Array [ids.length, H]
    const m = this.m, g = this.g, H = m.H;
    const idb = g.buffer(ids.length * 4, "rowids"), out = g.buffer(ids.length * H * 4, "rows");
    g.write(idb, Uint32Array.from(ids));
    g.run("gather", K.gather, [idb, m.embed.w, m.embed.s, out], ids.length, [ids.length, H, F(1), 0]);
    const r = new Float32Array(await g.read(out));
    g.free(idb); g.free(out);
    return r;
  }

  // Topic -> {words, q (unit vector in embedding space), targets: [{id, w, text, cos}] (up to 64 whole-word tokens,
  // weighted by similarity), display (a few distinct ones in the topic's script)}, or null when nothing usable.
  async concept(text, { max = 64 } = {}) {
    const tok = this.tok;
    const words = (text.toLowerCase().match(/[\p{L}\p{N}']+/gu) || []).filter(w => !STOP.has(w));
    const qids = new Set();
    for (const w of words.length ? words : [text.trim().toLowerCase()]) {
      for (const v of [" " + w, w, " " + cap(w), cap(w)]) {
        const ids = tok.encode(v, { special: false });
        if (ids.length === 1 || (ids.length && tok.piece(ids[0]).trim().length >= 3)) qids.add(ids[0]);
      }
    }
    if (!qids.size) return null;
    return this.lock(async () => {
      const m = this.m, g = this.g, H = m.H, V = m.V, st = m.st;
      const vs = await this._vocabStats();
      const q0 = [...qids];
      const rows = await this._rows(q0);
      const q = new Float32Array(H);
      q0.forEach((id, i) => { const nn = vs.norms[id] || 1; for (let k = 0; k < H; k++) q[k] += rows[i * H + k] / nn; });
      const qn = Math.hypot(...q) || 1;
      for (let k = 0; k < H; k++) q[k] /= qn;
      g.write(this.b.vec, q);
      m._mm(this.b.vec, m.embed, st.logits, 1);
      const sim = new Float32Array(await g.read(st.logits, V * 4));
      for (let i = 0; i < V; i++) sim[i] /= vs.norms[i] || 1;
      const top = partialTop(sim, 400, 1);
      const targets = new Map(q0.map(id => [id, 1]));  // the topic's own tokens
      const s0 = sim[top[0]];
      for (const id of top) {
        if (targets.size >= max || sim[id] < RELATED) break;
        if (targets.has(id) || !wordlike(tok.piece(id))) continue;
        targets.set(id, Math.max(0.05, (sim[id] - 0.2) / (s0 - 0.2)));
      }
      const list = [...targets].map(([id, w]) => ({ id, w, text: tok.piece(id), cos: sim[id] }));
      const latin = /\p{Script=Latin}/u.test(text);
      // shown to the user: distinct words (one per 4-letter stem, so not nine spellings of "music"), topic's script
      const seen = new Set(), display = [];
      for (const t of [...list].sort((x, y) => y.cos - x.cos)) {
        const k = t.text.trim().toLowerCase().slice(0, 4);
        const w0 = t.text.trim().toLowerCase();
        if (seen.has(k) || (latin && !/^\s?\p{Script=Latin}+$/u.test(t.text))) continue;
        if (!words.includes(w0) && isPiece(w0, list)) continue;
        seen.add(k); display.push(t.text.trim().toLowerCase());
      }
      return { words, q, targets: list, display: display.slice(0, 10) };
    });
  }

  // Candidates from the weights: [{layer, n, sign, z}] (sign +1: boosting favors the topic, -1: suppressing does).
  async shortlist(concept, { perBand = 3, bands = 6 } = {}) {
    await this.writeNorms();
    return this.lock(async () => {
      const m = this.m, g = this.g, H = m.H, I = m.I, L = m.L;
      const vs = await this._vocabStats();
      const ids = concept.targets.map(t => t.id), w = concept.targets.map(t => t.w);
      const rows = await this._rows(ids);
      const u = new Float32Array(H), W = w.reduce((a, b) => a + b, 0);
      ids.forEach((_, i) => { for (let k = 0; k < H; k++) u[k] += w[i] * rows[i * H + k] * vs.gain[k] / W; });
      for (let k = 0; k < H; k++) u[k] -= vs.mean[k];
      g.write(this.b.vec, u);
      for (let l = 0; l < L; l++) {
        const D = m.layers[l].down;
        g.run("mvT", K.matvecT, [this.b.vec, D.w, D.s, { buffer: this.b.out, offset: l * I * 4, size: I * 4 }], Math.ceil(I / 4 / 64), [H, I, 0, 0]);
      }
      const a = new Float32Array(await g.read(this.b.out, L * I * 4));
      const z = new Float32Array(L * I);
      for (let l = 0; l < L; l++) {
        let mu = 0, sd = 0;
        for (let n = 0; n < I; n++) { const v = a[l * I + n] / (this.colNorm.get(l * 1e6 + n) || 1); a[l * I + n] = v; mu += v; }
        mu /= I;
        for (let n = 0; n < I; n++) sd += (a[l * I + n] - mu) ** 2;
        sd = Math.sqrt(sd / I) || 1;
        for (let n = 0; n < I; n++) z[l * I + n] = (a[l * I + n] - mu) / sd;
      }
      const out = [], per = Math.ceil(L / bands);
      for (let b0 = 0; b0 < L; b0 += per) {
        const lo = b0 * I, hi = Math.min(L, b0 + per) * I, band = z.subarray(lo, hi);
        for (const sign of [1, -1]) {
          for (const i of partialTop(band, perBand, sign)) out.push({ layer: Math.floor((lo + i) / I), n: (lo + i) % I, sign, z: sign * band[i] });
        }
      }
      return out;
    });
  }

  // Pushes each candidate (rows of one batch, three at a time) in short contexts and measures what happens to the
  // topic's words. Adds to each candidate: dp (probability mass gained), gain (log-odds), kl, rec (recommended
  // strength), adds (top words it raises), score. Sorted by score.
  async probe(cands, concept, { dose = 0.8, onProgress = () => {}, cancel = { stop: false } } = {}) {
    const m = this.m, tok = this.tok, V = m.V;
    await this.init();
    const tid = concept.targets.map(t => t.id), lw = concept.targets.map(t => Math.log(t.w));
    const ctxs = PROBE_CONTEXTS.map(c => tok.encode(c, { special: false }));
    const base = [];
    const acc = cands.map(() => ({ dp: 0, gain: 0, kl: 0, d: null }));
    const jobs = [];
    ctxs.forEach((ids, c) => { jobs.push({ c, k: -1 }); cands.forEach((_, k) => jobs.push({ c, k })); });
    let done = 0;
    for (let j = 0; j < jobs.length && !cancel.stop; j += 3) {
      const group = jobs.slice(j, j + 3).filter(x => x.c === jobs[j].c);  // one context per batch (same length)
      j -= 3 - group.length;
      const ids = ctxs[group[0].c];
      const lps = await this.lock(async () => {
        const pushes = {};
        group.forEach((x, r) => {
          if (x.k < 0) return;
          const cd = cands[x.k], v = cd.sign * dose * this.residP50[cd.layer] / (this.colNorm.get(cd.layer * 1e6 + cd.n) || 1);
          const vals = [0, 0, 0, 0]; vals[r] = v;
          (pushes[cd.layer] ||= []).push({ n: cd.n, mode: "add", vals });
        });
        if (Object.values(pushes).some(p => p.length > 8)) throw new Error("too many pushes in one layer");
        for (let r = 0; r < 3; r++) m.reset(r);
        const all = [];
        for (let r = 0; r < 3; r++) all.push(...ids);
        m._forward(all, 0, 3, ids.length, { pushes });
        m._logits(3);
        const lg = new Float32Array(await m.gpu.read(m.st.logits, 3 * V * 4));
        for (let r = 0; r < 3; r++) m.reset(r);
        return group.map((_, r) => logSoftmax(lg.subarray(r * V, (r + 1) * V)));
      });
      group.forEach((x, r) => {
        const lp = lps[r];
        if (x.k < 0) { base[x.c] = lp; return; }
        const b = base[x.c], a = acc[x.k];
        let mass = 0, mb = 0;
        const t = tid.map((id, i) => { mass += Math.exp(lp[id]); mb += Math.exp(b[id]); return lp[id] + lw[i]; });
        const tb = tid.map((id, i) => b[id] + lw[i]);
        a.dp += (mass - mb) / ctxs.length;
        a.gain += (lse(t) - lse(tb)) / ctxs.length;
        a.kl += kl(lp, b) / ctxs.length;
        if (!a.d) a.d = new Float32Array(V);
        for (let i = 0; i < V; i++) a.d[i] += lp[i] - b[i];
        done++;
        onProgress({ done, total: cands.length * ctxs.length });
      });
    }
    if (cancel.stop) return null;
    const out = cands.map((cd, k) => {
      const a = acc[k];
      const adds = partialTop(a.d, 60, 1).filter(i => wordlike(tok.piece(i))).slice(0, 6).map(i => tok.piece(i));
      const rec = Math.min(1.2, Math.max(0.1, dose * Math.sqrt(0.8 / Math.max(a.kl, 1e-6))));  // aim for KL about 0.8
      const score = a.dp * (a.kl < 2.5 ? 1 : 0.5) + 0.02 * a.gain;
      return { ...cd, dp: a.dp, gain: a.gain, kl: a.kl, rec: +rec.toFixed(2), adds, score };
    });
    return out.sort((x, y) => y.score - x.score);
  }

  // The start of a short story with each result pushed at its recommended strength (three per batch). Adds
  // preview (text), hits (distinct words in it that are close to the topic in embedding space) and repeats (share of
  // repeated word pairs) to each.
  async preview(results, concept, { prompt = "Tell me a short story.", maxNew = 32, onProgress = () => {}, cancel = { stop: false } } = {}) {
    const m = this.m, tok = this.tok;
    const ids = tok.encode(tok.chat([{ role: "user", content: prompt }]));
    for (let j = 0; j < results.length && !cancel.stop; j += 3) {
      const group = results.slice(j, j + 3);
      const texts = await this.lock(async () => {
        const pushes = {};
        group.forEach((cd, r) => {
          const v = cd.sign * cd.rec * this.residP50[cd.layer] / (this.colNorm.get(cd.layer * 1e6 + cd.n) || 1);
          const vals = [0, 0, 0, 0]; vals[r] = v;
          (pushes[cd.layer] ||= []).push({ n: cd.n, mode: "add", vals });
        });
        for (let r = 0; r < 3; r++) { m.reset(r); m.prefill(r, ids, { pushes }); }
        const out = [[], [], []], fin = [false, false, false];
        for (let s = 0; s < maxNew && !cancel.stop; s++) {
          const nx = await m.argmaxAll();
          nx.forEach((t, r) => { if (t === this.eos) fin[r] = true; else if (!fin[r]) out[r].push(t); });
          if (fin.every(Boolean)) break;
          m.step(nx, { pushes });
        }
        for (let r = 0; r < 3; r++) m.reset(r);
        // which generated words sit near the topic: cosine of their embedding with the topic vector
        const uniq = [...new Set(out.flat())].filter(id => wordlike(tok.piece(id)));
        const near = new Set();
        if (uniq.length) {
          const vs = await this._vocabStats(), rows = await this._rows(uniq), H = m.H;
          uniq.forEach((id, i) => {
            let d = 0; for (let k = 0; k < H; k++) d += rows[i * H + k] * concept.q[k];
            if (d / (vs.norms[id] || 1) >= RELATED) near.add(id);
          });
        }
        return out.map(o => ({ text: tok.decode(o).trim(), hits: new Set(o.filter(id => near.has(id))).size }));
      });
      group.forEach((cd, r) => {
        const { text, hits } = texts[r], w = text.toLowerCase().match(/[\p{L}\p{N}']+/gu) || [];
        const pairs = w.slice(1).map((x, i) => w[i] + " " + x);
        cd.preview = text;
        cd.hits = hits;
        cd.repeats = pairs.length ? 1 - new Set(pairs).size / pairs.length : 0;
        onProgress({ done: j + r + 1, total: results.length, item: cd });
      });
    }
    return results;
  }

  // ---------- calibration ----------

  // Runs a corpus through the model in chunks (each its own sequence; position 0, the attention sink, is skipped) and
  // keeps each neuron's largest and most negative activation and where they happened (top `keep` chunks per sign).
  // Takes the GPU lock one chunk at a time, so it can run in the background while the explorer is in use (anything
  // else queued gets the GPU between chunks).
  async calibrate(texts, { onProgress = () => {}, keep = 4, cancel = { stop: false }, hold = null } = {}) {
    const m = this.m, g = this.g, L = m.L, I = m.I;
    const chunks = [];
    for (const t of texts) {
      const ids = this.tok.encode(t);
      for (let i = 0; i < ids.length; i += m.maxT - 1) {
        const c = ids.slice(i, i + m.maxT);
        if (c.length > 8) chunks.push(c);
      }
    }
    const pos = new Float32Array(L * I).fill(0), neg = new Float32Array(L * I).fill(0);
    const topP = new Array(L * I), topN = new Array(L * I);
    const cap = g.buffer(L * m.maxT * I * 4, "calcap");
    try {
      for (let c = 0; c < chunks.length && !cancel.stop; c++) {
        while (hold && hold()) await new Promise(r => setTimeout(r, 300));  // e.g. wait while a steered run decodes
        const e = await this.lock(async () => {
          const ids = chunks[c], T = ids.length;
          m.reset(0);
          m.prefill(0, ids, { capture: cap, capT: T });
          g.run("extremes", K.tokenExtremes, [cap, this.b.ext], [Math.ceil(I / 256), L], [L, T, I, 1]);
          const out = new Float32Array(await g.read(this.b.ext, L * I * 16));
          m.reset(0);
          return out;
        });
        for (let i = 0; i < L * I; i++) {
          const mx = e[i * 4], amx = e[i * 4 + 1], mn = e[i * 4 + 2], amn = e[i * 4 + 3];
          if (mx > pos[i]) pos[i] = mx;
          if (mn < neg[i]) neg[i] = mn;
          insertTop(topP, i, [mx, c, amx], keep, (a, b) => b[0] - a[0]);
          insertTop(topN, i, [mn, c, amn], keep, (a, b) => a[0] - b[0]);
        }
        onProgress({ frac: (c + 1) / chunks.length, chunk: c + 1, chunks: chunks.length });
      }
    } finally {
      await this.lock(async () => g.free(cap));
    }
    if (cancel.stop) return null;
    this.scale = { pos, neg };
    this.examples = { chunks, topP, topN };
    return this.scale;
  }

  // A neuron's strongest calibration contexts: [{sign, act, ids, at}] (window of tokens around the max)
  contexts(layer, n, win = 24) {
    if (!this.examples) return null;
    const i = layer * this.I + n, out = [];
    for (const [sign, list] of [["+", this.examples.topP[i]], ["-", this.examples.topN[i]]]) {
      for (const [act, c, at] of list || []) {
        if ((sign === "+" && act <= 0) || (sign === "-" && act >= 0)) continue;
        const ids = this.examples.chunks[c];
        const s = Math.max(1, at - win), e = Math.min(ids.length, at + 6);
        out.push({ sign, act, chunk: c, at, s, e });
      }
    }
    return out;
  }
}

// ---------- helpers ----------

const STOP = new Set("the a an of to and or in on at for about with more less be is are sound sounds like talk talks make makes model it its that this some any very".split(" "));
const cap = w => w.charAt(0).toUpperCase() + w.slice(1);
// a whole word: " word" (word-initial piece), "Word" (capitalized, sentence start), or a run of CJK characters
const wordlike = s => /^ \p{L}[\p{L}'-]+$/u.test(s) || /^\p{Lu}\p{Ll}+$/u.test(s) || /^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+$/u.test(s);
// a short word-initial piece of a longer listed word ("oce" of ocean, "seaw" of seaweed), not a word with its own
// listed plural ("sea", "seas")
const isPiece = (w0, list) => {
  if (w0.length > 4) return false;
  const all = list.map(o => o.text.trim().toLowerCase());
  return all.some(w => w.length > w0.length && w.startsWith(w0)) && !all.includes(w0 + "s");
};
const lse = a => { let mx = -Infinity; for (const v of a) if (v > mx) mx = v; let s = 0; for (const v of a) s += Math.exp(v - mx); return mx + Math.log(s); };
// short contexts that end where a topic word could come next (plain text, no chat template)
const PROBE_CONTEXTS = ["My favorite thing in the world is", "Today I want to tell you about"];
// embedding cosine above which a word counts as related to the topic (sea~ocean 0.45, boat 0.25, song~music 0.40,
// kitchen~cooking 0.42; unrelated city 0.10-0.15, rain 0.06-0.20, the ~0)
const RELATED = 0.22;

function insertTop(arr, i, item, keep, cmp) {
  const l = arr[i] || (arr[i] = []);
  if (l.length < keep) { l.push(item); l.sort(cmp); return; }
  if (cmp(item, l[l.length - 1]) < 0) { l[l.length - 1] = item; l.sort(cmp); }
}

export function topk(arr, k, tok) {
  const hi = partialTop(arr, k, 1), lo = partialTop(arr, k, -1);
  return { up: hi.map(i => tok.piece(i)), down: lo.map(i => tok.piece(i)), upIds: hi, downIds: lo };
}

function partialTop(arr, k, sign) {
  const best = [];
  for (let i = 0; i < arr.length; i++) {
    const v = sign * arr[i];
    if (best.length < k) { best.push([v, i]); if (best.length === k) best.sort((a, b) => a[0] - b[0]); continue; }
    if (v > best[0][0]) { best[0] = [v, i]; best.sort((a, b) => a[0] - b[0]); }
  }
  return best.sort((a, b) => b[0] - a[0]).map(x => x[1]);
}

function logSoftmax(x) {
  let mx = -Infinity;
  for (const v of x) if (v > mx) mx = v;
  let s = 0;
  for (const v of x) s += Math.exp(v - mx);
  const lse = mx + Math.log(s);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - lse;
  return out;
}

function kl(lp, lq) {  // KL(p || q) from log-probabilities
  let s = 0;
  for (let i = 0; i < lp.length; i++) { const p = Math.exp(lp[i]); if (p > 1e-12) s += p * (lp[i] - lq[i]); }
  return s;
}

function sample(logits, temperature) {
  let mx = -Infinity;
  for (const v of logits) if (v > mx) mx = v;
  let s = 0;
  const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / temperature); s += p[i]; }
  let r = Math.random() * s;
  for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }
  return p.length - 1;
}
