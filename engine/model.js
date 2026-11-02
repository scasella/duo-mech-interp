// Qwen3.5 text decoder on WebGPU with interpretability hooks.
//
// Weights stream straight from the Hugging Face checkpoint (bf16 safetensors), are quantized to int8 on the GPU as the
// bytes arrive, and are cached in the browser (Cache Storage) for the next visit. Linear layers are fused per block:
// full attention [q+gate | k | v], linear attention [qkv | z | b | a], MLP [gate | up].
//
// Hooks (all exact, at the input of down_proj where neuron activations silu(gate) * up live):
//   pushes   [{layer, n, mode: "add"|"mul", vals: [per row]}]   n = null: every neuron of the layer (mul only)
//   capture  every neuron at every token of a forward call -> Float32Array [layers, T, inter]
//   heads    per-head output scale on linear-attention layers (ablation)
//   resid    residual stream after each layer -> Float32Array [layers, T, hidden]

import { GPU, F } from "./gpu.js";
import * as K from "./kernels.js";
import * as Q from "./kernels_qwen35.js";

let HF = "https://huggingface.co";
export function setHubBase(url) { HF = url.replace(/\/$/, ""); }  // tests serve a local mirror

export const MODELS = {
  "Qwen/Qwen3.5-0.8B": { file: "model.safetensors-00001-of-00001.safetensors", prefix: "model.language_model." },
  "Qwen/Qwen3.5-2B": { file: "model.safetensors-00001-of-00001.safetensors", prefix: "model.language_model." },
};

export class Model {
  constructor(gpu, repo, cfg) {
    this.gpu = gpu;
    this.repo = repo;
    const c = cfg.text_config || cfg;
    this.cfg = c;
    this.L = c.num_hidden_layers;
    this.H = c.hidden_size;
    this.I = c.intermediate_size;
    this.V = c.vocab_size;
    this.eps = c.rms_norm_eps;
    this.types = c.layer_types;
    this.nh = c.num_attention_heads;
    this.nkv = c.num_key_value_heads;
    this.hd = c.head_dim;
    const rp = c.rope_parameters || {};
    this.theta = rp.rope_theta || c.rope_theta || 1e7;
    this.rot = Math.round(this.hd * (rp.partial_rotary_factor ?? c.partial_rotary_factor ?? 1));
    this.Hk = c.linear_num_key_heads;
    this.Hv = c.linear_num_value_heads;
    this.dk = c.linear_key_head_dim;
    this.dv = c.linear_value_head_dim;
    this.KS = c.linear_conv_kernel_dim;
    this.Kd = this.Hk * this.dk;
    this.Vd = this.Hv * this.dv;
    this.C = 2 * this.Kd + this.Vd;  // conv channels (q, k, v)
    if (this.dk !== 128 || this.dv !== 128) throw new Error("linear attention kernels assume 128-dim heads");
    this.Wfull = this.nh * 2 * this.hd + 2 * this.nkv * this.hd;
    this.Wlin = this.C + this.Vd + 2 * this.Hv;
    this.layers = [];
    this.rows = 0;
  }

  // ---------- loading ----------

  static async load(repo, { onProgress = () => {}, gpu = null, signal, cache = true } = {}) {
    gpu = gpu || await GPU.create();
    const cfg = await (await fetch(`${HF}/${repo}/resolve/main/config.json`, { signal })).json();
    const m = new Model(gpu, repo, cfg);
    m.useCache = cache;
    await m._loadWeights(onProgress, signal);
    return m;
  }

  _plan() {
    // destination of every checkpoint tensor we use: {kind: "q", buf, row} for quantized rows, {kind: "f", key} small
    const P = MODELS[this.repo]?.prefix ?? "model.language_model.";
    const plan = new Map();
    const qbuf = (rows, cols, label) => ({ rows, cols, label });
    this.embed = qbuf(this.V, this.H, "embed");
    plan.set(P + "embed_tokens.weight", { kind: "q", dst: this.embed, row: 0 });
    plan.set(P + "norm.weight", { kind: "f", set: v => (this.finalNorm = v) });
    for (let l = 0; l < this.L; l++) {
      const lin = this.types[l] === "linear_attention";
      const ly = { lin };
      const p = `${P}layers.${l}.`;
      plan.set(p + "input_layernorm.weight", { kind: "f", set: v => (ly.inNorm = v) });
      plan.set(p + "post_attention_layernorm.weight", { kind: "f", set: v => (ly.postNorm = v) });
      ly.gu = qbuf(2 * this.I, this.H, `L${l}.gu`);
      ly.down = qbuf(this.H, this.I, `L${l}.down`);
      plan.set(p + "mlp.gate_proj.weight", { kind: "q", dst: ly.gu, row: 0 });
      plan.set(p + "mlp.up_proj.weight", { kind: "q", dst: ly.gu, row: this.I });
      plan.set(p + "mlp.down_proj.weight", { kind: "q", dst: ly.down, row: 0 });
      if (lin) {
        const a = p + "linear_attn.";
        ly.proj = qbuf(this.Wlin, this.H, `L${l}.lin`);
        ly.out = qbuf(this.H, this.Vd, `L${l}.out`);
        plan.set(a + "in_proj_qkv.weight", { kind: "q", dst: ly.proj, row: 0 });
        plan.set(a + "in_proj_z.weight", { kind: "q", dst: ly.proj, row: this.C });
        plan.set(a + "in_proj_b.weight", { kind: "q", dst: ly.proj, row: this.C + this.Vd });
        plan.set(a + "in_proj_a.weight", { kind: "q", dst: ly.proj, row: this.C + this.Vd + this.Hv });
        plan.set(a + "out_proj.weight", { kind: "q", dst: ly.out, row: 0 });
        plan.set(a + "conv1d.weight", { kind: "f", set: v => (ly.conv = v) });
        plan.set(a + "A_log", { kind: "f", set: v => (ly.alog = v) });
        plan.set(a + "dt_bias", { kind: "f", set: v => (ly.dtb = v) });
        plan.set(a + "norm.weight", { kind: "f", set: v => (ly.gnorm = v) });
      } else {
        const a = p + "self_attn.";
        ly.proj = qbuf(this.Wfull, this.H, `L${l}.attn`);
        ly.out = qbuf(this.H, this.nh * this.hd, `L${l}.o`);
        plan.set(a + "q_proj.weight", { kind: "q", dst: ly.proj, row: 0 });
        plan.set(a + "k_proj.weight", { kind: "q", dst: ly.proj, row: this.nh * 2 * this.hd });
        plan.set(a + "v_proj.weight", { kind: "q", dst: ly.proj, row: this.nh * 2 * this.hd + this.nkv * this.hd });
        plan.set(a + "o_proj.weight", { kind: "q", dst: ly.out, row: 0 });
        plan.set(a + "q_norm.weight", { kind: "f", set: v => (ly.qNorm = v) });
        plan.set(a + "k_norm.weight", { kind: "f", set: v => (ly.kNorm = v) });
      }
      this.layers.push(ly);
    }
    return plan;
  }

  _allocQ(d) {
    const g = this.gpu;
    if (d.cols % K.QG) throw new Error(`${d.label}: ${d.cols} columns not a multiple of ${K.QG}`);
    d.w = g.buffer(d.rows * d.cols, d.label + ".w");
    d.s = g.buffer(d.rows * d.cols / K.QG * 4, d.label + ".s");
  }

  _qbufs() {
    const out = [this.embed];
    for (const ly of this.layers) out.push(ly.proj, ly.out, ly.gu, ly.down);
    return out;
  }

  _small() {  // every f32 tensor, as [key, getter, setter]
    const out = [["final", () => this.finalNorm, v => (this.finalNorm = v)]];
    this.layers.forEach((ly, l) => {
      const keys = ["inNorm", "postNorm", ...(ly.lin ? ["conv", "alog", "dtb", "gnorm"] : ["qNorm", "kNorm"])];
      for (const k of keys) out.push([`L${l}.${k}`, () => ly[k], v => (ly[k] = v)]);
    });
    return out;
  }

  _cacheName() { return `duo-mech-interp:${this.repo}:q8v1`; }

  async _loadWeights(onProgress, signal) {
    const plan = this._plan();
    for (const d of this._qbufs()) this._allocQ(d);
    if (this.useCache && await this._loadCached(onProgress)) { this._finish(); return; }
    await this._stream(plan, onProgress, signal);
    this._finish();
    if (!this.useCache) return;
    onProgress({ phase: "caching", frac: 0 });
    try { await this._saveCache(onProgress); } catch (e) { console.warn("could not cache weights", e); }
  }

  async _stream(plan, onProgress, signal) {
    const g = this.gpu;
    const file = MODELS[this.repo]?.file ?? "model.safetensors";
    const res = await fetch(`${HF}/${this.repo}/resolve/main/${file}`, { signal });
    if (!res.ok) throw new Error(`download failed: ${res.status}`);
    const reader = res.body.getReader();
    let buf = new Uint8Array(0), at = 0;  // unconsumed bytes, absolute file position of buf[0]
    const pull = async () => {
      const { done, value } = await reader.read();
      if (done) throw new Error("checkpoint ended early");
      const nb = new Uint8Array(buf.length + value.length);
      nb.set(buf); nb.set(value, buf.length);
      buf = nb;
    };
    while (buf.length < 8) await pull();
    const hlen = Number(new DataView(buf.buffer).getBigUint64(0, true));
    while (buf.length < 8 + hlen) await pull();
    const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hlen)));
    delete header.__metadata__;
    const base = 8 + hlen;
    const todo = Object.entries(header).filter(([k]) => plan.has(k))
      .map(([k, v]) => ({ name: k, dtype: v.dtype, shape: v.shape, s: base + v.data_offsets[0], e: base + v.data_offsets[1], ...plan.get(k) }))
      .sort((a, b) => a.s - b.s);
    const missing = [...plan.keys()].filter(k => !header[k]);
    if (missing.length) throw new Error("checkpoint is missing " + missing.slice(0, 3).join(", "));
    const total = todo.reduce((s, t) => s + (t.e - t.s), 0);
    let doneBytes = 0;
    const STAGE = 16 << 20;
    const stage = g.buffer(STAGE, "stage");
    const pending = new Uint8Array(STAGE);
    // drop consumed bytes from buf
    const consume = (upto) => { buf = buf.subarray(upto - at); at = upto; };
    consume(base);
    for (const t of todo) {
      // skip bytes before this tensor
      while (at + buf.length < t.s) { consume(at + buf.length); await pull(); }
      consume(t.s);
      if (t.kind === "f") {
        while (buf.length < t.e - t.s) await pull();
        t.set(toF32(buf.subarray(0, t.e - t.s), t.dtype));
        consume(t.e);
      } else {
        const cols = t.shape[t.shape.length - 1];
        const rows = t.shape.length === 1 ? 1 : t.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const es = t.dtype === "F32" ? 4 : 2;
        const rowBytes = cols * es;
        const perBatch = Math.max(1, Math.floor(STAGE / rowBytes));
        const src = t.dtype === "F32" ? "f32" : t.dtype === "BF16" ? "bf16" : "f16";
        for (let r0 = 0; r0 < rows; r0 += perBatch) {
          const nr = Math.min(perBatch, rows - r0);
          const need = nr * rowBytes;
          let filled = 0;
          while (filled < need) {
            if (buf.length === 0) await pull();
            const take = Math.min(buf.length, need - filled);
            pending.set(buf.subarray(0, take), filled);
            filled += take;
            consume(at + take);
          }
          g.write(stage, pending.subarray(0, need));
          const groups = nr * cols / K.QG;
          const stride = Math.min(groups, 65535);
          g.run("quant_" + src, K.quantize(src), [stage, t.dst.w, t.dst.s], [stride, Math.ceil(groups / stride)],
            [groups, cols / K.QG, t.row + r0, stride]);
          g.submit();
          // keep queued uploads bounded (each batch is a 16 MB staging copy until the GPU consumes it)
          if ((r0 / perBatch) % 4 === 3) await g.idle();
          doneBytes += need;
          onProgress({ phase: "download", frac: doneBytes / total, bytes: doneBytes, total });
          if (g.lost) throw new Error("GPU device lost while loading");
        }
        // throttle: let the GPU catch up so queued uploads don't pile up in memory
        await g.idle();
      }
    }
    reader.cancel().catch(() => {});
    g.free(stage);
  }

  async _saveCache(onProgress) {
    if (!("caches" in self)) return;
    const cache = await caches.open(this._cacheName());
    const g = this.gpu;
    const bufs = this._qbufs();
    let i = 0;
    const CH = 32 << 20;
    for (const d of bufs) {
      for (const part of ["w", "s"]) {
        const b = d[part];
        for (let o = 0, k = 0; o < b.size; o += CH, k++) {
          const data = await g.read(b, Math.min(CH, b.size - o), o);
          await cache.put(`/w/${d.label}.${part}.${k}`, new Response(data));
        }
      }
      onProgress({ phase: "caching", frac: ++i / bufs.length });
    }
    const small = {};
    for (const [k, get] of this._small()) { const v = get(); if (v) small[k] = Array.from(v); }
    await cache.put("/w/small.json", new Response(JSON.stringify(small)));
    await cache.put("/w/complete", new Response("1"));
  }

  async _loadCached(onProgress) {
    if (!("caches" in self)) return false;
    const cache = await caches.open(this._cacheName());
    if (!(await cache.match("/w/complete"))) return false;
    const g = this.gpu;
    const bufs = this._qbufs();
    let i = 0;
    const CH = 32 << 20;
    for (const d of bufs) {
      for (const part of ["w", "s"]) {
        const b = d[part];
        for (let o = 0, k = 0; o < b.size; o += CH, k++) {
          const r = await cache.match(`/w/${d.label}.${part}.${k}`);
          if (!r) return false;
          g.write(b, new Uint8Array(await r.arrayBuffer()), o);
        }
      }
      onProgress({ phase: "cache", frac: ++i / bufs.length });
      if (i % 8 === 0) await g.idle();
    }
    const small = await (await cache.match("/w/small.json")).json();
    for (const [k, , set] of this._small()) if (small[k]) set(new Float32Array(small[k]));
    return true;
  }

  static async clearCache(repo) { await caches.delete(`duo-mech-interp:${repo}:q8v1`); }

  _finish() {
    const g = this.gpu;
    const up = (arr, label) => { const b = g.buffer(arr.byteLength, label); g.write(b, arr); return b; };
    this.finalNormB = up(this.finalNorm, "final_norm");
    for (const [l, ly] of this.layers.entries()) {
      ly.inNormB = up(ly.inNorm, `L${l}.in_norm`);
      ly.postNormB = up(ly.postNorm, `L${l}.post_norm`);
      if (ly.lin) {
        ly.convB = up(ly.conv, `L${l}.conv`);
        ly.alogB = up(ly.alog, `L${l}.alog`);
        ly.dtbB = up(ly.dtb, `L${l}.dtb`);
        ly.gnormB = up(ly.gnorm, `L${l}.gnorm`);
        ly.headScale = new Float32Array(this.Hv).fill(1);
        ly.headScaleB = up(ly.headScale, `L${l}.hs`);
      } else {
        ly.qNormB = up(ly.qNorm, `L${l}.qn`);
        ly.kNormB = up(ly.kNorm, `L${l}.kn`);
      }
    }
    this.noPush = g.buffer(32, "nopush");
  }

  // ---------- state ----------

  // rows: independent sequences decoded together (3 for the -, 0, + chat); maxLen: KV cache length per row; maxT:
  // tokens per forward call (prefill is chunked to this).
  alloc(rows = 3, maxLen = 1024, maxT = 128) {
    const g = this.gpu;
    if (this.rows) this.freeState();
    Object.assign(this, { rows, maxLen, maxT });
    const st = this.st = {};
    const Mx = Math.max(maxT, rows);
    st.x = g.buffer(Mx * this.H * 4, "x");
    st.xn = g.buffer(Mx * this.H * 4, "xn");
    st.y = g.buffer(Mx * Math.max(this.Wlin, this.Wfull, 2 * this.I) * 4, "y");
    st.q = g.buffer(Mx * this.nh * this.hd * 4, "q");
    st.c = g.buffer(Mx * this.C * 4, "c");
    st.o = g.buffer(Mx * Math.max(this.Vd, this.nh * this.hd) * 4, "o");
    st.a = g.buffer(Mx * this.I * 4, "a");
    st.sc = g.buffer(Mx * this.nh * maxLen * 4, "scores");
    st.ids = g.buffer(Mx * 4, "ids");
    st.pos = g.buffer(16, "pos");
    st.last = g.buffer(rows * this.H * 4, "last");
    st.logits = g.buffer(rows * this.V * 4, "logits");
    st.amax = g.buffer(rows * 8, "argmax");
    st.push = g.buffer(this.L * 256, "pushes");  // 8 push records per layer
    st.kv = [];
    st.lin = [];
    for (const ly of this.layers) {
      if (ly.lin) {
        st.lin.push({ conv: g.buffer(rows * (this.KS - 1) * this.C * 4, "conv_state"), S: g.buffer(rows * this.Hv * this.dk * this.dv * 4, "S") });
        st.kv.push(null);
      } else {
        st.kv.push({ k: g.buffer(rows * maxLen * this.nkv * this.hd * 4, "k"), v: g.buffer(rows * maxLen * this.nkv * this.hd * 4, "v") });
        st.lin.push(null);
      }
    }
    this.len = new Array(rows).fill(0);
    this.zero = new Float32Array(Math.max(this.Hv * this.dk * this.dv, (this.KS - 1) * this.C));
  }

  freeState() {
    const st = this.st;
    for (const k of Object.keys(st)) if (st[k] && st[k].destroy) this.gpu.free(st[k]);
    for (const x of st.kv) if (x) { this.gpu.free(x.k); this.gpu.free(x.v); }
    for (const x of st.lin) if (x) { this.gpu.free(x.conv); this.gpu.free(x.S); }
    this.rows = 0;
  }

  reset(r) {  // start row r over (zero its recurrent state; the KV cache is overwritten as positions refill)
    const g = this.gpu;
    const sz = this.Hv * this.dk * this.dv, cs = (this.KS - 1) * this.C;
    for (const x of this.st.lin) if (x) {
      g.write(x.S, this.zero.subarray(0, sz), r * sz * 4);
      g.write(x.conv, this.zero.subarray(0, cs), r * cs * 4);
    }
    this.len[r] = 0;
  }

  setHeadScales(scales) {  // {layer: Float32Array(Hv)} for linear layers; anything missing is 1
    for (const [l, ly] of this.layers.entries()) {
      if (!ly.lin) continue;
      const s = scales?.[l];
      ly.headScale = s ? Float32Array.from(s) : new Float32Array(this.Hv).fill(1);
      this.gpu.write(ly.headScaleB, ly.headScale);
    }
  }

  // ---------- forward ----------

  _mm(xBuf, d, yBuf, M, add = false, xoff = 0) {
    const g = this.gpu;
    if (M <= 4) {
      const wgs = Math.ceil(d.rows / 4);
      const stride = Math.min(wgs, 32768);
      g.run(`mv${M}${add ? "a" : ""}`, K.matvec(M, add), [xBuf, d.w, d.s, yBuf], [stride, Math.ceil(wgs / stride)],
        [d.rows, d.cols, stride, xoff / 4]);
    } else {
      if (xoff) throw new Error("matmul offset");
      g.run(`mm${add ? "a" : ""}`, K.matmul(add), [xBuf, d.w, d.s, yBuf], [Math.ceil(d.rows / 64), Math.ceil(M / 64)],
        [M, d.rows, d.cols, 0]);
    }
  }

  _norm(x, w, y, M, D, offset = true) {
    this.gpu.run(offset ? "rms1" : "rms0", K.rmsnorm(offset), [x, w, y], M, [D, F(this.eps), 0, 0]);
  }

  // One forward call: `rows` consecutive sequences starting at rowBase, each with T new tokens (ids.length = rows*T),
  // continuing from this.len[row]. Returns nothing; leaves the final hidden state of each row's last token in st.last.
  _forward(ids, rowBase, rows, T, opts = {}) {
    const g = this.gpu, st = this.st, M = rows * T, H = this.H;
    if (T > this.maxT && rows > 1) throw new Error("T too large");
    g.write(st.ids, Uint32Array.from(ids));
    const pos = new Uint32Array(4);
    for (let r = 0; r < this.rows; r++) pos[r] = this.len[r];
    g.write(st.pos, pos);
    g.run("gather", K.gather, [st.ids, this.embed.w, this.embed.s, st.x], M, [M, H, F(1), 0]);
    const pushByLayer = opts.pushes || {};
    for (const [l, ps] of Object.entries(pushByLayer)) {
      if (!ps.length) continue;
      if (ps.length > 8) throw new Error("at most 8 pushes per layer");
      const rec = new ArrayBuffer(ps.length * 32);
      const u = new Uint32Array(rec), f = new Float32Array(rec);
      ps.forEach((p, i) => {
        u[i * 8] = p.n == null ? 0xffffffff : p.n;
        u[i * 8 + 1] = p.mode === "mul" ? 1 : 0;
        for (let r = 0; r < 4; r++) f[i * 8 + 4 + r] = p.vals[r] ?? (p.mode === "mul" ? 1 : 0);
      });
      g.write(st.push, new Uint8Array(rec), l * 256);
    }
    const lastLayer = opts.stopAfter ?? this.L - 1;
    for (let l = 0; l <= lastLayer; l++) {
      const ly = this.layers[l];
      this._norm(st.x, ly.inNormB, st.xn, M, H);
      this._mm(st.xn, ly.proj, st.y, M);
      if (ly.lin) {
        const s = st.lin[l];
        g.run(`conv${this.KS}`, Q.conv(this.KS), [st.y, ly.convB, s.conv, st.c], [Math.ceil(this.C / 256), M],
          [T, this.C, this.Wlin, rowBase]);
        g.run(`convst${this.KS}`, Q.convState(this.KS), [st.y, s.conv], [Math.ceil(this.C / 256), rows],
          [T, this.C, this.Wlin, rowBase, rows, 0, 0, 0]);
        g.run("delta", Q.delta, [st.c, st.y, ly.alogB, ly.dtbB, ly.gnormB, ly.headScaleB, s.S, st.o], [this.Hv, rows],
          [T, this.Hk, this.Hv, this.dk, this.dv, this.Wlin, this.C, rowBase, F(this.eps), this.Kd, 0, 0]);
      } else {
        const kv = st.kv[l];
        g.run("attnprep", Q.attnPrep, [st.y, ly.qNormB, ly.kNormB, st.pos, st.q, kv.k, kv.v], [M, this.nh + this.nkv],
          [T, this.nh, this.nkv, this.hd, this.rot, this.maxLen, rowBase, F(this.eps), F(this.theta), this.Wfull, 0, 0]);
        g.run("attn", Q.attention, [st.q, kv.k, kv.v, st.pos, st.y, st.sc, st.o], [M, this.nh],
          [T, this.nh, this.nkv, this.hd, this.maxLen, rowBase, F(1 / Math.sqrt(this.hd)), this.Wfull]);
      }
      this._mm(st.o, ly.out, st.x, M, true);
      if (opts.resid === "attn") this._snapResid(l, M, opts);
      this._norm(st.x, ly.postNormB, st.xn, M, H);
      this._mm(st.xn, ly.gu, st.y, M);
      const ps = pushByLayer[l] || [];
      const pbuf = ps.length ? { buffer: st.push, offset: l * 256, size: 256 } : this.noPush;
      const ci = !opts.capture ? -1 : opts.capLayers ? opts.capLayers.indexOf(l) : l;
      if (ci >= 0) {
        g.run("swiglu_c", Q.swiglu(true), [st.y, pbuf, st.a, opts.capture], [Math.ceil(this.I / 256), M],
          [M, this.I, T, rowBase, ps.length, (ci * opts.capT + opts.capT0) * this.I, 0, 0]);
      } else {
        g.run("swiglu", Q.swiglu(false), [st.y, pbuf, st.a], [Math.ceil(this.I / 256), M],
          [M, this.I, T, rowBase, ps.length, 0, 0, 0]);
      }
      this._mm(st.a, ly.down, st.x, M, true);
      if (opts.resid === true) this._snapResid(l, M, opts);
    }
    // last token of each row -> final norm
    for (let r = 0; r < rows; r++) g.copy(st.x, (r * T + T - 1) * H * 4, st.last, (rowBase + r) * H * 4, H * 4);
    for (let r = 0; r < rows; r++) this.len[rowBase + r] += T;
  }

  _snapResid(l, M, opts) {
    this.gpu.copy(this.st.x, 0, opts.residBuf, (l * opts.capT + opts.capT0) * this.H * 4, M * this.H * 4);
  }

  _logits(n, r0 = 0) {  // logits for rows r0..r0+n-1 of st.last -> st.logits rows 0..n-1
    const g = this.gpu, st = this.st, H = this.H;
    g.run("rms1", K.rmsnorm(true), [{ buffer: st.last, offset: r0 * H * 4, size: n * H * 4 }, this.finalNormB, st.xn], n,
      [H, F(this.eps), 0, 0]);
    this._mm(st.xn, this.embed, st.logits, n);
  }

  // logits of one row's last token (after prefill or step)
  async logitsRow(r) {
    this._logits(1, r);
    return new Float32Array(await this.gpu.read(this.st.logits, this.V * 4));
  }

  // Prefill row r with ids (from position this.len[r]); returns nothing. Chunked by maxT.
  prefill(r, ids, opts = {}) {
    for (let i = 0; i < ids.length; i += this.maxT) {
      const chunk = ids.slice(i, i + this.maxT);
      this._forward(chunk, r, 1, chunk.length, { ...opts, capT0: (opts.capT0 || 0) + i });
    }
  }

  // Greedy next tokens for all rows given their last-prefilled hidden states (st.last rows 0..rows-1 must be set).
  async argmaxAll() {
    const g = this.gpu, st = this.st;
    this._logits(this.rows);
    g.run("argmax", K.argmax, [st.logits, st.amax], this.rows, [this.V, 0, 0, 0]);
    const out = new Uint32Array(await g.read(st.amax, this.rows * 8));
    return Array.from({ length: this.rows }, (_, r) => out[r * 2]);
  }

  async logitsAll() {
    this._logits(this.rows);
    return new Float32Array(await this.gpu.read(this.st.logits, this.rows * this.V * 4));
  }

  // decode one token for every row (ids[r] for row r)
  step(ids, opts = {}) { this._forward(ids, 0, this.rows, 1, opts); }
}

function toF32(bytes, dtype) {
  const n = bytes.length / (dtype === "F32" ? 4 : 2);
  const out = new Float32Array(n);
  const al = new Uint8Array(bytes);  // aligned copy
  if (dtype === "F32") return new Float32Array(al.buffer);
  const u16 = new Uint16Array(al.buffer);
  const tmp = new Uint32Array(1), tf = new Float32Array(tmp.buffer);
  for (let i = 0; i < n; i++) {
    if (dtype === "BF16") { tmp[0] = u16[i] << 16; out[i] = tf[0]; }
    else out[i] = f16(u16[i]);
  }
  return out;
}

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
