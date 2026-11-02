// Kernels specific to the Qwen3.5 text decoder: gated full attention (q/gate split, per-head q/k RMSNorm, partial
// rotary, GQA over a KV cache) and Gated DeltaNet linear attention (causal depthwise conv with carried state, l2-normed
// q/k, delta-rule recurrence, gated RMSNorm), plus the SwiGLU MLP activation where the neuron hook lives.
//
// Token layout everywhere: activations are [M, width] with M = rows * T (row-major by row, then token). `rowOf(m)`
// = rowBase + m / T. Per-row positions live in a small storage buffer `pos` (start position of this call's tokens).

const common = /* wgsl */ `
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn softplus(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 20.0); }
`;

// ---- full attention ----

// From the fused projection y[M, Wqkv] = [q+gate (H * 2hd) | k (KV * hd) | v (KV * hd)]: per (token, head) q RMSNorm
// + rotary -> q[M, H, hd]; per (token, kv head) k RMSNorm + rotary and v -> caches at the row's position.
export const attnPrep = /* wgsl */ `
struct P { T: u32, H: u32, KV: u32, hd: u32, rot: u32, maxLen: u32, rowBase: u32, eps: f32, theta: f32, W: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> qn: array<f32>;
@group(0) @binding(3) var<storage, read> kn: array<f32>;
@group(0) @binding(4) var<storage, read> pos: array<u32>;
@group(0) @binding(5) var<storage, read_write> q: array<f32>;
@group(0) @binding(6) var<storage, read_write> kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> vc: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> buf: array<f32, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let m = wg.x;                     // token
  let hh = wg.y;                    // 0..H-1: query heads, H..H+KV-1: key/value heads
  let r = p.rowBase + m / p.T;
  let ps = pos[r] + m % p.T;
  let isQ = hh < p.H;
  var src: u32;
  if (isQ) { src = m * p.W + hh * 2u * p.hd; } else { src = m * p.W + p.H * 2u * p.hd + (hh - p.H) * p.hd; }
  var s = 0.0;
  for (var i = li.x; i < p.hd; i += 256u) { let v = y[src + i]; buf[i] = v; s += v * v; }
  red[li.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li.x < k) { red[li.x] += red[li.x + k]; } workgroupBarrier(); }
  let rr = inverseSqrt(red[0] / f32(p.hd) + p.eps);
  for (var i = li.x; i < p.hd; i += 256u) {
    let w = select(kn[i], qn[i], isQ);
    buf[256u + i] = buf[i] * rr * (1.0 + w);
  }
  workgroupBarrier();
  let half = p.rot / 2u;
  for (var i = li.x; i < p.hd; i += 256u) {
    var v = buf[256u + i];
    if (i < p.rot) {
      let j = i % half;
      let freq = pow(p.theta, -f32(2u * j) / f32(p.rot));
      let ang = f32(ps) * freq;
      let c = cos(ang); let sn = sin(ang);
      if (i < half) { v = v * c - buf[256u + i + half] * sn; } else { v = v * c + buf[256u + i - half] * sn; }
    }
    if (isQ) {
      q[(m * p.H + hh) * p.hd + i] = v;
    } else {
      let kvh = hh - p.H;
      let at = ((r * p.maxLen + ps) * p.KV + kvh) * p.hd + i;
      kc[at] = v;
      vc[at] = y[m * p.W + p.H * 2u * p.hd + p.KV * p.hd + kvh * p.hd + i];
    }
  }
}`;

// Causal softmax attention for each (token, head) over its row's cache, output gated by sigmoid(gate):
// o[M, H * hd]. Scores go through a global scratch row so the context length is not bounded by workgroup memory.
export const attention = /* wgsl */ `
struct P { T: u32, H: u32, KV: u32, hd: u32, maxLen: u32, rowBase: u32, scale: f32, W: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read> pos: array<u32>;
@group(0) @binding(5) var<storage, read> y: array<f32>;
@group(0) @binding(6) var<storage, read_write> sc: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let m = wg.x;
  let h = wg.y;
  let r = p.rowBase + m / p.T;
  let n = pos[r] + m % p.T + 1u;    // keys 0..n-1
  let kvh = h / (p.H / p.KV);
  for (var i = li.x; i < p.hd; i += 256u) { qs[i] = q[(m * p.H + h) * p.hd + i]; }
  workgroupBarrier();
  let srow = (m * p.H + h) * p.maxLen;
  var mx = -3.4e38;
  for (var j = li.x; j < n; j += 256u) {
    let kb = ((r * p.maxLen + j) * p.KV + kvh) * p.hd;
    var d = 0.0;
    for (var i = 0u; i < p.hd; i++) { d += qs[i] * kc[kb + i]; }
    d *= p.scale;
    sc[srow + j] = d;
    mx = max(mx, d);
  }
  red[li.x] = mx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li.x < k) { red[li.x] = max(red[li.x], red[li.x + k]); } workgroupBarrier(); }
  let gmax = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = li.x; j < n; j += 256u) { let e = exp(sc[srow + j] - gmax); sc[srow + j] = e; sum += e; }
  red[li.x] = sum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li.x < k) { red[li.x] += red[li.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  storageBarrier();
  workgroupBarrier();
  for (var i = li.x; i < p.hd; i += 256u) {
    var acc = 0.0;
    for (var j = 0u; j < n; j++) { acc += sc[srow + j] * vc[((r * p.maxLen + j) * p.KV + kvh) * p.hd + i]; }
    let g = y[m * p.W + h * 2u * p.hd + p.hd + i];
    o[(m * p.H + h) * p.hd + i] = acc * inv * (1.0 / (1.0 + exp(-g)));
  }
}`;

// ---- linear attention (Gated DeltaNet) ----

// Causal depthwise conv (kernel KS) over the qkv part of the fused projection, with the row's carried state of the
// previous KS-1 inputs, then SiLU: c[M, C]. y row layout: [qkv (C) | z (Vd) | b (Hv) | a (Hv)], width W.
export const conv = (KS) => /* wgsl */ `
struct P { T: u32, C: u32, W: u32, rowBase: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;       // [C, KS]
@group(0) @binding(3) var<storage, read> cst: array<f32>;      // [rows, KS-1, C]
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
${common}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  let m = gid.y;
  if (ch >= p.C) { return; }
  let t = m % p.T;
  let r = p.rowBase + m / p.T;
  var acc = 0.0;
  for (var i = 0u; i < ${KS}u; i++) {
    // input at token t - (KS-1) + i
    let back = ${KS - 1}u - i;
    var v: f32;
    if (t >= back) { v = y[(m - back) * p.W + ch]; }
    else { v = cst[(r * ${KS - 1}u + (${KS - 1}u - (back - t))) * p.C + ch]; }
    acc += cw[ch * ${KS}u + i] * v;
  }
  c[m * p.C + ch] = silu(acc);
}`;

// Shift the conv state forward by this call's T inputs (run after conv). One row at a time (rows of a call share T).
export const convState = (KS) => /* wgsl */ `
struct P { T: u32, C: u32, W: u32, rowBase: u32, rows: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> cst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  let rr = gid.y;
  if (ch >= p.C || rr >= p.rows) { return; }
  let r = p.rowBase + rr;
  var old: array<f32, ${KS - 1}>;
  for (var i = 0u; i < ${KS - 1}u; i++) { old[i] = cst[(r * ${KS - 1}u + i) * p.C + ch]; }
  // the sequence old ++ new, keep the last KS-1
  for (var i = 0u; i < ${KS - 1}u; i++) {
    let idx = p.T + i;             // index into (old ++ new) of the kept element, counted from old[0]... shifted
    // kept element j (0..KS-2) is (old ++ new)[T + j]
    var v: f32;
    if (idx < ${KS - 1}u) { v = old[idx]; } else { v = y[(rr * p.T + (idx - ${KS - 1}u)) * p.W + ch]; }
    cst[(r * ${KS - 1}u + i) * p.C + ch] = v;
  }
}`;

// The delta-rule recurrence for one (row, value head): thread j owns column j of the state S[dk, dv]. Per token:
// S *= exp(g); u = S^T k; S += k (beta (v - u))^T; o = S^T q; then the gated RMSNorm o * rms * w * silu(z).
// Optional per-head scale (ablation / head steering) multiplies the output before out_proj.
export const delta = /* wgsl */ `
struct P { T: u32, Hk: u32, Hv: u32, dk: u32, dv: u32, W: u32, C: u32, rowBase: u32, eps: f32, Kd: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> c: array<f32>;        // conv output [M, C]: q (Kd) | k (Kd) | v (Vd)
@group(0) @binding(2) var<storage, read> y: array<f32>;        // fused projection, for z, b, a
@group(0) @binding(3) var<storage, read> alog: array<f32>;
@group(0) @binding(4) var<storage, read> dtb: array<f32>;
@group(0) @binding(5) var<storage, read> nw: array<f32>;
@group(0) @binding(6) var<storage, read> hs: array<f32>;       // per-head output scale [Hv]
@group(0) @binding(7) var<storage, read_write> S: array<f32>;  // [rows, Hv, dk, dv]
@group(0) @binding(8) var<storage, read_write> o: array<f32>;  // [M, Hv * dv]
${common}
var<workgroup> qk: array<f32, 256>;
var<workgroup> red: array<f32, 128>;
var<workgroup> nrm: array<f32, 2>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let h = wg.x;
  let rr = wg.y;
  let r = p.rowBase + rr;
  let j = li.x;                    // value column (dv == 128 == workgroup size)
  let kh = h / (p.Hv / p.Hk);
  let sb = ((r * p.Hv + h) * p.dk) * p.dv + j;
  var s: array<f32, 128>;
  for (var i = 0u; i < p.dk; i++) { s[i] = S[sb + i * p.dv]; }
  let A = -exp(alog[h]);
  for (var t = 0u; t < p.T; t++) {
    let m = rr * p.T + t;
    // load q and k of this head, l2-normalise (q also by 1/sqrt(dk))
    let qv = c[m * p.C + kh * p.dk + j];
    let kv = c[m * p.C + p.Kd + kh * p.dk + j];
    qk[j] = qv; qk[128u + j] = kv;
    red[j] = qv * qv;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
    if (j == 0u) { nrm[0] = inverseSqrt(red[0] + 1e-6) / sqrt(f32(p.dk)); }
    workgroupBarrier();
    red[j] = kv * kv;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
    if (j == 0u) { nrm[1] = inverseSqrt(red[0] + 1e-6); }
    workgroupBarrier();
    let qs = nrm[0]; let ks = nrm[1];
    let base = m * p.W + p.C + p.Hv * p.dv;
    let beta = sigm(y[base + h]);
    let g = A * softplus(y[base + p.Hv + h] + dtb[h]);
    let dec = exp(g);
    var u = 0.0;
    for (var i = 0u; i < p.dk; i++) { s[i] *= dec; u += s[i] * qk[128u + i]; }
    let v = c[m * p.C + 2u * p.Kd + h * p.dv + j];
    let dl = (v - u * ks) * beta;
    var out = 0.0;
    for (var i = 0u; i < p.dk; i++) { s[i] += qk[128u + i] * ks * dl; out += s[i] * qk[i]; }
    out *= qs;
    // gated RMSNorm over this head's dv outputs
    red[j] = out * out;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
    let rms = inverseSqrt(red[0] / f32(p.dv) + p.eps);
    let z = y[m * p.W + p.C + h * p.dv + j];
    o[m * p.Hv * p.dv + h * p.dv + j] = out * rms * nw[j] * silu(z) * hs[h];
    workgroupBarrier();
  }
  for (var i = 0u; i < p.dk; i++) { S[sb + i * p.dv] = s[i]; }
}`;

// ---- MLP ----

// a[M, I] = silu(gate) * up from the fused gate|up projection gu[M, 2I], then the neuron hook: pushes for this layer
// (add a per-row value, or multiply by one; neuron 0xffffffff = every neuron), then optional capture of the result.
// Push record: (neuron, mode, pad, pad, vals: vec4<f32> per row).
export const swiglu = (capture) => /* wgsl */ `
struct P { M: u32, I: u32, T: u32, rowBase: u32, nPush: u32, capOff: u32, pad0: u32, pad1: u32 };
struct Push { n: u32, mode: u32, p0: u32, p1: u32, v: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> gu: array<f32>;
@group(0) @binding(2) var<storage, read> pushes: array<Push>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
${capture ? "@group(0) @binding(4) var<storage, read_write> cap: array<f32>;" : ""}
${common}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= p.I || m >= p.M) { return; }
  var v = silu(gu[m * 2u * p.I + n]) * gu[m * 2u * p.I + p.I + n];
  let r = p.rowBase + m / p.T;
  for (var k = 0u; k < p.nPush; k++) {
    let ps = pushes[k];
    if (ps.n == n || ps.n == 0xffffffffu) {
      let val = ps.v[min(r, 3u)];
      if (ps.mode == 0u) { v += val; } else { v *= val; }
    }
  }
  a[m * p.I + n] = v;
  ${capture ? "cap[p.capOff + m * p.I + n] = v;" : ""}
}`;
