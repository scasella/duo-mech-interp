// WGSL kernels. Weights are int8, symmetric, one f32 scale per group of QG values along a row ("q8"); activations are
// f32 throughout, so nothing here needs the shader-f16 feature. Kernels are generated as strings so row counts and
// flags are compile-time constants.

export const QG = 128;  // quantization group along the input dimension

const unpack = /* wgsl */ `
fn q4(w: u32) -> vec4<f32> {
  return vec4<f32>(f32((bitcast<i32>(w << 24u)) >> 24u), f32((bitcast<i32>(w << 16u)) >> 24u),
                   f32((bitcast<i32>(w << 8u)) >> 24u), f32(bitcast<i32>(w) >> 24u));
}`;

// ---------- load-time conversion ----------

// raw bf16/f16/f32 rows -> int8 + per-group scales. One workgroup of 32 threads per group of 128 values.
export const quantize = (src) => /* wgsl */ `
struct P { groups: u32, gpr: u32, row0: u32, stride: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
var<workgroup> mx: array<f32, 32>;
fn val(i: u32) -> f32 {
  ${src === "f32" ? "return bitcast<f32>(raw[i]);"
    : src === "bf16" ? "let w = raw[i >> 1u]; return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (i & 1u) == 1u));"
    : "let w = unpack2x16float(raw[i >> 1u]); return select(w.x, w.y, (i & 1u) == 1u);"}
}
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let g = wg.x + wg.y * p.stride;
  if (g >= p.groups) { return; }
  let base = g * 128u + li.x * 4u;
  let v = vec4<f32>(val(base), val(base + 1u), val(base + 2u), val(base + 3u));
  mx[li.x] = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (li.x < s) { mx[li.x] = max(mx[li.x], mx[li.x + s]); }
    workgroupBarrier();
  }
  let m = mx[0];
  let scale = select(m / 127.0, 1.0, m == 0.0);
  let r = vec4<i32>(round(v / scale));
  let word = (u32(r.x) & 0xffu) | ((u32(r.y) & 0xffu) << 8u) | ((u32(r.z) & 0xffu) << 16u) | ((u32(r.w) & 0xffu) << 24u);
  let og = p.row0 * p.gpr + g;
  q[og * 32u + li.x] = word;
  if (li.x == 0u) { sc[og] = scale; }
}`;

// ---------- matrix products ----------

// y[M, N] (+)= x[M, K] . W[N, K]^T for small M (decode): each workgroup computes 4 output rows, 64 lanes per row
// stride over K. ADD: accumulate into y (residual add fused).
export const matvec = (M, add) => /* wgsl */ `
struct P { N: u32, K: u32, stride: u32, xoff: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${unpack}
var<workgroup> red: array<array<f32, 256>, ${M}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x >> 6u;
  let lane = li.x & 63u;
  let n = (wg.x + wg.y * p.stride) * 4u + r;
  let K4 = p.K >> 2u;
  var acc: array<f32, ${M}>;
  if (n < p.N) {
    let wrow = n * K4;
    let srow = n * (p.K / 128u);
    for (var k = lane; k < K4; k += 64u) {
      let wv = q4(w[wrow + k]) * sc[srow + (k >> 5u)];
      for (var m = 0u; m < ${M}u; m++) { acc[m] += dot(wv, x[p.xoff + m * K4 + k]); }
    }
  }
  for (var m = 0u; m < ${M}u; m++) { red[m][li.x] = acc[m]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var m = 0u; m < ${M}u; m++) { red[m][li.x] += red[m][li.x + s]; } }
    workgroupBarrier();
  }
  if (lane == 0u && n < p.N) {
    for (var m = 0u; m < ${M}u; m++) { ${add ? "y[m * p.N + n] += red[m][li.x];" : "y[m * p.N + n] = red[m][li.x];"} }
  }
}`;

// y[M, N] (+)= x[M, K] . W[N, K]^T for any M (prefill): 64x64 output tiles, 16-deep K steps, 4x4 per thread.
export const matmul = (add) => /* wgsl */ `
struct P { M: u32, N: u32, K: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${unpack}
var<workgroup> xs: array<array<f32, 65>, 16>;
var<workgroup> ws: array<array<f32, 65>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let tid = li.y * 16u + li.x;
  let m0 = wg.y * 64u;
  let n0 = wg.x * 64u;
  var acc: array<array<f32, 4>, 4>;
  let K4 = p.K >> 2u;
  let G = p.K / 128u;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    // x tile: 64 rows x 16 k, 4 values per thread
    for (var i = 0u; i < 4u; i++) {
      let e = tid * 4u + i;
      let mm = e >> 4u;
      let kk = e & 15u;
      let gm = m0 + mm;
      xs[kk][mm] = select(0.0, x[gm * p.K + k0 + kk], gm < p.M);
    }
    // w tile: 64 rows x 16 k = 64 x 4 words, one word per thread
    {
      let nn = tid >> 2u;
      let kw = tid & 3u;
      let gn = n0 + nn;
      var v = vec4<f32>(0.0);
      if (gn < p.N) {
        let kidx = (k0 >> 2u) + kw;
        v = q4(w[gn * K4 + kidx]) * sc[gn * G + (k0 >> 7u)];
      }
      ws[kw * 4u][nn] = v.x; ws[kw * 4u + 1u][nn] = v.y; ws[kw * 4u + 2u][nn] = v.z; ws[kw * 4u + 3u][nn] = v.w;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = xs[kk][li.y * 4u + i]; b[i] = ws[kk][li.x * 4u + i]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] += a[i] * b[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let gm = m0 + li.y * 4u + i;
    if (gm >= p.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let gn = n0 + li.x * 4u + j;
      if (gn < p.N) { ${add ? "y[gm * p.N + gn] += acc[i][j];" : "y[gm * p.N + gn] = acc[i][j];"} }
    }
  }
}`;

// y[N_in] = W[N_out=R, K]^T applied the other way: y[k] = sum_r W[r, k] * x[r]. One thread per 4 columns.
export const matvecT = /* wgsl */ `
struct P { R: u32, K: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;
${unpack}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let kw = gid.x;
  let K4 = p.K >> 2u;
  if (kw >= K4) { return; }
  let G = p.K / 128u;
  var acc = vec4<f32>(0.0);
  for (var r = 0u; r < p.R; r++) { acc += q4(w[r * K4 + kw]) * (sc[r * G + (kw >> 5u)] * x[r]); }
  y[kw] = acc;
}`;

// one column of W[R, K] (dequantized): y[r] = W[r, col] * mul
export const column = /* wgsl */ `
struct P { R: u32, K: u32, col: u32, mul: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${unpack}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= p.R) { return; }
  let v = q4(w[r * (p.K >> 2u) + (p.col >> 2u)]);
  y[r] = v[p.col & 3u] * sc[r * (p.K / 128u) + (p.col >> 7u)] * p.mul;
}`;

// rows of a quantized table (embedding lookup): y[t, :] = E[ids[t], :] * mul
export const gather = /* wgsl */ `
struct P { T: u32, K: u32, mul: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<vec4<f32>>;
${unpack}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x;
  let K4 = p.K >> 2u;
  let row = ids[t];
  for (var k = li.x; k < K4; k += 64u) {
    y[t * K4 + k] = q4(w[row * K4 + k]) * (sc[row * (p.K / 128u) + (k >> 5u)] * p.mul);
  }
}`;

// ---------- elementwise / reductions ----------

// RMSNorm over rows of width D. OFFSET: weight is stored as (w - 1), i.e. y = x * rms * (1 + w).
export const rmsnorm = (offset) => /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let base = wg.x * p.D;
  var s = 0.0;
  for (var i = li.x; i < p.D; i += 256u) { let v = x[base + i]; s += v * v; }
  red[li.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li.x < k) { red[li.x] += red[li.x + k]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li.x; i < p.D; i += 256u) { y[base + i] = x[base + i] * r * (${offset ? "1.0 + w[i]" : "w[i]"}); }
}`;

// a[i] += b[i]
export const addInto = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * 256u;
  if (i < p.n) { a[i] += b[i]; }
}`;

// Per-row argmax over the last dimension: one workgroup per row. Writes (index, value) pairs.
export const argmax = /* wgsl */ `
struct P { V: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let base = wg.x * p.V;
  var best = -3.4e38;
  var idx = 0u;
  for (var i = li.x; i < p.V; i += 256u) { let v = x[base + i]; if (v > best) { best = v; idx = i; } }
  bv[li.x] = best; bi[li.x] = idx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li.x < k) {
      let o = bv[li.x + k];
      if (o > bv[li.x] || (o == bv[li.x] && bi[li.x + k] < bi[li.x])) { bv[li.x] = o; bi[li.x] = bi[li.x + k]; }
    }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[wg.x * 2u] = bi[0]; out[wg.x * 2u + 1u] = bitcast<u32>(bv[0]); }
}`;

// Per (layer, neuron) over the tokens of a capture [L, T, I], skipping tokens before t0:
// out[(l * I + n) * 4 + 0..3] = max, argmax, min, argmin (indices as f32).
export const tokenExtremes = /* wgsl */ `
struct P { L: u32, T: u32, I: u32, t0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cap: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let l = gid.y;
  if (n >= p.I || l >= p.L) { return; }
  var mx = -3.4e38; var mn = 3.4e38; var amx = 0u; var amn = 0u;
  for (var t = p.t0; t < p.T; t++) {
    let v = cap[(l * p.T + t) * p.I + n];
    if (v > mx) { mx = v; amx = t; }
    if (v < mn) { mn = v; amn = t; }
  }
  let o = (l * p.I + n) * 4u;
  out[o] = mx; out[o + 1u] = f32(amx); out[o + 2u] = mn; out[o + 3u] = f32(amn);
}`;

// y[i] = x[i] * w[i] (+ 1 when OFFSET): scale a vector by a norm weight
export const scaleBy = (offset) => /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < p.n) { y[i] = x[i] * (${offset ? "1.0 + w[i]" : "w[i]"}); }
}`;
