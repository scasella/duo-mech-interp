// Scales: word families whose values the model holds in order (tools/shapes/family_sweep.py), steered by adding a
// value's average to the residual stream. scales.json lists them; scales.bin holds the vectors as float16, one per
// value, in the order listed (tools/export_scales.py writes both).

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

export async function loadScales(base = new URL("./", import.meta.url).href) {
  const [meta, buf] = await Promise.all([
    fetch(base + "scales.json").then(r => { if (!r.ok) throw new Error(`scales.json: ${r.status}`); return r.json(); }),
    fetch(base + "scales.bin").then(r => { if (!r.ok) throw new Error(`scales.bin: ${r.status}`); return r.arrayBuffer(); }),
  ]);
  const H = meta.hidden, raw = new Uint16Array(buf);
  const byId = new Map(meta.scales.map(s => [s.id, s]));
  // average of value i of scale id, times strength
  const vec = (id, i, strength = 1) => {
    const s = byId.get(id), out = new Float32Array(H), o = (s.offset + i) * H;
    for (let h = 0; h < H; h++) out[h] = f16(raw[o + h]) * strength;
    return out;
  };
  // the same fraction of the way along the straight line from the first value's average to the last one's
  const chord = (id, i, strength = 1) => {
    const s = byId.get(id), V = s.vals.length, t = i / (V - 1);
    const a = vec(id, 0), b = vec(id, V - 1), out = new Float32Array(H);
    for (let h = 0; h < H; h++) out[h] = (a[h] + t * (b[h] - a[h])) * strength;
    return out;
  };
  return { list: meta.scales, get: id => byId.get(id), vec, chord, hidden: H };
}
