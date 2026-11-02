// Byte-level BPE tokenizer for Qwen tokenizer.json files (NFC, regex split, GPT-2 byte map, ranked merges), plus the
// Qwen3.5 chat template with thinking disabled. Checked against Hugging Face `tokenizers` by tests/tokenizer.mjs.

const byteToChar = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const m = new Array(256);
  bs.forEach((b, i) => (m[b] = String.fromCodePoint(cs[i])));
  return m;
})();
const charToByte = new Map(byteToChar.map((c, b) => [c, b]));

export class Tokenizer {
  constructor(json) {
    const m = json.model;
    if (m.type !== "BPE") throw new Error("only BPE tokenizers are supported");
    this.vocab = new Map(Object.entries(m.vocab));
    this.ranks = new Map();
    m.merges.forEach((mg, i) => this.ranks.set(Array.isArray(mg) ? mg.join(" ") : mg, i));
    this.idToTok = [];
    for (const [t, id] of this.vocab) this.idToTok[id] = t;
    this.special = new Map();
    for (const a of json.added_tokens) { this.special.set(a.content, a.id); this.idToTok[a.id] = a.content; }
    this.specialIds = new Set(json.added_tokens.filter(a => a.special).map(a => a.id));
    const pre = json.pre_tokenizer?.pretokenizers?.find(p => p.type === "Split")?.pattern?.Regex;
    // JS lacks the (?i:...) modifier in older engines: spell the contractions out
    const src = (pre || "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
      .replace("(?i:'s|'t|'re|'ve|'m|'ll|'d)", "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]");
    this.split = new RegExp(src, "gu");
    const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    this.specialRe = new RegExp("(" + [...this.special.keys()].sort((a, b) => b.length - a.length).map(esc).join("|") + ")");
    this.cache = new Map();
    this.enc = new TextEncoder();
    this.size = this.idToTok.length;
  }

  static async fromUrl(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`tokenizer download failed: ${r.status}`);
    return new Tokenizer(await r.json());
  }

  _bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const merged = parts[at] + parts[at + 1];
      const next = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] + " " + parts[i + 1] === parts[at] + " " + parts[at + 1]) { next.push(merged); i++; }
        else next.push(parts[i]);
      }
      parts = next;
    }
    const ids = parts.map(p => this.vocab.get(p));
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  // text -> ids; special tokens (e.g. <|im_start|>) in the text are recognised
  encode(text, { special = true } = {}) {
    const out = [];
    const pieces = special ? text.split(this.specialRe) : [text];
    for (const piece of pieces) {
      if (!piece) continue;
      if (special && this.special.has(piece)) { out.push(this.special.get(piece)); continue; }
      const norm = piece.normalize("NFC");
      for (const m of norm.matchAll(this.split)) {
        const bytes = this.enc.encode(m[0]);
        let w = "";
        for (const b of bytes) w += byteToChar[b];
        for (const id of this._bpe(w)) out.push(id);
      }
    }
    return out;
  }

  // ids -> bytes of their text (special tokens included unless skipSpecial)
  bytes(ids, skipSpecial = true) {
    const out = [];
    for (const id of ids) {
      if (skipSpecial && this.specialIds.has(id)) continue;
      const t = this.idToTok[id];
      if (t === undefined) continue;
      if (this.special.has(t)) { out.push(...this.enc.encode(t)); continue; }
      for (const ch of t) { const b = charToByte.get(ch); if (b !== undefined) out.push(b); }
    }
    return new Uint8Array(out);
  }

  decode(ids, skipSpecial = true) { return new TextDecoder().decode(this.bytes(ids, skipSpecial)); }

  // display string for one token (byte pieces of a split character show as hex)
  piece(id) {
    const b = this.bytes([id], false);
    const s = new TextDecoder("utf-8", { fatal: false }).decode(b);
    return s.includes("�") ? [...b].map(x => "<" + x.toString(16).padStart(2, "0") + ">").join("") : s;
  }

  // Groups token ids into displayable units: the byte pieces of one multi-byte character join the token that
  // completes it. Returns [{text, idx: [token indices]}].
  units(ids) {
    const units = [];
    let pending = [], prev = "";
    const all = [];
    for (let i = 0; i < ids.length; i++) {
      all.push(ids[i]);
      pending.push(i);
      const full = this.decode(all, false);
      if (full.endsWith("�") && i < ids.length - 1) continue;
      units.push({ text: full.startsWith(prev) ? full.slice(prev.length) : this.decode(pending.map(j => ids[j]), false), idx: pending });
      prev = full; pending = [];
    }
    return units;
  }

  id(tok) { return this.special.get(tok) ?? this.vocab.get(tok); }

  // Qwen3.5 chat template, thinking disabled (matches apply_chat_template(..., enable_thinking=False))
  chat(messages, { generation = true } = {}) {
    let s = "";
    for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
    if (generation) s += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return s;
  }
}
