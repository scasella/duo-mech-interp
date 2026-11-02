# python tools/find_steering.py <local Qwen3.5-0.8B checkpoint dir> "<topic>" <out.json>
#
# Reference (PyTorch, float32) for the app's "Steer toward" search (engine/analysis.js: concept, shortlist, probe,
# preview), and a comparison with the naive approach (neurons that fire hardest on the topic word):
#   1. topic words + nearest whole-word neighbours in the tied embedding space
#   2. every neuron's down column through the final norm and unembedding, scored against those words (centred),
#      z-scored per layer, top 3 per band of 4 layers and sign
#   3. each candidate pushed (0.8) in two short plain-text contexts: probability mass and log-odds gained by the
#      topic words, KL; recommended strength aims for KL about 0.6-0.8
#   4. short stories with the top candidates at their recommended strength (and the naive ones at 0.8): topic-word
#      hits and repeated word pairs
# Needs: pip install torch transformers regex
import sys, json, re, torch, torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
torch.set_num_threads(4)
path, concept = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)
m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
lm = m.model.language_model
L, I = len(lm.layers), lm.layers[0].mlp.down_proj.in_features
E = lm.embed_tokens.weight.detach()                         # [V, H], tied with lm_head
g = 1.0 + lm.norm.weight.detach()                           # final norm gain
REF = ("The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
       "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. ")
resid = {}
hs = [lm.layers[i].register_forward_hook(lambda mod, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0, 1:].norm(dim=-1).median().item())) for i in range(L)]
with torch.inference_mode(): lm(input_ids=torch.tensor([tok(REF).input_ids]))
for h in hs: h.remove()
push, cap = {}, {}
def ph(i):
    def f(mod, inp):
        x = inp[0]
        if i in cap: cap[i] = x[0, -1].clone()
        if i in push:
            x = x.clone()
            for n, v in push[i]: x[..., n] += v
            return (x,)
    return f
for i in range(L): lm.layers[i].mlp.down_proj.register_forward_pre_hook(ph(i))
Wd = [lm.layers[l].mlp.down_proj.weight.detach() for l in range(L)]   # [H, I]
dn = torch.stack([w.norm(dim=0) for w in Wd])                          # [L, I]
unit = lambda l, n: resid[l] / dn[l, n].item()

# 1. concept tokens + neighbours in embedding space
STOP = set("the a an of to and in on about more less be sound like talk make model it its".split())
words = [w for w in re.findall(r"[\w']+", concept.lower()) if w not in STOP] or [concept]
q_ids = []
for w in words:
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        ids = tok(v, add_special_tokens=False).input_ids
        if len(ids) == 1 or (ids and len(tok.decode([ids[0]]).strip()) >= 3): q_ids.append(ids[0])
q_ids = sorted(set(q_ids))
En = E / E.norm(dim=1, keepdim=True)
q = En[q_ids].mean(0); q = q / q.norm()
sim = En @ q
top = sim.topk(80)
import regex
def wordlike(s):
    return bool(regex.fullmatch(r" \p{L}[\p{L}'-]+", s) or regex.fullmatch(r"\p{Lu}\p{Ll}+", s) or regex.fullmatch(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]+", s))
targets = {}
for t in q_ids: targets[t] = 1.0
for s_, t in zip(top.values.tolist(), top.indices.tolist()):
    if len(targets) >= 24: break
    if t in targets or not wordlike(tok.decode([t])): continue
    targets[t] = max(0.0, s_ / top.values[0].item())
print("targets:", [tok.decode([t]) for t in targets])
tid = torch.tensor(list(targets)); tw = torch.tensor(list(targets.values()))

# 2. write alignment of every neuron: centred target direction through the final norm, cos with down column
u = (tw[:, None] * (E[tid] * g)).sum(0) / tw.sum() - (E * g).mean(0)
align = torch.stack([(u @ Wd[l]) / dn[l] for l in range(L)])          # [L, I]
z = (align - align.mean(1, keepdim=True)) / align.std(1, keepdim=True)
cands = []
for lo in range(0, L, 4):                                               # per band of 4 layers, both signs
    band = z[lo:lo + 4]
    for sign in (1, -1):
        vals, idx = (sign * band).flatten().topk(3)
        cands += [(lo + int(i) // I, int(i) % I, sign) for i in idx]
# naive: neurons that fire hardest on the concept word itself
cap.update({i: None for i in range(L)})
with torch.inference_mode(): lm(input_ids=torch.tensor([tok(" " + words[0]).input_ids]))
acts = torch.stack([cap[i] for i in range(L)]); cap.clear()
naive_vals, naive_idx = (acts.abs() / acts.abs().amax(1, keepdim=True)).flatten().topk(6)
naive = [(int(i) // I, int(i) % I, 1 if acts.flatten()[i] > 0 else -1) for i in naive_idx]

# 3. causal probe: push (sign-aware, 0.8) on two plain contexts, target log-odds gain and KL
CTX = ["My favorite thing in the world is", "Today I want to tell you about"]
def dist(pushes):
    push.clear(); push.update(pushes)
    with torch.inference_mode():
        out = [F.log_softmax(m(input_ids=torch.tensor([tok(c).input_ids])).logits[0, -1], -1) for c in CTX]
    push.clear()
    return out
base = dist({})
def tlp(lp): return torch.logsumexp(lp[tid] + tw.log(), 0)
def probe(l, n, sign, dose=0.8):
    lps = dist({l: [(n, sign * dose * unit(l, n))]})
    gain = sum(float(tlp(a) - tlp(b)) for a, b in zip(lps, base)) / len(CTX)
    dp = sum(float(a[tid].exp().sum() - b[tid].exp().sum()) for a, b in zip(lps, base)) / len(CTX)
    kl = sum(float((a.exp() * (a - b)).sum()) for a, b in zip(lps, base)) / len(CTX)
    d = sum(a - b for a, b in zip(lps, base))
    top = [tok.decode([t]) for t in d.topk(40).indices.tolist() if wordlike(tok.decode([t]))][:6]
    return gain, kl, top, dp
res = []
for l, n, s in cands + naive:
    gain, kl, top, dp = probe(l, n, s)
    rec = float(min(1.2, max(0.2, 0.8 * (0.6 / max(kl, 1e-6)) ** 0.5)))
    res.append({"layer": l, "n": n, "sign": s, "gain": gain, "kl": kl, "dp": dp, "rec": rec, "top": top, "src": "naive" if (l, n, s) in naive else "align"})
score = lambda r: r["dp"] * (1 if r["kl"] < 2.5 else 0.5) + 0.02 * r["gain"]
ranked = sorted([r for r in res if r["src"] == "align"], key=score, reverse=True)
for r in ranked[:8] + [r for r in res if r["src"] == "naive"]:
    print(f'{r["src"]:5} L{r["layer"]:2} N{r["n"]:4} {"+" if r["sign"]>0 else "-"}  dp {r["dp"]:.3f} gain {r["gain"]:5.2f}  kl {r["kl"]:5.2f} rec {r["rec"]:.2f}  {r["top"]}')

# 4. does it steer a whole answer? generate a short story with the top 3 probed vs top 3 naive
def gen(l, n, sign, prompt="Tell me a short story.", new=48, dose=0.8):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"]
    push.clear()
    if n is not None: push[l] = [(n, sign * dose * unit(l, n))]
    with torch.inference_mode():
        o = m.generate(input_ids=ids, max_new_tokens=new, do_sample=False)
    push.clear()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
tw_set = {tok.decode([t]).strip().lower() for t in targets}
def hits(text): return sum(1 for w in re.findall(r"[\w']+", text.lower()) if w in tw_set)
print("\nbaseline:", repr(gen(0, None, 1)[:200]))
out = {"concept": concept, "targets": [tok.decode([t]) for t in targets], "probed": [], "naive": []}
def distinct(text):
    w = re.findall(r"\w+", text.lower())
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / max(1, len(bg))
for name, lst in (("probed", ranked[:5]), ("naive", [r for r in res if r["src"] == "naive"][:2])):
    for r in lst:
        d = r["rec"] if name == "probed" else 0.8
        txt = gen(r["layer"], r["n"], r["sign"], dose=d)
        out[name].append({**r, "story": txt, "hits": hits(txt), "distinct": distinct(txt), "dose": d})
        print(f'{name} L{r["layer"]} N{r["n"]} {"+" if r["sign"]>0 else "-"} @{d:.2f} hits={hits(txt)} distinct={distinct(txt):.2f}: {txt[:170]!r}')
json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
