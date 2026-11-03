# python tools/find_scales.py <local Qwen3.5-0.8B checkpoint dir> <family_sweep run dir> <out.json> [layers] [strengths]
#
# Reference (PyTorch, float32) for the app's scale steering: add one value's average (from tools/shapes/family_sweep.py,
# template-centred, so it is "this value minus the family average") to the residual stream after a layer, at every
# position of a chat turn, and generate. Two paths through a scale:
#   path   the value's own average (the real words)
#   chord  the same fraction of the way along the straight line from the first value's average to the last one's
# For each family, value, layer and strength: does the answer mention that value (word match, plurals allowed), which
# family value it mentions first, and how repetitive it gets (share of repeated word pairs).
# Needs: pip install torch transformers
import json, os, re, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes"))
from family_sweep import FAMILIES

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
DEV = os.environ.get("DEVICE", "cpu")
path, run, out = sys.argv[1:4]
LAYERS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "6,12").split(",")]
STRENGTHS = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,4").split(",")]
PROMPTS = {
    "wind": "What is the weather like at the coast today? Answer in one sentence.",
    "settlement": "Describe the place where you grew up in one sentence.",
    "water body": "Where did you go swimming last summer? Answer in one sentence.",
    "geographic scale": "How far did the news spread? Answer in one sentence.",
    "nobility": "Who lives in the castle on the hill? Answer in one sentence.",
    "corporate rank": "Who approved the new budget at your company? Answer in one sentence.",
    "life stage": "Who is sitting in the waiting room? Answer in one sentence.",
    "letter grade": "What grade did you get on your math test? Answer in one sentence.",
    "school level": "What kind of school does your sister teach at? Answer in one sentence.",
    "schedule": "How often is the newsletter published? Answer in one sentence.",
    "closeness": "Who did you have dinner with last night? Answer in one sentence.",
    "decade of life": "How old is your neighbor? Answer in one sentence.",
    "(temperature)": "How is the water in the lake today? Answer in one sentence.",
    "(size)": "How big is the house you live in? Answer in one sentence.",
    "(how soon)": "When will the repairs be finished? Answer in one sentence.",
    "(weekday)": "Which day is the meeting? Answer in one sentence.",
    "(month)": "When is your birthday? Answer in one sentence.",
}
only = os.environ.get("ONLY")
fams = [f for f in PROMPTS if not only or f in only.split(",")]

tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval().to(DEV)
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
st = {"vec": None, "layer": None}
def hook(l):
    def f(mod, inp, outp):
        if st["vec"] is None or st["layer"] != l: return
        h = outp[0] if isinstance(outp, tuple) else outp
        h = h + st["vec"][:, None, :]
        return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
    return f
for l in LAYERS: lm.layers[l].register_forward_hook(hook(l))
chat = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# typical residual norm per layer (median over the tokens of a neutral chat turn), to report push sizes
norms = {}
cap = {}
hs = [lm.layers[l].register_forward_hook(lambda mod, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].norm(dim=-1).median().item())) for l in LAYERS]
with torch.inference_mode(): lm(input_ids=torch.tensor([tok(chat("Tell me about your day."), add_special_tokens=False).input_ids], device=DEV))
for h in hs: h.remove()
norms = dict(cap)


def mentions(text, vals):
    t = text.lower(); found = []
    for i, v in enumerate(vals):
        mm = re.search(r"(?<![a-z])" + re.escape(v.lower()) + r"(e?s)?(?![a-z])", t)
        if mm: found.append((mm.start(), i))
    return [i for _, i in sorted(found)]


def rep(text):
    w = re.findall(r"[a-z']+", text.lower()); pairs = list(zip(w, w[1:]))
    return 1 - len(set(pairs)) / len(pairs) if pairs else 0.0


def generate(q, vecs, L, n=40):
    ids = torch.tensor([tok(chat(q), add_special_tokens=False).input_ids] * len(vecs), device=DEV)
    st["vec"], st["layer"] = torch.tensor(np.stack(vecs), dtype=torch.float32, device=DEV), L
    with torch.inference_mode():
        o = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False)
    st["vec"] = None
    return [tok.decode(r[ids.shape[1]:], skip_special_tokens=True) for r in o]


res = dict(norms=norms, families={})
for f in fams:
    vals = FAMILIES[f]["vals"]; V = len(vals); cyc = FAMILIES[f]["order"] == "cyclic"
    base = generate(PROMPTS[f], [np.zeros(lm.config.hidden_size)], LAYERS[0])[0]
    R = dict(prompt=PROMPTS[f], vals=vals, base=base, base_mentions=[vals[i] for i in mentions(base, vals)], runs=[])
    print(f"\n== {f}: {PROMPTS[f]}\n   as is: {base.strip()[:120]}")
    for L in LAYERS:
        mu = np.load(f"{run}/{f.replace(' ', '_')}_L{L}_means.npy").astype(np.float32)
        size = float(np.linalg.norm(mu, axis=1).mean())
        for s in STRENGTHS:
            texts = generate(PROMPTS[f], [s * mu[v] for v in range(V)], L)
            ends = (0, V // 2) if cyc else (0, V - 1)
            chord = [s * (mu[ends[0]] + (v - ends[0]) / (ends[1] - ends[0]) * (mu[ends[1]] - mu[ends[0]])) for v in range(ends[0], ends[1] + 1)]
            ctexts = generate(PROMPTS[f], chord, L)
            ment = [mentions(t, vals) for t in texts]; cment = [mentions(t, vals) for t in ctexts]
            hit = [v in ment[v] for v in range(V)]
            first = [ment[v][0] == v if ment[v] else False for v in range(V)]
            inner = range(ends[0] + 1, ends[1])
            row = dict(layer=L, strength=s, push_norm=round(s * size, 2), resid_norm=round(norms[L], 2),
                       hit=float(np.mean(hit)), first=float(np.mean(first)),
                       inner_path=float(np.mean([hit[v] for v in inner])) if len(inner) else None,
                       inner_chord=float(np.mean([v in cment[v - ends[0]] for v in inner])) if len(inner) else None,
                       rep=float(np.mean([rep(t) for t in texts])), texts=texts, chord_texts=ctexts)
            R["runs"].append(row)
            pct = lambda x: "n/a" if x is None else f"{x:.0%}"
            print(f"   L{L:2d} x{s:<3g} push {s * size:5.1f} (resid {norms[L]:.1f}): mentions own value {np.mean(hit):.0%}, first {np.mean(first):.0%}; "
                  f"middle values: path {pct(row['inner_path'])} vs chord {pct(row['inner_chord'])}; repetition {row['rep']:.2f}", flush=True)
            for v in (0, V // 2, V - 1):
                print(f"        {vals[v]:>12s}: {texts[v].strip()[:110]!r}")
    res["families"][f] = R
    json.dump(res, open(out, "w"), indent=1)
os._exit(0)
