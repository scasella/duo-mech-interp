# Causal test of a cross-domain alignment: push along one scale's axis (e.g. dirty -> clean, with the good-bad axis
# removed) while the model judges something in another domain ("Lying to protect yourself is ..."), and measure the shift
# in the next word between that domain's poles (log P(high words) - log P(low words)), against random pushes of the
# same size, the good-bad axis itself, and the target scale's own axis (an upper bound).
#   python tools/shapes/cross_steer.py <model dir> <words run dir> <HF cache or ""> <out.json> [layer] [src>tgt,...]
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from axis_atlas import SCALES
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir, run, cache, out = sys.argv[1:5]
L = int(sys.argv[5]) if len(sys.argv) > 5 else 12
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
TESTS = {   # target domain: prompts, low readout words, high readout words
    "morality": (["Stealing money from a charity is", "Lying to protect yourself is", "Helping a stranger in need is", "Cheating on an exam is",
                  "Telling the truth even when it hurts is", "The way the company treated its workers was", "In my opinion, what he did was",
                  "Breaking a promise to a friend is"],
                 "evil wicked immoral corrupt unethical dishonest cruel wrong", "moral righteous noble honest ethical decent honorable right"),
    "cleanliness": (["After the long hike, his boots were", "When we arrived, the hotel room was", "The water in the old well was",
                     "After the party, the kitchen was", "Her hands after gardening were", "The windows of the abandoned house were",
                     "The hospital floor was", "After the flood, the streets were"],
                    "dirty filthy messy dusty polluted muddy", "clean tidy pristine sterile fresh washed"),
    "mood": (["When she heard the news, she felt", "After the funeral, everyone was", "On the first day of summer, the children were",
              "He stared out of the window, feeling", "Winning the prize made him", "Lately I have been feeling", "The mood in the room was",
              "After the long winter, people felt"],
             "sad unhappy depressed miserable", "happy cheerful joyful glad"),
    "seriousness": (["The tone of the meeting was", "The judge's expression was", "Her speech at the ceremony was", "The atmosphere at the party was",
                     "His voice became", "The style of the email was", "The documentary was", "The comedian's act was"],
                    "playful silly casual humorous informal funny", "serious grave solemn stern formal strict"),
    "friendliness": (["The new neighbours were very", "The shopkeeper was", "When I asked for help, the receptionist was", "The dog seemed",
                      "The way he spoke to the waiter was", "Our tour guide was", "The locals were", "Her reply to my message was"],
                     "hostile rude cruel nasty harsh", "friendly kind gentle nice caring"),
}
PAIRS = [("cleanliness", "morality"), ("morality", "cleanliness"), ("brightness", "mood"), ("hardness", "seriousness"),
         ("temperature", "friendliness")]
if len(sys.argv) > 6: PAIRS = [tuple(x.split(">")) for x in sys.argv[6].split(",")]
ALPHAS = [1.0, 2.0, 4.0]
N_RAND = 24

tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
E = lm.embed_tokens.weight.detach()
st = {"add": None}
def hook(mod, inp, outp):
    if st["add"] is None: return
    h = outp[0] if isinstance(outp, tuple) else outp
    h = h + st["add"][:, None, :]
    return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
lm.layers[L].register_forward_hook(hook)

ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
row = {int(t): r for r, t in enumerate(ids)}
M = np.load(f"{run}/L{L}_word_means.npy").astype(np.float32)
pool = np.array([r for r in range(len(ids)) if counts[r] >= 50]); gm = M[pool].mean(0)
tok_s = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache or None)
unit = lambda v: v / np.linalg.norm(v)
def vec(w):
    got = [(M[row[t[0]]], counts[row[t[0]]]) for s in (" " + w, " " + w.capitalize())
           for t in [tok_s(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
    if not got: return None
    wt = np.array([c for _, c in got], float); return (np.stack([v for v, _ in got]) * wt[:, None]).sum(0) / wt.sum() - gm
def diff(name):
    lo, hi = SCALES[name]
    lo_v = np.stack([v for v in (vec(x) for x in lo.split()) if v is not None]); hi_v = np.stack([v for v in (vec(x) for x in hi.split()) if v is not None])
    return hi_v.mean(0) - lo_v.mean(0)
val = unit(diff("valence"))
def no_val(v): return v - (v @ val) * val
rng = np.random.default_rng(0)
def single(words):
    out = []
    for w in words.split():
        t = tok(" " + w, add_special_tokens=False).input_ids
        if len(t) == 1: out.append(t[0])
    return out

res = {}
for src, tgt in PAIRS:
    prompts, lo_w, hi_w = TESTS[tgt]
    lo_ids, hi_ids = single(lo_w), single(hi_w)
    d_src = diff(src); scale = float(np.linalg.norm(d_src))            # push size: the source scale's own pole gap
    axes = {"source (valence removed)": unit(no_val(d_src)), "source": unit(d_src), "valence": val,
            "target (upper bound)": unit(diff(tgt))}
    rand = [unit(rng.normal(size=M.shape[1]).astype(np.float32)) for _ in range(N_RAND)]
    names = list(axes) + [f"random{i}" for i in range(N_RAND)]
    vecs = list(axes.values()) + rand
    shifts = {n: {a: [] for a in ALPHAS} for n in names}
    for p in prompts:
        ptoks = tok(p, add_special_tokens=False).input_ids
        rows, meta = [], []
        for a in ALPHAS:
            for n, v in zip(names, vecs):
                rows.append(torch.tensor(a * scale * v, dtype=torch.float32)); meta.append((n, a))
        st["add"] = None
        with torch.inference_mode():
            lp0 = torch.log_softmax(lm(input_ids=torch.tensor([ptoks])).last_hidden_state[:, -1] @ E.T, -1)[0]
            st["add"] = torch.stack(rows)
            lp = torch.log_softmax(lm(input_ids=torch.tensor([ptoks] * len(rows))).last_hidden_state[:, -1] @ E.T, -1)
        st["add"] = None
        def lo_hi(l): return float(torch.logsumexp(l[..., hi_ids], -1).mean() - torch.logsumexp(l[..., lo_ids], -1).mean()) if l.dim() == 1 else (torch.logsumexp(l[:, hi_ids], -1) - torch.logsumexp(l[:, lo_ids], -1))
        base = float(torch.logsumexp(lp0[hi_ids], -1) - torch.logsumexp(lp0[lo_ids], -1))
        cur = lo_hi(lp)
        for k, (n, a) in enumerate(meta): shifts[n][a].append(float(cur[k]) - base)
    summ = {}
    print(f"\n{src} -> {tgt} (layer {L}; push = alpha x the {src} pole gap; shift in log-odds {tgt} high vs low words, mean over {len(prompts)} prompts)")
    for a in ALPHAS:
        rnd = np.array([np.mean(shifts[f"random{i}"][a]) for i in range(N_RAND)])
        line = []
        for n in axes:
            s = float(np.mean(shifts[n][a])); z = (s - rnd.mean()) / (rnd.std() + 1e-9)
            summ.setdefault(n, {})[a] = dict(shift=round(s, 3), z=round(z, 2))
            line.append(f"{n} {s:+.2f} (z {z:+.1f})")
        summ.setdefault("random", {})[a] = dict(mean=round(float(rnd.mean()), 3), sd=round(float(rnd.std()), 3))
        print(f"  x{a:<3} " + " | ".join(line) + f" | random {rnd.mean():+.2f} ± {rnd.std():.2f}")
    res[f"{src}->{tgt}"] = summ
json.dump(res, open(out, "w"), indent=1)
