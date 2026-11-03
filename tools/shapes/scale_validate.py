# Validate candidate scales found by discover.py: does the model hold the words in the hypothesised order, and is the
# scale a straight line? (On a straight line, a straight push from one end to the other passes through the middle
# words in order. On a curve, like the weekday circle, it skips them.)
#   python tools/shapes/scale_validate.py <model dir> <out dir> [layers]
# Steps: template sentences for each scale (residual at the scale word, per layer); order of the per-value averages
# on their top principal axis vs the hypothesised order; then, with a copy prompt, a straight push from the first
# word's average to the last word's (and back), reading which scale word the model would repeat at each point.
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

SCALES = {
    "how soon": dict(order="linear", vals="someday eventually later soon shortly immediately now".split(), templates=[
        "We will fix the problem {X}.", "They promised to call {X}.", "The results will arrive {X}.", "I will finish the report {X}.",
        "She plans to move abroad {X}.", "The bridge will reopen {X}.", "He said he would apologize {X}.",
        "The company will announce the decision {X}."], copy="She said they would leave {X}, and he agreed they would leave"),
    "outbreak size": dict(order="linear", vals="case cluster outbreak epidemic pandemic".split(), templates=[
        "Health officials announced a new {X} on Monday.", "The hospital reported a {X} of the disease.", "Scientists studied the {X} closely.",
        "The {X} spread quickly.", "Officials confirmed the {X} last week.", "The report described the {X} in detail.",
        "Doctors were alarmed by the {X}.", "The town was shaken by the {X}."], copy="Officials first called it a {X}, and the report also called it a"),
    "speed": dict(order="linear", vals="sluggish slow moderate steady brisk fast rapid".split(), templates=[
        "The car moved at a {X} pace.", "The project made {X} progress.", "The river had a {X} current.", "He walked at a {X} speed.",
        "The economy saw {X} growth.", "The train kept a {X} pace.", "The dancers moved in {X} steps.", "The recovery was {X}."],
        copy="The car moved at a {X} pace, and the bus also moved at a"),
    "togetherness": dict(order="linear", vals="alone separately individually independently jointly together collectively".split(), templates=[
        "The students worked {X} on the project.", "They decided to travel {X}.", "The partners signed the contract {X}.",
        "The team members prepared {X}.", "The families celebrated {X}.", "We will review the proposals {X}.",
        "The researchers wrote the paper {X}.", "The brothers lived {X} for years."],
        copy="The students worked {X} on the project, and the teachers also worked"),
}
ALPHAS = np.linspace(0, 1, 13)

model_dir, out = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "6,12").split(",")]
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
os.makedirs(out, exist_ok=True)
tok, lm_probe = core.load_model("cpu", torch.float32, max(LAYERS), path=model_dir)
fams = {k: dict(order=v["order"], vals=v["vals"], templates=v["templates"]) for k, v in SCALES.items()}
meta = core.probe(tok, lm_probe, {}, out, "cpu", families=fams, layers=LAYERS)
del lm_probe
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
E = lm.embed_tokens.weight.detach()
st = dict(delta=None, pos=None, layer=None)
def hook(l):
    def f(mod, inp, outp):
        if st["delta"] is None or st["layer"] != l: return
        h = (outp[0] if isinstance(outp, tuple) else outp).clone(); h[:, st["pos"]] += st["delta"]
        return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
    return f
for l in LAYERS: lm.layers[l].register_forward_hook(hook(l))

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

res = {}
for name, F in SCALES.items():
    vals = meta[name]["vals"]; V = len(vals)
    vid = [tok(" " + v, add_special_tokens=False).input_ids[0] for v in vals]
    for L in LAYERS:
        R = np.load(f"{out}/probe_{name}_L{L}.npz")["resid"].astype(np.float32)
        mu = (R - R.mean(1, keepdims=True)).mean(0)
        Vt = np.linalg.svd(mu - mu.mean(0), full_matrices=False)[2]
        pcs = (mu - mu.mean(0)) @ Vt[:3].T
        rho = [spearman(pcs[:, k], np.arange(V)) for k in range(3)]
        null = [abs(spearman(np.random.default_rng(i).permutation(V), np.arange(V))) for i in range(2000)]
        p = float(np.mean(np.array(null) >= abs(rho[0]) - 1e-9))
        walks = []
        for s, t in ((0, V - 1), (V - 1, 0)):
            ids = tok(F["copy"].replace("{X}", vals[s]), add_special_tokens=False).input_ids
            st["pos"], st["layer"] = ids.index(vid[s]), L
            st["delta"] = torch.tensor(np.stack([a * (mu[t] - mu[s]) for a in ALPHAS]), dtype=torch.float32)
            with torch.inference_mode():
                h = lm(input_ids=torch.tensor([ids] * len(ALPHAS))).last_hidden_state[:, -1]
            st["delta"] = None
            q = torch.softmax(h @ E.T, -1)[:, vid].numpy()
            walks.append(dict(start=vals[s], words=[vals[int(i)] for i in q.argmax(1)], mass=[round(float(x), 3) for x in q.sum(1)],
                              idx=[int(i) for i in q.argmax(1)]))
        # does the straight line pass through the middle words in order?
        inner = set(range(1, V - 1))
        visited = sorted(set(walks[0]["idx"]) & inner)
        mono = spearman(np.arange(len(ALPHAS)), np.array(walks[0]["idx"])) if len(set(walks[0]["idx"])) > 1 else 0.0
        res.setdefault(name, {})[L] = dict(vals=vals, rho_pc=[round(r, 3) for r in rho], p_pc1=round(p, 4), walks=walks,
                                           visited=[vals[i] for i in visited], monotone=round(mono, 3))
        print(f"{name:13s} L{L:2d}  order on top axis: rank corr {rho[0]:+.2f} (p {p:.3f}; next axes {rho[1]:+.2f} {rho[2]:+.2f})")
        for w in walks:
            path = [w["words"][0]] + [x for i, x in enumerate(w["words"][1:], 1) if x != w["words"][i - 1]]
            print(f"      straight push from '{w['start']}': {' → '.join(path)}   (scale-word mass {min(w['mass']):.2f}-{max(w['mass']):.2f})")
        print(f"      middle words reached on the way: {len(visited)}/{V - 2}  {res[name][L]['visited']}")
json.dump(res, open(f"{out}/scale_validate.json", "w"), indent=1)
