# Steering along a concept's shape versus straight through it (the claim from Goodfire's interview: a contrastive
# vector "cuts through the middle of the circle", which is off the manifold and out of distribution for the model).
#
#   python tools/shapes/manifold_steer.py <model dir> <probe dir> <time-unit probe dir> <out.json> [layers]
#
# At the family word of a prompt ("Today is Monday. Tomorrow is"), the residual is moved from the source value's
# average (Monday) toward a target (Friday) by alpha in [0, 2] along four paths:
#   chord        straight line between the two averages in the full space (a contrastive / difference-of-means push)
#   through      piecewise line through the real intermediate averages (Tuesday, Wednesday, Thursday), wrapping round
#   plane-chord  the chord restricted to the family's top-2 principal plane (the rest of the residual is untouched)
#   plane-arc    the circle in that plane: angle and radius interpolated through the real values (for a line-shaped
#                family, the curve through the values' projections)
# Both full-space paths end at the target's average at alpha = 1, as do both plane paths. Past 1 they overshoot:
# chords keep going straight, the shape-following paths keep going round the circle (or stop at the end of a line).
# Measured from the next-token distribution: mass on the family's words (does it still answer with a day?), and
# whether the most likely family word is the value the path is passing through at that point.
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

model_dir, probe_dir, time_dir, out = sys.argv[1:5]
LAYERS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "6,9,12").split(",")]
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
ALPHAS = [0, .125, .25, .375, .5, .625, .75, .875, 1, 1.25, 1.5, 1.75, 2]
PATHS = ["chord", "through", "plane-chord", "plane-arc"]
FAMS = {
    "weekday": dict(dir=probe_dir, steps=4, tasks={"copy": ("The meeting was moved to {X}, so the meeting is now on", 0),
                                                   "next": ("Today is {X}. Tomorrow is", 1)}),
    "month": dict(dir=probe_dir, steps=6, tasks={"next": ("This month is {X}. Next month is", 1),
                                                 "copy": ("The festival was moved to {X}, so it now takes place in", 0)}),
    "time_unit": dict(dir=time_dir, steps=2, tasks={"copy": ("The timer was set for one {X}, so it will ring in one", 0),
                                                    "next": ("One {X} is shorter than one", 1)}),
}

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
E = lm.embed_tokens.weight.detach()
st = dict(delta=None, pos=None, layer=None)
def hook(l):
    def f(mod, inp, outp):
        if st["delta"] is None or st["layer"] != l: return
        h = (outp[0] if isinstance(outp, tuple) else outp).clone()
        h[:, st["pos"]] += st["delta"]
        return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
    return f
for l in LAYERS: lm.layers[l].register_forward_hook(hook(l))


def paths_for(mu, cyclic, s, steps):
    """Points (len(ALPHAS) x 4 paths x d) moving from value s toward s+steps."""
    V = len(mu)
    Vt = np.linalg.svd(mu - mu.mean(0), full_matrices=False)[2][:2]          # family plane
    c = (mu - mu.mean(0)) @ Vt.T
    ang = np.unwrap(np.arctan2(c[:, 1], c[:, 0])); rad = np.linalg.norm(c, axis=1)
    sgn = np.sign(ang[-1] - ang[0]) or 1.0
    def at(j):                                                               # value index j (may pass the end)
        if cyclic: return mu[j % V], ang[j % V] + 2 * math.pi * sgn * (j // V), rad[j % V], c[j % V]
        j = min(max(j, 0), V - 1); return mu[j], None, None, c[j]
    t = (s + steps) % V if cyclic else min(s + steps, V - 1)
    out = np.zeros((len(ALPHAS), len(PATHS), mu.shape[1]), np.float32)
    for ai, a in enumerate(ALPHAS):
        k = a * steps; j0 = math.floor(k); f = k - j0
        m0, a0, r0, c0 = at(s + j0); m1, a1, r1, c1 = at(s + j0 + 1)
        out[ai, 0] = mu[s] + a * (mu[t] - mu[s])                              # chord (extrapolates past t)
        out[ai, 1] = (1 - f) * m0 + f * m1                                   # through the real values
        out[ai, 2] = mu[s] + a * ((mu[t] - mu[s]) @ Vt.T) @ Vt                # chord in the plane
        if cyclic:
            th, r = (1 - f) * a0 + f * a1, (1 - f) * r0 + f * r1
            cc = r * np.array([math.cos(th), math.sin(th)])
        else:
            cc = (1 - f) * c0 + f * c1
        out[ai, 3] = mu[s] + (cc - c[s]) @ Vt                                # arc / curve in the plane
    return out


def expected(s, a, steps, V, cyclic, shift):
    j = s + round(a * steps)
    if not cyclic:
        j = min(j, V - 1)
        return j + shift if j + shift < V else None
    return (j + shift) % V


res = {}
for fam, F in FAMS.items():
    meta = json.load(open(f"{F['dir']}/probes.json"))[fam]
    vals = meta["vals"]; V = len(vals); cyclic = meta["order"] == "cyclic"
    vid = [tok(" " + v, add_special_tokens=False).input_ids[0] for v in vals]
    for L in LAYERS:
        R = np.load(f"{F['dir']}/probe_{fam}_L{L}.npz")["resid"].astype(np.float32)
        mu = (R - R.mean(1, keepdims=True)).mean(0)
        spacing = float(np.mean([np.linalg.norm(mu[(i + 1) % V] - mu[i]) for i in range(V if cyclic else V - 1)]))
        sources = list(range(V)) if cyclic else list(range(V - 2 * F["steps"]))
        for task, (tpl, shift) in F["tasks"].items():
            acc = {p: {a: dict(mass=0.0, hit=0, n=0, cond=0.0) for a in ALPHAS} for p in PATHS}
            off = {p: {a: 0.0 for a in ALPHAS} for p in PATHS}
            for s in sources:
                ids = tok(tpl.replace("{X}", vals[s]), add_special_tokens=False).input_ids
                st["pos"], st["layer"] = ids.index(vid[s]), L
                P = paths_for(mu, cyclic, s, F["steps"])                     # (alphas, paths, d)
                delta = torch.tensor(P - mu[s]).reshape(-1, mu.shape[1])
                st["delta"] = delta
                with torch.inference_mode():
                    h = lm(input_ids=torch.tensor([ids] * len(delta))).last_hidden_state[:, -1]
                st["delta"] = None
                prob = torch.softmax(h @ E.T, -1)[:, vid].numpy().reshape(len(ALPHAS), len(PATHS), V)
                for ai, a in enumerate(ALPHAS):
                    want = expected(s, a, F["steps"], V, cyclic, shift)
                    for pi, p in enumerate(PATHS):
                        q = prob[ai, pi]
                        d = acc[p][a]; d["mass"] += float(q.sum()); d["n"] += 1
                        if want is not None:
                            d["hit"] += int(q.argmax() == want); d["cond"] += float(q[want] / max(q.sum(), 1e-9))
                        off[p][a] += float(np.min(np.linalg.norm(mu - P[ai, pi], axis=1)) / spacing)
            summ = {p: {str(a): dict(mass=round(acc[p][a]["mass"] / acc[p][a]["n"], 3),
                                     hit=round(acc[p][a]["hit"] / acc[p][a]["n"], 3),
                                     cond=round(acc[p][a]["cond"] / acc[p][a]["n"], 3),
                                     off=round(off[p][a] / acc[p][a]["n"], 2)) for a in ALPHAS} for p in PATHS}
            res.setdefault(fam, {}).setdefault(task, {})[L] = summ
            print(f"\n{fam} {task} L{L}  (from {vals[sources[0]]} toward {vals[(sources[0] + F['steps']) % V]}, "
                  f"{len(sources)} sources)  cells: family-word mass / right value / distance to nearest real value")
            print("   alpha " + " ".join(f"{a:>13}" for a in ALPHAS))
            for p in PATHS:
                print(f"   {p:11s}" + " ".join(f"{summ[p][str(a)]['mass']:.2f}/{summ[p][str(a)]['hit']:.2f}/{summ[p][str(a)]['off']:.1f}".rjust(13) for a in ALPHAS), flush=True)
json.dump(res, open(out, "w"))
