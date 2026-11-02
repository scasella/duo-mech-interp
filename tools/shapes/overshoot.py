# Where does straight-line steering break? (Goodfire's "sweet spot": push too little and nothing happens, too much and
# the model turns to gibberish.) At the weekday word, push Monday along the Monday->Friday chord to alpha times its
# length, or keep following the real days round the week for the same number of steps, and read out: mass on day
# words, KL from the unsteered next-token distribution, and a short greedy continuation.
#   python tools/shapes/overshoot.py <model dir> <probe dir> <out.json> [layer]
import json, math, os, sys
import numpy as np, torch
model_dir, probe_dir, out = sys.argv[1:4]
L = int(sys.argv[4]) if len(sys.argv) > 4 else 9
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
E = lm.embed_tokens.weight.detach()
meta = json.load(open(f"{probe_dir}/probes.json"))["weekday"]
days = meta["vals"]; V = len(days)
vid = [tok(" " + d, add_special_tokens=False).input_ids[0] for d in days]
R = np.load(f"{probe_dir}/probe_weekday_L{L}.npz")["resid"].astype(np.float32)
mu = (R - R.mean(1, keepdims=True)).mean(0)
st = dict(delta=None, pos=None)
def hook(mod, inp, outp):
    if st["delta"] is None: return
    h = (outp[0] if isinstance(outp, tuple) else outp).clone()
    if h.shape[1] > st["pos"]: h[:, st["pos"]] += st["delta"]               # only while the prompt is processed
    return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
lm.layers[L].register_forward_hook(hook)
ALPHAS = [1, 1.5, 2, 3, 4, 6, 8, 12]
STEPS = 4                                                                   # Monday -> Friday
res = {}
for task, tpl, shift in (("next", "Today is {X}. Tomorrow is", 1), ("copy", "The meeting was moved to {X}, so the meeting is now on", 0)):
    rows = {"chord": [], "through": []}
    for s in range(V):
        ids = tok(tpl.replace("{X}", days[s]), add_special_tokens=False).input_ids
        st["pos"] = ids.index(vid[s]); st["delta"] = None
        with torch.inference_mode():
            base = torch.log_softmax(lm(input_ids=torch.tensor([ids])).last_hidden_state[:, -1] @ E.T, -1)[0]
        for path in ("chord", "through"):
            for a in ALPHAS:
                k = a * STEPS
                if path == "chord":
                    point = mu[s] + a * (mu[(s + STEPS) % V] - mu[s])
                    want = None
                else:
                    j0 = math.floor(k); f = k - j0
                    point = (1 - f) * mu[(s + j0) % V] + f * mu[(s + j0 + 1) % V]
                    want = (s + round(k) + shift) % V
                st["delta"] = torch.tensor(point - mu[s])
                with torch.inference_mode():
                    lp = torch.log_softmax(lm(input_ids=torch.tensor([ids])).last_hidden_state[:, -1] @ E.T, -1)[0]
                    gen = m.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False) if s == 0 else None
                st["delta"] = None
                p = lp.exp()
                rows[path].append(dict(alpha=a, source=days[s], mass=float(p[vid].sum()), kl=float((p * (lp - base)).sum()),
                                       top_is_day=bool(int(p.argmax()) in vid), top=tok.decode([int(p.argmax())]),
                                       right=None if want is None else bool(int(p[vid].argmax()) == want),
                                       text=tok.decode(gen[0, len(ids):]) if gen is not None else None))
    res[task] = rows
    print(f"\n{task} (layer {L}): mean over 7 source days")
    for path in ("chord", "through"):
        print(f"  {path}")
        for a in ALPHAS:
            rs = [r for r in rows[path] if r["alpha"] == a]
            ex = next(r for r in rs if r["text"] is not None)
            right = "" if path == "chord" else f"  right day {np.mean([r['right'] for r in rs]):.0%}"
            print(f"    x{a:<4} day mass {np.mean([r['mass'] for r in rs]):.2f}  KL {np.mean([r['kl'] for r in rs]):5.2f}  "
                  f"top token is a day {np.mean([r['top_is_day'] for r in rs]):.0%}{right}  | Monday -> {ex['text']!r}", flush=True)
json.dump(res, open(out, "w"))
