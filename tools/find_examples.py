# python tools/find_examples.py <local Qwen3.5-0.8B checkpoint dir> <out.json>
# How app/examples.js was chosen: next-word KL under a +-0.8 push for 336 random neurons (layers 5-18), then greedy
# answers for the top ten. Read the answers and pick neurons whose push changes them visibly but readably.
# Finds neurons whose push visibly changes Qwen3.5-0.8B's answers, for the app's onboarding examples.
# Same dose unit as the app: dose x (median residual norm leaving the layer on REFERENCE_TEXT) / |down column|.
import json, sys, random, torch, torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
torch.set_num_threads(4)
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path)
m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
lm = m.model.language_model
L = len(lm.layers)
REF = ("The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
       "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
       "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
       "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. ")
resid = {}
def rhook(i):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        resid[i] = h[0, 1:].norm(dim=-1).median().item()
    return f
hs = [lm.layers[i].register_forward_hook(rhook(i)) for i in range(L)]
with torch.inference_mode():
    lm(input_ids=torch.tensor([tok(REF).input_ids[:128]]))
for h in hs: h.remove()
push = {}
def phook(i):
    def f(mod, inp):
        if i in push:
            h = inp[0].clone()
            for n, v in push[i]: h[..., n] += v
            return (h,)
    return f
for i in range(L): lm.layers[i].mlp.down_proj.register_forward_pre_hook(phook(i))
def unit(l, n): return resid[l] / lm.layers[l].mlp.down_proj.weight[:, n].norm().item()
def chat(p): return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
ids = torch.tensor([tok(chat("Who are you?")).input_ids])
with torch.inference_mode():
    base = F.log_softmax(m(input_ids=ids).logits[0, -1], -1)
random.seed(0)
cands = [(l, n) for l in range(5, 19) for n in random.sample(range(3584), 24)]
scores = []
with torch.inference_mode():
    for k, (l, n) in enumerate(cands):
        u = unit(l, n)
        kls = []
        for s in (1, -1):
            push.clear(); push[l] = [(n, s * 0.8 * u)]
            lp = F.log_softmax(m(input_ids=ids).logits[0, -1], -1)
            kls.append(float((lp.exp() * (lp - base)).sum()))
        scores.append((max(kls), l, n, kls))
        if k % 100 == 0: print("scanned", k, file=sys.stderr, flush=True)
push.clear()
scores.sort(reverse=True)
def greedy(p, l, n, dose, new=28):
    ids = tok(chat(p), return_tensors="pt").input_ids
    push.clear()
    if dose: push[l] = [(n, dose * unit(l, n))]
    with torch.inference_mode():
        o = m.generate(input_ids=ids, max_new_tokens=new, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    push.clear()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
open(sys.argv[2] + ".kl", "w").write(json.dumps(scores[:40]))
out = []
for kl, l, n, kls in scores[:10]:
    rec = {"layer": l, "n": n, "kl_up": kls[0], "kl_down": kls[1], "unit": unit(l, n), "runs": {}}
    for p in ("Who are you?", "Describe your perfect day."):
        rec["runs"][p] = [greedy(p, l, n, d) for d in (0.8, 0, -0.8)]
    out.append(rec)
    with open(sys.argv[2] + ".lines", "a") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
