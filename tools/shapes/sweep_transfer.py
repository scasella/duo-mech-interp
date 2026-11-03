# Causal test of shared order directions: push along a direction built from OTHER families and read the target family.
#   python tools/shapes/sweep_transfer.py <model dir> <sweep dir> <summary json> <out.json> [layer]
# For each target family with a working copy prompt (node check >= 50%):
#   group direction    mean order direction of the other families in its group (leave-one-out; see sweep_shared.py)
#   generic direction  mean order direction of every ordered family outside its group
#   null               the same construction from the same families with their ranks shuffled (60 draws)
# The residual at the target word in the copy prompt is pushed by +/- a * |last - first| along the direction, from the
# middle value; effect = change in the expected position of the repeated word (in values, among the family's values).
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_sweep import FAMILIES
from sweep_shared import GROUPS, member, direction, unit
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir, sd, summ, out = sys.argv[1:5]
L = int(sys.argv[5]) if len(sys.argv) > 5 else 12
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
rows = {r["name"]: r for r in json.load(open(summ))}
zipf = {n: np.array(v["zipf"]) for n, v in json.load(open(f"{sd}/family_sweep.json")).items()}
fams = [n for n, r in rows.items() if r["order"] == "linear" and r["verdict"] != "not ordered" and r["V"] >= 4]
M = {n: np.load(f"{sd}/{n.replace(' ', '_')}_L{L}_means.npy").astype(np.float32) for n in fams}
X = np.concatenate([M[n] - M[n].mean(0) for n in fams]); y = np.concatenate([zipf[n] - zipf[n].mean() for n in fams])
fdir = unit(y @ X); Mf = {n: m - np.outer(m @ fdir, fdir) for n, m in M.items()}
rng = np.random.default_rng(0)
dirs = {n: direction(Mf[n], np.arange(len(M[n]), dtype=float)) for n in fams}

tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
U = m.get_output_embeddings().weight.detach()
st = dict(delta=None, pos=None)
def hook(mod, inp, outp):
    if st["delta"] is None: return
    h = (outp[0] if isinstance(outp, tuple) else outp).clone(); h[:, st["pos"]] += st["delta"]
    return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
lm.layers[L].register_forward_hook(hook)
enc = lambda s: tok(s, add_special_tokens=False).input_ids
A = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def readout(F, vals, mid, D):
    tpl = F["copy"]; pre = tpl[:tpl.index("{X}")]
    ids, ip, iv = enc(tpl.replace("{X}", vals[mid])), enc(pre.rstrip()), enc(" " + vals[mid])
    st["pos"] = len(ip) + len(iv) - 1
    first = [enc(" " + v)[0] for v in vals]
    st["delta"] = torch.tensor(D, dtype=torch.float32)
    with torch.inference_mode(): h = lm(input_ids=torch.tensor([ids] * len(D)), use_cache=False).last_hidden_state[:, -1]
    st["delta"] = None
    q = torch.softmax(h @ U.T, -1)[:, first].numpy(); q = q / q.sum(1, keepdims=True)
    return q @ np.arange(len(vals))                                              # expected position per push


res = {}
targets = [n for n in fams if n in member and rows[n]["node"] is not None and rows[n]["node"] >= 0.5]
for t in targets:
    F = FAMILIES[t]; vals = rows[t]["vals"]; V = len(vals); mid = (V - 1) // 2
    scale = float(np.linalg.norm(M[t][-1] - M[t][0]))
    g_others = [f for f in fams if member.get(f) == member[t] and f != t and not set(rows[f]["vals"]) & set(vals)]
    o_others = [f for f in fams if f in member and member[f] != member[t] and not set(rows[f]["vals"]) & set(vals)]
    out_t = {}
    for kind, others in (("group", g_others), ("generic", o_others)):
        if not others: continue
        d = unit(np.mean([dirs[f] for f in others], 0))
        e = readout(F, vals, mid, np.outer(A * scale, d)); eff = float(e[-1] - e[0])
        null = []
        for _ in range(60):
            dn = unit(np.mean([direction(Mf[f], rng.permutation(len(M[f])).astype(float)) for f in others], 0))
            en = readout(F, vals, mid, np.outer(A * scale, dn)); null.append(float(en[-1] - en[0]))
        null = np.array(null)
        own = readout(F, vals, mid, np.outer(A * scale, dirs[t]))                 # the family's own direction, for scale
        out_t[kind] = dict(n_families=len(others), effect=round(eff, 3), z=round(float((eff - null.mean()) / (null.std() + 1e-9)), 2),
                           null_mean=round(float(null.mean()), 3), null_sd=round(float(null.std()), 3), curve=e.round(3).tolist(),
                           own_effect=round(float(own[-1] - own[0]), 3))
    res[t] = dict(family_group=member[t], **out_t)
    txt = " | ".join(f"{k}: shift {v['effect']:+.2f} values (z {v['z']:+.1f}; own direction {v['own_effect']:+.2f})" for k, v in out_t.items())
    print(f"{t:18s} [{member[t]}] {txt}", flush=True)
# pooled per group: Stouffer z over the group's targets (each target's own null)
pooled = {}
for kind in ("group", "generic"):
    for g in sorted({v["family_group"] for v in res.values()}):
        zs = [v[kind]["z"] for v in res.values() if v["family_group"] == g and kind in v]
        if zs: pooled.setdefault(kind, {})[g] = dict(n=len(zs), z=round(float(np.sum(zs) / np.sqrt(len(zs))), 2), positive=int(sum(z > 0 for z in zs)))
    print(f"pooled ({kind} direction): " + "; ".join(f"{g}: z {v['z']:+.1f}, {v['positive']}/{v['n']} positive" for g, v in pooled.get(kind, {}).items()))
json.dump(dict(targets=res, pooled=pooled, layer=L), open(out, "w"), indent=1)
os._exit(0)
