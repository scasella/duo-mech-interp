# Score a trained block-sparse featurizer (and its SAE baseline) against structures we already know, then look for
# blocks that carry new ones.
#   python tools/shapes/bsf_analyze.py <bsf run dir> <model dir> <out.json> <probe dir>[,<probe dir>...]
# Known structures: probe families (template sentences, residual at the family word, same layer as the featurizer).
#   coverage    share of the family's sentences on which the family's most common block (or SAE feature) is active
#   dedicated   share of that block's firings on 2M tokens of web text that land on the family's own words
#   order       does the block's own 3-D coordinate put the values in order (cyclic or linear), as the full residual does
#   overlap     mean cosine of the principal angles between the block's 3-D subspace and the family's own top-3 subspace
# Discovery: blocks whose firings concentrate on a few word types; per-word mean coordinates inside the block; blocks
# whose words spread over 2-3 dimensions in a structured way are listed for inspection.
import json, math, os, sys
from collections import Counter
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsf import GrassmannianBSF
from transformers import AutoTokenizer

run, model_dir, out = sys.argv[1:4]
probe_dirs = sys.argv[4].split(",")
tok = AutoTokenizer.from_pretrained(model_dir)


def load(name):
    ck = torch.load(f"{run}/{name}.pt", map_location="cpu", weights_only=False)
    m = GrassmannianBSF(len(ck["mu"]), *ck["config"]); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["mu"], ck["scale"], ck["layer"]


bsf, mu, scale, L = load("bsf3")
sae, mu_s, scale_s, _ = load("sae")
norm = lambda x: (torch.tensor(x, dtype=torch.float32) - mu) * scale
toks = np.load(f"{run}/bsf3_tokens.npy"); top = np.load(f"{run}/bsf3_top.npy").astype(np.int64); coords = np.load(f"{run}/bsf3_coords.npy").astype(np.float32)
fires = np.bincount(top.reshape(-1), minlength=bsf.G)


def order_score(c, order):
    n = len(c)
    if order == "cyclic":
        c2 = c[:, :2] - c[:, :2].mean(0); ang = np.arctan2(c2[:, 1], c2[:, 0]); rank = {v: r for r, v in enumerate(np.argsort(ang))}
        return sum((rank[(v + 1) % n] - rank[v]) % n in (1, n - 1) for v in range(n)), n
    r = np.argsort(np.argsort(c[:, 0])); return round(abs(float(np.corrcoef(r, np.arange(n))[0, 1])), 3), 1.0


def pca(X, k=3):
    Xc = X - X.mean(0); _, S, Vt = np.linalg.svd(Xc, full_matrices=False); return Xc @ Vt[:k].T, Vt[:k], S ** 2 / max((S ** 2).sum(), 1e-12)


res = {"known": {}, "discover": []}
print(f"featurizer: {bsf.G} blocks of {bsf.b}, {bsf.k} active; SAE: {sae.G} features, {sae.k} active; layer {L}")
corpus = {"blocks": (toks, top)}
if os.path.exists(f"{run}/sae_top.npy"):
    corpus["SAE"] = (np.load(f"{run}/sae_tokens.npy"), np.load(f"{run}/sae_top.npy").astype(np.int64))
rng = np.random.default_rng(0)
with torch.no_grad():
    frames = bsf.frames().numpy(); gamma = bsf.log_gamma.exp().numpy()
for pdir in probe_dirs:
    meta = json.load(open(f"{pdir}/probes.json"))
    for fam, F in meta.items():
        path = f"{pdir}/probe_{fam}_L{L}.npz"
        if not os.path.exists(path): continue
        R = np.load(path)["resid"].astype(np.float32); nT, nV = R.shape[:2]
        Xn = norm(R.reshape(-1, R.shape[-1]))
        with torch.no_grad():
            tb, _ = bsf.encode_top(Xn); ts, _ = sae.encode_top((torch.tensor(R.reshape(-1, R.shape[-1])) - mu_s) * scale_s)
        codes = {"blocks": tb.numpy(), "SAE": ts.numpy()}
        word_ids = set()
        for v in F["vals"]:
            for w in (" " + v, " " + v.lower(), " " + v.capitalize(), v):
                t = tok(w, add_special_tokens=False).input_ids
                if len(t) == 1: word_ids.add(t[0])
        row = {}
        for kind, T in codes.items():
            nfeat = bsf.G if kind == "blocks" else sae.G
            on = (T[:, :, None] == np.arange(1)[None, None, :]).any(-1) if False else None
            item_cov = np.zeros(nfeat); 
            for i in range(len(T)): item_cov[np.unique(T[i])] += 1
            item_cov /= len(T)
            cands = np.where(item_cov >= 0.3)[0]
            entry = dict(candidates=int(len(cands)))
            if kind in corpus and len(cands):
                ctoks, ctop = corpus[kind]
                fam_tok = np.isin(ctoks, list(word_ids))
                ded = np.array([fam_tok[(ctop == f).any(1)].mean() if (ctop == f).any() else 0.0 for f in cands])
                best = int(cands[np.argmax(ded * item_cov[cands])])
                entry.update(best=best, coverage=round(float(item_cov[best]), 3), dedicated=round(float(ded.max() if len(ded) else 0), 3),
                             dedicated_best=round(float(ded[list(cands).index(best)]), 3),
                             n_dedicated=int((ded >= 0.5).sum()))
                if kind == "blocks":
                    # the block's coordinates of each value's (template-centred) average, against random 3-frames
                    Xv = Xn.numpy().reshape(nT, nV, -1).mean(0)
                    C = gamma[best] * Xv @ frames[best]
                    cc, _, _ = pca(C, 3); sc = order_score(cc, F["order"])[0]
                    null = []
                    for _ in range(500):
                        Q = np.linalg.qr(rng.normal(size=(Xv.shape[1], bsf.b)))[0]
                        cr, _, _ = pca(Xv @ Q, 3); null.append(order_score(cr, F["order"])[0])
                    entry["order"] = sc; entry["order_p"] = round(float(np.mean(np.array(null) >= sc - 1e-9)), 3)
                    mu_v = (R - R.mean(1, keepdims=True)).mean(0); _, Vo, _ = pca(mu_v, 3)
                    Qo = np.linalg.qr(Vo.T)[0].T; Qb = np.linalg.qr(frames[best])[0].T
                    entry["overlap"] = round(float(np.linalg.svd(Qo @ Qb.T, compute_uv=False).mean()), 3)
            row[kind] = entry
        res["known"][fam] = row
        b, s_ = row["blocks"], row.get("SAE", {})
        if "best" in b:
            omax = nV if F["order"] == "cyclic" else 1.0
            otxt = f"{b['order']}/{nV}" if F["order"] == "cyclic" else f"{b['order']:.2f}"
            stxt = (f"SAE: {s_['n_dedicated']} dedicated features, best dedicated {s_['dedicated_best']:.0%} covering {s_['coverage']:.0%}"
                    if "best" in s_ else "SAE: n/a")
            print(f"  {fam:13s} blocks: {b['n_dedicated']} dedicated (>=50% of firings on the family), best: dedicated {b['dedicated_best']:.0%}, "
                  f"covers {b['coverage']:.0%}, in-block order {otxt} (p {b['order_p']:.2f}), overlap {b['overlap']:.2f} | {stxt}")

# --- discovery: blocks concentrated on few word types, with structured in-block geometry
print("\nblocks whose firings concentrate on a handful of words, laid out over 2-3 dimensions:")
cands = []
flat_top = top.reshape(-1); flat_tok = np.repeat(toks, bsf.k); flat_c = coords.reshape(-1, bsf.b)
order = np.argsort(flat_top, kind="stable"); starts = np.searchsorted(flat_top[order], np.arange(bsf.G + 1))
for g in range(bsf.G):
    if fires[g] < 300: continue
    sel = order[starts[g]:starts[g + 1]]
    wt = Counter(flat_tok[sel].tolist())
    words = [(w, c) for w, c in wt.most_common(12) if c >= 15]
    if len(words) < 5: continue
    share = sum(c for _, c in words) / len(sel)
    if share < 0.5: continue
    cents = np.stack([flat_c[sel][flat_tok[sel] == w].mean(0) for w, _ in words])
    _, _, evf = pca(cents, 3)
    labels = [tok.decode([w]).strip() for w, _ in words]
    if len(set(l.lower() for l in labels)) < 5: continue
    cc, _, _ = pca(cents, 3)
    cands.append(dict(block=g, fires=int(fires[g]), share=round(float(share), 2), dims=round(float(1 / (evf ** 2).sum()), 2),
                      words=labels, xy=[[round(float(a), 3), round(float(b), 3)] for a, b in cc[:, :2]]))
cands.sort(key=lambda c: (-c["dims"], -c["share"]))
for c in cands[:40]:
    print(f"  block {c['block']:5d} fires {c['fires']:6d}  top words {c['share']:.0%} of firings  spread over {c['dims']:.1f} dims :: {' '.join(c['words'])}")
res["discover"] = cands
json.dump(res, open(out, "w"), indent=1)
