# Automated axis atlas from WordNet antonyms: hundreds of scales instead of hand-picked ones.
#   python tools/shapes/antonym_atlas.py <words run, model A> <words run, model B> <HF cache or ""> <nltk_data dir> <out.json> [layer]
# Scales: every WordNet adjective antonym pair, each pole = the adjective's synset lemmas + its "similar to" satellites,
# kept if a single token seen at least 50 times; at least 3 words per pole. Axis = mean(pole B) - mean(pole A).
# 1. Master dials: top eigenvectors of sum(axis axis^T) over all scales (sign-free), vs random word-set axes; which
#    scales load on each, and do Osgood's evaluation / potency / activity marker scales line up with them?
# 2. Pair-specific alignments: remove the top dials and a word-frequency axis, then the cosine between every pair of
#    scales from different topics (low similarity between the scales' own words), as a z-score against random axes,
#    required in both models with the same sign.
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from transformers import AutoTokenizer

run_a, run_b, cache, nltk_dir, out = sys.argv[1:6]
L = sys.argv[6] if len(sys.argv) > 6 else "12"
K_DIALS = 5
import nltk
nltk.data.path.insert(0, nltk_dir)
from nltk.corpus import wordnet as wn
tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache or None)
rng = np.random.default_rng(0)
unit = lambda v: v / np.linalg.norm(v)


def load(run):
    ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
    row = {int(t): r for r, t in enumerate(ids)}
    M = np.load(f"{run}/L{L}_word_means.npy").astype(np.float32)
    pool = np.array([r for r in range(len(ids)) if counts[r] >= 50]); gm = M[pool].mean(0)
    Xp = M[pool] - gm; y = np.log(counts[pool]); y -= y.mean()
    _, _, Vt = np.linalg.svd(Xp[rng.choice(len(Xp), 8000, replace=False)], full_matrices=False)
    Z = Xp @ Vt[:256].T; freq = unit(np.linalg.solve(Z.T @ Z + 1e3 * np.eye(256), Z.T @ y) @ Vt[:256])
    def vec(w):
        t = tok(" " + w, add_special_tokens=False).input_ids
        if len(t) != 1 or t[0] not in row or counts[row[t[0]]] < 50: return None
        return M[row[t[0]]] - gm
    return vec, M, pool, gm, freq


vec_a, Ma, pool_a, gm_a, freq_a = load(run_a)
vec_b, Mb, pool_b, gm_b, freq_b = load(run_b)
# --- build scales
scales, seen = [], set()
for s in wn.all_synsets("a"):
    for lem in s.lemmas():
        for ant in lem.antonyms():
            sa, sb = s, ant.synset()
            def pole(syn, head):
                words = {head} | {l.name() for l in syn.lemmas()} | {l.name() for t in syn.similar_tos() for l in t.lemmas()}
                return sorted(w.lower() for w in words if w.isalpha() and w.islower())
            pa, pb = pole(sa, lem.name()), pole(sb, ant.name())
            both = set(pa) & set(pb); pa = [w for w in pa if w not in both]; pb = [w for w in pb if w not in both]
            pa = [w for w in pa if vec_a(w) is not None and vec_b(w) is not None]
            pb = [w for w in pb if vec_a(w) is not None and vec_b(w) is not None]
            key = frozenset([frozenset(pa), frozenset(pb)])
            if len(pa) >= 3 and len(pb) >= 3 and key not in seen and lem.name().isalpha() and ant.name().isalpha():
                seen.add(key); scales.append(dict(name=f"{lem.name()}–{ant.name()}", a=pa, b=pb))
print(f"layer {L}: {len(scales)} scales with at least 3 single-token words per pole")
OSGOOD = {"evaluation": ["good", "pleasant", "beautiful", "nice", "kind"], "potency": ["strong", "large", "heavy", "hard", "powerful"],
          "activity": ["fast", "active", "quick", "lively", "energetic"]}


def axes_for(vec, freq, remove=None):
    A = []
    for sc in scales:
        a = np.stack([vec(w) for w in sc["a"]]).mean(0); b = np.stack([vec(w) for w in sc["b"]]).mean(0)
        v = b - a; v = v - (v @ freq) * freq
        if remove is not None: v = v - remove.T @ (remove @ v)
        A.append(unit(v))
    return np.stack(A)


def rand_axes(M, pool, gm, freq, n, remove=None):
    out = []
    for _ in range(n):
        v = (M[rng.choice(pool, 4)] - gm).mean(0) - (M[rng.choice(pool, 4)] - gm).mean(0); v = v - (v @ freq) * freq
        if remove is not None: v = v - remove.T @ (remove @ v)
        out.append(unit(v))
    return np.stack(out)


res = dict(layer=L, scales=[s["name"] for s in scales])
dials = {}
for tag, vec, M, pool, gm, freq in (("A", vec_a, Ma, pool_a, gm_a, freq_a), ("B", vec_b, Mb, pool_b, gm_b, freq_b)):
    A = axes_for(vec, freq)
    ev, U = np.linalg.eigh(A.T @ A); ev, U = ev[::-1], U[:, ::-1]
    share = ev[:K_DIALS] / ev.sum()
    R = rand_axes(M, pool, gm, freq, len(scales)); evr = np.linalg.eigvalsh(R.T @ R)[::-1]
    null_share = evr[:K_DIALS] / evr.sum()
    dials[tag] = U[:, :K_DIALS].T
    load = A @ U[:, :K_DIALS]                                                   # scale x dial
    print(f"\nmodel {tag}: top {K_DIALS} dials hold " + ", ".join(f"{s:.1%}" for s in share) + " of the scales' variance (random axes: "
          + ", ".join(f"{s:.1%}" for s in null_share) + ")")
    for k in range(K_DIALS):
        o = np.argsort(load[:, k])
        neg = [f"{scales[i]['name']}" for i in o[:5]]; pos = [f"{scales[i]['name']}" for i in o[::-1][:5]]
        print(f"  dial {k + 1}: + {', '.join(pos)}\n          - {', '.join(neg)}")
    # Osgood markers: which dial does each marker family's direction fall on (cosine with the dial)?
    for fac, words in OSGOOD.items():
        idx = [i for i, sc in enumerate(scales) if any(w in sc["a"] or w in sc["b"] for w in words)]
        if not idx: continue
        cos = np.abs(load[idx]).mean(0)
        print(f"  Osgood {fac:10s} ({len(idx)} scales): mean |loading| on dials 1-{K_DIALS}: " + " ".join(f"{c:.2f}" for c in cos))
    res[f"dials_{tag}"] = dict(share=share.round(4).tolist(), null=null_share.round(4).tolist(), loadings=load.round(3).tolist())

# --- pair-specific alignments after removing the top dials (each model's own), required in both models
def specific(vec, M, pool, gm, freq, D):
    A = axes_for(vec, freq, remove=D)
    R = rand_axes(M, pool, gm, freq, 600, remove=D)
    sd = float(np.std(R[:300] @ R[300:].T))
    topic = np.stack([unit(np.stack([vec(w) for w in sc["a"] + sc["b"]]).mean(0)) for sc in scales])
    return A @ A.T / sd, topic @ topic.T
Za, Ta = specific(vec_a, Ma, pool_a, gm_a, freq_a, dials["A"])
Zb, Tb = specific(vec_b, Mb, pool_b, gm_b, freq_b, dials["B"])
n = len(scales); iu = np.triu_indices(n, 1)
words = [set(s["a"]) | set(s["b"]) for s in scales]
cand = []
topic_cut = np.quantile(Ta[iu], 0.5)
for i, j in zip(*iu):
    if words[i] & words[j] or Ta[i, j] > topic_cut: continue
    za, zb = Za[i, j], Zb[i, j]
    if np.sign(za) == np.sign(zb): cand.append((min(abs(za), abs(zb)) * np.sign(za), i, j, za, zb))
cand.sort(key=lambda c: -abs(c[0]))
print(f"\npair-specific alignments between different-topic scales ({len(cand)} pairs kept; {n * (n - 1) // 2} total), "
      f"top {K_DIALS} dials removed, both models agree:")
pairs_out = []
for zmin, i, j, za, zb in cand[:40]:
    si, sj = scales[i], scales[j]
    # orient so the pair reads "a-pole of i goes with a-pole of j" when positive
    print(f"  z {za:+5.1f} / {zb:+5.1f}  {si['name']:24s} {'~' if zmin > 0 else '~ reversed'} {sj['name']:24s}  "
          f"[{', '.join(si['b'][:4])} | {', '.join(sj['b'][:4]) if zmin > 0 else ', '.join(sj['a'][:4])}]")
    pairs_out.append(dict(i=si["name"], j=sj["name"], za=round(float(za), 2), zb=round(float(zb), 2), a_i=si["a"], b_i=si["b"], a_j=sj["a"], b_j=sj["b"]))
expected_max = float(np.sqrt(2 * np.log(len(cand))))
print(f"  (with {len(cand)} pairs, the largest |z| expected by chance in one model is about {expected_max:.1f})")
res["pairs"] = pairs_out; res["scale_words"] = scales
json.dump(res, open(out, "w"))
