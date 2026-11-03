# Unsupervised search for ordered structure in word families (no labels, no chosen orders).
#   python tools/shapes/discover.py <families.json> <run dir A> <run dir B> <HF cache or ""> <out.json> [layer]
# For each family from analyze.py words (found on model A), the members' context-averaged residuals are ordered by
# the shortest path through them (open path and closed loop, 2-opt). A family is ordered if, along that path,
# distance grows with the number of steps between two words (Spearman of distance vs path gap). That is scored
# against random sets of the same number of words. The same words and ordering are then re-scored on model B
# (another model with the same tokenizer). Families made of known structures are flagged, not dropped.
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from transformers import AutoTokenizer

fam_path, run_a, run_b, cache, out = sys.argv[1:6]
L = sys.argv[6] if len(sys.argv) > 6 else "12"
tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache or None)
rng = np.random.default_rng(0)

KNOWN = {
    "weekday/month": "monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
                     "august september october november december jan feb mar apr jun jul aug sep sept oct nov dec",
    "number": "zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy "
              "eighty ninety hundred thousand million billion 0 1 2 3 4 5 6 7 8 9 first second third fourth fifth sixth "
              "seventh eighth ninth tenth",
    "colour": "red orange yellow green blue purple violet pink brown black white gray grey",
    "emotion": "happy sad angry afraid calm excited bored anxious nervous relaxed content depressed joyful scared furious",
    "element": "hydrogen helium lithium carbon nitrogen oxygen sodium iron copper gold silver zinc tin lead mercury uranium",
    "place": "north south east west london paris berlin tokyo china india france germany japan russia america",
}
KNOWN_SETS = {k: set(v.split()) for k, v in KNOWN.items()}


def loader(run):
    ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
    row = {int(t): r for r, t in enumerate(ids)}
    M = np.load(f"{run}/L{L}_word_means.npy")
    gm = M[rng.choice(len(M), 20000, replace=False)].astype(np.float32).mean(0)
    def vec(w):
        out = [(M[row[t[0]]].astype(np.float32), counts[row[t[0]]]) for s in (" " + w, " " + w.capitalize(), " " + w.upper(), w)
               for t in [tok(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
        if not out: return None
        wt = np.array([c for _, c in out], float)
        return (np.stack([v for v, _ in out]) * wt[:, None]).sum(0) / wt.sum() - gm
    pool = [r for r in range(len(ids)) if counts[r] >= 50]
    return vec, M, gm, pool


def tour_length(D, t, closed):
    s = sum(D[t[i], t[i + 1]] for i in range(len(t) - 1))
    return s + (D[t[-1], t[0]] if closed else 0.0)


def best_order(D, closed, starts=8):
    """Shortest closed loop or open path through all points: nearest-neighbour starts + 2-opt."""
    n = len(D)
    if not closed:                                        # open path = loop through an extra free node
        D = np.pad(D, ((0, 1), (0, 1))); n += 1
    best, best_len = None, np.inf
    for s0 in rng.choice(n, min(starts, n), replace=False):
        t, left = [int(s0)], set(range(n)) - {int(s0)}
        while left:
            nxt = min(left, key=lambda j: D[t[-1], j]); t.append(nxt); left.remove(nxt)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                        t[i:j + 1] = t[i:j + 1][::-1]; improved = True
        ln = tour_length(D, t, True)
        if ln < best_len: best, best_len = t, ln
    if not closed:
        k = best.index(n - 1); best = best[k + 1:] + best[:k]
    return best


def orderedness(X, order, closed):
    """Spearman between pairwise distance and number of path steps apart (circular for loops)."""
    n = len(order)
    pos = np.empty(n, int); pos[order] = np.arange(n)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    iu = np.triu_indices(n, 1)
    gap = np.abs(pos[:, None] - pos[None])[iu]
    if closed: gap = np.minimum(gap, n - gap)
    d = D[iu]
    ra, rb = np.argsort(np.argsort(gap)), np.argsort(np.argsort(d))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(X):
    Xc = X - X.mean(0); Xc /= np.linalg.norm(Xc, axis=1).mean()
    D = np.linalg.norm(Xc[:, None] - Xc[None], axis=-1)
    res = {}
    for closed in (False, True):
        o = best_order(D, closed)
        res["loop" if closed else "path"] = (orderedness(Xc, o, closed), o)
    return res


fams = json.load(open(fam_path))[L]
vec_a, Ma, gma, pool_a = loader(run_a)
vec_b, Mb, gmb, pool_b = loader(run_b)
null = {}
def null_for(n, M, gm, pool, key, trials=60):
    if (key, n) not in null:
        vals = {"path": [], "loop": []}
        for _ in range(trials):
            X = M[rng.choice(pool, n, replace=False)].astype(np.float32) - gm
            s = score(X)
            for k in vals: vals[k].append(s[k][0])
        null[(key, n)] = {k: (float(np.mean(v)), float(np.std(v) + 1e-6)) for k, v in vals.items()}
    return null[(key, n)]

rows = []
for f in fams:
    words = [w for w in f["words"] if vec_a(w) is not None and vec_b(w) is not None]
    n = len(words)
    if n < 5 or n > 30: continue
    Xa = np.stack([vec_a(w) for w in words]); Xb = np.stack([vec_b(w) for w in words])
    sa = score(Xa)
    kind = max(("path", "loop"), key=lambda k: (sa[k][0] - null_for(n, Ma, gma, pool_a, "a")[k][0]) / null_for(n, Ma, gma, pool_a, "a")[k][1])
    r_a, order = sa[kind]
    za = (r_a - null_for(n, Ma, gma, pool_a, "a")[kind][0]) / null_for(n, Ma, gma, pool_a, "a")[kind][1]
    # replication: the same order on model B, against B's own null
    Xbc = Xb - Xb.mean(0); Xbc /= np.linalg.norm(Xbc, axis=1).mean()
    r_b = orderedness(Xbc, order, kind == "loop")
    zb = (r_b - null_for(n, Mb, gmb, pool_b, "b")[kind][0]) / null_for(n, Mb, gmb, pool_b, "b")[kind][1]
    lw = {w.lower() for w in words}
    known = [k for k, s in KNOWN_SETS.items() if len(lw & s) >= 3]
    rows.append(dict(family=f["family"], n=n, kind=kind, r_a=round(r_a, 3), z_a=round(za, 2), r_b=round(r_b, 3), z_b=round(zb, 2),
                     known=known, order=[words[i] for i in order]))
rows.sort(key=lambda r: -min(r["z_a"], r["z_b"]))
json.dump(rows, open(out, "w"), indent=1)
print(f"layer {L}: scored {len(rows)} families of 5-30 words")
print("most ordered (ranked by the weaker of the two models' z-scores; known structures marked):")
for r in rows[:60]:
    tag = f"  [known: {','.join(r['known'])}]" if r["known"] else ""
    print(f"  f{r['family']:4d} {r['kind']:4s} n {r['n']:2d}  z {r['z_a']:5.1f} / {r['z_b']:5.1f}{tag} :: {' → '.join(r['order'][:18])}")
