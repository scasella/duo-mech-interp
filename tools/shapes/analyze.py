# Find concept shapes in Qwen-Scope SAE codes collected by core.harvest / core.probe.
#   python tools/shapes/analyze.py probes --run runs/probe --layers 6,12,18 [--cache HF_CACHE]
#   python tools/shapes/analyze.py groups --run runs/full --layers 12 [--probe-run runs/probe] [--cache HF_CACHE]
#
# probes: for families with a known order (weekdays, months, ...), does the model hold them as a shape (the
#   "oracle" subspace from per-value averages, as in the 0.8B test), how many SAE features carry the family, does
#   each feature cover only some values (tiling), and does the reconstruction from just those features keep the
#   shape (order around the circle / along the line, and overlap with the oracle subspace)?
# groups: unsupervised. Features are linked when their decoder directions point the same way or when they fire
#   together more than chance (normalised PMI); communities in that graph are candidate shapes. For each one, the
#   reconstruction from only its features over the stored tokens: dimensionality (participation ratio), how often
#   several of its features fire together, which words it covers, and the order of any known family among them.
import argparse, json, math, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


# ---------------------------------------------------------------- shape measures
def pca(X, k=3):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2
    coords = Xc @ Vt[:k].T
    if coords.shape[1] < k:                                 # fewer dimensions than asked (e.g. a 2-feature group)
        coords = np.pad(coords, ((0, 0), (0, k - coords.shape[1])))
        ev = np.pad(ev, (0, k - len(ev)))
    return coords, ev / max(ev.sum(), 1e-12), Vt[:k]


def participation(evfrac):
    return float(1.0 / max((evfrac ** 2).sum(), 1e-12))


def order_score(coords, order):
    """coords: per-value centroids (n, >=2) in family order. cyclic: how many neighbours (v, v+1) are also
    neighbours in angle around the centroid (n = perfect circle in the right order). linear: |Spearman| of PC1."""
    n = len(coords)
    if order == "cyclic":
        c = coords[:, :2] - coords[:, :2].mean(0)
        ang = np.arctan2(c[:, 1], c[:, 0])
        rank = {v: r for r, v in enumerate(np.argsort(ang))}
        return sum((rank[(v + 1) % n] - rank[v]) % n in (1, n - 1) for v in range(n)), n
    r = np.argsort(np.argsort(coords[:, 0]))
    return round(abs(float(np.corrcoef(r, np.arange(n))[0, 1])), 3), 1.0


def cyclic_null(n):
    """chance that random points come out in the right cyclic order: 2n / n! arrangements... = 1 / ((n-1)!/2)"""
    return 2.0 / math.factorial(n - 1)


def principal_cos(A, B):
    """mean cosine of the principal angles between row spaces of A and B (orthonormal rows)."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return float(s.mean())


# ---------------------------------------------------------------- probes
SHARED = ["weekday", "month", "number", "ordinal"]        # families whose k-th value is "position k"


def shared_code(run, L, meta):
    """Do families share a position code? For each pair, the cosine between value i of one and value j of the
    other (template-centred per-value means): same index vs different index, and how often value i's nearest
    match in the other family is value i."""
    mu = {}
    for f in SHARED:
        if f not in meta: continue
        R = np.load(f"{run}/probe_{f}_L{L}.npz")["resid"].astype(np.float32)
        m = (R - R.mean(1, keepdims=True)).mean(0)
        mu[f] = m / np.linalg.norm(m, axis=1, keepdims=True)
    out = {}
    names = list(mu)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = mu[names[a]], mu[names[b]]
            n = min(len(A), len(B))
            C = A[:n] @ B[:n].T
            out[f"{names[a]}-{names[b]}"] = dict(same=round(float(np.diag(C).mean()), 3),
                                                  other=round(float(C[~np.eye(n, dtype=bool)].mean()), 3),
                                                  match=round(float((C.argmax(1) == np.arange(n)).mean()), 3),
                                                  chance=round(1 / n, 3))
    return out


def probes(run, layers, cache):
    meta = json.load(open(f"{run}/probes.json"))
    report = {}
    for L in layers:
        has_sae = "idx" in np.load(f"{run}/probe_{next(iter(meta))}_L{L}.npz").files
        Wd = core.load_sae(L, "cpu", cache_dir=cache)["W_decT"].numpy() if has_sae else None
        for name, F in meta.items():
            z = np.load(f"{run}/probe_{name}_L{L}.npz")
            R = z["resid"].astype(np.float32)
            nT, nV = R.shape[:2]
            # oracle shape: template-centred per-value means
            Rc = R - R.mean(1, keepdims=True)
            mu = Rc.mean(0)
            co, evf, Vo = pca(mu, 3)
            o_score, o_max = order_score(co, F["order"])
            null = f"  (chance {cyclic_null(nV):.1e})" if F["order"] == "cyclic" else ""
            if not has_sae:
                report.setdefault(name, {})[L] = dict(oracle_order=o_score, max=o_max, oracle_pc12=round(float(evf[:2].sum()), 3))
                print(f"L{L:2d} {name:12s} oracle order {o_score}/{o_max}{null}  pc1-2 {evf[:2].sum():.2f}")
                continue
            I, V = z["idx"].astype(np.int64), z["val"].astype(np.float32)
            # SAE view: features firing on the family word
            fires = np.zeros((nV, core.D_SAE), np.float32)
            for t in range(nT):
                for v in range(nV):
                    on = V[t, v] > 0
                    fires[v, I[t, v][on]] += 1.0 / nT
            fam = np.where(fires.max(0) >= 0.25)[0]
            frac = fires[:, fam]
            specific = fam[(frac >= 0.25).sum(0) <= max(1, nV // 3)]
            broad = fam[(frac >= 0.25).sum(0) >= max(2, int(0.8 * nV))]
            prefs = Counter(int(fires[:, f].argmax()) for f in specific)
            # reconstruction from the family features only
            Xg = np.zeros((nT, nV, core.D_MODEL), np.float32)
            fam_set = set(fam.tolist())
            for t in range(nT):
                for v in range(nV):
                    keep = [(i, a) for i, a in zip(I[t, v], V[t, v]) if a > 0 and int(i) in fam_set]
                    if keep:
                        ii, aa = zip(*keep)
                        Xg[t, v] = np.asarray(aa) @ Wd[list(ii)]
            Xc = Xg - Xg.mean(1, keepdims=True)
            cg, evg, Vg = pca(Xc.mean(0), 3)
            g_score, _ = order_score(cg, F["order"])
            r = dict(oracle_order=o_score, max=o_max, oracle_pc12=round(float(evf[:2].sum()), 3),
                     n_features=int(len(fam)), n_specific=int(len(specific)), n_broad=int(len(broad)),
                     values_covered_by_specific=len(prefs), sae_order=g_score,
                     subspace_overlap=round(principal_cos(Vo, Vg), 3),
                     features=[int(f) for f in fam], specific=[int(f) for f in specific],
                     feature_prefs={int(f): F["vals"][int(fires[:, f].argmax())] for f in specific})
            report.setdefault(name, {})[L] = r
            print(f"L{L:2d} {name:12s} oracle order {o_score}/{o_max}{null}  pc1-2 {r['oracle_pc12']:.2f} | "
                  f"SAE: {r['n_features']} features ({r['n_specific']} value-specific covering "
                  f"{r['values_covered_by_specific']}/{nV} values, {r['n_broad']} broad) | "
                  f"SAE-only reconstruction order {g_score}/{o_max}  overlap with oracle {r['subspace_overlap']:.2f}")
        sc = shared_code(run, L, meta)
        report.setdefault("_shared_position_code", {})[L] = sc
        for pair, s in sc.items():
            print(f"L{L:2d} shared code {pair:16s} same-index cos {s['same']:+.2f} vs other {s['other']:+.2f}, "
                  f"nearest match {s['match']:.0%} (chance {s['chance']:.0%})")
    json.dump(report, open(f"{run}/probe_report.json", "w"), indent=1)
    return report


# ---------------------------------------------------------------- unsupervised groups
def build_index(idx, val, valid):
    """feature -> rows of (flat token position, activation) for every firing in the stored codes."""
    S, T, Kk = idx.shape
    pos = np.repeat(np.arange(S * T, dtype=np.int64), Kk)
    f = idx.reshape(-1).astype(np.int64); a = val.reshape(-1).astype(np.float32)
    ok = (a > 0) & np.repeat(valid.reshape(-1), Kk)
    f, a, pos = f[ok], a[ok], pos[ok]
    o = np.argsort(f, kind="stable")
    f, a, pos = f[o], a[o], pos[o]
    starts = np.searchsorted(f, np.arange(core.D_SAE + 1))
    return starts, pos, a


def decoder_neighbours(Wd, alive, k=16, chunk=2048):
    Wn = Wd / np.linalg.norm(Wd, axis=1, keepdims=True)
    A = Wn[alive]
    nb_i, nb_c = np.zeros((len(alive), k), np.int64), np.zeros((len(alive), k), np.float32)
    for r0 in range(0, len(alive), chunk):
        s = A[r0:r0 + chunk] @ A.T
        for j in range(s.shape[0]): s[j, r0 + j] = -1
        i = np.argpartition(-s, k, axis=1)[:, :k]
        nb_i[r0:r0 + chunk] = alive[i]; nb_c[r0:r0 + chunk] = np.take_along_axis(s, i, 1)
    return nb_i, nb_c


def communities(alive, nb_i, nb_c, p_i, p_v, p_c, cos_min, npmi_min, min_count, max_size, seed=0):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(alive.tolist())
    aset = set(alive.tolist())
    for row, f in enumerate(alive):
        for j, c in zip(nb_i[row], nb_c[row]):
            if c >= cos_min and int(j) in aset:
                w = float(c)
                if G.has_edge(int(f), int(j)): G[int(f)][int(j)]["weight"] = max(G[int(f)][int(j)]["weight"], w)
                else: G.add_edge(int(f), int(j), weight=w)
        for j, v, c in zip(p_i[f], p_v[f], p_c[f]):
            if v >= npmi_min and c >= min_count and int(j) in aset and int(j) != int(f):
                w = float(v)
                if G.has_edge(int(f), int(j)): G[int(f)][int(j)]["weight"] = max(G[int(f)][int(j)]["weight"], w)
                else: G.add_edge(int(f), int(j), weight=w)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    out, todo = [], [(set(c), 1.0) for c in nx.connected_components(G)]
    while todo:
        nodes, res = todo.pop()
        if len(nodes) <= max_size:
            if len(nodes) >= 2: out.append(sorted(nodes))
            continue
        parts = nx.community.louvain_communities(G.subgraph(nodes), weight="weight", resolution=res, seed=seed)
        if len(parts) == 1:
            todo.append((nodes, res * 2)); continue
        todo.extend((set(p), res * 1.5) for p in parts)
    return out, G.number_of_edges()


STOP = set("the a an of to and in on at for is was are be by with as that this it its from or not but have has had "
           "he she they we you i his her their our your my which who what when where will would can could do does "
           "did so if than then there these those been were also more most other some such no only into over".split())


def is_filler(w):
    w = w.strip().lower()
    return not w or w in STOP or not any(ch.isalnum() for ch in w)


def label_structure(X, labels, min_count=3):
    """How much of the group's geometry is organised by which word it fired on: the share of variance between
    word centroids, and the dimensionality (participation ratio) of those centroids. Filler words are ignored."""
    keep = [i for i, l in enumerate(labels) if not is_filler(l)]
    cnt = Counter(labels[i] for i in keep)
    words = [w for w, c in cnt.items() if c >= min_count]
    if len(words) < 4:
        return 0.0, 1.0, len(words)
    rows = [i for i in keep if cnt[labels[i]] >= min_count]
    Xs = X[rows]
    mu = Xs.mean(0)
    tot = float(((Xs - mu) ** 2).sum())
    lab_id = {w: j for j, w in enumerate(words)}
    lid = np.fromiter((lab_id[labels[i]] for i in rows), np.int64, len(rows))
    wts = np.bincount(lid, minlength=len(words)).astype(np.float64)
    C = np.zeros((len(words), X.shape[1]), np.float64)
    np.add.at(C, lid, Xs)
    C /= wts[:, None]
    between = float((wts[:, None] * (C - mu) ** 2).sum()) / max(tot, 1e-12)
    _, evc, _ = pca(C, 3)
    return between, participation(evc), len(words)


def groups(run, layers, cache, probe_run=None, min_fire=None, max_rate=0.005, cos_min=0.35, npmi_min=0.35,
           min_count=5, max_size=80, max_points=4000):
    from transformers import AutoTokenizer
    meta = json.load(open(f"{run}/meta.json"))
    tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache)
    tokens, valid = np.load(f"{run}/tokens.npy"), np.load(f"{run}/valid.npy")
    flat_tok = tokens.reshape(-1)
    T = tokens.shape[1]
    vocab = {}
    def word(t):
        if t not in vocab: vocab[t] = tok.decode([int(t)])
        return vocab[t]
    def context(p, w=8):
        s, i = divmod(int(p), T)
        a, b = max(0, i - w), min(T, i + 3)
        return tok.decode(tokens[s, a:i].tolist()) + "[[" + word(tokens[s, i]) + "]]" + tok.decode(tokens[s, i + 1:b].tolist())
    fams = probe_report = None
    if probe_run:
        fams = json.load(open(f"{probe_run}/probes.json"))
        if os.path.exists(f"{probe_run}/probe_report.json"):
            probe_report = json.load(open(f"{probe_run}/probe_report.json"))
    fam_of = {}
    for name, F in (fams or core.FAMILIES).items():
        for i, v in enumerate(F["vals"]): fam_of.setdefault(v.lower(), []).append((name, i))
    for L in layers:
        sae = core.load_sae(L, "cpu", cache_dir=cache)
        Wd = sae["W_decT"].numpy()
        freq = np.load(f"{run}/L{L}_freq.npy")
        p_i, p_v, p_c = (np.load(f"{run}/L{L}_partners_{k}.npy") for k in ("idx", "npmi", "cnt"))
        p_v = p_v.astype(np.float32)
        mf = min_fire or max(10, int(2e-6 * meta["n_tokens"]))
        alive = np.where((freq >= mf) & (freq <= max_rate * meta["n_tokens"]))[0]      # drop always-on features
        nb_i, nb_c = decoder_neighbours(Wd, alive)
        grps, n_edges = communities(alive, nb_i, nb_c, p_i, p_v, p_c, cos_min, npmi_min, min_count, max_size)
        print(f"L{L}: {len(alive):,} features fire {mf}..{int(max_rate * meta['n_tokens']):,} times; "
              f"graph {n_edges:,} edges; {len(grps):,} groups")
        idx, val = np.load(f"{run}/L{L}_idx.npy"), np.load(f"{run}/L{L}_val.npy")
        starts, pos, act = build_index(idx, val, valid)
        rng = np.random.default_rng(0)
        results = []
        for gi, g in enumerate(grps):
            pp = np.concatenate([pos[starts[f]:starts[f + 1]] for f in g])
            aa = np.concatenate([act[starts[f]:starts[f + 1]] for f in g])
            cc = np.concatenate([np.full(starts[f + 1] - starts[f], c) for c, f in enumerate(g)])
            P, inv = np.unique(pp, return_inverse=True)
            n_tok = len(P)
            if n_tok < 30: continue
            if n_tok > max_points:                                  # sample tokens, keep all their group features
                sel = np.sort(rng.choice(n_tok, max_points, replace=False))
                remap = np.full(n_tok, -1); remap[sel] = np.arange(len(sel))
                m = remap[inv] >= 0
                P, inv, aa, cc = P[sel], remap[inv][m], aa[m], cc[m]
            A = np.zeros((len(P), len(g)), np.float32)
            A[inv, cc] = aa
            # the reconstruction A @ Wd[g] lives in the span of the group's decoder rows: with Wd[g]^T = QR it equals
            # (A R^T) Q^T, and Q^T keeps distances, so all geometry is computed on the small (tokens x size) matrix
            _, Rg = np.linalg.qr(Wd[g].T)
            X = A @ Rg.T
            coords, evf, _ = pca(X, 3)
            pr = participation(evf)
            mix = float(((A > 0).sum(1) >= 2).mean())
            labels = [word(flat_tok[p]).strip() for p in P]
            lc = Counter(labels)
            filler = sum(c for w, c in lc.items() if is_filler(w)) / len(labels)
            between, cpr, n_labels = label_structure(X, labels)
            # order of a known family among the group's words
            fam_hits = {}
            for lab, cnt in lc.items():
                for name, vi in fam_of.get(lab.lower(), []): fam_hits.setdefault(name, {})[vi] = lab
            fam_order = None
            for name, hit in fam_hits.items():
                F = (fams or core.FAMILIES)[name]
                if len(hit) >= 4:
                    vis = sorted(hit)
                    cent = np.stack([coords[[i for i, l in enumerate(labels) if l.lower() == F["vals"][v].lower()]].mean(0) for v in vis])
                    sc, mx = order_score(cent, F["order"]) if F["order"] == "linear" or len(vis) == len(F["vals"]) else (None, None)
                    fam_order = dict(family=name, values=len(vis), of=len(F["vals"]), order=sc, max=mx)
            feats = []
            for f in g:
                s0, s1 = starts[f], starts[f + 1]
                top = np.argsort(-act[s0:s1])[:6]
                feats.append(dict(f=int(f), fires=int(freq[f]), top=[word(flat_tok[pos[s0 + t]]).strip() for t in top]))
            # a shape candidate: words sit in distinct places (between) arranged in more than one dimension (cpr)
            score = 0.0 if filler > 0.5 else between * min(cpr, 4.0) * min(n_labels, 12) / 12.0
            sample = rng.choice(len(P), min(300, len(P)), replace=False)
            results.append(dict(layer=L, group=gi, size=len(g), tokens=int(n_tok), pr=round(pr, 2),
                                pc123=[round(float(x), 3) for x in evf[:3]], mix=round(mix, 3), n_labels=n_labels,
                                between=round(between, 3), centroid_pr=round(cpr, 2), filler=round(filler, 3),
                                top_labels=lc.most_common(12), family=fam_order, score=round(score, 4), features=feats,
                                points=[[round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3),
                                         round(float(coords[i, 2]), 3), labels[i]] for i in sample],
                                examples=[context(P[i]) for i in np.argsort(-np.linalg.norm(X, axis=1))[:5]]))
        results.sort(key=lambda r: -r["score"])
        json.dump(results, open(f"{run}/groups_L{L}.json", "w"))
        print(f"L{L}: {len(results)} groups with >= 30 tokens; top by shape score:")
        for r in results[:15]:
            fo = f" {r['family']['family']} {r['family']['order']}/{r['family']['max']}" if r["family"] else ""
            print(f"  g{r['group']:5d} size {r['size']:2d} tokens {r['tokens']:5d} between {r['between']:.2f} "
                  f"centroid-PR {r['centroid_pr']:.2f} words {r['n_labels']:3d}{fo} :: "
                  f"{' '.join(w for w, _ in r['top_labels'][:10])}")
        if probe_report:                                    # where did each family's value-specific features end up?
            where = {f: r["group"] for r in results for f in [x["f"] for x in r["features"]]}
            for name, by_layer in probe_report.items():
                if name.startswith("_"): continue
                fs = by_layer.get(str(L), {}).get("specific", [])
                c = Counter(where.get(f) for f in fs)
                best, n = max(((g, n) for g, n in c.items() if g is not None), key=lambda x: x[1], default=(None, 0))
                print(f"  probe family {name:12s}: {len(fs)} features; best group g{best} holds {n}; ungrouped {c.get(None, 0)}")


# ---------------------------------------------------------------- word families (no SAE)
def null_planarity(n, d=256, trials=200, seed=0):
    """Top-1 and top-2 variance share of n random isotropic points (centred): what a shapeless family scores."""
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    for _ in range(trials):
        X = rng.normal(size=(n, d)); X -= X.mean(0)
        ev = np.linalg.svd(X, compute_uv=False) ** 2; ev /= ev.sum()
        s1 += ev[0]; s2 += ev[:2].sum()
    return s1 / trials, s2 / trials


def words(run, layers, cache, min_count=50, drop_pcs=2, knn=10, max_size=40, families_out=None):
    """Families of words whose context-averaged residuals form a low-dimensional shape. Words are linked to their
    nearest neighbours (cosine, after removing the shared mean and the main shared axes), communities are split
    to at most `max_size`, and each family's averages are measured: how much variance its top one and two axes
    hold, compared with what random points of the same count would give (so 0 = no shape, 1 = a perfect line
    or plane)."""
    import re, networkx as nx
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache)
    ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
    texts = [tok.decode([int(i)]) for i in ids]
    wordlike = re.compile(r"^ [A-Za-z][a-z]{2,}$|^ [A-Z][a-z]+$|^ ?[0-9]$")
    keep = np.array([bool(wordlike.match(t)) and c >= min_count for t, c in zip(texts, counts)])
    ids, counts = ids[keep], counts[keep]
    labels = [t.strip() for t, k in zip(texts, keep) if k]
    fam_of = {v: (name, i) for name, F in core.FAMILIES.items() for i, v in enumerate(F["vals"])}
    nulls = {}
    out = {}
    # merge case variants (" his" / " His") and plain plurals (" brains" -> " brain") into one word, weighted by
    # count, so capitalisation and number do not show up as "shapes"
    lower = {w.lower() for w in labels}
    def base(w):
        if w in fam_of: return w
        w = w.lower()
        return w[:-1] if len(w) >= 5 and w.endswith("s") and not w.endswith("ss") and w[:-1] in lower else w
    groups_ = {}
    for r, w in enumerate(labels):
        groups_.setdefault(base(w), []).append(r)
    merged = list(groups_.items())
    labels = [w for w, _ in merged]
    wcount = np.array([counts[rows].sum() for _, rows in merged])
    for L in layers:
        X0 = np.load(f"{run}/L{L}_word_means.npy")[keep].astype(np.float32)
        X = np.stack([(X0[rows] * counts[rows, None]).sum(0) / counts[rows].sum() for _, rows in merged])
        X -= X.mean(0)
        if drop_pcs:
            _, _, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)],
                                     full_matrices=False)
            X -= (X @ Vt[:drop_pcs].T) @ Vt[:drop_pcs]
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        G = nx.Graph(); G.add_nodes_from(range(len(Xn)))
        for r0 in range(0, len(Xn), 2048):
            S = Xn[r0:r0 + 2048] @ Xn.T
            for j in range(S.shape[0]): S[j, r0 + j] = -1
            nb = np.argpartition(-S, knn, axis=1)[:, :knn]
            for j in range(S.shape[0]):
                for t in nb[j]:
                    if S[j, t] > 0.2: G.add_edge(r0 + j, int(t), weight=float(S[j, t]))
        fams, todo = [], [(set(c), 1.0) for c in nx.connected_components(G)]
        while todo:
            nodes, res = todo.pop()
            if len(nodes) <= max_size:
                if len(nodes) >= 4: fams.append(sorted(nodes))
                continue
            parts = nx.community.louvain_communities(G.subgraph(nodes), weight="weight", resolution=res, seed=0)
            if len(parts) == 1: todo.append((nodes, res * 2)); continue
            todo.extend((set(p), res * 1.5) for p in parts)
        results = []
        for fi, fam in enumerate(fams):
            Y = X[fam]
            coords, evf, _ = pca(Y, 3)
            n = len(fam)
            if n not in nulls: nulls[n] = null_planarity(n)
            n1, n2 = nulls[n]
            line = (evf[0] - n1) / (1 - n1)
            plane = (evf[:2].sum() - n2) / (1 - n2)
            words_ = [labels[i] for i in fam]
            known = Counter(fam_of[w][0] for w in words_ if w in fam_of)
            fam_name, fam_n = known.most_common(1)[0] if known else (None, 0)
            order = None
            if fam_name and fam_n >= 4:
                F = core.FAMILIES[fam_name]
                pos = [(fam_of[w][1], r) for r, w in enumerate(words_) if w in fam_of and fam_of[w][0] == fam_name]
                pos.sort()
                cent = np.stack([coords[r] for _, r in pos])
                if F["order"] == "linear" or len(pos) == len(F["vals"]):
                    order = order_score(cent, F["order"])
            results.append(dict(layer=L, family=fi, n=n, line=round(float(line), 3), plane=round(float(plane), 3),
                                pc123=[round(float(x), 3) for x in evf[:3]], words=words_,
                                counts=[int(wcount[i]) for i in fam], known=fam_name, known_n=fam_n,
                                order=order, xyz=[[round(float(v), 3) for v in c] for c in coords]))
        results.sort(key=lambda r: -r["plane"])
        out[L] = results
        print(f"L{L}: {len(labels):,} words, {len(fams)} families of 4-{max_size}")
        print("  most planar families of 6+ words (top-2 axes hold far more than chance):")
        for r in [x for x in results if x["n"] >= 6][:30]:
            kn = f"  [{r['known']} {r['known_n']}{' order ' + str(r['order'][0]) + '/' + str(r['order'][1]) if r['order'] else ''}]" if r["known"] else ""
            print(f"    f{r['family']:4d} n {r['n']:2d} plane {r['plane']:.2f} line {r['line']:.2f}{kn} :: {' '.join(r['words'][:14])}")
        for name in ("weekday", "month", "number", "ordinal", "color", "season", "compass"):
            where = Counter()
            for r in results:
                for w in r["words"]:
                    if fam_of.get(w, ("",))[0] == name: where[r["family"]] += 1
            tot = sum(1 for w in labels if fam_of.get(w, ("",))[0] == name)
            best = where.most_common(1)
            if best:
                r = next(x for x in results if x["family"] == best[0][0])
                print(f"  {name:8s}: {best[0][1]}/{tot} words in f{best[0][0]} (n {r['n']}, plane {r['plane']:.2f}, "
                      f"order {r['order']}) :: {' '.join(r['words'][:16])}")
    json.dump({str(k): v for k, v in out.items()}, open(families_out or f"{run}/word_families.json", "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probes", "groups", "words"])
    ap.add_argument("--drop-pcs", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layers", default="12")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--probe-run", default=None)
    ap.add_argument("--min-fire", type=int, default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    if a.what == "probes": probes(a.run, layers, a.cache)
    elif a.what == "words": words(a.run, layers, a.cache, min_count=a.min_count, drop_pcs=a.drop_pcs)
    else: groups(a.run, layers, a.cache, a.probe_run, a.min_fire)
