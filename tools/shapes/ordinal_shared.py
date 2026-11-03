# Is there a domain-general ordinal direction? Reconciling our pooled result (sweep_shared.py: a weak "later in the
# sequence" direction shared across ordered families) with arXiv 2606.01269, which found per-domain probe directions for
# digits, months and size adjectives in Qwen2.5-1.5B "nearly orthogonal at every layer (|cos| < 0.2)" and concluded
# there is no single ordinality direction. That test has no null, and in 1,536 dimensions a real shared component of
# cosine 0.03-0.05 is far below 0.2.
#   python tools/shapes/ordinal_shared.py isolated <model> <out.json>
#       Their protocol: each item tokenized on its own (no special tokens), residual averaged over its sub-tokens, every
#       layer; per-domain direction by a ridge probe of rank (small and large ridge) and by covariance with rank; pairwise
#       cosines with a null from shuffled ranks (2,000 shuffles), and the three pairs pooled.
#   python tools/shapes/ordinal_shared.py power <summary json> <out.json> <sweep dir>:<layer>[:<name>] ...
#       In our contextual family averages: the pooled test over all cross-group pairs, the paper's three domains
#       (number words, months, sizes) alone, and how often a test on three random families from different groups
#       would detect the shared component (p < 0.05) or show any |cos| > 0.2.
import json, os, sys
import numpy as np

unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
rng = np.random.default_rng(0)
DOMAINS = {"digits": [str(i) for i in range(10)],
           "months": "January February March April May June July August September October November December".split(),
           "sizes": "tiny little small medium large huge enormous gigantic".split()}


def cov_dir(X, r):
    r = r - r.mean(); return unit(r @ (X - X.mean(0)))


def ridge_dir(X, r, lam):
    Xc = X - X.mean(0); rc = r - r.mean()
    K = Xc @ Xc.T
    a = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K)), rc)   # lam relative to the mean eigenvalue
    return unit(Xc.T @ a)


def pair_stats(D, how, n_perm=2000):
    """D: {domain: X (items x d)}. Cosine of each pair of rank directions, with a shuffled-rank null; and the pooled mean."""
    names = list(D)
    dirs = {n: how(D[n], np.arange(len(D[n]), dtype=float)) for n in names}
    perms = {n: [how(D[n], rng.permutation(len(D[n])).astype(float)) for _ in range(n_perm)] for n in names}
    out, pooled_obs, pooled_null = {}, [], np.zeros(n_perm)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = float(dirs[a] @ dirs[b]); null = np.array([pa @ pb for pa, pb in zip(perms[a], perms[b])])
            out[f"{a}~{b}"] = dict(cos=round(c, 4), z=round(float((c - null.mean()) / null.std()), 2), p=float(np.mean(null >= c)),
                                   null_sd=round(float(null.std()), 4))
            pooled_obs.append(c); pooled_null += null
    pooled_null /= len(pooled_obs); m = float(np.mean(pooled_obs))
    out["pooled"] = dict(cos=round(m, 4), z=round(float((m - pooled_null.mean()) / pooled_null.std()), 2), p=float(np.mean(pooled_null >= m)))
    return out


def isolated(model, out):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval().to(dev)
    reps = {}
    for dom, items in DOMAINS.items():
        H = []
        for w in items:
            ids = tok(w, add_special_tokens=False).input_ids
            with torch.inference_mode():
                hs = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states   # embeddings + each layer
            H.append(torch.stack([h[0].mean(0) for h in hs[1:]]).float().cpu().numpy())        # layers x d, mean over sub-tokens
        reps[dom] = np.stack(H, 1)                                                            # layers x items x d
    n_layers = reps["digits"].shape[0]
    res = dict(model=model, layers=n_layers, by_layer=[])
    for l in range(n_layers):
        D = {d: reps[d][l] for d in DOMAINS}
        row = dict(layer=l, cov=pair_stats(D, cov_dir), ridge_small=pair_stats(D, lambda X, r: ridge_dir(X, r, 1e-3)),
                   ridge_large=pair_stats(D, lambda X, r: ridge_dir(X, r, 10.0)))
        res["by_layer"].append(row)
        c = row["cov"]; s = row["ridge_small"]
        print(f"L{l:2d}  covariance dirs: " + "  ".join(f"{k} {v['cos']:+.3f} (z {v['z']:+.1f})" for k, v in c.items())
              + f"   | small-ridge probe pooled {s['pooled']['cos']:+.3f} (z {s['pooled']['z']:+.1f})", flush=True)
    for how in ("cov", "ridge_small", "ridge_large"):
        mx = max(abs(r[how][k]["cos"]) for r in res["by_layer"] for k in r[how] if k != "pooled")
        npos = sum(r[how]["pooled"]["p"] < 0.05 for r in res["by_layer"])
        print(f"{how:12s}: largest |cos| over pairs and layers {mx:.3f}; pooled cosine p < 0.05 at {npos}/{n_layers} layers")
        res[f"summary_{how}"] = dict(max_abs_cos=mx, layers_pooled_significant=npos)
    json.dump(res, open(out, "w"), indent=1)


def power(summ, out, specs):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sweep_shared import member, direction
    rows = {r["name"]: r for r in json.load(open(summ))}
    keep = [n for n, r in rows.items() if r["order"] == "linear" and r["verdict"] != "not ordered" and r["V"] >= 4 and n in member]
    res = {}
    for spec in specs:
        d, L, *nm = spec.split(":"); name = nm[0] if nm else d
        M = {n: np.load(f"{d}/{n.replace(' ', '_')}_L{L}_means.npy").astype(np.float32) for n in keep + ["(month)"]
             if os.path.exists(f"{d}/{n.replace(' ', '_')}_L{L}_means.npy")}
        zipf = {n: np.array(v["zipf"]) for n, v in json.load(open(f"{d}/family_sweep.json")).items() if n in M}
        X = np.concatenate([M[n] - M[n].mean(0) for n in M]); y = np.concatenate([zipf[n] - zipf[n].mean() for n in M])
        f = unit(y @ X); M = {n: v - np.outer(v @ f, f) for n, v in M.items()}            # pooled frequency direction removed
        fams = [n for n in keep if n in M]
        P = 1000
        dirs = {n: direction(M[n], np.arange(len(M[n]), dtype=float)) for n in M}
        perms = {n: np.stack([direction(M[n], rng.permutation(len(M[n])).astype(float)) for _ in range(P)]) for n in M}
        cross = [(a, b) for i, a in enumerate(fams) for b in fams[i + 1:] if member[a] != member[b] and not set(rows[a]["vals"]) & set(rows[b]["vals"])]
        obs = np.array([dirs[a] @ dirs[b] for a, b in cross])
        null = np.mean([np.einsum("pd,pd->p", perms[a], perms[b]) for a, b in cross], 0)
        rec = dict(n_families=len(fams), n_pairs=len(cross), mean_cos=round(float(obs.mean()), 4),
                   z=round(float((obs.mean() - null.mean()) / null.std()), 2), p=float(np.mean(null >= obs.mean())),
                   share_pairs_abs_cos_over_0_2=round(float(np.mean(np.abs(obs) > 0.2)), 3), max_abs_cos=round(float(np.abs(obs).max()), 3))
        # the paper's three domains, in our contextual averages
        trio = ["(number)", "(month)", "(size)"]
        if all(t in M for t in trio):
            prs = [(trio[0], trio[1]), (trio[0], trio[2]), (trio[1], trio[2])]
            o3 = np.array([dirs[a] @ dirs[b] for a, b in prs]); n3 = np.mean([np.einsum("pd,pd->p", perms[a], perms[b]) for a, b in prs], 0)
            rec["paper_trio"] = dict(cos={f"{a}~{b}": round(float(dirs[a] @ dirs[b]), 3) for a, b in prs}, mean=round(float(o3.mean()), 4),
                                     p=float(np.mean(n3 >= o3.mean())))
        # power: random triples of families from three different groups
        groups = sorted({member[n] for n in fams}); hits, over, trials = 0, 0, 2000
        by = {g: [n for n in fams if member[n] == g] for g in groups}
        for _ in range(trials):
            gs = rng.choice(groups, 3, replace=False); t3 = [rng.choice(by[g]) for g in gs]
            prs = [(t3[0], t3[1]), (t3[0], t3[2]), (t3[1], t3[2])]
            o3 = np.array([dirs[a] @ dirs[b] for a, b in prs]); n3 = np.mean([np.einsum("pd,pd->p", perms[a], perms[b]) for a, b in prs], 0)
            hits += np.mean(n3 >= o3.mean()) < 0.05; over += np.any(np.abs(o3) > 0.2)
        rec["random_trios"] = dict(detected_share=round(hits / trials, 3), any_abs_cos_over_0_2_share=round(over / trials, 3))
        res[name] = rec
        t = rec.get("paper_trio", {})
        print(f"{name:10s} L{L}: {rec['n_pairs']} cross-group pairs of {rec['n_families']} families, mean cos {rec['mean_cos']:+.3f} "
              f"(z {rec['z']:+.1f}, p {rec['p']:.3f}); |cos| > 0.2 in {rec['share_pairs_abs_cos_over_0_2']:.0%} of pairs (max {rec['max_abs_cos']:.2f})\n"
              f"           paper's trio (number words, months, sizes): {t.get('cos')} mean {t.get('mean')} p {t.get('p')}\n"
              f"           three random families from different groups: shared component detected {rec['random_trios']['detected_share']:.0%} of the time; "
              f"some |cos| > 0.2 in {rec['random_trios']['any_abs_cos_over_0_2_share']:.0%}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "isolated": isolated(sys.argv[2], sys.argv[3])
    else: power(sys.argv[2], sys.argv[3], sys.argv[4:])
