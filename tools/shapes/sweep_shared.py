# Do the ordered families from family_sweep.py share their order direction (as weekdays, months and numbers share a
# position code)?
#   python tools/shapes/sweep_shared.py <sweep dir, model A> <sweep dir, model B> <summary json> <out.json> [layer]
# Order direction of a family: covariance of its template-centred value means with the value's rank (unit length).
# For each pair of families: cosine of their directions, as a z-score against the same cosine with both families'
# ranks shuffled (1000 shuffles). Families sharing words are skipped. Kept: z >= 3 in both models.
# Groups of families (social rank, letter sequence, magnitude, age and time): mean cosine within the group against
# shuffled ranks, and its excess over the mean cosine of pairs from different groups. A pooled word-frequency direction
# is removed first (pass --keep-freq to skip that).
import json, os, sys
import numpy as np

unit = lambda v: v / (np.linalg.norm(v) + 1e-12)


def direction(mu, rank):
    r = rank - rank.mean(); return unit(r @ (mu - mu.mean(0)))


GROUPS = {"social rank": ["military rank", "nobility", "corporate rank", "academic rank", "church rank", "school year", "school level", "degree"],
          "letter sequence": ["letter grade", "greek letter", "NATO alphabet"],
          "magnitude": ["settlement", "geographic scale", "water body", "wind", "(size)", "(number)", "coffee size", "clothing size", "rock size", "ensemble"],
          "age and time": ["life stage", "decade of life", "generation", "(how soon)", "schedule", "historical era"]}
GROUPS["magnitude, new families only"] = [f for f in GROUPS["magnitude"] if not f.startswith("(")]
member = {f: g for g, fs in GROUPS.items() if not g.startswith("magnitude,") for f in fs}


def main():
    da, db, summ, out = sys.argv[1:5]
    L = sys.argv[5] if len(sys.argv) > 5 else "12"
    rows = {r["name"]: r for r in json.load(open(summ))}
    keep = [n for n, r in rows.items() if r["order"] == "linear" and r["verdict"] != "not ordered" and r["V"] >= 4]
    rng = np.random.default_rng(0)

    def load(d, n):
        return np.load(f"{d}/{n.replace(' ', '_')}_L{L}_means.npy").astype(np.float32)

    NOFREQ = "--keep-freq" not in sys.argv
    res, groups = {}, {}
    for tag, d in (("A", da), ("B", db)):
        M = {n: load(d, n) for n in keep}
        if NOFREQ:
            # remove one pooled word-frequency direction (covariance of within-family centred means with Zipf frequency)
            zipf = {n: np.array(v["zipf"]) for n, v in json.load(open(f"{d}/family_sweep.json")).items() if n in keep}
            X = np.concatenate([M[n] - M[n].mean(0) for n in keep]); y = np.concatenate([zipf[n] - zipf[n].mean() for n in keep])
            fdir = unit(y @ X); M = {n: m - np.outer(m @ fdir, fdir) for n, m in M.items()}
        dirs = {n: direction(M[n], np.arange(len(M[n]), dtype=float)) for n in keep}
        perms = {n: [direction(M[n], rng.permutation(len(M[n])).astype(float)) for _ in range(1000)] for n in keep}
        Z = {}
        for i, a in enumerate(keep):
            for b in keep[i + 1:]:
                if set(rows[a]["vals"]) & set(rows[b]["vals"]): continue
                null = np.array([pa @ pb for pa, pb in zip(perms[a], perms[b])])
                Z[(a, b)] = ((dirs[a] @ dirs[b]) - null.mean()) / null.std(), float(dirs[a] @ dirs[b])
        res[tag] = Z
        # group test: mean within-group cosine vs the same mean with every family's ranks shuffled; and its excess over
        # the mean cosine of pairs across groups (a direction shared by every ordered family would raise both)
        grouped = [f for f in dirs if f in member]
        cross = [(a, b) for i, a in enumerate(grouped) for b in grouped[i + 1:]
                 if member[a] != member[b] and not set(rows[a]["vals"]) & set(rows[b]["vals"])]
        cross_obs = np.mean([dirs[a] @ dirs[b] for a, b in cross])
        cross_null = np.array([np.mean([perms[a][i] @ perms[b][i] for a, b in cross]) for i in range(1000)])
        for g, fams in GROUPS.items():
            fs = [f for f in fams if f in dirs]
            pr = [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:] if not set(rows[a]["vals"]) & set(rows[b]["vals"])]
            if g == "social rank":
                groups.setdefault("(across groups)", {})[tag] = dict(families=grouped, pairs=len(cross), mean_cos=round(float(cross_obs), 3),
                    z=round(float((cross_obs - cross_null.mean()) / cross_null.std()), 2), p=float(np.mean(cross_null >= cross_obs)))
            if len(pr) < 2: continue
            obs = np.mean([dirs[a] @ dirs[b] for a, b in pr])
            null = np.array([np.mean([perms[a][i] @ perms[b][i] for a, b in pr]) for i in range(1000)])
            ex, exn = obs - cross_obs, null - cross_null
            groups.setdefault(g, {})[tag] = dict(families=fs, pairs=len(pr), mean_cos=round(float(obs), 3),
                                                 z=round(float((obs - null.mean()) / null.std()), 2), p=float(np.mean(null >= obs)),
                                                 excess=round(float(ex), 3), p_excess=float(np.mean(exn >= ex)))
    pairs = []
    for k in res["A"]:
        za, ca = res["A"][k]; zb, cb = res["B"][k]
        pairs.append(dict(a=k[0], b=k[1], za=round(float(za), 2), zb=round(float(zb), 2), cos_a=round(ca, 3), cos_b=round(cb, 3)))
    pairs.sort(key=lambda p: -min(abs(p["za"]), abs(p["zb"])) * (np.sign(p["za"]) == np.sign(p["zb"])))
    n = len(pairs)
    print(f"layer {L}: {len(keep)} ordered families, {n} pairs (largest |z| expected by chance in one model ≈ {np.sqrt(2 * np.log(max(n, 2))):.1f})")
    for p in pairs[:30]:
        flag = "  <- both models" if min(abs(p["za"]), abs(p["zb"])) >= 3 and np.sign(p["za"]) == np.sign(p["zb"]) else ""
        print(f"  z {p['za']:+5.1f} / {p['zb']:+5.1f}  cos {p['cos_a']:+.2f} / {p['cos_b']:+.2f}  {p['a']:18s} ~ {p['b']}{flag}")
    print("groups (mean cosine of order directions within the group, vs shuffled ranks):")
    for g, v in groups.items():
        print(f"  {g:16s} " + " | ".join(f"{t}: cos {x['mean_cos']:+.3f} z {x['z']:+.1f} p {x['p']:.3f} ({x['pairs']} pairs)"
                                           + (f" excess over across-group {x['excess']:+.3f} p {x['p_excess']:.3f}" if "excess" in x else "") for t, x in v.items())
              + f"   [{', '.join(v['A']['families'])}]")
    json.dump(dict(layer=L, families=keep, pairs=pairs, groups=groups), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
