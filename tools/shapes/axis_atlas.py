# Axis atlas: does the model reuse one internal axis across domains ("more is up", "affection is warmth",
# "the future is ahead")? Each scale is a set of low-pole and high-pole words; its axis is mean(high) - mean(low) of the
# words' context-averaged residuals (unit length). Alignment of two scales = cosine of their axes, as a z-score against
# axes built from random word sets (5 vs 5 words). (An AUC of one axis on the other scale's words saturates with this
# few words, even for random axes, so it is not used.)
# Three views:
#   raw         the cosine as is
#   controlled  with the good-bad (valence) axis and a word-frequency axis projected out of every word first
#   specific    additionally with the scales' shared first component removed (a common "more vs less" axis, if any),
#               so what remains is alignment particular to that pair
#   python tools/shapes/axis_atlas.py <run dir> <HF cache or ""> <out.json> [layers]
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from transformers import AutoTokenizer

SCALES = {  # low pole | high pole
    "vertical": ("down below beneath bottom lower underneath", "up above top upper overhead upward"),
    "quantity": ("fewer less decrease reduced shortage scarce", "more increase abundant plenty surplus extra"),
    "size": ("tiny small little miniature compact", "huge large big enormous giant massive"),
    "importance": ("unimportant negligible insignificant irrelevant inconsequential", "important crucial vital essential significant"),
    "valence": ("bad terrible awful horrible lousy", "good great excellent wonderful fantastic"),
    "near-far": ("near nearby close adjacent neighboring", "far distant remote faraway outlying"),
    "soon-later": ("soon shortly imminent immediately promptly", "eventually someday ultimately afterward belatedly"),
    "temperature": ("cold cool chilly freezing icy", "hot warm heated boiling scorching"),
    "friendliness": ("hostile rude cruel nasty harsh", "friendly kind gentle nice caring welcoming"),
    "weight": ("lighter lightweight thin", "heavy heavier hefty bulky"),
    "seriousness": ("frivolous playful silly casual lighthearted", "serious grave solemn somber stern"),
    "brightness": ("dark dim gloomy shadowy murky", "bright luminous radiant shiny glowing"),
    "intelligence": ("stupid dumb foolish ignorant idiotic", "smart clever intelligent wise shrewd"),
    "mood": ("sad unhappy depressed miserable sorrowful", "happy cheerful joyful glad delighted"),
    "power": ("weak powerless feeble helpless subordinate", "powerful strong mighty dominant superior"),
    "past-future": ("past previous earlier former prior", "future upcoming subsequent forthcoming coming"),
    "behind-ahead": ("behind back rear backward backwards", "ahead forward front forwards onward"),
    "speed": ("slow sluggish leisurely gradual", "fast quick rapid swift speedy"),
    "loudness": ("quiet silent soft hushed muted", "loud noisy deafening booming thunderous"),
    "age": ("young youthful infant juvenile newborn", "old elderly aged ancient senior"),
    "wealth": ("poor impoverished broke needy destitute", "rich wealthy affluent prosperous lavish"),
    "cost": ("cheap inexpensive affordable discounted", "expensive costly pricey luxurious premium"),
    "difficulty": ("easy simple effortless straightforward", "difficult hard tough challenging arduous"),
    "certainty": ("impossible unlikely doubtful improbable uncertain", "certain likely sure definite inevitable"),
    "frequency": ("never rarely seldom occasionally sometimes", "always often frequently constantly usually"),
    "morality": ("evil wicked sinful immoral corrupt", "virtuous moral righteous noble honest"),
    "health": ("sick ill diseased unhealthy injured", "healthy fit robust vigorous thriving"),
    "cleanliness": ("dirty filthy messy dusty grimy", "clean tidy spotless neat pristine"),
    "hardness": ("soft fluffy squishy flexible supple", "rigid stiff solid firm sturdy"),
    "beginning-end": ("start beginning begin initial opening", "end ending finish final closing"),
}
METAPHORS = [("vertical", "quantity", "more is up"), ("vertical", "valence", "good is up"), ("vertical", "mood", "happy is up"),
             ("vertical", "power", "power is up"), ("size", "importance", "important is big"), ("temperature", "friendliness", "affection is warmth"),
             ("weight", "seriousness", "serious is heavy"), ("weight", "importance", "important is heavy"), ("brightness", "intelligence", "smart is bright"),
             ("brightness", "mood", "happy is bright"), ("near-far", "soon-later", "time is distance"), ("behind-ahead", "past-future", "the future is ahead"),
             ("hardness", "difficulty", "difficult is hard"), ("cleanliness", "morality", "moral is clean"), ("loudness", "size", "loud is big"),
             ("wealth", "power", "rich is powerful"), ("cost", "importance", "valuable is important"), ("speed", "intelligence", "smart is quick")]

def main(run, cache, out, LAYERS):
    tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache or None)
    ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
    row = {int(t): r for r, t in enumerate(ids)}
    rng = np.random.default_rng(0)
    unit = lambda v: v / np.linalg.norm(v)
    res = {}
    for L in LAYERS:
        M = np.load(f"{run}/L{L}_word_means.npy").astype(np.float32)
        pool = np.array([r for r in range(len(ids)) if counts[r] >= 50])
        gm = M[pool].mean(0)
        # word-frequency axis: ridge regression of log count on the (centred) averages over the pool
        Xp = M[pool] - gm; y = np.log(counts[pool]); y = y - y.mean()
        U, S, Vt = np.linalg.svd(Xp[rng.choice(len(Xp), min(len(Xp), 8000), replace=False)], full_matrices=False)
        Z = Xp @ Vt[:256].T
        w = np.linalg.solve(Z.T @ Z + 1e3 * np.eye(256), Z.T @ y)
        freq_axis = unit(w @ Vt[:256])
        def vec(wd):
            got = [(M[row[t[0]]], counts[row[t[0]]]) for s in (" " + wd, " " + wd.capitalize())
                   for t in [tok(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
            if not got: return None
            wt = np.array([c for _, c in got], float)
            return (np.stack([v for v, _ in got]) * wt[:, None]).sum(0) / wt.sum() - gm
        poles = {}
        for name, (lo, hi) in SCALES.items():
            lo_v = [v for v in (vec(x) for x in lo.split()) if v is not None]
            hi_v = [v for v in (vec(x) for x in hi.split()) if v is not None]
            if len(lo_v) >= 3 and len(hi_v) >= 3: poles[name] = (np.stack(lo_v), np.stack(hi_v))
        names = list(poles)
        def axes_with(remove):
            R = np.stack(remove) if remove else None
            def proj(X):
                if R is None: return X
                Q, _ = np.linalg.qr(R.T)
                return X - (X @ Q) @ Q.T
            ax = {n: unit(proj(poles[n][1]).mean(0) - proj(poles[n][0]).mean(0)) for n in names}
            rand = [unit(proj(M[rng.choice(pool, 5)] - gm).mean(0) - proj(M[rng.choice(pool, 5)] - gm).mean(0)) for _ in range(400)]
            return ax, np.stack(rand), proj
        out_L = {}
        ax_raw, rand_raw, _ = axes_with([])
        val = ax_raw["valence"]
        ax_c, rand_c, proj_c = axes_with([val, freq_axis])
        # shared first component of the scale axes (controlled), excluding valence
        names_nv = [n for n in names if n != "valence"]
        A = np.stack([ax_c[n] for n in names_nv])
        _, Sv, Vv = np.linalg.svd(A, full_matrices=False)
        shared = Vv[0] * np.sign((A @ Vv[0]).sum())
        share = float(Sv[0] ** 2 / (Sv ** 2).sum())
        Rn = np.stack([unit(rand_c[i]) for i in range(len(names_nv))])                   # null: same number of random axes
        null_share = []
        for _ in range(200):
            Rr = rand_c[rng.choice(len(rand_c), len(names_nv), replace=False)]
            s = np.linalg.svd(Rr, compute_uv=False); null_share.append(s[0] ** 2 / (s ** 2).sum())
        ax_s, rand_s, _ = axes_with([val, freq_axis, shared])
        loadings = {n: round(float(ax_c[n] @ shared), 3) for n in names_nv}
        views = {}
        for view, ax, rand in (("raw", ax_raw, rand_raw), ("controlled", ax_c, rand_c), ("specific", ax_s, rand_s)):
            nn = names if view == "raw" else names_nv
            C = np.array([[ax[a] @ ax[b] for b in nn] for a in nn])
            sd = float(np.std(rand[: len(rand) // 2] @ rand[len(rand) // 2:].T))          # cosine spread between random axes
            views[view] = dict(names=nn, cos=np.round(C, 3).tolist(), sd=round(sd, 4))
        out_L = dict(views=views, shared_share=round(share, 3), shared_null95=round(float(np.percentile(null_share, 95)), 3),
                     loadings=loadings)
        res[L] = out_L
        print(f"\nlayer {L}: {len(names)} scales. Random-axis cosine spread (sd): raw {views['raw']['sd']:.3f}, "
              f"controlled {views['controlled']['sd']:.3f}, specific {views['specific']['sd']:.3f}")
        print(f"  shared first component of the scale axes holds {share:.0%} of their variance (random axes: 95th pct {np.percentile(null_share, 95):.0%})")
        print("  loading on it: " + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(loadings.items(), key=lambda x: -x[1])))
        def z(view, a, b):
            v = views[view]
            if a not in v["names"] or b not in v["names"]: return None
            return v["cos"][v["names"].index(a)][v["names"].index(b)] / v["sd"]
        print("  metaphor pairs: cosine z-score  raw / controlled (valence, frequency out) / specific (shared axis out too)")
        for a, b, label in METAPHORS:
            zs = [z(v, a, b) for v in ("raw", "controlled", "specific")]
            print(f"    {label:22s} {a:12s} {b:12s} " + " / ".join("  -  " if x is None else f"{x:+5.1f}" for x in zs))
        v = views["specific"]; nn = v["names"]; C = np.array(v["cos"])
        pairs = sorted(((C[i, j] / v["sd"], nn[i], nn[j]) for i in range(len(nn)) for j in range(i + 1, len(nn))), key=lambda x: -abs(x[0]))
        print("  strongest pair-specific alignments (shared axis, valence and frequency removed):")
        for zz, a, b in pairs[:16]:
            print(f"    {a:13s} {b:13s} z {zz:+5.1f}")
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else ["6", "12", "18"])
