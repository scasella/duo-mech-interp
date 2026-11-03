# Re-test the atlas's pair-specific alignments with held-out word lists that share no words with the originals.
# Same measure as axis_atlas.py: cosine between scale axes as a z-score against random-word axes, after removing the
# valence and word-frequency axes ("controlled") and also the scales' shared "more vs less" component ("specific").
#   python tools/shapes/atlas_validate.py <run dir> <HF cache or ""> [layers]
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from axis_atlas import SCALES
from transformers import AutoTokenizer

HELD_OUT = {
    "seriousness": ("humorous informal fun funny comic witty", "earnest sober formal strict severe"),
    "hardness": ("tender elastic limp floppy", "brittle hardened dense"),
    "morality": ("unethical dishonest greedy unfair", "ethical decent honorable fair"),
    "cleanliness": ("polluted contaminated stained muddy", "washed sterile sanitary fresh"),
}
PAIRS = [("seriousness", "hardness"), ("morality", "cleanliness"), ("temperature", "friendliness"), ("weight", "seriousness"),
         ("weight", "importance"), ("brightness", "mood"), ("loudness", "size")]

run, cache = sys.argv[1], (sys.argv[2] or None)
LAYERS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["6", "12", "18"]
tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache)
ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
row = {int(t): r for r, t in enumerate(ids)}
rng = np.random.default_rng(1)
unit = lambda v: v / np.linalg.norm(v)
for L in LAYERS:
    M = np.load(f"{run}/L{L}_word_means.npy").astype(np.float32)
    pool = np.array([r for r in range(len(ids)) if counts[r] >= 50]); gm = M[pool].mean(0)
    Xp = M[pool] - gm; y = np.log(counts[pool]); y -= y.mean()
    _, _, Vt = np.linalg.svd(Xp[rng.choice(len(Xp), 8000, replace=False)], full_matrices=False)
    Zp = Xp @ Vt[:256].T; freq_axis = unit(np.linalg.solve(Zp.T @ Zp + 1e3 * np.eye(256), Zp.T @ y) @ Vt[:256])
    def vec(w):
        got = [(M[row[t[0]]], counts[row[t[0]]]) for s in (" " + w, " " + w.capitalize())
               for t in [tok(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
        if not got: return None
        wt = np.array([c for _, c in got], float); return (np.stack([v for v, _ in got]) * wt[:, None]).sum(0) / wt.sum() - gm
    def pole_axis(lo, hi, proj):
        lo_v = [v for v in (vec(x) for x in lo.split()) if v is not None]; hi_v = [v for v in (vec(x) for x in hi.split()) if v is not None]
        if len(lo_v) < 3 or len(hi_v) < 3: return None
        return unit(proj(np.stack(hi_v)).mean(0) - proj(np.stack(lo_v)).mean(0))
    def projector(R):
        Q, _ = np.linalg.qr(np.stack(R).T); return lambda X: X - (X @ Q) @ Q.T
    val = pole_axis(*SCALES["valence"], lambda X: X)
    pc = projector([val, freq_axis])
    orig = {n: pole_axis(*SCALES[n], pc) for n in SCALES if n != "valence"}
    orig = {n: a for n, a in orig.items() if a is not None}
    A = np.stack(list(orig.values())); _, _, Vv = np.linalg.svd(A, full_matrices=False)
    shared = Vv[0] * np.sign((A @ Vv[0]).sum())
    ps = projector([val, freq_axis, shared])
    def rand_sd(proj):
        r = np.stack([unit(proj(M[rng.choice(pool, 5)] - gm).mean(0) - proj(M[rng.choice(pool, 5)] - gm).mean(0)) for _ in range(400)])
        return float(np.std(r[:200] @ r[200:].T))
    sd_c, sd_s = rand_sd(pc), rand_sd(ps)
    print(f"\nlayer {L}: z-scores, controlled / specific (o = original words, h = held-out words)")
    for a, b in PAIRS:
        cells = []
        for la, sa in (("o", SCALES.get(a)), ("h", HELD_OUT.get(a))):
            for lb, sb in (("o", SCALES.get(b)), ("h", HELD_OUT.get(b))):
                if sa is None or sb is None: continue
                ca, cb = pole_axis(*sa, pc), pole_axis(*sb, pc)
                sa_, sb_ = pole_axis(*sa, ps), pole_axis(*sb, ps)
                if ca is None or cb is None: cells.append(f"{la}{lb} n/a"); continue
                cells.append(f"{la}{lb} {ca @ cb / sd_c:+4.1f}/{sa_ @ sb_ / sd_s:+4.1f}")
        print(f"  {a:12s} ~ {b:12s} " + "   ".join(cells))
