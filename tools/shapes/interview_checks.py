# python tools/shapes/interview_checks.py <dir holding words30m and words08 runs> <HF cache dir>
# Two structures from the Goodfire interview, checked on our existing 30M-token word averages (2B and 0.8B):
# the periodic table (atomic number, period, group) and the affective circumplex (valence x arousal).
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import core
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=sys.argv[2])
EL = {"hydrogen": (1, 1, 1), "helium": (2, 1, 18), "lithium": (3, 2, 1), "beryllium": (4, 2, 2), "boron": (5, 2, 13), "carbon": (6, 2, 14),
      "nitrogen": (7, 2, 15), "oxygen": (8, 2, 16), "fluorine": (9, 2, 17), "neon": (10, 2, 18), "sodium": (11, 3, 1), "magnesium": (12, 3, 2),
      "aluminum": (13, 3, 13), "silicon": (14, 3, 14), "phosphorus": (15, 3, 15), "sulfur": (16, 3, 16), "chlorine": (17, 3, 17), "argon": (18, 3, 18),
      "potassium": (19, 4, 1), "calcium": (20, 4, 2), "titanium": (22, 4, 4), "chromium": (24, 4, 6), "manganese": (25, 4, 7), "iron": (26, 4, 8),
      "cobalt": (27, 4, 9), "nickel": (28, 4, 10), "copper": (29, 4, 11), "zinc": (30, 4, 12), "arsenic": (33, 4, 15), "selenium": (34, 4, 16),
      "bromine": (35, 4, 17), "krypton": (36, 4, 18), "silver": (47, 5, 11), "tin": (50, 5, 14), "iodine": (53, 5, 17), "xenon": (54, 5, 18),
      "tungsten": (74, 6, 6), "platinum": (78, 6, 10), "gold": (79, 6, 11), "mercury": (80, 6, 12), "lead": (82, 6, 14), "radon": (86, 6, 18),
      "radium": (88, 7, 2), "uranium": (92, 7, 3), "plutonium": (94, 7, 3), "lithium-": None}
EMO = {"excited": (1, 1), "thrilled": (1, 1), "delighted": (1, 1), "enthusiastic": (1, 1), "elated": (1, 1), "joyful": (1, 1), "ecstatic": (1, 1),
       "calm": (1, -1), "relaxed": (1, -1), "content": (1, -1), "serene": (1, -1), "peaceful": (1, -1), "satisfied": (1, -1), "comfortable": (1, -1),
       "angry": (-1, 1), "afraid": (-1, 1), "scared": (-1, 1), "anxious": (-1, 1), "nervous": (-1, 1), "tense": (-1, 1), "furious": (-1, 1), "terrified": (-1, 1),
       "sad": (-1, -1), "depressed": (-1, -1), "bored": (-1, -1), "tired": (-1, -1), "gloomy": (-1, -1), "miserable": (-1, -1), "lonely": (-1, -1), "exhausted": (-1, -1)}
def loo_ridge(Z, Y, lam=1.0):
    pred = np.zeros_like(Y, dtype=float)
    for i in range(len(Z)):
        tr = np.arange(len(Z)) != i; A = np.c_[Z[tr], np.ones(tr.sum())]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[tr]); pred[i] = np.r_[Z[i], 1.0] @ W
    return pred
for tag, run in (("2B", sys.argv[1] + "/words30m"), ("0.8B", sys.argv[1] + "/words08")):
    ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy"); row = {int(t): r for r, t in enumerate(ids)}
    for L in (6, 12, 18):
        M = np.load(f"{run}/L{L}_word_means.npy")
        gm = M[np.random.default_rng(0).choice(len(M), 20000, replace=False)].astype(np.float32).mean(0)
        def vec(w):
            out = [(M[row[t[0]]].astype(np.float32), counts[row[t[0]]]) for s in (" " + w, " " + w.capitalize())
                   for t in [tok(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
            if not out: return None
            wt = np.array([c for _, c in out], float); return (np.stack([v for v, _ in out]) * wt[:, None]).sum(0) / wt.sum()
        rng = np.random.default_rng(0)
        # periodic table
        els = [e for e, v in EL.items() if v and vec(e) is not None]
        X = np.stack([vec(e) for e in els]) - gm; X -= X.mean(0)
        Vt = np.linalg.svd(X, full_matrices=False)[2]; Z = X @ Vt[:12].T
        Y = np.array([EL[e] for e in els], float)
        pred = loo_ridge(Z, Y)
        r = [np.corrcoef(pred[:, k], Y[:, k])[0, 1] for k in range(3)]
        null = np.percentile([np.mean([np.corrcoef(p[:, k], Ys[:, k])[0, 1] for k in range(3)]) for Ys in [Y[rng.permutation(len(Y))] for _ in range(100)] for p in [loo_ridge(Z, Ys)]], 95)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True); S = Xn @ Xn.T; np.fill_diagonal(S, -9)
        nn = S.argmax(1); grp = Y[:, 2]
        same = np.mean(grp[nn] == grp); chance = np.mean([np.mean(grp[rng.permutation(len(grp))] == grp) for _ in range(200)])
        print(f"{tag:4s} L{L:2d} periodic table: {len(els)} elements | held-out r: atomic number {r[0]:.2f}, period {r[1]:.2f}, group {r[2]:.2f} "
              f"(shuffled 95th pct {null:.2f}) | nearest neighbour in the same group {same:.0%} (chance {chance:.0%})")
        # affective circumplex
        ws = [w for w in EMO if vec(w) is not None]
        X = np.stack([vec(w) for w in ws]) - gm; X -= X.mean(0)
        Vt = np.linalg.svd(X, full_matrices=False)[2]; Z = X @ Vt[:8].T
        Y = np.array([EMO[w] for w in ws], float)
        pred = loo_ridge(Z, Y)
        acc = [(np.sign(pred[:, k]) == Y[:, k]).mean() for k in range(2)]
        nulls = [np.mean([(np.sign(p[:, k]) == Ys[:, k]).mean() for k in range(2)]) for Ys in [Y[rng.permutation(len(Y))] for _ in range(100)] for p in [loo_ridge(Z, Ys)]]
        pc = X @ Vt[:2].T
        quad_ok = np.mean([(np.sign(np.corrcoef(pc[:, 0], Y[:, 0])[0, 1]) * pc[i, 0] * Y[i, 0] > 0) for i in range(len(ws))])
        print(f"{tag:4s} L{L:2d} emotion wheel: {len(ws)} words | held-out sign accuracy: valence {acc[0]:.0%}, arousal {acc[1]:.0%} "
              f"(shuffled 95th pct {np.percentile(nulls, 95):.0%}) | PC1 corr with valence {np.corrcoef(pc[:, 0], Y[:, 0])[0, 1]:+.2f}, "
              f"PC2 with arousal {np.corrcoef(pc[:, 1], Y[:, 1])[0, 1]:+.2f}")
