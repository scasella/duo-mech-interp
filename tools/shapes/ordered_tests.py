# python tools/shapes/ordered_tests.py <words run dir> <HF cache dir or ""> <out.json>
# Do discovered word families line up with a real-world quantity? Uses the 30M-token word averages (2B base).
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import core
from transformers import AutoTokenizer
run, cache = sys.argv[1], (sys.argv[2] or None)
tok = AutoTokenizer.from_pretrained(core.BASE, cache_dir=cache)
ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
row = {int(t): r for r, t in enumerate(ids)}
def vec(M, word):
    out = []
    for s in (" " + word, " " + word.capitalize(), word):
        t = tok(s, add_special_tokens=False).input_ids
        if len(t) == 1 and t[0] in row: out.append((M[row[t[0]]].astype(np.float32), counts[row[t[0]]]))
    if not out: return None
    w = np.array([c for _, c in out], np.float64)
    return (np.stack([v for v, _ in out]) * w[:, None]).sum(0) / w.sum()
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
TESTS = {
  "digits":  {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9},
  "tens":    {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1e6, "billion": 1e9},
  "volume":  {"teaspoon": 5, "tablespoon": 15, "ounce": 30, "cup": 240, "pint": 473, "quart": 946, "liter": 1000, "gallon": 3785, "barrel": 159000},
  "length":  {"millimeter": 0.001, "inch": 0.0254, "foot": 0.3048, "feet": 0.3048, "yard": 0.9144, "meter": 1, "kilometer": 1000, "mile": 1609},
  "time":    {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2.6e6, "year": 3.15e7, "decade": 3.15e8, "century": 3.15e9},
  "weight":  {"gram": 1, "ounce": 28, "pound": 454, "kilogram": 1000, "ton": 1e6},
  "luck":    {"unfortunately": -1, "sadly": -1, "regrettably": -1, "unluckily": -1, "luckily": 1, "fortunately": 1, "thankfully": 1, "happily": 1},
  "intensity": {"slightly": 1, "somewhat": 2, "moderately": 3, "considerably": 4, "substantially": 4, "significantly": 4, "dramatically": 5, "drastically": 5},
}
CYCLIC = {"weekday": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
          "month": "January February March April May June July August September October November December".split()}
CITIES = {"london": (51.5, -0.1), "paris": (48.9, 2.4), "berlin": (52.5, 13.4), "madrid": (40.4, -3.7), "rome": (41.9, 12.5),
  "moscow": (55.8, 37.6), "tokyo": (35.7, 139.7), "beijing": (39.9, 116.4), "shanghai": (31.2, 121.5), "delhi": (28.6, 77.2),
  "mumbai": (19.1, 72.9), "cairo": (30.0, 31.2), "sydney": (-33.9, 151.2), "melbourne": (-37.8, 145.0), "chicago": (41.9, -87.6),
  "boston": (42.4, -71.1), "seattle": (47.6, -122.3), "denver": (39.7, -105.0), "dallas": (32.8, -96.8), "houston": (29.8, -95.4),
  "miami": (25.8, -80.2), "atlanta": (33.7, -84.4), "toronto": (43.7, -79.4), "vancouver": (49.3, -123.1), "montreal": (45.5, -73.6),
  "mexico": (19.4, -99.1), "lima": (-12.0, -77.0), "santiago": (-33.4, -70.7), "dublin": (53.3, -6.3), "vienna": (48.2, 16.4),
  "prague": (50.1, 14.4), "athens": (38.0, 23.7), "istanbul": (41.0, 29.0), "tehran": (35.7, 51.4), "baghdad": (33.3, 44.4),
  "nairobi": (-1.3, 36.8), "lagos": (6.5, 3.4), "bangkok": (13.8, 100.5), "singapore": (1.35, 103.8), "seoul": (37.6, 127.0),
  "manila": (14.6, 121.0), "jakarta": (-6.2, 106.8), "stockholm": (59.3, 18.1), "oslo": (59.9, 10.8), "helsinki": (60.2, 24.9),
  "warsaw": (52.2, 21.0), "lisbon": (38.7, -9.1), "amsterdam": (52.4, 4.9), "brussels": (50.8, 4.4), "geneva": (46.2, 6.1),
  "phoenix": (33.4, -112.1), "portland": (45.5, -122.7), "detroit": (42.3, -83.0), "philadelphia": (40.0, -75.2), "washington": (38.9, -77.0)}
res = {}
for L in (6, 12, 18):
    M = np.load(f"{run}/L{L}_word_means.npy")
    gm = M[np.random.default_rng(0).choice(len(M), 20000, replace=False)].astype(np.float32).mean(0)
    for name, d in TESTS.items():
        ws = [w for w in d if vec(M, w) is not None]
        X = np.stack([vec(M, w) for w in ws]) - gm
        X -= X.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = X @ Vt[:3].T
        target = np.log10(np.abs([d[w] for w in ws]) + 1e-9) if name not in ("luck", "intensity", "digits") else np.array([d[w] for w in ws], float)
        rho = [spearman(pcs[:, k], target) for k in range(3)]
        k = int(np.argmax(np.abs(rho)))
        ev = S**2 / (S**2).sum()
        res.setdefault(name, {})[L] = dict(words=ws, rho_pc=[round(r, 3) for r in rho], best_pc=k + 1,
                                           ev=[round(float(e), 3) for e in ev[:3]],
                                           xy=[[round(float(a), 3), round(float(b), 3)] for a, b in pcs[:, :2]])
        print(f"L{L:2d} {name:9s} n {len(ws):2d}  Spearman with the real quantity on PC1/2/3 {rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f}  (top-2 PCs {ev[:2].sum():.0%})  missing {[w for w in d if w not in ws]}")
    for name, vals in CYCLIC.items():                       # do the known circles come back from web text?
        ws = [w for w in vals if vec(M, w) is not None]
        X = np.stack([vec(M, w) for w in ws]) - gm
        X -= X.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = X @ Vt[:2].T
        c = pcs - pcs.mean(0); ang = np.arctan2(c[:, 1], c[:, 0]); n = len(ws)
        rank = {v: r for r, v in enumerate(np.argsort(ang))}
        score = sum((rank[(v + 1) % n] - rank[v]) % n in (1, n - 1) for v in range(n))
        ev = S**2 / (S**2).sum()
        res.setdefault(name, {})[L] = dict(words=ws, cyclic=int(score), n=n, ev=[round(float(e), 3) for e in ev[:3]],
                                           xy=[[round(float(a), 3), round(float(b), 3)] for a, b in pcs])
        print(f"L{L:2d} {name:9s} n {n:2d}  {score}/{n} neighbours in cyclic order (top-2 PCs {ev[:2].sum():.0%})")
    # map: leave-one-out ridge from city averages to latitude and longitude
    cs = [c for c in CITIES if vec(M, c) is not None]
    X = np.stack([vec(M, c) for c in cs]) - gm
    Y = np.array([CITIES[c] for c in cs])
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:20].T                                       # 20 PCs of the cities, then ridge, leave-one-out
    pred = np.zeros_like(Y)
    for i in range(len(cs)):
        tr = np.arange(len(cs)) != i
        A = np.c_[Z[tr], np.ones(tr.sum())]; lam = 1.0
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[tr])
        pred[i] = np.r_[Z[i], 1.0] @ W
    r_lat = float(np.corrcoef(pred[:, 0], Y[:, 0])[0, 1]); r_lon = float(np.corrcoef(pred[:, 1], Y[:, 1])[0, 1])
    # null: shuffle city labels
    null = []
    rng = np.random.default_rng(0)
    for _ in range(50):
        Ys = Y[rng.permutation(len(cs))]; ps = np.zeros_like(Ys)
        for i in range(len(cs)):
            tr = np.arange(len(cs)) != i; A = np.c_[Z[tr], np.ones(tr.sum())]
            W = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ Ys[tr]); ps[i] = np.r_[Z[i], 1.0] @ W
        null.append((np.corrcoef(ps[:, 0], Ys[:, 0])[0, 1] + np.corrcoef(ps[:, 1], Ys[:, 1])[0, 1]) / 2)
    res.setdefault("map", {})[L] = dict(cities=cs, true=Y.tolist(), pred=np.round(pred, 2).tolist(), r_lat=round(r_lat, 3), r_lon=round(r_lon, 3),
                                         null95=round(float(np.percentile(null, 95)), 3))
    print(f"L{L:2d} map       {len(cs)} cities: held-out latitude r {r_lat:.2f}, longitude r {r_lon:.2f} (shuffled-label 95th pct {np.percentile(null, 95):.2f})")
json.dump(res, open(sys.argv[3], "w"))
