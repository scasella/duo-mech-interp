# Combine family_sweep.py runs on two models into one verdict per family.
#   python tools/shapes/sweep_summary.py <sweep json, model A> <sweep json, model B> <out.json>
# Ordered: order along the top principal axis (linear) or a circle on the top plane (cyclic) at p < 0.05 at some layer,
# in both models (the same test as scale_validate.py). Secondary: the best of the top 3 axes / planes, with a
# permutation null that also takes the best of 3.
# With 4 values the smallest possible p is 0.083, so 4-value families are reported as "too few values" unless perfectly
# ordered at every layer of both models. Frequency: flagged when the known order itself tracks word frequency
# (|rank corr| >= 0.8) and the order no longer holds after partialling frequency out (|partial| < 0.5).
# Bend: a second or third axis that bends as a quadratic of the order (Fisher-combined over layers and models; the
# tests share words, so treat the combined p as descriptive).
import json, math, sys
import numpy as np
from scipy.stats import chi2

A, B = (json.load(open(p)) for p in sys.argv[1:3])
out = sys.argv[3]
rows = []
for name, fa in A.items():
    fb = B.get(name)
    if fb is None: continue
    V = len(fa["vals"]); lin = fa["order"] == "linear"
    def per_model(f):
        r = []
        for L, row in f["layers"].items():
            o = row["main"]
            r.append(dict(L=int(L), p=o["p"], stat=o["rho"] if lin else o["score"], p_best=o["p_best"],
                          stat_best=o["rho_best"] if lin else o["score_best"],
                          axis=o["best_axis"] if lin else "".join(str(i + 1) for i in o["best_plane"]), ev=o["ev"],
                          partial=o.get("rho_partial"), ofreq=o.get("order_vs_freq"), bend_p=o.get("p_horseshoe"),
                          alt={k: (v["score"], v["p"]) for k, v in row.items() if k not in ("main", "steer")},
                          steer=row.get("steer", {}), pcs=o["pcs"]))
        return r
    ra, rb = per_model(fa), per_model(fb)
    def ok(r):
        best = min(r, key=lambda x: x["p"])
        return best["p"] < 0.05, best
    oa, ba = ok(ra); ob, bb = ok(rb)
    low = all(min(x["p_best"] for x in r) < 0.05 for r in (ra, rb))         # secondary: best of the top 3 axes / planes
    perfect = lin and all(abs(x["stat"]) > 0.999 for x in ra + rb)          # (principal-axis signs are arbitrary)
    if lin and V <= 4:
        verdict = "too few values, but perfectly ordered everywhere" if perfect else "too few values"
    elif oa and ob: verdict = "ordered in both models"
    elif low: verdict = "ordered on a lower axis in both models"
    elif oa or ob: verdict = "ordered in one model"
    else: verdict = "not ordered"
    freq_flag = lin and any(x["ofreq"] is not None and abs(x["ofreq"]) >= 0.8 and abs(x["partial"]) < 0.5 for x in ra + rb)
    bps = [x["bend_p"] for x in ra + rb if x["bend_p"] is not None]
    bend = float(chi2.sf(-2 * sum(math.log(max(p, 1e-4)) for p in bps), 2 * len(bps))) if bps else None
    st = next((x["steer"] for x in ra if x["L"] == 12 and "node_rate" in x["steer"]), {})
    alt = {k: [(x["alt"][k][0], x["alt"][k][1]) for x in ra + rb] for k in (ra[0]["alt"] if ra else {})}
    rows.append(dict(name=name, order=fa["order"], V=V, vals=fa["vals"], verdict=verdict, freq_flag=bool(freq_flag),
                     A=[{k: x[k] for k in ("L", "stat", "p", "axis", "stat_best", "p_best")} for x in ra],
                     B=[{k: x[k] for k in ("L", "stat", "p", "axis", "stat_best", "p_best")} for x in rb],
                     bend_p=bend, node=st.get("node_rate"), chord=st.get("rel_median"), alt=alt,
                     pcs_A=next((x["pcs"] for x in ra if x["L"] == 12), ra[0]["pcs"]),
                     pcs_B=next((x["pcs"] for x in rb if x["L"] == 12), rb[0]["pcs"]), control=name.startswith("(")))

fmt = lambda r, lin, V: " ".join((f"{x['stat']:+.2f}" if lin else f"{x['stat']}/{V}") + f"({x['p']:.2f})"
                               + (f"[PC{x['axis']} {x['stat_best']:+.2f} p{x['p_best']:.2f}]" if lin and x["axis"] != 1 and x["p_best"] < 0.05 else "") for x in r)
order = ["ordered in both models", "too few values, but perfectly ordered everywhere", "ordered on a lower axis in both models", "ordered in one model", "too few values", "not ordered"]
for ctl in (False, True):
    print("\n" + ("CONTROLS" if ctl else "NEW FAMILIES"))
    for r in sorted([r for r in rows if r["control"] == ctl], key=lambda r: (order.index(r["verdict"]), r["name"])):
        lin = r["order"] == "linear"
        extra = "".join(f"  [{k}: " + " ".join(f"{s}/{r['V']}({p:.2f})" for s, p in v) + "]" for k, v in r["alt"].items())
        node = "n/a" if r["node"] is None else f"{r['node']:.0%}"
        chord = "n/a" if r["chord"] is None else f"{r['chord']:.2f}"
        bend = "n/a" if r["bend_p"] is None else f"{r['bend_p']:.3f}"
        print(f"  {r['name']:18s} {r['V']:2d} {r['verdict']:48s} A {fmt(r['A'], lin, r['V'])} | B {fmt(r['B'], lin, r['V'])}"
              f"{'  FREQ?' if r['freq_flag'] else ''}  bend p {bend}  node {node}  chord {chord}{extra}")
json.dump(rows, open(out, "w"), indent=1)
