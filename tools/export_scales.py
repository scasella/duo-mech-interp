# python tools/export_scales.py <family_sweep run dir of 0.8B> <find_scales.py output json> [app dir]
#
# Writes app/scales.json (names, values, layer, strength, suggested questions, a starting point, and the check's
# numbers) and app/scales.bin (each value's template-centred average as float16, scale after scale, value after value).
# Families and settings were chosen from tools/find_scales.py (Modal, GPU, greedy): the layer and strength with the
# most answers naming the pushed word while repeating little (share of repeated word pairs <= 0.15).
import json, os, sys
import numpy as np

run, check = sys.argv[1], sys.argv[2]
app = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
CHOSEN = [  # family in the sweep, id, name, layer, strength, extra questions
    ("(weekday)", "weekday", "Days of the week", 12, 1.0, ["What day is the party?"]),
    ("life stage", "life-stage", "Life stages", 12, 1.2, ["Who is playing in the park?"]),
    ("wind", "wind", "Wind", 6, 1.0, ["Describe a morning at sea."]),
    ("geographic scale", "reach", "How far", 6, 1.2, ["Where is the new rule in force?"]),
    ("corporate rank", "job-title", "Job titles", 12, 1.2, ["Who called the meeting?"]),
    ("schedule", "how-often", "How often", 12, 0.8, ["How often do you back up your files?"]),
    ("closeness", "closeness", "Closeness", 12, 0.8, ["Who did you call when you got the news?"]),
    ("settlement", "settlement", "Places to live", 12, 0.6, ["Where does your grandmother live?"]),
    ("(temperature)", "temperature", "Temperature", 6, 1.2, ["How does the coffee taste?"]),
    ("nobility", "nobility", "Nobility", 12, 0.8, ["Who rode into the village?"]),
    ("water body", "water", "Water", 6, 1.3, ["Where did the children play?"]),
]
res = json.load(open(check))["families"]
meta, blobs, offset = [], [], 0
for fam, sid, name, layer, strength, extra in CHOSEN:
    mu = np.load(f"{run}/{fam.replace(' ', '_')}_L{layer}_means.npy").astype(np.float32)
    R = res[fam]; vals = R["vals"]; V = len(vals)
    run_ = min(R["runs"], key=lambda r: abs(r["layer"] - layer) + abs(r["strength"] - strength))
    # start on a middle word where the real-word path names it and the straight line does not (nearest the middle)
    words = lambda t: t.lower()
    path_hit = [vals[v].lower() in words(run_["texts"][v]) for v in range(V)]
    ends = (0, V // 2) if fam == "(weekday)" else (0, V - 1)
    chord_hit = {v: vals[v].lower() in words(run_["chord_texts"][v - ends[0]]) for v in range(ends[0] + 1, ends[1])}
    good = [v for v in chord_hit if path_hit[v] and not chord_hit[v]] or [v for v in range(1, V - 1) if path_hit[v]] or [V // 2]
    start = min(good, key=lambda v: abs(v - (V - 1) / 2))
    q = R["prompt"]
    meta.append(dict(id=sid, name=name, vals=vals, layer=layer, strength=strength, start=start, offset=offset,
                     suggest=[q, *[e + " Answer in one sentence." for e in extra]],  # without it the model tends to refuse ("I don't have real-time data")
                     check=dict(named=round(run_["hit"], 2), middle_path=run_["inner_path"], middle_chord=run_["inner_chord"])))
    blobs.append(mu.astype(np.float16))
    offset += V
H = blobs[0].shape[1]
np.concatenate(blobs).tofile(os.path.join(app, "scales.bin"))
json.dump(dict(hidden=H, model="Qwen/Qwen3.5-0.8B", scales=meta), open(os.path.join(app, "scales.json"), "w"), indent=1)
for m in meta:
    print(f"{m['name']:18s} L{m['layer']:2d} x{m['strength']:<4g} start {m['vals'][m['start']]:>12s}  named {m['check']['named']:.0%}  "
          f"middle {m['check']['middle_path']} vs {m['check']['middle_chord']}")
print(f"{offset} vectors of {H}: {offset * H * 2 / 1024:.0f} KB")
