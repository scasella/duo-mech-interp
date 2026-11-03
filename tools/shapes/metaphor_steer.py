# Causal tests of conceptual metaphors ("happy is bright", "north is up", "moral is clean", and others), version 2.
#   python tools/shapes/metaphor_steer.py <model dir or HF id> <words run dir> <out.json> [layer]
# Push the residual stream (after `layer`, every token) along a source scale's axis while the model completes a sentence
# about the target domain; read the shift in log-odds of the target's high vs low words. Changes from cross_steer.py:
#   * held-out source words with no target-domain sense (no "gloomy"/"radiant" for brightness -> mood, no "dirty"/"filthy"
#     for cleanliness -> morality), and held-out prompts and readout words for every target;
#   * every push has the same size (the median pole gap over all scales), so one random null per target serves every
#     source, and the hypothesised source can be ranked against all other scales pushing the same target;
#   * two controls: the good-bad (valence) axis removed, or a whole evaluation subspace removed (valence, health,
#     intelligence, wealth, morality, friendliness, mood, cleanliness, minus the pair's own scales);
#   * both directions of the main pairs, four classic metaphors, z against 64 random word-set contrasts (with the same
#     removal), a Bonferroni bar over the tested pairs.
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_atlas import SCALES as _S
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir, run, out = sys.argv[1:4]
L = int(sys.argv[4]) if len(sys.argv) > 4 else 12
DEV = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
N_RAND = int(os.environ.get("N_RAND", 64))
ALPHAS = [1.0, 2.0]
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))

SCALES = dict(_S)
SCALES["north-south"] = ("south southern southeast southwest southeastern", "north northern northeast northwest northeastern")
HELD = {  # held-out source lists (low | high): no word with a sense in the target domain
    "brightness": ("unlit dim murky dusky darkened shaded", "luminous illuminated sunlit lit glowing fluorescent"),
    "cleanliness": ("muddy grimy greasy stained soiled sooty dusty", "washed scrubbed laundered polished sanitized rinsed"),
    "mood": ("sorrowful heartbroken downcast tearful grieving", "delighted elated thrilled ecstatic overjoyed"),
    "morality": ("sinful shameful vile despicable villainous", "virtuous admirable principled commendable saintly"),
    "vertical": ("lower descending downstairs downhill sinking", "higher ascending upstairs uphill rising"),
    "north-south": ("southerly southward southbound", "northerly northward northbound"),
    "temperature": ("chilly frosty wintry frozen icy", "sweltering scorching steaming toasty balmy"),
    "size": ("minuscule teeny pocket petite microscopic", "colossal gigantic towering immense vast"),
}
T = lambda prompts, lo, hi: (prompts.split(" | "), lo.split(), hi.split())
TARGETS = {  # target: (prompts, low readout, high readout) x (original, held-out)
    "mood": (T("When she heard the news, she felt | After the funeral, everyone was | On the first day of summer, the children were | He stared out of the window, feeling | Winning the prize made him | Lately I have been feeling | The mood in the room was | After the long winter, people felt",
               "sad unhappy depressed miserable", "happy cheerful joyful glad"),
             T("Reading her letter, he became | By the end of the trip we were all | The puppy made the old man | Hearing the song again, she felt | At the reunion, the brothers were | After the exam, the students felt | Looking at the photos, I felt | When the rain stopped, the village was",
               "upset lonely hopeless devastated", "delighted thrilled excited pleased")),  # single tokens only
    "vertical": (T("The elevator was going | She looked | The balloon floated | The bird flew | He climbed the stairs, going | The road sloped | The sun was coming | The children ran",
                   "down downward downwards below beneath", "up upward upwards above overhead"),
                 T("The kite drifted | The water level kept | The plane began | The smoke rose | Prices moved | The diver swam | The rocket shot | He pointed",
                   "lower descending downstairs falling", "higher ascending upstairs rising")),
    "morality": (T("Stealing money from a charity is | Lying to protect yourself is | Helping a stranger in need is | Cheating on an exam is | Telling the truth even when it hurts is | The way the company treated its workers was | In my opinion, what he did was | Breaking a promise to a friend is",
                   "evil wicked immoral corrupt unethical dishonest cruel wrong", "moral righteous noble honest ethical decent honorable right"),
                 T("Keeping the money she found was | Most people think his decision was | Covering for a friend's mistake is | The senator's vote was | Taking credit for her work was | Donating his bonus to the shelter was | Returning the lost wallet was | Spreading rumors about a colleague is",
                   "sinful shameful vile despicable", "virtuous admirable principled commendable")),
    "brightness": (T("The room was | The light in the hallway was | At noon the sky was | The lamp made the kitchen | The stage looked | Inside the cave it was | The street at night was | Her screen was",
                     "dark dim gloomy shadowy murky", "bright luminous radiant shiny glowing"),
                   T("The basement was | Without the candles, the hall was | The new bulbs made the office | Under the trees the path was | The windows let the room stay | After sunset the garden was | The tunnel was | With every lamp on, the house was",
                     "unlit dusky darkened shaded", "illuminated sunlit lit fluorescent")),
    "cleanliness": (T("After the long hike, his boots were | When we arrived, the hotel room was | The water in the old well was | After the party, the kitchen was | Her hands after gardening were | The windows of the abandoned house were | The hospital floor was | After the flood, the streets were",
                      "dirty filthy messy dusty polluted muddy", "clean tidy pristine sterile fresh washed"),
                    T("The car after the rally was | The dishes in the sink were | His shirt after the game was | The carpet in the hall was | The beach after the storm was | The tools in the shed were | The bathroom at the station was | The child's face was",
                      "grimy greasy stained soiled", "spotless scrubbed polished sanitized")),
    "north-south": (T("The geese flew | Every winter the whales swim | We drove for hours, heading | The storm was moving | The river flows | From the capital, the train runs | The border lies further | The expedition set off",
                      "south southward southern", "north northward northern"),
                    T("The road continues | Migrating birds travel | The town lies a few miles | The cold front pushed | The trail heads | Their ship sailed | The mountains rise to the | The wind blew from the",
                      "southerly southbound southeast", "northerly northbound northeast")),
    "friendliness": (T("The new neighbours were very | The shopkeeper was | When I asked for help, the receptionist was | The dog seemed | The way he spoke to the waiter was | Our tour guide was | The locals were | Her reply to my message was",
                       "hostile rude cruel nasty harsh", "friendly kind gentle nice caring"),
                     T("The landlord's tone was | At the border, the guards were | My new teacher seems | The crowd at the match was | The nurse on duty was | Toward newcomers the club was | His manner on the phone was | The old woman next door is",
                       "unfriendly cold aggressive mean", "welcoming warm pleasant generous")),
    "importance": (T("The meeting tomorrow is | For the project, this detail is | His role in the team was | Getting enough sleep is | The new law is | The final exam is | Her contribution was | This decision is",
                     "unimportant minor trivial irrelevant", "important crucial vital essential"),
                   T("For the economy, the vote was | The paperwork is | In the end, the mistake was | For the patient, the test is | The missing letter was | The speech was | Your opinion is | The deadline is",
                     "negligible insignificant secondary marginal", "critical significant key central")),
    "intelligence": (T("The student's answer was | The new assistant seems | Her plan was | The dog in the film was | His decision to quit was | The detective was | The puzzle's solution was | The robot's reply was",
                       "stupid dumb foolish idiotic", "smart clever intelligent brilliant"),
                     T("The way she solved it was | The parrot is surprisingly | The board's strategy was | As a child he was | The essay's argument was | The chess move was | Their marketing idea was | The professor's student was",
                       "ignorant dim clueless senseless", "wise shrewd sharp gifted")),
}
PAIRS = [  # (source, target, name, group); every pair is read high pole -> high pole
    ("brightness", "mood", "happy is bright", "main"), ("north-south", "vertical", "north is up", "main"), ("cleanliness", "morality", "moral is clean", "main"),
    ("mood", "brightness", "bright is happy (reverse)", "reverse"), ("vertical", "north-south", "up is north (reverse)", "reverse"),
    ("morality", "cleanliness", "clean is moral (reverse)", "reverse"),
    ("vertical", "mood", "happy is up", "classic"), ("temperature", "friendliness", "affection is warmth", "classic"),
    ("size", "importance", "important is big", "classic"), ("brightness", "intelligence", "smart is bright", "classic"),
]
EVAL = ["valence", "health", "intelligence", "wealth", "morality", "friendliness", "mood", "cleanliness"]

tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval().to(DEV)
lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
U = m.get_output_embeddings().weight.detach()
st = {"add": None}
def hook(mod, inp, outp):
    if st["add"] is None: return
    h = outp[0] if isinstance(outp, tuple) else outp
    h = h + st["add"][:, None, :]
    return (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
lm.layers[L].register_forward_hook(hook)

ids, counts = np.load(f"{run}/word_ids.npy"), np.load(f"{run}/word_counts.npy")
row = {int(t): r for r, t in enumerate(ids)}
Mw = np.load(f"{run}/L{L}_word_means.npy").astype(np.float32)
pool = np.array([r for r in range(len(ids)) if counts[r] >= 50]); gm = Mw[pool].mean(0)
unit = lambda v: v / np.linalg.norm(v)
def vec(w):
    got = [(Mw[row[t[0]]], counts[row[t[0]]]) for s in (" " + w, " " + w.capitalize())
           for t in [tok(s, add_special_tokens=False).input_ids] if len(t) == 1 and t[0] in row]
    if not got: return None
    wt = np.array([c for _, c in got], float); return (np.stack([v for v, _ in got]) * wt[:, None]).sum(0) / wt.sum() - gm
used = {}
def gap(lo, hi, name):
    lo_v = [(w, v) for w, v in ((w, vec(w)) for w in lo.split()) if v is not None]
    hi_v = [(w, v) for w, v in ((w, vec(w)) for w in hi.split()) if v is not None]
    used[name] = ([w for w, _ in lo_v], [w for w, _ in hi_v])
    if len(lo_v) < 3 or len(hi_v) < 3: return None
    return np.stack([v for _, v in hi_v]).mean(0) - np.stack([v for _, v in lo_v]).mean(0)
axes = {n: gap(*SCALES[n], n) for n in SCALES}
held = {n: gap(*HELD[n], n + " (held-out)") for n in HELD}
SIZE = float(np.median([np.linalg.norm(a) for a in axes.values() if a is not None]))
val = unit(axes["valence"])
def remover(names):
    Q, _ = np.linalg.qr(np.stack([unit(axes[n]) for n in names]).T)
    return lambda v: v - Q @ (Q.T @ v)
rng = np.random.default_rng(0)
rand_raw = [(Mw[rng.choice(pool, 5)] - gm).mean(0) - (Mw[rng.choice(pool, 5)] - gm).mean(0) for _ in range(N_RAND)]
def single(words):
    o = []
    for w in words:
        t = tok(" " + w, add_special_tokens=False).input_ids
        if len(t) == 1: o.append(t[0])
    return o

res = dict(layer=L, push_size=SIZE, words_used=used, targets={})
for tgt in sorted({p[1] for p in PAIRS}):
    hyp = [p for p in PAIRS if p[1] == tgt]
    rm_val = lambda v: v - (v @ val) * val
    excl = {tgt, "valence"}
    # sources: every scale (valence removed), plus the hypothesised sources' held-out lists and the evaluation-removed variant
    src = {}
    for n, a in axes.items():
        if a is None or n in excl: continue
        src[f"{n}"] = unit(rm_val(a))
    for s, _, name, _ in hyp:
        if s in held and held[s] is not None: src[f"{s} (held-out)"] = unit(rm_val(held[s]))
        rm_ev = remover([e for e in EVAL if e not in (s, tgt)])
        src[f"{s} (evaluation removed)"] = unit(rm_ev(axes[s]))
        if s in held and held[s] is not None: src[f"{s} (held-out, evaluation removed)"] = unit(rm_ev(held[s]))
    nulls = {"valence removed": [unit(rm_val(r)) for r in rand_raw]}
    for s, _, _, _ in hyp:
        rm_ev = remover([e for e in EVAL if e not in (s, tgt)])
        nulls[f"evaluation removed ({s})"] = [unit(rm_ev(r)) for r in rand_raw]
    names = list(src) + [f"null|{k}|{i}" for k, v in nulls.items() for i in range(len(v))]
    vecs = list(src.values()) + [x for v in nulls.values() for x in v]
    V = torch.tensor(np.stack(vecs), dtype=torch.float32, device=DEV)
    out_t = {}
    for ver, (prompts, lo_w, hi_w) in zip(("original", "held-out"), TARGETS[tgt]):
        lo_ids, hi_ids = single(lo_w), single(hi_w)
        sh = {a: np.zeros((len(prompts), len(names))) for a in ALPHAS}
        for pi, p in enumerate(prompts):
            pt = tok(p, add_special_tokens=False).input_ids
            with torch.inference_mode():
                st["add"] = None
                lp0 = torch.log_softmax(lm(input_ids=torch.tensor([pt], device=DEV)).last_hidden_state[:, -1] @ U.T, -1)[0]
                base = float(torch.logsumexp(lp0[hi_ids], -1) - torch.logsumexp(lp0[lo_ids], -1))
                for a in ALPHAS:
                    vals = []
                    for b in range(0, len(names), 128):
                        st["add"] = a * SIZE * V[b:b + 128]
                        lp = torch.log_softmax(lm(input_ids=torch.tensor([pt] * len(st["add"]), device=DEV)).last_hidden_state[:, -1] @ U.T, -1)
                        vals.append((torch.logsumexp(lp[:, hi_ids], -1) - torch.logsumexp(lp[:, lo_ids], -1)).cpu().numpy() - base)
                    sh[a][pi] = np.concatenate(vals)
            st["add"] = None
        summ = {}
        for a in ALPHAS:
            mean = sh[a].mean(0); eff = dict(zip(names, mean))
            nz = {k: np.array([eff[f"null|{k}|{i}"] for i in range(len(v))]) for k, v in nulls.items()}
            z = {}
            for n in src:
                k = next((f"evaluation removed ({s})" for s, _, _, _ in hyp if n.startswith(s + " (") and "evaluation" in n), "valence removed")
                z[n] = dict(shift=round(float(eff[n]), 3), z=round(float((eff[n] - nz[k].mean()) / (nz[k].std() + 1e-9)), 2))
            plain = sorted([n for n in src if "(" not in n], key=lambda n: -z[n]["z"])
            summ[str(a)] = dict(z=z, rank={n: plain.index(n) + 1 for n in plain}, n_sources=len(plain),
                                null={k: [round(float(v.mean()), 3), round(float(v.std()), 3)] for k, v in nz.items()})
        out_t[ver] = summ
    res["targets"][tgt] = out_t

bar = 2.58  # one-sided p < 0.005: Bonferroni over 10 pairs at 0.05
print(f"layer {L}; push size {SIZE:.2f} (median pole gap); z vs {N_RAND} random word-set contrasts; bar z > {bar}")
res["pairs"] = []
for s, t, name, grp in PAIRS:
    rec = dict(source=s, target=t, name=name, group=grp)
    for ver in ("original", "held-out"):
        for a in ("1.0", "2.0"):
            S = res["targets"][t][ver][a]
            g = lambda k: S["z"].get(k, {}).get("z")
            rec[f"{ver}|{a}"] = dict(z=g(s), z_heldout_source=g(f"{s} (held-out)"), z_eval_removed=g(f"{s} (evaluation removed)"),
                                     z_heldout_eval_removed=g(f"{s} (held-out, evaluation removed)"), rank=S["rank"].get(s), of=S["n_sources"],
                                     top=sorted(S["rank"], key=S["rank"].get)[:3])
    res["pairs"].append(rec)
    o, h = rec["original|1.0"], rec["held-out|1.0"]
    f = lambda x: "  -  " if x is None else f"{x:+5.1f}"
    print(f"\n{name:28s} [{grp}] push {s} -> read {t}   (x1; x2 in the json)")
    print(f"   original prompts:  source z {f(o['z'])} | held-out source {f(o['z_heldout_source'])} | evaluation removed {f(o['z_eval_removed'])}"
          f" | both {f(o['z_heldout_eval_removed'])} | rank {o['rank']}/{o['of']} (top: {', '.join(o['top'])})")
    print(f"   held-out prompts:  source z {f(h['z'])} | held-out source {f(h['z_heldout_source'])} | evaluation removed {f(h['z_eval_removed'])}"
          f" | both {f(h['z_heldout_eval_removed'])} | rank {h['rank']}/{h['of']} (top: {', '.join(h['top'])})")
json.dump(res, open(out, "w"), indent=1)
