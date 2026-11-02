# Concept shapes with Qwen-Scope SAEs

Looking for concepts that a language model stores as a *shape*, not a single direction. Examples are a circle of
weekdays or a line of temperatures. The tools use Alibaba's public Qwen-Scope sparse autoencoders
([`Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50)) on
Qwen3.5-2B-Base. Each SAE is TopK-50 with 32,768 features, reading the residual stream at one of all 24 layers.
No Qwen-Scope SAE exists for the 0.8B model the app runs. Shapes found in 2B are checked on 0.8B with template
sentences instead (`local.py probe --model … --no-sae`).

## Files

| File | What it does |
|---|---|
| `core.py` | Model and SAE loading, the web-text harvest, and the probe families (shared by both entry points) |
| `local.py` | Runs the steps on this machine (CPU is fine for the probes and for about 1M tokens) |
| `modal_app.py` | The full run on a Modal GPU: probes, then harvest, then analysis; outputs go to the volume `qwen-scope-shapes` |
| `analyze.py` | `probes`: known families. `groups`: unsupervised search for shapes |

## Running

```sh
# probes (known families) on 2B with SAEs, and on the app's 0.8B without
python tools/shapes/local.py probe --layers 6,12,18 --out runs/probe
python tools/shapes/analyze.py probes --run runs/probe --layers 6,12,18
python tools/shapes/local.py probe --model Qwen/Qwen3.5-0.8B --no-sae --layers 6,12,18 --out runs/probe08

# web text: about 300 tokens/s on a 4-core CPU at layer 12, so 1M tokens takes about an hour
python tools/shapes/local.py corpus --layers 12 --tokens 1000000 --store 4000 --seq 256 --out runs/cpu1m
python tools/shapes/analyze.py groups --run runs/cpu1m --layers 12 --probe-run runs/probe

# GPU (needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or `modal token set`)
modal run tools/shapes/modal_app.py --run pilot --layers 12 --tokens 2000000
modal run tools/shapes/modal_app.py --run full --layers 6,12,18 --tokens 20000000 --store 4000
modal volume get qwen-scope-shapes runs/full ./runs/full
```

## How shapes are found

1. **Harvest.** Stream web text (FineWeb-Edu) through the model. At each layer, record which SAE features fire on
   each token and how often every pair fires together (dense co-activation counts). Keep the full codes for the
   first few thousand sequences.
2. **Group.** Link two features when their decoder directions point the same way, or when they fire together more
   than chance (normalised PMI). Features that fire on more than 0.5% of tokens are left out. Communities in this
   graph (Louvain, split until at most 80 features each) are the candidate shapes.
3. **Measure.** For each group, rebuild the residual from only that group's features on the tokens where it fires,
   and take its PCA. Two numbers rank the groups:
   - how much of the geometry is explained by which word the group fired on (variance between word centroids);
   - how many dimensions those word centroids spread over (participation ratio).

   Filler words are ignored when scoring.
4. **Check.** The probe families come with a known order (weekdays, months, numbers, ordinals, seasons, times of
   day, colours, temperatures, sizes, compass points, planets). For each one, the analysis tests:
   - whether the model holds it in order;
   - how many SAE features carry it;
   - whether the reconstruction from only those features keeps the order;
   - whether the grouping step puts those features together.

## First results (CPU, probe sentences)

- **2B:** weekdays sit on a circle in the right order at layers 6, 12 and 18 (chance about 0.3%), and months do too
  (chance about 5e-8). Numbers, ordinals, temperatures and sizes lie on a line in order. Colours, seasons and compass
  points do not come out in their natural order.
- **The SAE spreads each shape over many features:** 36–160 value-specific features per family, plus 35–80 broad
  ones. The reconstruction from only those features keeps the weekday and month circles at layers 6 and 12, but
  not at 18.
- **A shared position code.** Several SAE features fire on the *k*-th item of every list. Feature 330 fires on
  Thursday, April, four and fourth; feature 13888 on Friday, May, five and fifth; feature 1348 on July, seven and
  seventh. In the raw residuals, value *i* of one family is nearest to value *i* of another:
  - 100% of the time for every pair of weekday, month, number and ordinal, at layers 6, 12 and 18 (chance 8–14%);
  - still 57–80% after removing the three main shared axes, so this is more than a single frequency line.

  The app's 0.8B model has the same code (86–100% at layers 6–18).
