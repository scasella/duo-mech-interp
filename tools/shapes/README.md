# Concept shapes with Qwen-Scope SAEs

Looking for concepts that a language model stores as a *shape*, not a single direction: a circle of weekdays, a line of
durations, a map of cities. The search runs on Qwen3.5-2B-Base, using Alibaba's public Qwen-Scope sparse autoencoders
([`Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50)): TopK-50,
32,768 features, residual stream, all 24 layers. Each finding is then checked on the 0.8B model the app runs, which
has no SAE.

## Files

| File | What it does |
|---|---|
| `core.py` | Model and SAE loading; the web-text harvests (SAE codes and co-activation, or per-word averages); probe families |
| `local.py` | The same steps on this machine (CPU is fine for the probes and for about 1M tokens) |
| `modal_app.py` | GPU runs on Modal: `main` (probes + SAE harvest + analysis) and `::words` (per-word averages, any checkpoint) |
| `analyze.py` | `probes` (known families), `groups` (SAE-feature grouping), `words` (word families and their shapes) |
| `ordered_tests.py` | Do the found families line up with a real quantity (durations, volumes, intensity, digits, luck)? And a city map |

## Running

```sh
# known families on 2B (with SAEs) and on the app's 0.8B (without)
python tools/shapes/local.py probe --layers 6,12,18 --out runs/probe
python tools/shapes/analyze.py probes --run runs/probe --layers 6,12,18
python tools/shapes/local.py probe --model Qwen/Qwen3.5-0.8B --no-sae --layers 6,12,18 --out runs/probe08

# GPU (needs a Modal token; behind an HTTP proxy, install modal[api-proxy-support])
modal run --detach tools/shapes/modal_app.py --run full --layers 6,12,18 --tokens 30000000 --store 4000
modal run --detach tools/shapes/modal_app.py::words --run words30m --layers 6,12,18 --tokens 30000000
modal run --detach tools/shapes/modal_app.py::words --run words08 --model Qwen/Qwen3.5-0.8B --tokens 30000000
modal volume get qwen-scope-shapes runs/words30m ./runs/

python tools/shapes/analyze.py words --run runs/words30m --layers 12
python tools/shapes/ordered_tests.py runs/words30m <hf cache or ""> ordered.json
```

An A100 harvests about 25,000 tokens/s with SAEs at three layers, and 30,000–36,000 tokens/s for word averages.
30M tokens take 15–20 minutes.

## What worked and what did not

- **Grouping SAE features did not find shapes.** The Qwen-Scope SAE has almost no features dedicated to one list.
  Over 1M tokens of web text, no feature fires mainly on weekdays, and only four fire mainly on months. Each shape is
  carried by about a hundred broad features. Grouping features by decoder direction or co-activation (`analyze.py
  groups`) gave spelling clusters ("authority, authorities, author") rather than shapes, and scattered the weekday
  features across groups.
- **Word families worked.** `harvest_centroids` averages the residual of every frequent word over the text. `analyze.py
  words` links words to their nearest neighbours (after removing the shared mean and two main axes, and merging case
  and plural variants) and splits them into families. With no labels, weekdays (7/7), months (12/12), number words,
  ordinals, colours, seasons and compass points each came out as one family. New families appeared next to them:
  digits, tens, units, ordering words ("firstly, secondly, lastly"), stance adverbs and place names.

## Results

**Probe sentences (2B; same on 0.8B):**
- Weekdays and months sit on circles in the right order at layers 6, 12 and 18.
- Numbers, ordinals, temperatures and sizes lie on lines in order.
- Colours, seasons and compass points do not follow their natural order.

**A shared position code.** Weekdays, months, numbers and ordinals share one code: in the residuals, value *i* of one
list is nearest to value *i* of another.
- 2B matches 100% of the time for every pair at all three layers (chance 8–14%).
- 2B still matches 57–80% after removing the three main shared axes.
- 0.8B matches 86–100%.
- It is causal in 0.8B. Adding "four minus one" taken from number sentences to "Monday" in "Today is Monday.
  Tomorrow is" turns the answer into "Friday" 57–71% of the time, against 0–7% for a random push.

**Found in web text** (2B, layer 12, rank correlation on the best of the top three axes):

| Shape | Rank correlation |
|---|---|
| Units of time (second … century) | 0.98 |
| Intensity adverbs (slightly … drastically) | 0.90 |
| Units of volume (teaspoon … barrel) | 0.73 |
| Digits 0–9 | 0.62 (0.84 at layer 6) |
| Luck adverbs (fortunately vs unfortunately) | 0.77 |
| City map (held-out linear readout) | latitude r 0.60, longitude r 0.82 (shuffled baseline about 0.15) |
