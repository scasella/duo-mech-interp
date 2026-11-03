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
| `interview_checks.py` | The periodic table and the emotion wheel (valence × arousal) in the word averages |
| `manifold_steer.py` | Steering along a shape (through the real values) versus a straight push, on 0.8B |
| `overshoot.py` | How far a straight push goes before the model stops answering in kind, versus following the shape |
| `family_sweep.py` | Round three: 56 ordered/cyclic word families plus 7 known controls; order, bend, node check and straight push |
| `sweep_summary.py` | Combines two models' sweeps into one verdict per family |
| `sweep_shared.py` | Do ordered families share their order direction (within groups, across groups)? |
| `sweep_transfer.py` | Causal version: push along a direction built from other families, read the target family |
| `ordinal_shared.py` | Round four: the isolated-word protocol of arXiv 2606.01269 with a null, and a power analysis |
| `metaphor_steer.py` | Round four: metaphor steering with held-out words, specificity ranks and evaluation removed |

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

## Steering along the shape (0.8B, the app's model)

At the family word ("Today is Monday. Tomorrow is"), the internal state was moved from Monday toward Friday along two
paths: a straight line, or a path through the real Tuesday–Thursday averages that then continues round the week.

**In between source and target** (share of right answers, layers 6/9/12):

| Family | Through the real values | Straight line |
|---|---|---|
| Weekdays | 86–100% | 0–5% |
| Months | 95–100% | 0% |
| Units of time | 67–73% | 0% |

**Past the target:**
- Through the real values: 89–100% for weekdays and months, 10–30% for units of time.
- Straight line: 0–4%. It stops moving, then breaks from about 4× its length. Day-word probability falls from 0.85 to
  0.34 at 12×, and the top answer is a day only 14% of the time.
- Following the days 48 steps round the week still gives the right "tomorrow" every time.

**Moves confined to the circle's 2-D plane** change the answer only 0–19% of the time. The steerable shape is the
curve through the real averages in the full 1,024 dimensions.

**Also in the word averages** (`interview_checks.py`):
- Element names give atomic number (held-out r 0.61–0.86 in 2B, 0.64–0.71 in 0.8B) and period. Shuffled labels score
  about 0.25.
- Emotion words give valence and arousal. A held-out readout gets the sign right 89–96% of the time (shuffled about 60%).
  But these are not the top two axes among emotion words, so the wheel is a hidden plane, not the main shape.

## New-structure search: axes reused across domains (0.8B and 2B)

`discover.py` (label-free ordering of word families) mostly finds synonym chains and verb-form loops. A few candidate
scales are not yet validated: how soon (future → imminent → brink), outbreak size (outbreak → epidemic → pandemic),
speed, and togetherness.

`axis_atlas.py` builds 30 scales from opposite word sets and tests whether their axes are shared. It uses cosine
z-scores against random word axes, with valence and word frequency removed (an AUC version saturates with this few
words and was dropped). `atlas_validate.py` re-tests with held-out word lists. `cross_steer.py` pushes along one
scale while the model judges another domain, against random word-pair contrasts.

**Shared "more" axis:**
- Power, size, quantity, importance, wealth, loudness and cost share a weak common axis.
- It holds 13–15% of the scales' variance at layers 6 and 12; random axes give 10–13%.
- "More is up", "rich is powerful", "important is big" and "loud is big" vanish once it is removed.

**Happy is bright** (weakened in round four):
- The axes align at every layer in both models.
- Pushing dark→bright moves 0.8B toward happy words (z 4.5 at 1×). It does nothing to moral words (z −0.5).
- Round four: most of this came from words with a mood sense ("gloomy", "radiant"). Without them it is z 1.9–2.3 in
  both models, below the multiple-comparison bar.

**Moral is clean** (double meanings, round four):
- The alignment holds with held-out word lists.
- Round four: the push vanishes once "dirty" and "filthy", which have moral senses, are replaced (z −1.0 to 0.9).
- Clean→moral is causal (z 3.6), but so are smart→moral (3.9) and healthy→moral (2.2). This is a broad "good
  qualities" direction, not a specific metaphor.

**Rigid → serious:**
- Strong (z 4.7), likely shared meaning (rigid, stiff and firm also mean strict).
- Quiet→serious and slow→serious are borderline (|z| about 2). Heavy→serious and big→serious are not significant.

**Not found:**
- Affection is warmth (warm→friendly z −1.1; axes unaligned).
- Serious or important is heavy.
- Time is distance.

## Round two (candidate scales, WordNet atlas, block-sparse featurizer)

**Candidate scales** (`scale_validate.py`; order of the per-word averages on their top axis):

| Scale | 0.8B (layers 6, 12) | 2B (layers 6, 12, 18) | Straight push |
|---|---|---|---|
| How soon (someday → eventually → later → soon → shortly → immediately → now) | 1.00, 0.96 | 0.96, 0.96, 0.86 | skips all 5 middle words* |
| Outbreak size (case → cluster → outbreak → epidemic → pandemic) | 0.90 | 0.90–1.00 | inconclusive |
| Speed | 0.93 at layer 12 | 0.54–0.86 | skips most of the middle |
| Togetherness | not clearly ordered | not clearly ordered | inconclusive |

- How soon is a new ordered scale. The path through the real words walks through all seven.
- \*Correction (round three): a straight push skips the middle words for nearly every family, including number
  words and months, so skipping is not evidence that a scale is curved. The bend test finds no reliable bend in how
  soon (1 of 4 tests). Treat it as ordered, shape unknown.
- Outbreak size and togetherness are inconclusive on the straight-push test because the copy prompt can't produce their
  middle words.

**WordNet atlas** (`antonym_atlas.py`; 253 scales from adjective antonyms, both models):
- The top shared direction is Osgood's evaluation: 11–12% of scale variance vs 7–8% for random axes; evaluation
  markers load 0.48.
- Potency appears only as a weaker intensity dial. There is no activity dial.
- No cross-domain pair beats the multiple-comparison bar on its own. The coherent candidate, north–south ~ up–down
  and ascending–descending, is causal in 0.8B (`cross_steer.py`): south→north raises "up" words, z 4.0 with valence
  removed; slow→fast does nothing.
- Round four: the robust direction is the reverse. Up → north passes every control in both models; north → up is
  weaker and not specific.

**Block-sparse featurizer** (`bsf.py`, `modal_app.py::blocks`, `bsf_analyze.py`; 0.8B layer 12, 60M tokens of
FineWeb-Edu):
- Setup: 8,192 blocks of 3 dimensions (32 active) reach R² 0.80. A matched TopK SAE (24,576 features, 96 active)
  reaches 0.84.
- No family gets a dedicated unit in either. The best blocks are 3–32% dedicated, the best SAE features 4–28%.
- Months (7/12, p 0.02) and ordinals (0.96, p 0.01) are partly ordered inside one block. Weekdays are not (2/7).
- Blocks do capture common graded concepts with an ordered internal axis:
  - two size blocks put small against large (rank correlation 0.81 and 0.82);
  - a commonness block puts rare against common (0.60);
  - a good/bad block's first axis is degree (better/best vs good), not polarity.

## Round three: a targeted sweep of 56 word families

**Prior work.** That words with a real-world order line up along a direction is established, so these families are
new examples, not a new kind of structure:
- Grand et al. 2022: semantic projection recovers size, danger and wealth from word embeddings.
- Heinzerling & Inui 2024: monotonic, causal directions for numeric properties in LLM hidden states.
- Scalar adjectives: BERT 2020, LLMs 2024.
- Linear number values (COLING 2025).
- Successor heads: one position code shared by numbers, days, months and letters.
- A 2026 granularity axis for social roles.
- 2026 work on ordinal probes and steering to points on a scale.

A quick search found none of these specific word lists named. That is weak evidence, and anyone with the same lists
would likely get the same result.

`family_sweep.py` tests 56 families whose order is known from the world, not from the model:
- ranks: military, church, nobility, corporate, academic, chess, cards, status tiers, credit ratings;
- school and life: school year, grade, school level, degree, life stage, generation, decade of life, relationship,
  closeness, grief, insect stage;
- sizes: clothing, coffee, coins, settlement, geographic scale, water body, rock, ensemble, taxonomy, quantifiers,
  wealth;
- nature: wind, precipitation, state of matter, moon phase, brightness;
- time: meals, historical era, schedule, tournament round, project stage, sleep;
- degree: frequency, probability, hunger, anger, quality;
- music: tempo, dynamics, karate belts;
- rote sequences: Greek letters, NATO alphabet;
- cycles: zodiac, Chinese zodiac, solfège, musical notes (scale order and circle of fifths).

Each family has 8 template sentences and a copy prompt. Words of several tokens are read at their last token. Seven
families with known shapes run alongside as controls. Both models are tested (0.8B, 2B-Base), at layers 6 and 12.

```sh
python tools/shapes/family_sweep.py <0.8B dir> runs/sweep08 6,12
python tools/shapes/family_sweep.py Qwen/Qwen3.5-2B-Base runs/sweep2b 6,12 --cache <HF cache>
python tools/shapes/sweep_summary.py runs/sweep08/family_sweep.json runs/sweep2b/family_sweep.json runs/sweep_summary.json
python tools/shapes/sweep_shared.py runs/sweep08 runs/sweep2b runs/sweep_summary.json runs/sweep_shared.json 12
python tools/shapes/sweep_transfer.py <0.8B dir> runs/sweep08 runs/sweep_summary.json runs/sweep_transfer08.json 12
```

The order test is the round-two test: rank correlation along the top principal axis, or neighbours that stay adjacent
on the top plane for cycles, with a permutation p. A family counts as ordered at p < 0.05 at some layer in both
models. By chance about 0.5 of 56 families would pass. Word frequency (Zipf, from `wordfreq`) is partialled out as a
check. With 4 values the smallest possible p is 0.083, so 4-value families can only be reported as consistent.

**Ordered in both models (14 new families)**. Rank correlation at the best layer, 0.8B / 2B:

| Family | Values | 0.8B | 2B |
|---|---|---|---|
| Wind | calm → breeze → wind → gale → storm → hurricane | 1.00 | 1.00 |
| Settlement | hamlet → village → town → city → metropolis | 1.00 | 1.00 |
| Water body | puddle → pond → lake → sea → ocean | 1.00 | 1.00 |
| Geographic scale | street → neighborhood → city → county → state → country → continent | 0.96 | 1.00 |
| Letter grade | A → B → C → D → F | 1.00 | 1.00 |
| School level | preschool → kindergarten → elementary → college → graduate | 1.00 | 1.00 |
| Corporate rank | intern → assistant → manager → director → CEO | 1.00 | 1.00 |
| Decade of life | teens → twenties → … → eighties | 1.00 | 1.00 |
| Schedule | hourly → daily → weekly → monthly → quarterly → annually | 0.94 | 1.00 |
| Nobility | baron → viscount → earl → marquis → duke → prince → king → emperor | 0.98 | 0.86 |
| Closeness | stranger → acquaintance → colleague → friend → partner → spouse | 0.94 | 0.94 |
| Life stage | infant → baby → toddler → child → teenager → adult → senior | 0.89 | 0.96 |
| Taxonomy rank | species → genus → family → order → class → phylum → kingdom | 0.86 | 0.93 |
| Greek letters | alpha → beta → … → theta | 0.88 | 0.83 |

- Word frequency does not explain them. The order survives partialling out frequency, with partial correlations
  ≥ 0.54; the lowest are Greek letters (0.54) and nobility (0.66) in 2B. School level's order tracks frequency
  closely (0.9), so there it can't be fully separated.
- Perfectly ordered at every layer of both models, but with only 4 values: school year, state of matter, coffee size.
  Coffee size also tracks word frequency.
- Ordered on the second axis in both models: military rank (0.90 / 0.86) and clothing size (0.94 / 1.00). The test
  allows for picking the best of three axes. Their top axis is something else.
- Ordered in one model only: NATO alphabet, academic rank, church rank, status tier, ensemble, frequency adverbs,
  quality, zodiac, Chinese zodiac, musical notes.
- Not ordered: chess pieces by value, card ranks, coins by value, credit ratings, grief stages, project stages, moon
  phases, tempo, karate belts, relationship stages, quantifiers, probability, hunger, anger, meals, precipitation,
  historical era, solfège.
- Inconclusive, 4 values and not perfectly ordered: brightness, degree, dynamics, generation, insect stage, rock size,
  sleep, tournament round, wealth.

**Cycles are mostly absent.**
- No circle for the Chinese zodiac: 0–2 of 12 neighbours kept in 0.8B, and 5 of 12 at one layer of 2B (p 0.05).
  None for solfège either (0–2 of 7).
- The Western zodiac keeps 4–6 of 12 neighbours (p 0.01–0.16), not a clean circle.
- Musical notes keep 5 of 7 neighbours in scale order in 0.8B only. Nothing follows the circle of fifths (1–2 of 7).
- So the model learns circles for weekdays and months, which text repeats constantly, but not for conventional cycles
  it reads about less often.

**A straight push is not a curvature test.**
- The node check works for most families: placing the residual at a middle word's average makes the copy prompt
  repeat that word.
- But a straight push from the first word's average to the last skips the middle words almost everywhere, known
  lines included. At the best point on the push, middle words get a median of 1–9% of the share they get at their own
  average, for number words, months, how soon and speed (temperature and size 22%). Weekdays are the exception
  (65%).
- Middle words are not blends of the end words. Each has its own direction, and the order lives in a few shared
  axes.
- The readout-free test is the bend: does a second or third axis curve as a quadratic of the order (permutation p)?
  - It bends in all 4 tests for geographic scale, closeness and frequency adverbs, and in 3 of 4 for settlement and
    decade of life.
  - Frequency adverbs are not a clean line on the top axis. Instead a second axis puts the two absolutes, never and
    always, on one side and sometimes on the other.

**Shared directions** (`sweep_shared.py`; order direction = covariance of the value averages with rank; a pooled
word-frequency direction removed; layer 12, 0.8B / 2B):
- **A general "later in the sequence" direction.** The order directions of all ordered families point slightly the
  same way, across unrelated groups: mean cosine 0.044 / 0.031, z 8.1 / 6.6 against shuffled ranks. It replicates
  at layer 6. This extends the shared position code (weekdays, months, numbers, ordinals) to ranks, sizes and
  alphabets.
- **Magnitude families share more than that.** Settlement, geographic scale, water body, wind, the size scales and
  number words: excess over the general direction p 0.001 / 0.004. With only the new families it is p 0.02 / 0.03.
  This is the round-one "more" axis again, now reaching nouns for places, water and weather.
- **Social ranks share a little more** (excess p 0.09 / 0.035; layer 6: 0.08 / 0.014). Letter sequences are
  borderline (p 0.035 / 0.086). Age and time families share nothing beyond the general direction.
- No single pair of families passes the multiple-comparison bar in both models.

**Causal check** (`sweep_transfer.py`, 0.8B layer 12):
- Setup: push the residual at the target word along a direction built only from the other families in its group,
  and read which value the copy prompt repeats. The null builds the same direction from those families with their
  ranks shuffled (60 draws). Families whose copy prompt fails the node check are left out.
- **Social rank**: pooled z +2.9, and 6 of 7 families move up their own scale. The largest moves are nobility,
  academic rank and school year.
- **Magnitude**: z +2.2, 7 of 8 move up.
- **Letter sequences**: z +1.8, 3 of 3 move up. The general direction moves none of them (0 of 3), so letters have a
  shared direction of their own.
- **Age and time**: nothing (z −0.5).
- The effects are modest. A borrowed direction moves a family typically less than half as far as its own direction
  does, for example 0.6 vs 2.7 values for geographic scale. The pooled z treats families as independent, so read it
  as a summary, not a strict test.

## Round four: a shared ordinal direction, and metaphors with clean word lists

### Is there a domain-general ordinal direction?

arXiv 2606.01269 ("Emergent Ordinal Geometry in Transformers Trained on Local Comparisons") reports for Qwen2.5-1.5B:
- the per-domain probe directions for digits, months and size adjectives are "nearly orthogonal at every layer
  (|cos| < 0.2 throughout)";
- so the model "does not appear to encode a single, domain-general ordinality direction".

That test has no null distribution. In 1,536 dimensions a real shared component of cosine 0.03 sits far below 0.2.
`ordinal_shared.py` checks both protocols.

```sh
python tools/shapes/ordinal_shared.py isolated Qwen/Qwen2.5-1.5B iso.json          # their protocol, with a null
python tools/shapes/family_sweep.py Qwen/Qwen2.5-1.5B runs/sweepq25 7,14,21 --no-steer
python tools/shapes/ordinal_shared.py power runs/sweep_summary.json power.json runs/sweepq25:14:Qwen2.5
python tools/shapes/sweep_shared.py runs/sweepq25 runs/sweepq25 runs/sweep_summary.json shared.json 14
```

**Their protocol on their model** (each word tokenized on its own, sub-tokens averaged, every layer):
- With a ridge probe like theirs, the largest pairwise |cos| is 0.21. Their observation reproduces.
- The three pairs average a cosine of 0.02–0.05, positive at every layer. It beats shuffled ranks (p < 0.05) at 16
  of 28 layers. In Qwen3.5-0.8B it does so at 11–13 of 24 layers.
- Raw covariance directions of isolated words all point the same way (cosine 0.96–1.0 at layers 1–25), and so do
  shuffled ones. A word on its own sits at the first position, whose few huge activations swamp the rest. Reading
  words in context avoids this.

**In context** (our template sentences, 26 ordered families, 238 pairs from different groups, frequency removed):

| model | mean cosine | z | the paper's three domains, mean (p) | random three-family tests that detect it |
|---|---|---|---|---|
| Qwen3.5-0.8B, layer 12 | 0.047 | 8.7 | 0.21 (0.002) | 12% |
| Qwen3.5-2B, layer 12 | 0.034 | 7.5 | 0.20 (0.004) | 12% |
| Qwen2.5-1.5B, layer 7 / 14 / 21 | 0.026 / 0.033 / 0.019 | 6.2 / 6.7 / 4.0 | 0.14 (0.02) / 0.14 (0.03) / 0.10 (0.09) | 8% / 10% / 4% |

- Within their three domains, number words and months share strongly (cosine 0.26–0.36 in all three models), while
  sizes share little (0.0–0.17).
- The group structure replicates in Qwen2.5-1.5B (layer 14), beyond the general direction:
  - magnitude: p 0.002 (new families only: 0.015);
  - letter sequences: p 0.006;
  - social ranks: p 0.025;
  - age and time: nothing.

**Reading.** Both results hold. Most of each family's order direction is its own, at an average cosine of about 0.03.
A small shared part exists in all three models, and only pooling reveals it: a test on three random families finds it
4–12% of the time. "No single domain-general direction" overstates their evidence; "no strong one" is right.

### Metaphors, version 2

`metaphor_steer.py` reruns the causal test with these changes:
- held-out source words with no sense in the target domain (no "gloomy" or "radiant" for brightness, no "dirty" or
  "filthy" for cleanliness);
- held-out prompts and readout words;
- one push size for every source, so each hypothesised source is ranked against 29 others;
- a condition with a whole set of good/bad directions removed;
- both directions of each pair, 64 random word-set contrasts as the null, and the bar at z 2.58 (Bonferroni over 10
  pairs).

Run on Modal (`modal_app.py::job`) for Qwen3.5-0.8B and 2B, at layer 12.

| pair (push → read) | original words | everything held out | evaluation also removed | rank of 29 | both models |
|---|---|---|---|---|---|
| **up → north** | z 4.8 / 6.5 | 4.8 / 5.0 | 4.5 / 4.3 | 1st in 3 of 4, 2nd in the other | **yes** |
| north → up | 3.5 / 3.1 | — (no clean word list) | 3.2 / 3.0 | 2nd–7th | weak, not specific |
| bright → happy | 4.6 / 4.6 | 2.2 / 1.9 | 2.3 / 2.3 | 1st–2nd | below the bar |
| happy → bright | 6.1 / 8.4 | 2.3 / 0.8 (held-out readout) | 2.3 / 1.1 | 1st / 9th | no |
| clean → moral | 3.7 / 2.7 | 0.0 / 0.9 | 0.0 / 0.1 | 1st–5th | no (double meanings) |
| bright → smart | 1.9 / 1.9 | 0.4 / 0.7 | −0.1 / 0.0 | 3rd–5th | no (double meanings) |
| big → important | 2.0 / 1.9 | 2.6 / 2.7 (held-out prompts) | 2.9 / 3.4 | 4th–8th | present, not specific |
| up → happy | 0.4 / 0.3 | 0.0 / 0.0 | −0.7 / −0.1 | 13th–16th | no |
| warm → friendly | −0.9 / −1.7 | 0.1 / 0.4 | −0.4 / 0.0 | 9th–28th | no |

(0.8B / 2B. "Everything held out" uses held-out source words with held-out prompts and readouts, where they exist.)

- **Up → north** is the one robust cross-domain mapping. It survives every control in both models and runs one way:
  pushing "up" makes "north" likelier, while north → up is weak and loudness and temperature push "up" harder.
- **Collocation caveat:** English says "up north" and "down south". This test can't separate a mapping between
  concepts from the model having learned those phrases, which may themselves trace back to maps drawn with north up.
- **Happy is bright** shrinks to about half once no word has a mood sense. It stays positive in both models but under
  the bar.
- **Moral is clean** and **smart is bright** were double meanings ("dirty tricks", "a bright student").
