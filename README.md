# duo neuron explorer

A steering-first neuron explorer for **Qwen3.5-0.8B** that runs entirely in Safari on an iPhone (built around the
iPhone Duo's two displays and fold), on WebGPU, with no server. It is an on-device port of
[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (MIT), which does the same for
Qwen3.5-4B with a Python server and a GPU sweep.

Every MLP neuron (24 layers × 3,584) can be pushed up and down while the model chats, read on your own text, and traced
through the weights:

- **chat** (writes): three conversations decoded side by side from the same prompt: neuron pushed **up**, **untouched**,
  pushed **down**. Doses are in the reference's unit, a fraction of the median residual norm leaving the layer (`add`),
  or a multiplier on the activation (`×`). Multi-turn: each column keeps its own history.
- **reads you**: your text, each token shaded by the neuron's activation.
- **lens**: logit lens of the neuron's down column (what it writes, read through the final norm and unembedding),
  steer lens (next-token log-prob change when pushed ±0.8 on three prompts, and the KL), and direct-path connections
  to earlier and later neurons (gate·up weights, z-scored per layer).
- **examples**: the neuron's strongest contexts on a small built-in corpus, shaded live.
- **fires**: paste text, tap a token, see which neurons fire hardest there (scored against each neuron's calibrated
  range); the map lights up with the same scores.
- **map**: the 24 MLP sheets stacked (full-attention layers marked); tap a layer for its 64 × 56 sheet, drag to preview
  neurons (logit lens in the tooltip), release to open. The open neuron's connection web is drawn on top.
- names you give neurons stay in the browser (localStorage); `#L12N1234` links open a neuron.

## How it runs on the phone

| | |
|---|---|
| model | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0), the newest small Qwen; same family as the reference's 4B |
| architecture | 24 layers: 18 Gated DeltaNet (linear attention) + 6 gated full attention, SwiGLU MLP, tied 248k vocab |
| weights | streamed from Hugging Face as bf16 (only the text decoder: the first 1.5 GB of the 1.75 GB file; the vision tower and MTP head are never downloaded), quantized to int8 (per 128 values) on the GPU as the bytes arrive, then stored in Cache Storage so later visits load from the device |
| GPU memory | about 0.78 GB of weights + about 0.15 GB of caches and scratch, under Safari's per-tab budget |
| engine | hand-written WGSL (`engine/`): no ONNX/TVM black box, so every layer has hooks. f32 math; needs neither `shader-f16` nor `subgroups` (Safari on iOS has no subgroups) |
| hooks | at the input of `down_proj`, where neuron activations `silu(gate) * up` live: add or multiply per batch row (the three chat columns are three rows of one batch), capture any layer at every token, per-head output scale on DeltaNet layers |

Why not Gemma 4: its smallest model, E2B, is 5.1B parameters with its per-layer embedding table (about 2.5 GB even
in the QAT mobile build), and its post-FFN norms and per-layer residual scalars make a single neuron's write
nonlinear, which muddies steering and the logit lens. Qwen3.6 and 3.8 ship only at 27B and up.

## iPhone Duo layout

Safari on iPhone Duo gives pages 382–951 CSS px of width (WebKit, *Safari 27.1*). The page adapts by width and,
where enabled, by the fold itself:

| where | layout |
|---|---|
| outer display (382 × ~600), split view (385 wide), outer landscape | one pane at a time with a switcher; the three chat columns become a swipeable pager |
| inner display, landscape (951 × ~600) | model on the left page, neuron on the right page, split at the fold |
| inner display folded like a book | the same, split exactly at the reported fold |
| inner display, portrait (669 × ~880) | neuron (three chat columns) on top, composer and model below |
| inner display folded like a laptop | the same: the top half is the screen, the bottom half the deck |

The fold is read from the **Viewport Segments API** (`window.viewport.segments`) and the **Device Posture API**
(`navigator.devicePosture`), which Safari 27.1 ships behind feature flags: Settings › Apps › Safari › Advanced ›
Feature Flags › *Viewport Segments API* and *Device Posture API*. Without them the layout splits at the middle of the
inner display, which is where the fold is. `?duo=book` or `?duo=laptop` simulates the segments in a desktop browser.

## Run it

It is a static site. Serve the folder over HTTPS (WebGPU needs a secure context) and open it in Safari:

- **GitHub Pages**: Settings › Pages › Deploy from a branch › this branch, `/ (root)`. Then open
  `https://<user>.github.io/duo-mech-interp/` on the phone.
- **locally**: `python3 -m http.server 8000` and open `http://localhost:8000` (localhost counts as secure).

The first load downloads 1.5 GB from Hugging Face (use Wi-Fi); "forget downloaded weights" on the start screen removes
the stored copy. Needs iOS 26 or later (WebGPU is on by default since Safari 26).

## Tests

```
python tests/tokenizer_ref.py tokenizer.json expected.json && node tests/tokenizer.mjs tokenizer.json expected.json
python tests/engine_ref.py <local checkpoint dir> engine_expected.json   # HF transformers, float32, CPU
node tests/engine.mjs <served url> <expected url> <hub mirror url>        # same checks in headless Chromium WebGPU
node tests/screens.mjs <served url> <hub mirror url> <out dir>           # screenshots at iPhone Duo viewport sizes
```

`engine.mjs` compares the WGSL engine with Hugging Face transformers on the same prompt: next-token logits, MLP
activations at layers 0/3/11/23, and greedy continuations with a neuron pushed up, untouched and down.

## What is not ported

The reference's offline GPU sweep (every neuron × doses × prompts, ranked by word-cluster mass against random
controls), its UMAP rollout map, FineFineWeb exemplars, R / J++ / S lenses, attention-head analysis and the Claude
agent tab need either a GPU pod or a server. Here the corpus-based pieces (activation ranges, examples) come from a
small on-device calibration pass instead.

## License

MIT. Qwen3.5 weights are Apache-2.0 and are downloaded from Hugging Face at run time.
