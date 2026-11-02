# duo neuron explorer

A steering-first neuron explorer for **Qwen3.5-0.8B** that runs entirely in Safari on an iPhone (built around the
iPhone Duo's two displays and fold), on WebGPU, with no server. It is an on-device port of
[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (MIT), which does the same for
Qwen3.5-4B with a Python server and a GPU sweep.

Every MLP neuron (24 layers × 3,584) can be turned up or down while the model answers. The app has two places:

- **Steer** (the experiment): ask once, read three answers: the neuron **boosted**, left **as is**, and
  **suppressed**. One strength slider (in the reference's unit: a fraction of the median residual norm leaving the
  layer), suggested questions, stop at any time. Switching neurons offers to ask the last question again, so the same
  question can be compared across neurons.
- **Explore** (choose and understand):
  - **Find**: type any text; see the neurons that react most to it (scored against each neuron's range on a small
    built-in corpus, measured once in the background), tap a word for just that word's neurons.
  - **Neuron**: the words boosting or suppressing it favors (logit lens), a name, and behind collapsed sections:
    where it fires in sample texts, how it reacts to your own text, its effect on the next word (steer lens), its
    connected neurons (direct path through the weights), and advanced options (multiply instead of push, keep a
    multi-turn conversation).
  - **Browse**: your saved and recent neurons, a map of all 24 layers (tap a band for its sheet, slide to preview,
    lift to open), and opening a neuron by number.

First visit: a three-step welcome and one button (download, 1.5 GB once). After loading: **Try an example** (a
neuron whose push visibly changes answers), **Find neurons in your own words**, or **Surprise me**. `#L12N1234` links
open a neuron; names and saved neurons stay in the browser.

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
| outer display (382 × ~600), split view (385 wide), outer landscape | Steer and Explore as two tabs; the three answers stacked |
| inner display, landscape (951 × ~600) | Explore on the left page, Steer on the right page, split at the fold |
| inner display folded like a book | the same, split exactly at the reported fold |
| inner display, portrait (669 × ~880) | the three answers side by side on top; the question box and Explore below |
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
the stored copy. Safari deletes a website's stored data after about a week without a visit; adding the page to the Home
Screen keeps it. Needs iOS 26 or later (WebGPU is on by default since Safari 26).

## Tests

```
python tests/tokenizer_ref.py tokenizer.json expected.json && node tests/tokenizer.mjs tokenizer.json expected.json
python tests/engine_ref.py <local checkpoint dir> engine_expected.json   # HF transformers, float32, CPU
node tests/engine.mjs <served url> <expected url> <hub mirror url>        # same checks in headless Chromium WebGPU
node tests/smoke.mjs <served url> <hub mirror url>                       # every explorer operation once, tiny inputs
node tests/keyboard.mjs <served url>                                     # text fields keep focus when the keyboard opens
SMALL_CORPUS=1 node tests/e2e.mjs <site url> <out dir>                   # end-user run of every feature on the deployed site
node tests/profile.mjs <served url> <hub mirror url>                     # time per kernel type
node tests/screens.mjs <served url> <hub mirror url> <out dir>           # screenshots at iPhone Duo viewport sizes
```

`engine.mjs` compares the WGSL engine with Hugging Face transformers (float32) on the same chat prompt. Last run
(headless Chromium, SwiftShader WebGPU):

| check | result |
|---|---|
| tokenizer, 13 strings (code, CJK, emoji ZWJ, RTL, chat template) | 13/13 identical ids, exact round trip |
| next-token logits | same top-1, top-20 overlap 20/20, relative error 1.9%, correlation 0.9998 |
| neuron activations (input of `down_proj`), layers 0 / 3 / 11 / 23 | relative error 1.6–4.2% (int8 weights) |
| greedy decode, three rows in one batch: L12 N1234 pushed +8 / untouched / −8 | +8 and untouched: tokens match; −8: first 3 match, the 4th is a near tie in the reference itself ("3" 18.050 vs "wen" 18.011) |

## What is not ported

The reference's offline GPU sweep (every neuron × doses × prompts, ranked by word-cluster mass against random
controls), its UMAP rollout map, FineFineWeb exemplars, R / J++ / S lenses, attention-head analysis and the Claude
agent tab need either a GPU pod or a server. Here the corpus-based pieces (activation ranges, examples) come from a
small on-device calibration pass instead.

## Files

| | |
|---|---|
| `index.html`, `app/` | UI: `main.js` (panels, chat, fires, lens, examples), `layout.js` (fold / posture / width modes), `map.js` (layer stack and sheets on a canvas), `corpus.js` (calibration passages written for this project) |
| `engine/gpu.js` | WebGPU device, buffers, cached pipelines / uniforms / bind groups |
| `engine/kernels.js`, `engine/kernels_qwen35.js` | WGSL: int8 quantize on load, matvec / tiled matmul / transposed matvec on int8 weights, RMSNorm, gated attention with partial RoPE over a KV cache, Gated DeltaNet conv + delta rule + gated norm, SwiGLU with the neuron hook |
| `engine/model.js` | Qwen3.5 decoder: streaming loader, Cache Storage, batched rows, forward with pushes / capture / residual snapshots |
| `engine/tokenizer.js` | byte-level BPE from `tokenizer.json`, Qwen chat template |
| `engine/analysis.js` | explorer operations: steer, read, lens, steer lens, connections, calibration, fires scoring |
| `tests/` | reference scripts (HF tokenizers / transformers) and headless-Chromium checks |

## License

MIT. Qwen3.5 weights are Apache-2.0 and are downloaded from Hugging Face at run time.
