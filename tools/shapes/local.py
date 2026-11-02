# Run the shape-finding steps on this machine (CPU is fine at small scale):
#   python tools/shapes/local.py probe  --layers 6,12,18 --out runs/probe
#   python tools/shapes/local.py corpus --layers 12 --tokens 20000 --store 64 --out runs/smoke
# --cache points at a Hugging Face cache that already holds Qwen3.5-2B-Base and the Qwen-Scope SAEs.
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["probe", "corpus", "words"])
ap.add_argument("--layers", default="12")
ap.add_argument("--out", required=True)
ap.add_argument("--cache", default=None)
ap.add_argument("--tokens", type=int, default=20000)
ap.add_argument("--store", type=int, default=64, help="sequences whose full codes are kept")
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--text-file", default=None)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--model", default=core.BASE, help="another checkpoint (e.g. the app's 0.8B) for probes without SAEs")
ap.add_argument("--no-sae", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
layers = [int(x) for x in a.layers.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok, lm = core.load_model(dev, torch.bfloat16, max(layers), path=a.model, cache_dir=a.cache)
saes = {} if (a.no_sae or a.what == "words") else {L: core.load_sae(L, dev, cache_dir=a.cache) for L in layers}
if a.what == "probe":
    core.probe(tok, lm, saes, a.out, dev, layers=layers)
elif a.what == "words":
    batches = core.corpus_batches(tok, a.seq, a.batch, text_file=a.text_file)
    core.harvest_centroids(tok, lm, layers, batches, a.tokens, a.out, dev, min_count=2)
else:
    batches = core.corpus_batches(tok, a.seq, a.batch, text_file=a.text_file)
    core.harvest(tok, lm, saes, batches, a.tokens, a.store, a.out, dev)
sys.stdout.flush()
os._exit(0)          # the streaming dataset leaves background threads that keep the process alive
