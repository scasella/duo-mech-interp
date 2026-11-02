# Full run on a Modal GPU: probe families + web-text harvest of Qwen-Scope SAE codes, then the analysis.
#   pip install modal && modal token set ...      (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment)
#   modal run tools/shapes/modal_app.py --run pilot --layers 12 --tokens 2000000
#   modal run tools/shapes/modal_app.py --run full --layers 6,12,18 --tokens 20000000
#   modal volume get qwen-scope-shapes runs/<run> ./runs/<run>
# GPU type: SHAPES_GPU (default A100-80GB). Weights and outputs live on the volume "qwen-scope-shapes".
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("SHAPES_GPU", "A100-80GB")
app = modal.App("qwen-scope-shapes")
vol = modal.Volume.from_name("qwen-scope-shapes", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.22.1", "datasets", "huggingface_hub", "numpy",
                      "networkx", "safetensors")
         .env({"HF_HOME": "/data/hf"})
         .add_local_file(os.path.join(HERE, "core.py"), "/root/core.py")
         .add_local_file(os.path.join(HERE, "analyze.py"), "/root/analyze.py"))


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, memory=65536, timeout=8 * 3600)
def harvest(run: str, layers: list, n_tokens: int, n_store: int, seq_len: int, batch: int, full_coact: bool):
    import torch, core
    torch.backends.cuda.matmul.allow_tf32 = True
    out = f"/data/runs/{run}"
    tok, lm = core.load_model("cuda", torch.bfloat16, max(layers), cache_dir="/data/hf")
    saes = {L: core.load_sae(L, "cuda", cache_dir="/data/hf") for L in layers}
    vol.commit()                                                     # keep the downloaded weights
    core.probe(tok, lm, saes, out + "/probe", "cuda")
    batches = core.corpus_batches(tok, seq_len, batch)
    meta = core.harvest(tok, lm, saes, batches, n_tokens, n_store, out, "cuda", save_full_coact=full_coact,
                        log=lambda s: print(s, flush=True))
    vol.commit()
    return meta


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, memory=65536, timeout=4 * 3600)
def centroids(run: str, layers: list, n_tokens: int, seq_len: int, batch: int, min_count: int):
    import torch, core
    out = f"/data/runs/{run}"
    tok, lm = core.load_model("cuda", torch.bfloat16, max(layers), cache_dir="/data/hf")
    batches = core.corpus_batches(tok, seq_len, batch)
    meta = core.harvest_centroids(tok, lm, layers, batches, n_tokens, out, "cuda", min_count=min_count,
                                  log=lambda s: print(s, flush=True))
    vol.commit()
    return meta


@app.local_entrypoint()
def words(run: str = "words", layers: str = "6,12,18", tokens: int = 30_000_000, seq: int = 512, batch: int = 32,
          min_count: int = 20):
    """Average residual per vocabulary word (no SAE): modal run tools/shapes/modal_app.py::words"""
    print(centroids.remote(run, [int(x) for x in layers.split(",")], tokens, seq, batch, min_count))
    print(f"done. fetch with: modal volume get qwen-scope-shapes runs/{run} ./runs/{run}")


@app.function(image=image, cpu=8, memory=98304, volumes={"/data": vol}, timeout=6 * 3600)
def analyze(run: str, layers: list):
    import analyze as A
    out = f"/data/runs/{run}"
    A.probes(out + "/probe", layers, "/data/hf")
    A.groups(out, layers, "/data/hf", probe_run=out + "/probe")
    vol.commit()


@app.local_entrypoint()
def main(run: str = "pilot", layers: str = "12", tokens: int = 2_000_000, store: int = 2000, seq: int = 512,
         batch: int = 16, full_coact: bool = False, skip_harvest: bool = False, skip_analyze: bool = False):
    L = [int(x) for x in layers.split(",")]
    if not skip_harvest:
        print(harvest.remote(run, L, tokens, store, seq, batch, full_coact))
    if not skip_analyze:
        analyze.remote(run, L)
    print(f"done. fetch with: modal volume get qwen-scope-shapes runs/{run} ./runs/{run}")
