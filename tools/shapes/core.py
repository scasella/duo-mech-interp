"""Shared code for finding concept shapes with the Qwen-Scope SAEs (Qwen3.5-2B-Base, residual stream, TopK 50).

Used by both entry points:
  modal_app.py  - the full run on a Modal GPU
  local.py      - the same steps on CPU at small scale (smoke tests, probe families)

Two kinds of data are collected:
  corpus  - SAE codes for web text: how often each feature fires, which features fire together (co-activation),
            and the full codes of the first sequences (for reconstructions, PCA and example contexts)
  probes  - template sentences for families with a known order (weekdays, months, numbers, colours, ...): the
            residual and the SAE code at the family word, used to check the analysis against shapes we can verify
"""
import json, math, os, time
import numpy as np
import torch

BASE = "Qwen/Qwen3.5-2B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
K, D_SAE, D_MODEL = 50, 32768, 2048
SKIP_FIRST = 4            # the first positions of a packed sequence carry attention-sink outliers


# ---------------------------------------------------------------- model + SAE
def load_model(device="cpu", dtype=torch.bfloat16, max_layer=23, path=BASE, cache_dir=None):
    """Tokenizer and the text model truncated after `max_layer` (no LM head: only residuals are needed)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, cache_dir=cache_dir).eval()
    lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
    lm.layers = torch.nn.ModuleList(list(lm.layers)[: max_layer + 1])
    return tok, lm.to(device)


def load_sae(layer, device="cpu", repo=SAE_REPO, cache_dir=None, path=None):
    from huggingface_hub import hf_hub_download
    p = path or hf_hub_download(repo, f"layer{layer}.sae.pt", cache_dir=cache_dir)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    sae = {k: v.to(device=device, dtype=torch.float32) for k, v in sd.items()}
    sae["W_decT"] = sae.pop("W_dec").T.contiguous()           # (D_SAE, D_MODEL): one row per feature
    return sae


def encode(x, sae):
    """As in the Qwen-Scope model card: pre = x W_enc^T + b_enc, keep the top 50. x: (N, D_MODEL) float32."""
    vals, idx = (x @ sae["W_enc"].T + sae["b_enc"]).topk(K, dim=-1)
    return vals, idx


def decode(vals, idx, sae):
    return torch.einsum("nk,nkd->nd", vals, sae["W_decT"][idx]) + sae["b_dec"]


class Capture:
    """Residual stream after the given decoder layers (Qwen-Scope's hook point, resid_post)."""
    def __init__(self, lm, layers):
        self.out = {}
        self.hooks = [lm.layers[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def f(mod, inp, out):
            self.out[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    def remove(self):
        for h in self.hooks: h.remove()


# ---------------------------------------------------------------- corpus
def corpus_batches(tok, seq_len, batch_size, dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT",
                   split="train", text_key="text", text_file=None):
    """Packed sequences of documents separated by <|endoftext|> (streamed; or a local text file, one doc per
    blank-line-separated paragraph, for offline smoke tests)."""
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    if text_file:
        docs = (d for d in open(text_file, encoding="utf-8").read().split("\n\n") if d.strip())
    else:
        from datasets import load_dataset
        docs = (ex[text_key] for ex in load_dataset(dataset, config, split=split, streaming=True))
    buf, rows = [], []
    for d in docs:
        buf.extend(tok(d, add_special_tokens=False).input_ids)
        buf.append(eos)
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(rows) == batch_size:
                yield torch.tensor(rows)
                rows = []


@torch.inference_mode()
def harvest(tok, lm, saes, batches, n_tokens, n_store_seqs, out_dir, device, partners=256, min_count=5,
            save_full_coact=False, log=print):
    """Stream text through the model; per layer accumulate feature frequencies and pairwise co-activation counts
    (dense D_SAE x D_SAE float32, 4.3 GB per layer); keep full codes for the first `n_store_seqs` sequences."""
    os.makedirs(out_dir, exist_ok=True)
    layers = sorted(saes)
    cap = Capture(lm, layers)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    iu = torch.triu_indices(K, K, 1, device=device)
    st = {L: dict(freq=torch.zeros(D_SAE, dtype=torch.float64, device=device),
                  coact=torch.zeros(D_SAE * D_SAE, dtype=torch.float32, device=device),
                  idx=[], val=[], ss_res=0.0, ss_tot=0.0, neg=0, n=0) for L in layers}
    tok_store, valid_store = [], []
    seen, nb, t0 = 0, 0, time.time()
    for ids in batches:
        ids = ids.to(device)
        B, T = ids.shape
        lm(input_ids=ids, use_cache=False)
        valid = torch.ones(B, T, dtype=torch.bool, device=device)
        valid[:, :SKIP_FIRST] = False
        valid &= ids != eos
        vflat = valid.reshape(-1)
        keep = len(tok_store) < n_store_seqs
        if keep:
            tok_store.append(ids.int().cpu()); valid_store.append(valid.cpu())
        for L in layers:
            s, sae = st[L], saes[L]
            x = cap.out[L].reshape(-1, D_MODEL).float()
            vals, idx = encode(x, sae)
            fire = (vals > 0) & vflat[:, None]
            s["neg"] += int(((vals <= 0) & vflat[:, None]).sum()); s["n"] += int(vflat.sum()) * K
            f_idx = idx[fire]
            s["freq"].index_add_(0, f_idx, torch.ones(f_idx.numel(), dtype=torch.float64, device=device))
            ok = fire[:, iu[0]] & fire[:, iu[1]]
            a, b = idx[:, iu[0]][ok].long(), idx[:, iu[1]][ok].long()
            ones = torch.ones(a.numel(), dtype=torch.float32, device=device)
            s["coact"].index_add_(0, a * D_SAE + b, ones)
            s["coact"].index_add_(0, b * D_SAE + a, ones)
            if keep:
                s["idx"].append(idx.view(B, T, K).short().cpu()); s["val"].append(vals.view(B, T, K).half().cpu())
            if nb % 10 == 0:                                       # reconstruction check on a few valid tokens
                sel = vflat.nonzero().squeeze(1)[:256]
                xs, xh = x[sel], decode(vals[sel], idx[sel], sae)
                s["ss_res"] += float(((xs - xh) ** 2).sum()); s["ss_tot"] += float(((xs - xs.mean(0)) ** 2).sum())
        seen += int(vflat.sum()); nb += 1
        if nb % 20 == 0 or seen >= n_tokens:
            r2 = {L: round(1 - st[L]["ss_res"] / max(st[L]["ss_tot"], 1e-9), 3) for L in layers}
            log(f"{seen:,} tokens  {seen / (time.time() - t0):,.0f} tok/s  recon R2 {r2}")
        if seen >= n_tokens:
            break
    if hasattr(batches, "close"): batches.close()
    cap.remove()
    meta = dict(base=BASE, sae=SAE_REPO, layers=layers, n_tokens=seen, seconds=round(time.time() - t0),
                recon_r2={}, nonpositive_topk_frac={})
    if tok_store:
        np.save(f"{out_dir}/tokens.npy", torch.cat(tok_store).numpy())
        np.save(f"{out_dir}/valid.npy", torch.cat(valid_store).numpy())
    for L in layers:
        s = st[L]
        meta["recon_r2"][L] = round(1 - s["ss_res"] / max(s["ss_tot"], 1e-9), 4)
        meta["nonpositive_topk_frac"][L] = round(s["neg"] / max(s["n"], 1), 5)
        np.save(f"{out_dir}/L{L}_freq.npy", s["freq"].cpu().numpy())
        if s["idx"]:
            np.save(f"{out_dir}/L{L}_idx.npy", torch.cat(s["idx"]).numpy())
            np.save(f"{out_dir}/L{L}_val.npy", torch.cat(s["val"]).numpy())
        C = s["coact"].view(D_SAE, D_SAE)
        pi, pv, pc = top_partners(C, s["freq"], seen, partners, min_count)
        np.save(f"{out_dir}/L{L}_partners_idx.npy", pi); np.save(f"{out_dir}/L{L}_partners_npmi.npy", pv)
        np.save(f"{out_dir}/L{L}_partners_cnt.npy", pc)
        if save_full_coact:
            np.save(f"{out_dir}/L{L}_coact.npy", C.cpu().numpy())
        del s["coact"], C
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
    log(json.dumps(meta))
    return meta


@torch.inference_mode()
def harvest_centroids(tok, lm, layers, batches, n_tokens, out_dir, device, min_count=20, model=BASE, log=print):
    """Average residual of every vocabulary token over its occurrences in the text (one row per token that occurs
    at least `min_count` times), per layer. Families of words whose averages form a circle or a line are searched
    for in analyze.py words; this needs no SAE."""
    os.makedirs(out_dir, exist_ok=True)
    layers = sorted(layers)
    cap = Capture(lm, layers)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    V, D = lm.config.vocab_size, lm.config.hidden_size
    sums = {L: torch.zeros(V, D, dtype=torch.float32, device=device) for L in layers}
    cnt = torch.zeros(V, dtype=torch.float32, device=device)
    seen, nb, t0 = 0, 0, time.time()
    for ids in batches:
        ids = ids.to(device)
        lm(input_ids=ids, use_cache=False)
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, :SKIP_FIRST] = False
        valid &= ids != eos
        v = valid.reshape(-1)
        t = ids.reshape(-1)[v]
        cnt.index_add_(0, t, torch.ones(t.numel(), dtype=torch.float32, device=device))
        for L in layers:
            sums[L].index_add_(0, t, cap.out[L].reshape(-1, D)[v].float())
        seen += int(v.sum()); nb += 1
        if nb % 50 == 0 or seen >= n_tokens:
            log(f"{seen:,} tokens  {seen / (time.time() - t0):,.0f} tok/s")
        if seen >= n_tokens:
            break
    if hasattr(batches, "close"): batches.close()
    cap.remove()
    keep = (cnt >= min_count).nonzero().squeeze(1)
    np.save(f"{out_dir}/word_ids.npy", keep.cpu().numpy().astype(np.int32))
    np.save(f"{out_dir}/word_counts.npy", cnt[keep].cpu().numpy().astype(np.int64))
    for L in layers:
        np.save(f"{out_dir}/L{L}_word_means.npy", (sums[L][keep] / cnt[keep, None]).half().cpu().numpy())
    meta = dict(base=model, layers=layers, n_tokens=seen, words=int(keep.numel()), min_count=min_count,
                seconds=round(time.time() - t0))
    json.dump(meta, open(f"{out_dir}/words_meta.json", "w"), indent=1)
    log(json.dumps(meta))
    return meta


@torch.inference_mode()
def top_partners(C, freq, n_tokens, k, min_count, chunk=1024):
    """For every feature, the `k` features it fires with most often relative to chance (normalised PMI),
    counting only pairs seen at least `min_count` times."""
    D = C.shape[0]
    p = (freq / n_tokens).float()
    out_i = np.zeros((D, k), np.int32); out_v = np.zeros((D, k), np.float16); out_c = np.zeros((D, k), np.int32)
    for r0 in range(0, D, chunk):
        c = C[r0:r0 + chunk].float()
        pij = c / n_tokens
        npmi = torch.log(pij / (p[r0:r0 + chunk, None] * p[None, :]).clamp_min(1e-30)) / (-torch.log(pij.clamp_min(1e-30)))
        npmi = torch.where(c >= min_count, npmi, torch.full_like(npmi, -2.0))
        v, i = npmi.topk(k, dim=1)
        out_i[r0:r0 + chunk] = i.cpu().numpy(); out_v[r0:r0 + chunk] = v.cpu().numpy()
        out_c[r0:r0 + chunk] = torch.gather(c, 1, i).cpu().numpy().astype(np.int32)
    return out_i, out_v, out_c


# ---------------------------------------------------------------- probe families (known order, for checking)
FAMILIES = {
    "weekday": dict(order="cyclic", vals="Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(), templates=[
        "I'll see you on {X}.", "The meeting is on {X} morning.", "Every {X} we go to the market.",
        "She was born on a {X} in spring.", "The store is closed on {X}.", "Let's meet again next {X}.",
        "It rained all day on {X}.", "The package should arrive by {X}.", "On {X} afternoon he called his mother.",
        "We have soccer practice every {X}.", "The report is due {X} at noon.", "Last {X} was a busy day.",
        "See you {X}!", "The concert was moved to {X}.", "Nothing much happened on {X}.", "The museum is free on {X}."]),
    "month": dict(order="cyclic", vals=("January February March April May June July August September October "
                                        "November December").split(), templates=[
        "My birthday is in {X}.", "The festival takes place every {X}.", "In {X} the weather turns cold.",
        "She moved to the city last {X}.", "The school year starts in {X}.", "We visited Paris in {X}.",
        "The deadline is the end of {X}.", "It snowed heavily in {X}.", "Sales were strong in {X} this year.",
        "He was born on {X} 3rd.", "The garden blooms in late {X}.", "Prices rose sharply in {X}.",
        "See you in {X}!", "The project started in {X} and ended later.", "Every {X} they travel north.",
        "The election is held in {X}."]),
    "number": dict(order="linear", vals="one two three four five six seven eight nine ten eleven twelve".split(), templates=[
        "I bought {X} apples at the market.", "There were {X} people in the room.", "She has {X} cousins.",
        "We waited for {X} hours.", "The recipe needs {X} eggs.", "He scored {X} goals this season.",
        "They planted {X} trees in the yard.", "The box contains {X} books.", "It took {X} days to finish.",
        "Only {X} students passed the test."]),
    "ordinal": dict(order="linear", vals="first second third fourth fifth sixth seventh eighth ninth tenth".split(), templates=[
        "She finished in {X} place.", "This is the {X} time I have seen it.", "He lives on the {X} floor.",
        "It was their {X} anniversary.", "The {X} chapter is the best.", "Turn left at the {X} street.",
        "Our {X} child was born in May.", "The {X} attempt finally worked."]),
    "season": dict(order="cyclic", vals="spring summer autumn winter".split(), templates=[
        "The garden looks best in {X}.", "We always travel in {X}.", "The {X} months are my favorite.",
        "Last {X} was unusually wet.", "Prices drop every {X}.", "She was born in the {X} of 1990.",
        "The festival happens each {X}.", "It gets dark early in {X}."]),
    "time_of_day": dict(order="cyclic", vals="morning afternoon evening night".split(), templates=[
        "Every {X} he reads the news.", "The {X} sky was clear.", "She called me that {X}.",
        "The {X} was cold and quiet.", "We spent the whole {X} at the beach.", "The {X} shift starts soon.",
        "I like to walk in the {X}.", "The {X} air smelled of rain."]),
    "color": dict(order="cyclic", vals="red orange yellow green blue purple".split(), templates=[
        "She painted the door {X}.", "He wore a {X} shirt to the party.", "The car is {X} with black stripes.",
        "My favorite color is {X}.", "The walls were bright {X}.", "A {X} light flashed twice.",
        "They chose {X} flowers for the table.", "The sky turned {X} at sunset."]),
    "temperature": dict(order="linear", vals="freezing cold cool warm hot".split(), templates=[
        "The water was {X} this morning.", "It felt {X} outside today.", "The soup is still {X}.",
        "We had a {X} day at the lake.", "The room was {X} all night.", "Her hands were {X}.",
        "It will be {X} tomorrow.", "The tea is {X} now."]),
    "size": dict(order="linear", vals="tiny small large huge".split(), templates=[
        "They live in a {X} house.", "She found a {X} spider in the bath.", "It was a {X} mistake.",
        "He ordered a {X} pizza.", "The dog is {X} for its age.", "We saw a {X} ship in the harbor.",
        "A {X} crowd gathered.", "The box was {X}."]),
    "compass": dict(order="cyclic", vals="north east south west".split(), templates=[
        "The wind blew from the {X}.", "They drove {X} for hours.", "The house faces {X}.",
        "Birds fly {X} in winter.", "The {X} side of the island is rocky.", "We hiked toward the {X}.",
        "The river flows {X}.", "The {X} gate was closed."]),
    "time_unit": dict(order="linear", vals="second minute hour day week month year decade century".split(), templates=[
        "It only took one {X}.", "We waited one {X} for the bus.", "The warranty lasts one {X}.",
        "Give me one {X} to think.", "He was gone for one {X}.", "The process takes about one {X}.",
        "It happens once every {X}.", "Prices doubled in one {X}."]),
    "planet": dict(order="linear", vals="Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split(), templates=[
        "The probe flew past {X} last year.", "Scientists studied the surface of {X}.", "A mission to {X} is planned.",
        "The moons of {X} are fascinating.", "Light from {X} reached the telescope.", "We read a book about {X}.",
        "The orbit of {X} is well known.", "They pointed the camera at {X}."]),
}


@torch.inference_mode()
def probe(tok, lm, saes, out_dir, device, families=FAMILIES, layers=None, log=print):
    """Residual and SAE code at the family word, for every template x value (values must be single tokens).
    With no SAEs (e.g. for the 0.8B model the app runs), pass `layers` and only residuals are saved."""
    os.makedirs(out_dir, exist_ok=True)
    layers = sorted(saes) if saes else sorted(layers)
    cap = Capture(lm, layers)
    meta = {}
    for name, F in families.items():
        vals = [v for v in F["vals"] if len(tok(" " + v, add_special_tokens=False).input_ids) == 1]
        vids = [tok(" " + v, add_special_tokens=False).input_ids[0] for v in vals]
        d_model = lm.config.hidden_size
        R = {L: np.zeros((len(F["templates"]), len(vals), d_model), np.float16) for L in layers}
        I = {L: np.zeros((len(F["templates"]), len(vals), K), np.int16) for L in layers}
        V = {L: np.zeros((len(F["templates"]), len(vals), K), np.float16) for L in layers}
        for ti, tpl in enumerate(F["templates"]):
            rows = [tok(tpl.replace("{X}", v), add_special_tokens=False).input_ids for v in vals]
            if len(set(map(len, rows))) != 1:
                raise ValueError(f"{name}: template splits differently across values: {tpl}")
            pos = rows[0].index(vids[0])
            lm(input_ids=torch.tensor(rows, device=device), use_cache=False)
            for L in layers:
                x = cap.out[L][:, pos].float()
                R[L][ti] = x.cpu().numpy()
                if saes:
                    v, i = encode(x, saes[L])
                    I[L][ti] = i.short().cpu().numpy(); V[L][ti] = v.half().cpu().numpy()
        for L in layers:
            extra = dict(idx=I[L], val=V[L]) if saes else {}
            np.savez_compressed(f"{out_dir}/probe_{name}_L{L}.npz", resid=R[L], **extra)
        meta[name] = dict(order=F["order"], vals=vals, n_templates=len(F["templates"]))
        log(f"probe {name}: {len(vals)} values x {len(F['templates'])} templates")
    cap.remove()
    json.dump(meta, open(f"{out_dir}/probes.json", "w"), indent=1)
    return meta
