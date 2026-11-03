"""Block-sparse featurizer (Grassmannian variant, after Fel et al. 2026, "Structuring Sparsity") trained online on a
model's residual stream, plus the same thing with blocks of one dimension as a TopK-SAE baseline.

Each concept is an orthonormal b-frame D_g (b x d). The code is tied: z_g = gamma_g * x D_g^T; the k blocks with the
largest |z_g| are kept; the reconstruction is sum_g z_g D_g. Frames are re-orthonormalised (Gram-Schmidt) in every forward.
An auxiliary loss asks the next k blocks (by norm, among those not selected) to explain what the selected ones missed,
which keeps blocks from dying.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn


class GrassmannianBSF(nn.Module):
    def __init__(self, d, n_blocks, block, k):
        super().__init__()
        self.d, self.G, self.b, self.k = d, n_blocks, block, k
        self.B = nn.Parameter(torch.linalg.qr(torch.randn(n_blocks, d, block)).Q)    # (G, d, b)
        self.log_gamma = nn.Parameter(torch.zeros(n_blocks))

    def frames(self):
        """Orthonormal columns by Gram-Schmidt (b is small; much cheaper than batched QR on a GPU)."""
        cols = []
        for j in range(self.b):
            v = self.B[:, :, j]
            for q in cols:
                v = v - (v * q).sum(-1, keepdim=True) * q
            cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        return torch.stack(cols, -1)

    def codes(self, x, D=None):
        D = self.frames() if D is None else D
        z = (x @ D.permute(1, 0, 2).reshape(self.d, -1)).view(len(x), self.G, self.b)
        return z * self.log_gamma.exp()[None, :, None], D

    def decode(self, z, D):
        return z.reshape(len(z), -1) @ D.permute(0, 2, 1).reshape(-1, self.d)

    def forward(self, x, aux=True):
        z, D = self.codes(x)
        norms = z.norm(dim=-1)
        top = norms.topk(self.k, dim=1).indices
        keep = torch.zeros_like(norms).scatter_(1, top, 1.0)
        xhat = self.decode(z * keep[..., None], D)
        loss = (x - xhat).pow(2).sum(-1).mean()
        out = dict(xhat=xhat, top=top, z=z, loss=loss)
        if aux:
            masked = norms.masked_fill(keep.bool(), -1.0)
            top2 = masked.topk(self.k, dim=1).indices
            keep2 = torch.zeros_like(norms).scatter_(1, top2, 1.0)
            r = (x - xhat).detach()
            out["aux"] = (r - self.decode(z * keep2[..., None], D)).pow(2).sum(-1).mean()
            out["loss_total"] = loss + out["aux"] / self.k
        return out

    @torch.no_grad()
    def encode_top(self, x):
        """top-k block ids and their b coordinates, for analysis."""
        z, _ = self.codes(x)
        top = z.norm(dim=-1).topk(self.k, dim=1).indices
        return top, torch.gather(z, 1, top[..., None].expand(-1, -1, self.b))


def train_online(tok, lm, layer, batches, n_tokens, configs, out_dir, device, lr=4e-4, batch=4096, buffer=1 << 19,
                 steps_per_refill=4, log=print, eval_every=500):
    """Stream residuals of `layer` from `batches` into a shuffled buffer and train every featurizer in `configs`
    ({name: (n_blocks, block, k)}) on the same minibatches. Saves <out_dir>/<name>.pt with the normalisation."""
    from core import Capture, SKIP_FIRST
    os.makedirs(out_dir, exist_ok=True)
    cap = Capture(lm, [layer])
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    d = lm.config.hidden_size
    it = iter(batches)

    def harvest():
        ids = next(it).to(device)
        with torch.inference_mode():
            lm(input_ids=ids, use_cache=False)
        v = torch.ones_like(ids, dtype=torch.bool); v[:, :SKIP_FIRST] = False; v &= ids != eos
        return cap.out[layer][v].float()

    # normalisation from the first tokens: centre, then scale so the mean norm is sqrt(d)
    first = torch.cat([harvest() for _ in range(8)])
    mu = first.mean(0)
    scale = math.sqrt(d) / float((first - mu).norm(dim=-1).mean())
    norm = lambda x: (x - mu) * scale
    buf = torch.empty(buffer, d, device=device, dtype=torch.bfloat16)
    filled = 0
    while filled < buffer:
        x = norm(harvest()); n = min(len(x), buffer - filled); buf[filled:filled + n] = x[:n].bfloat16(); filled += n
    held = norm(harvest())[:4096]
    var = float((held - held.mean(0)).pow(2).sum(-1).mean())
    models = {name: GrassmannianBSF(d, *cfg).to(device) for name, cfg in configs.items()}
    opts = {name: torch.optim.Adam(m.parameters(), lr=lr) for name, m in models.items()}
    total_steps = int(n_tokens / batch)
    sched = lambda s: min(1.0, s / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    fired = {name: torch.zeros(cfg[0], device=device) for name, cfg in configs.items()}
    seen, step, t0, history = filled, 0, time.time(), []
    while step < total_steps:
        x = norm(harvest()); pos = torch.randint(0, buffer, (len(x),), device=device); buf[pos] = x.bfloat16(); seen += len(x)
        for _ in range(steps_per_refill):
            xb = buf[torch.randint(0, buffer, (batch,), device=device)].float()
            for name, m in models.items():
                for g in opts[name].param_groups: g["lr"] = lr * sched(step)
                o = m(xb)
                opts[name].zero_grad(set_to_none=True); o["loss_total"].backward(); opts[name].step()
                fired[name].index_add_(0, o["top"].reshape(-1), torch.ones(o["top"].numel(), device=device))
            step += 1
            if step % eval_every == 0 or step == total_steps:
                rec = {"step": step, "tokens": seen, "tok_s": round(seen / (time.time() - t0))}
                with torch.no_grad():
                    for name, m in models.items():
                        o = m(held, aux=False)
                        rec[name] = dict(r2=round(1 - float(o["loss"]) / var, 4), dead=int((fired[name] == 0).sum()))
                        fired[name].zero_()
                history.append(rec); log(json.dumps(rec))
    for name, m in models.items():
        torch.save(dict(state=m.state_dict(), config=configs[name], mu=mu.cpu(), scale=scale, layer=layer), f"{out_dir}/{name}.pt")
    json.dump(history, open(f"{out_dir}/train_log.json", "w"))
    cap.remove()
    return models, (mu, scale), it


@torch.no_grad()
def encode_corpus(tok, lm, layer, it, model, norm, n_tokens, out_dir, name, device, log=print):
    """Codes of `model` on fresh text: token ids, top block ids, block coordinates (for finding what blocks do)."""
    from core import Capture, SKIP_FIRST
    cap = Capture(lm, [layer]); eos = tok.convert_tokens_to_ids("<|endoftext|>")
    toks, tops, coords, n = [], [], [], 0
    while n < n_tokens:
        ids = next(it).to(device)
        lm(input_ids=ids, use_cache=False)
        v = torch.ones_like(ids, dtype=torch.bool); v[:, :SKIP_FIRST] = False; v &= ids != eos
        top, c = model.encode_top(norm(cap.out[layer][v].float()))
        toks.append(ids[v].int().cpu()); tops.append(top.short().cpu()); coords.append(c.half().cpu()); n += int(v.sum())
    cap.remove()
    np.save(f"{out_dir}/{name}_tokens.npy", torch.cat(toks).numpy())
    np.save(f"{out_dir}/{name}_top.npy", torch.cat(tops).numpy())
    np.save(f"{out_dir}/{name}_coords.npy", torch.cat(coords).numpy())
    log(f"encoded {n:,} tokens with {name}")
