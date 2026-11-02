# Reference outputs from Hugging Face transformers (float32, CPU) for tests/engine.mjs: next-token logits, MLP
# activations (input of down_proj) at every layer, greedy continuations with and without a neuron push.
import json, sys, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

path, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)
m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
lm = m.model.language_model
acts, push = {}, {}
def hook(i):
    def f(mod, inp):
        h = inp[0]
        if i in push:
            h = h.clone(); n, v = push[i]; h[..., n] += v
        acts[i] = h[0].detach().clone()
        return (h,)
    return f
for i, layer in enumerate(lm.layers):
    layer.mlp.down_proj.register_forward_pre_hook(hook(i))

prompt = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True,
                                 enable_thinking=False, tokenize=False)
ids = tok(prompt).input_ids
res = {"ids": ids}
with torch.inference_mode():
    out = m(input_ids=torch.tensor([ids]))
    lg = out.logits[0, -1]
    res["logits_top"] = lg.topk(20).indices.tolist()
    res["logits_top_vals"] = lg.topk(20).values.tolist()
    res["logits_sample"] = lg[:4096:7].tolist()
    res["acts"] = {str(l): {"last": acts[l][-1, :64].tolist(), "absmax": float(acts[l].abs().max()),
                            "n100": acts[l][:, 100].tolist()} for l in (0, 3, 11, 23)}

    def greedy(n_new=24):
        cur = list(ids)
        for _ in range(n_new):
            o = m(input_ids=torch.tensor([cur]))
            t = int(o.logits[0, -1].argmax())
            if t == tok.convert_tokens_to_ids("<|im_end|>"): break
            cur.append(t)
        return cur[len(ids):]
    res["greedy"] = greedy()
    push.update({12: (1234, 8.0)})
    res["greedy_push_up"] = greedy()
    push.update({12: (1234, -8.0)})
    res["greedy_push_down"] = greedy()
    push.clear()
# longer prompts: 17-32 tokens (two passes of the matrix-vector kernel) and over 32 (tiled matmul)
res["extra"] = []
for text in ["Tell me about your day, and please be very specific about the weather outside.",
             "Write a short paragraph about a lighthouse keeper who finds a message in a bottle on a stormy night, "
             "and describe what the message says and how it changes the rest of their life."]:
    p2 = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids2 = tok(p2).input_ids
    with torch.inference_mode():
        lg2 = m(input_ids=torch.tensor([ids2])).logits[0, -1]
    res["extra"].append({"text": text, "ids": ids2, "logits_top": lg2.topk(20).indices.tolist(), "logits_sample": lg2[:4096:7].tolist()})
    print("extra prompt", len(ids2), "tokens")
for k in ("greedy", "greedy_push_up", "greedy_push_down"):
    print(k, repr(tok.decode(res[k])))
json.dump(res, open(out_path, "w"))
