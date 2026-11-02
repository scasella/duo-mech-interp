# Writes expected token ids for a set of strings using Hugging Face `tokenizers` (the reference for engine/tokenizer.js).
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
cases = [
    "Hello world, café 123 naïve 日本語 🙂",
    "Who are you?",
    "I'm here. You're there; they've gone, we'll see. It's 2026!",
    "  leading spaces\n\n\ttabs and\r\nCRLF   trailing   ",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))\n    return out\n",
    "<|im_start|>user\nWho made you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "Ünïcödé façade — “quotes” and ‘apostrophes’… 👩‍👩‍👧‍👦 🇺🇸 ½ ⅔ 𝔘𝔫𝔦",
    "Здравствуйте, мир! مرحبا بالعالم שלום עולם नमस्ते दुनिया 안녕하세요",
    "URL: https://example.com/path?q=1&r=2#frag; email: a.b@c.de; $100.00 (approx.) 3.14159e-10",
    "AAAAAAAAAAAAAAAA aaaaaaaaaaaaa ............ !!!??? ```js\nconst x = 1;\n```",
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived.",
    "她问我对这个计划有什么看法，我诚实地告诉她，这似乎太仓促了。",
    "'S 'T 'RE 'VE 'M 'LL 'D 's 't",
]
out = [{"text": c, "ids": tok.encode(c).ids} for c in cases]
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(out), "cases")
