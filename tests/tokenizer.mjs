// node tests/tokenizer.mjs <tokenizer.json> <expected.json>: compares engine/tokenizer.js with HF tokenizers' output.
import { readFileSync } from "node:fs";
import { Tokenizer } from "../engine/tokenizer.js";
const tok = new Tokenizer(JSON.parse(readFileSync(process.argv[2], "utf8")));
const cases = JSON.parse(readFileSync(process.argv[3], "utf8"));
let bad = 0;
for (const c of cases) {
  const ids = tok.encode(c.text);
  const ok = ids.length === c.ids.length && ids.every((x, i) => x === c.ids[i]);
  const round = tok.decode(ids, false) === c.text.normalize("NFC");
  if (!ok || !round) { bad++; console.log("MISMATCH", JSON.stringify(c.text).slice(0, 60), "\n got", ids.slice(0, 30).join(","), "\n exp", c.ids.slice(0, 30).join(","), "round", round); }
}
console.log(`${cases.length - bad}/${cases.length} match`);
process.exit(bad ? 1 : 0);
