#!/usr/bin/env python3
"""Save independent production IDs; refuse to generate on a disagreement."""
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
import regex
from baselines import engines
from import_gpt2 import ROOT, PATTERN

FIXED = ["", " ", "  ", "\t\r\n", "  hello world  ", "a\x00b", "</w>", "literal </w> stays",
         "aaaa", "aaaaaaaaa", "abcabc", "I'm we're they've he'll I'd DON'T", "  x\t y\n\n", "你好，世界！", "हिन्दी में एक प्रश्न।", "বাংলা ভাষায় প্রশ্ন।",
         "e\u0301 vs é", "👩🏽‍💻 👨‍👩‍👧‍👦 🏏", "\u00a0\u2003\u2028\u2029\u0085", "\ufeffhello\u200b", "<|endoftext|>",
         "x<|endoftext|>y", "1234567890 ١٢٣ १२३", "fn main() {\n    println!(\"hello\");\n}\n", "\x1c\x1d\x1e\x1f", "\r\n" * 20]
DOMAINS = {
    "english": "The quick brown fox can't jump over every quiet river. We explain a byte tokenizer: punctuation, numbers 42, and spaces. ",
    "code": "fn main() { let data = vec![1, 2, 3]; println!(\"{:?}\", data); }\n# Python\ndef f(x):\n\treturn x ** 2\n",
    "hindi": "यह एक परीक्षण है। हिन्दी में शब्द, संख्याएँ १२३ और विराम चिह्न सुरक्षित रहने चाहिए। ",
    "bengali": "এটি একটি পরীক্ষা। বাংলা ভাষায় শব্দ, সংখ্যা ১২৩ এবং যতিচিহ্ন অক্ষত রাখতে হবে। ",
    "unicode": "你好 αβγ русский العربية é e\u0301 👩🏽‍💻 🏏 \x00\t\r\n\u00a0\u2003\u2028 ",
    "whitespace": " \t\n\r\v\f\u0085\u00a0\u2002\u2003\u2028\u2029",
}

def cases(count, seed):
    if count < len(FIXED):
        raise ValueError(f"count must be >= {len(FIXED)}")
    yield from (("fixed", text) for text in FIXED)
    rng = random.Random(seed)
    domains = list(DOMAINS)
    for index in range(count - len(FIXED)):
        domain = domains[index % len(domains)]
        alphabet = DOMAINS[domain]
        length = rng.randrange(1, 180)
        if index % 4 == 0:
            start = rng.randrange(len(alphabet))
            text = alphabet[start:] + f" [{index}] " + alphabet[:start]
        else:
            text = "".join(rng.choice(alphabet) for _ in range(length))
        yield domain, text

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures/gpt2.jsonl")
    args = parser.parse_args()
    tk, hf, hf_allowed = engines()
    pattern = regex.compile(PATTERN)
    rows = []
    for index, (domain, text) in enumerate(cases(args.count, args.seed)):
        ids = tk.encode(text, allowed_special=set(), disallowed_special=())
        hf_ids = hf.encode(text, add_special_tokens=False).ids
        if ids != hf_ids or tk.decode_bytes(ids) != text.encode() or hf.decode(hf_ids, skip_special_tokens=False) != text:
            raise AssertionError(f"production mismatch at case {index}: {text!r}; tiktoken={ids}, HF={hf_ids}")
        spans = [[len(text[:m.start()].encode()), len(text[:m.end()].encode())] for m in pattern.finditer(text)]
        rows.append({"case_id": index, "domain": domain, "text": text, "ids": ids, "spans": spans})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    special_rows = []
    for text in ["<|endoftext|>", "x<|endoftext|>y", "বাংলা<|endoftext|> हिन्दी", "<|endoftext|><|endoftext|>"]:
        ids = tk.encode(text, allowed_special={"<|endoftext|>"})
        assert ids == hf_allowed.encode(text, add_special_tokens=False).ids
        special_rows.append({"text": text, "ids": ids})
    (args.out.parent / "gpt2-special.json").write_text(json.dumps(special_rows, ensure_ascii=False, indent=2) + "\n")
    manifest = {"schema_version": 1, "cases": len(rows), "seed": args.seed,
                "fixture_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "model_file_sha256": hashlib.sha256((ROOT / "models/gpt2.json").read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {name: importlib.metadata.version(name) for name in ["tiktoken", "tokenizers", "regex"]},
                "settings": {"special": "ordinary", "prefix_space": False, "add_special_tokens": False, "hf_cache_capacity": 0},
                "data_kind": "fixed and seeded synthetic cases; not a representative performance corpus",
                "production_mismatches": 0}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
