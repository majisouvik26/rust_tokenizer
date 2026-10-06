#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from import_gpt2 import ROOT
from rust_tokenizer import Tokenizer
from fixture_io import read_jsonl


def equal(actual, expected, context):
    if actual != expected: raise AssertionError(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    args = parser.parse_args()
    tokenizer = Tokenizer(str(ROOT / "models/gpt2.json"))
    fixture = ROOT / "fixtures/gpt2.jsonl"
    rows = read_jsonl(fixture)
    variants = [{"backend": "reference"}, {"backend": "heap"}, {"backend": "auto", "heap_threshold": 32}]
    for variant in variants:
        for row in rows:
            ids = tokenizer.encode(row["text"], **variant)
            equal(ids, row["ids"], f"{variant}: case {row['case_id']}")
            equal(tokenizer.decode_bytes(ids), row["text"].encode(), "byte decode")
            equal(tokenizer.decode(ids), row["text"], "strict decode")
        print(f"Checked {len(rows)} frozen cases: {variant}", flush=True)
    batch = [row["text"] for row in rows[:256]]
    expected = [row["ids"] for row in rows[:256]]
    for variant in variants:
        for threads in [1, 2, 4]:
            for capacity in [0, 32]:
                equal(tokenizer.encode_batch(batch, threads=threads, cache_capacity=capacity, cache_bytes=4096, **variant), expected,
                      f"batch: {variant}, threads={threads}, cache={capacity}")
        for row in json.loads((ROOT / "fixtures/gpt2-special.json").read_text(encoding="utf-8")):
            equal(tokenizer.encode(row["text"], allowed_special=["<|endoftext|>"], **variant), row["ids"], "allowed special")
            try: tokenizer.encode(row["text"], reject_special=True, **variant)
            except ValueError: pass
            else: raise AssertionError("special rejection did not fail")
    for row in rows[:128]:
        equal(json.loads(tokenizer.trace(row["text"], backend="reference")), json.loads(tokenizer.trace(row["text"], backend="heap")), "merge trace")
    result = {"cases_per_backend": len(rows), "variants": variants, "batch_threads": [1, 2, 4], "batch_cache_capacities": [0, 32],
              "mismatches": 0, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "model_sha256": tokenizer.model_sha256(), "python_interface": "installed release wheel", "trace_cases": 128}
    if args.out:
        output = (ROOT / args.out).resolve()
        if not output.is_relative_to((ROOT / "docs").resolve()): parser.error("--out must be inside docs/")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
