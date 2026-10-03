#!/usr/bin/env python3
"""Check the installed release wheel on the same frozen Rust fixtures."""
import argparse
import hashlib
import json
from import_gpt2 import ROOT
from rust_tokenizer import Tokenizer
from fixture_io import read_jsonl

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    args = parser.parse_args()
    tokenizer = Tokenizer(str(ROOT / "models/gpt2.json"))
    fixture = ROOT / "fixtures/gpt2.jsonl"
    rows = read_jsonl(fixture)
    for row in rows:
        ids = tokenizer.encode(row["text"])
        assert ids == row["ids"], row["case_id"]
        assert tokenizer.decode_bytes(ids) == row["text"].encode()
        assert tokenizer.decode(ids) == row["text"]
    assert tokenizer.encode_batch([r["text"] for r in rows[:128]]) == [r["ids"] for r in rows[:128]]
    for row in json.loads((ROOT / "fixtures/gpt2-special.json").read_text()):
        assert tokenizer.encode(row["text"], allowed_special=["<|endoftext|>"]) == row["ids"]
        try:
            tokenizer.encode(row["text"], reject_special=True)
        except ValueError:
            pass
        else:
            raise AssertionError("special rejection did not fail")
    result = {"cases": len(rows), "mismatches": 0, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "model_sha256": tokenizer.model_sha256(), "python_interface": "installed release wheel"}
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
