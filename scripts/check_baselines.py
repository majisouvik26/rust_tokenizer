#!/usr/bin/env python3
"""Verify frozen full IDs against current installed pinned production engines."""
import json
from baselines import engines
from import_gpt2 import ROOT
from fixture_io import read_jsonl

tk, hf, hf_allowed = engines()
rows = read_jsonl(ROOT / "fixtures/gpt2.jsonl")
for row in rows:
    ids = tk.encode(row["text"], disallowed_special=())
    assert ids == row["ids"] == hf.encode(row["text"], add_special_tokens=False).ids, row["case_id"]
    assert tk.decode_bytes(ids) == row["text"].encode()
    assert hf.decode(ids, skip_special_tokens=False) == row["text"]
for row in json.loads((ROOT / "fixtures/gpt2-special.json").read_text()):
    assert tk.encode(row["text"], allowed_special={"<|endoftext|>"}) == row["ids"] == hf_allowed.encode(row["text"], add_special_tokens=False).ids
print(json.dumps({"cases": len(rows), "production_mismatches": 0}))
