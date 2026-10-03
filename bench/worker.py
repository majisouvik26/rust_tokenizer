#!/usr/bin/env python3
"""One fresh process, loaded model, ID-only encoding + strict byte decoding."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fixture_io import read_jsonl

def run(args):
    rows = read_jsonl(args.input)
    if args.engine == "rust_reference_python":
        from rust_tokenizer import Tokenizer
        tokenizer = Tokenizer(args.model)
        encode = tokenizer.encode
        decode_bytes = tokenizer.decode_bytes
    else:
        from baselines import engines
        tk, hf, _ = engines()
        if args.engine == "tiktoken":
            encode = lambda text: tk.encode(text, allowed_special=set(), disallowed_special=())
            decode_bytes = tk.decode_bytes
        else:
            encode = lambda text: hf.encode(text, add_special_tokens=False).ids
            # Decode bytes via preserved source IDs, avoiding UTF-8 cleanup.
            import base64
            model = json.loads(Path(args.model).read_text())
            vocabulary = {token["id"]: base64.b64decode(token["bytes_b64"]) for token in model["tokens"]}
            decode_bytes = lambda ids: b"".join(vocabulary[id_] for id_ in ids)
    for row in rows:  # preflight plus warmup, excluded from timing
        ids = encode(row["text"])
        assert ids == row["ids"], row["case_id"]
        assert decode_bytes(ids) == row["text"].encode()
    encode_seconds, decode_seconds = [], []
    for _ in range(args.iterations):
        start = time.perf_counter()
        outputs = [encode(row["text"]) for row in rows]
        encode_seconds.append(time.perf_counter() - start)
        assert outputs == [row["ids"] for row in rows]
        if args.engine != "huggingface":
            start = time.perf_counter()
            decoded = [decode_bytes(ids) for ids in outputs]
            decode_seconds.append(time.perf_counter() - start)
            assert decoded == [row["text"].encode() for row in rows]
    return {"schema_version": 1, "engine": args.engine, "threads": 1, "cache_capacity": 0,
            "documents": len(rows), "input_bytes": sum(len(row["text"].encode()) for row in rows),
            "tokens": sum(len(row["ids"]) for row in rows), "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
            "decode_note": "not timed: HF default string decode differs from the strict byte API" if args.engine == "huggingface" else "strict bytes, Python/native conversion included"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["rust_reference_python", "tiktoken", "huggingface"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    print(json.dumps(run(args)))
