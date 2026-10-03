# Rust BPE Tokenizer 

A lossless byte BPE tokenizer with deterministic training and a readable scan
encoder. GPT-2 import preserves the original token IDs. This is the first
reference milestone of the project; optimization and the hosted demo belong to later days.

## Quick start

```bash
cargo test --workspace --locked
cargo run --release --locked -p bpe-cli -- train \
  --input data/toy.jsonl --pretokenizer raw --vocab-size 258 --out model.json
cargo run --release --locked -p bpe-cli -- encode --model model.json --text "aaabbb"
cargo run --release --locked -p bpe-cli -- inspect --model model.json
printf '[256,97,257,98]' | cargo run --release --locked -p bpe-cli -- decode --model model.json
```

The toy encodes to `[256,97,257,98]` and decodes to `aaabbb`. Encoding accepts
`--input text.txt`, `--input records.jsonl --jsonl`, or stdin. Decoding accepts a
JSON ID array from `--ids-file ids.json` or stdin and writes the exact text
without adding a newline. Training records are JSONL objects `{"text":"..."}`;
each record is independent. Empty/multilingual text, whitespace, NUL, and literal
`</w>` are preserved.

## GPT-2 and the Day 1 gate

The archive includes the pinned GPT-2 model, source artifacts, and 10,000 frozen
fixtures. Regeneration requires Python baseline dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python scripts/import_gpt2.py --out models/gpt2.json
python scripts/generate_fixtures.py
cargo run --release --locked -p bpe-cli -- encode --model models/gpt2.json --text "This is some text"
python scripts/day1_gate.py
python bench/run.py --config bench/configs/smoke.json
```

The GPT-2 example returns `[1212,318,617,2420]`. Specials are ordinary text by
default; use `--special allow --allow-special '<|endoftext|>'` to recognize that
token, or `--special reject` to reject recognized special strings.

The gate builds and installs a release Python wheel, checks the same 10,000
cases from Python, builds WASM, runs formatting/clippy/tests, and trains a toy
model in 20 fresh processes. The benchmark records native Rust plus Rust Python,
tiktoken, and Hugging Face timings with frozen ID preflight. Results are bounded
synthetic smoke measurements, with no speed thresholds or production claims.

```python
from rust_tokenizer import Tokenizer
tokenizer = Tokenizer("models/gpt2.json")
ids = tokenizer.encode("বাংলা ও हिन्दी")
assert tokenizer.decode_bytes(ids) == "বাংলা ও हिन्दी".encode("utf-8")
```

## Project map

| Path | Purpose |
| --- | --- |
| `bpe/` | Validated model, preprocessing, special policy, scan encoder, trainer |
| `bpe-cli/` | train / encode / decode / inspect; discovered CLI tests |
| `bindings/` | Thin Python and single-threaded WASM adapters |
| `scripts/` | Import, fixture generation, parity and reproducibility gates |
| `fixtures/`, `models/` | Frozen IDs, preprocessing spans, hashes and GPT-2 sources |
