# Rust BPE Tokenizer

A lossless byte-level BPE tokenizer implemented in Rust, with GPT-2-compatible
token IDs, deterministic training, and Python and WebAssembly bindings.

The core provides reference and heap encoders, reference and incremental
trainers, reusable buffers, optional bounded caching, and ordered native batch
parallelism. Merge traces expose the actual ranked operations. Automatic backend
selection uses the reference scan unless an explicit byte threshold is supplied.

```bash
cargo fmt --all
python scripts/validate_project.py --rust-only --out docs/runs/native-validation
python scripts/validate_project.py --out docs/runs/validation
```

The full validator builds and installs a release Python wheel, checks frozen
GPT-2 IDs across backends, checks deterministic training and batch behavior, and
compiles WASM. Compilation alone does not establish browser runtime parity.

## Tokenize and train

```bash
cargo run --release --locked -p bpe-cli -- encode \
  --model models/gpt2.json --backend heap --text "This is some text"
cargo run --release --locked -p bpe-cli -- trace \
  --model models/gpt2.json --backend heap --text "abcabc"
```

Training consumes independent JSONL records of the form `{"text":"..."}` and
supports `--trainer reference` or `--trainer incremental`. Encoding accepts text,
a file, stdin, or ordered JSONL batches. Decoding concatenates token bytes before
validating UTF-8. Whitespace, Unicode, embedded NUL, and literal `</w>` are
preserved. Special tokens are ordinary text by default, with explicit allow and
reject policies; no BOS/EOS tokens are inserted automatically.

```python
from rust_tokenizer import Tokenizer

tokenizer = Tokenizer("models/gpt2.json")
ids = tokenizer.encode("বাংলা ও हिन्दी", backend="heap")
assert tokenizer.decode_bytes(ids) == "বাংলা ও हिन्दी".encode("utf-8")
outputs = tokenizer.encode_batch(["hello", "", "world"], backend="heap", threads=2)
```

## Project map

| Path | Purpose |
| --- | --- |
| `bpe/` | Validated model, merge engines, trainers, batch API and correctness tests |
| `bpe-cli/` | Train, encode, decode, inspect, trace and diagnostic profiling |
| `bindings/` | Python and serial WebAssembly adapters over the same Rust core |
| `scripts/` | Import, fixtures, compatibility checks and evidence collection |
| `bench/` | Development workloads, ablations, profiling and training comparisons |
| `fixtures/`, `models/` | Frozen IDs, preprocessing spans, hashes and GPT-2 sources |
