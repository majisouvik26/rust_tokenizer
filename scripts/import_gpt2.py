#!/usr/bin/env python3
"""Convert original GPT-2 artifacts; preserve IDs and recover exact token bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
SOURCES = {
    "encoder.json": ("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json", "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
    "vocab.bpe": ("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe", "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
}

def byte_to_unicode():
    # Mapping defined by OpenAI GPT-2 encoder.py (MIT); implemented afresh.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in visible}
    extra = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(extra)
            extra += 1
    return mapping

def get_source(name, supplied=None):
    path = Path(supplied) if supplied else ROOT / "models" / "sources" / name
    url, expected = SOURCES[name]
    if not path.exists():
        if supplied:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{name}: unexpected SHA-256 {digest}; expected pinned artifact {expected}")
    return path

def source_data(encoder=None, merges=None):
    encoder_path = get_source("encoder.json", encoder)
    merges_path = get_source("vocab.bpe", merges)
    vocab = json.loads(encoder_path.read_text())
    lines = merges_path.read_text().splitlines()
    if lines[0] != "#version: 0.2":
        raise ValueError("unexpected GPT-2 merge header")
    pairs = [tuple(line.split()) for line in lines[1:] if line]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("malformed source merge")
    return vocab, pairs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "models/gpt2.json")
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--merges", type=Path)
    args = parser.parse_args()
    vocab, pairs = source_data(args.encoder, args.merges)
    inverse = {character: byte for byte, character in byte_to_unicode().items()}
    ordinary = sorted(((symbol, id_) for symbol, id_ in vocab.items() if symbol != "<|endoftext|>"), key=lambda item: item[1])
    assert [id_ for _, id_ in ordinary] == list(range(50256))
    model = {"format_version": 1, "profile": "gpt2", "pretokenizer": {"kind": "gpt2", "pattern": PATTERN},
             "tokens": [{"id": id_, "bytes_b64": base64.b64encode(bytes(inverse[c] for c in symbol)).decode()} for symbol, id_ in ordinary],
             "merges": [{"left": vocab[left], "right": vocab[right], "out": vocab[left + right], "rank": rank} for rank, (left, right) in enumerate(pairs)],
             "special_tokens": {"<|endoftext|>": vocab["<|endoftext|>"]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode()
    args.out.write_bytes(payload)
    provenance = {"model_sha256": hashlib.sha256(payload).hexdigest(), "sources": {name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCES.items()},
                  "ordinary_tokens": len(ordinary), "merges": len(pairs), "special_tokens": model["special_tokens"], "license": "OpenAI GPT-2 MIT (see licenses/GPT2-MIT.txt)"}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))

if __name__ == "__main__":
    main()
