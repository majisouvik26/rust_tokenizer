#!/usr/bin/env python3
"""20 fresh CLI processes must produce identical canonical model files."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from import_gpt2 import ROOT

def main():
    subprocess.run(["cargo", "build", "--locked", "-p", "bpe-cli"], cwd=ROOT, check=True)
    hashes = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(20):
            output = Path(directory) / f"toy-{index}.json"
            subprocess.run([str(ROOT / "target/debug/bpe-cli"), "train", "--input", str(ROOT / "data/toy.jsonl"), "--pretokenizer", "raw", "--vocab-size", "258", "--out", str(output)], check=True, capture_output=True)
            hashes.append(hashlib.sha256(output.read_bytes()).hexdigest())
    assert len(set(hashes)) == 1, hashes
    print(json.dumps({"fresh_processes": len(hashes), "unique_models": len(set(hashes)), "model_sha256": hashes[0]}))

if __name__ == "__main__":
    main()
