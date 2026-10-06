#!/usr/bin/env python3
"""Retain canonical models from both trainers across fresh CLI processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from import_gpt2 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/bpe-cli.exe" if os.name == "nt" else "target/release/bpe-cli")
    parser.add_argument("--runs", type=int, default=20, help="fresh processes per trainer")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.runs < 1: parser.error("--runs must be positive")
    report_path = (ROOT / args.out).resolve()
    if not report_path.is_relative_to((ROOT / "docs").resolve()): parser.error("--out must be inside docs/")
    output = report_path.parent / "determinism-models"
    output.mkdir(parents=True, exist_ok=False)
    processes, hashes = [], set()
    for index in range(args.runs):
        for backend in ["reference", "incremental"]:
            model = output / f"{backend}-{index:02d}.json"
            argv = [str((ROOT / args.binary).resolve()), "train", "--input", str(ROOT / "data/toy.jsonl"),
                    "--pretokenizer", "raw", "--vocab-size", "258", "--trainer", backend, "--out", str(model)]
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
            (output / f"{backend}-{index:02d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode: raise RuntimeError(f"trainer failed: {backend}: {result.stderr}")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            hashes.add(digest)
            processes.append({"backend": backend, "run": index, "argv": argv, "model_sha256": digest})
    expected = "3c8fc29120c29d6f50837deecc9476b39fce0f833552280a9a8cc82129670da2"
    passed = hashes == {expected}
    report_path.write_text(json.dumps({"passed": passed, "fresh_processes": len(processes),
                           "unique_models": len(hashes), "expected_model_sha256": expected, "processes": processes}, indent=2) + "\n", encoding="utf-8")
    if not passed: raise AssertionError("trainer models disagree with the preserved toy baseline")
    print(json.dumps({"passed": True, "fresh_processes": len(processes), "unique_models": 1}))


if __name__ == "__main__": main()
