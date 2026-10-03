#!/usr/bin/env python3

"""Validate correctness, reproducibility, and build compatibility."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from import_gpt2 import ROOT

def main():
    output = ROOT / "results/day1-validation"
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        ["cargo", "fmt", "--all", "--", "--check"],
        ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings"],
        ["cargo", "test", "--workspace", "--locked"],
        ["cargo", "build", "--release", "--locked", "-p", "bpe-wasm", "--target", "wasm32-unknown-unknown"],
        [sys.executable, "-m", "maturin", "build", "--release", "--locked", "--manifest-path", "bindings/python/Cargo.toml", "--out", "dist"],
        [sys.executable, "scripts/check_baselines.py"],
        [sys.executable, "scripts/check_determinism.py"],
    ]
    checks = []
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1")
    def check(argv):
        print("Running " + " ".join(argv), flush=True)
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        log = output / f"{len(checks):02d}.log"
        log.write_text(result.stdout + result.stderr)
        checks.append({"argv": argv, "exit_code": result.returncode, "log": str(log.relative_to(ROOT)), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        if result.returncode:
            (output / "checks.json").write_text(json.dumps({"passed": False, "checks": checks}, indent=2) + "\n")
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
    for argv in commands:
        check(argv)
    wheels = list((ROOT / "dist").glob("rust_tokenizer_baseline-0.2.0-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected one Day 1 release wheel; clear stale dist/ builds")
    check([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])])
    check([sys.executable, "scripts/check_python.py", "--out", "results/day1-validation/python-parity.json"])
    result = {"schema_version": 1, "passed": True, "checks": checks,
              "wheel_sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
              "wasm_sha256": hashlib.sha256((ROOT / "target/wasm32-unknown-unknown/release/bpe_wasm.wasm").read_bytes()).hexdigest()}
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks)}))

if __name__ == "__main__":
    main()
