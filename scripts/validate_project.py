#!/usr/bin/env python3
"""Build and verify correctness, reproducibility and native/Python/WASM compatibility."""

import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from protocol import benchmark_environment, new_output, provenance, run_logged, sha, source_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--rust-only", action="store_true", help="initial native checks; full interface validation remains pending")
    args = parser.parse_args()
    output = new_output("validation", ROOT / args.out if args.out else None)
    env = dict(benchmark_environment(), CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "1"))
    report = {"schema_version": 2, "passed": False, "scope": "native" if args.rust_only else "native_python_wasm_build",
              "provenance": provenance(), "checks": [], "wasm_runtime": "not exercised; compile check only"}
    write_json(output / "checks.json", report)

    def check(argv):
        print("Running " + " ".join(str(arg) for arg in argv), flush=True)
        event, _ = run_logged(argv, output, f"{len(report['checks']):02d}", timeout=3600, env=env)
        report["checks"].append(event)
        write_json(output / "checks.json", report)
        if event["exit_code"] != 0: raise RuntimeError(f"check failed ({event['status']}): {output / event['log']}")

    try:
        for argv in [
            ["cargo", "fmt", "--all", "--", "--check"],
            ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"],
            ["cargo", "test", "-p", "bpe", "--test", "optimization", "--no-default-features", "--release", "--locked"],
            ["cargo", "test", "--workspace", "--all-features", "--release", "--locked"],
            ["cargo", "build", "--release", "--locked", "-p", "bpe-cli"],
            [sys.executable, "scripts/check_determinism.py", "--out", str(output / "determinism.json")],
        ]: check(argv)
        if not args.rust_only:
            check(["cargo", "build", "--release", "--locked", "-p", "bpe-wasm", "--target", "wasm32-unknown-unknown"])
            check([sys.executable, "-m", "maturin", "build", "--release", "--locked", "--manifest-path", "bindings/python/Cargo.toml", "--out", str(output / "wheels")])
            wheels = list((output / "wheels").glob("*.whl"))
            if len(wheels) != 1: raise RuntimeError("expected exactly one freshly built wheel")
            check([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])])
            check([sys.executable, "scripts/check_baselines.py"])
            check([sys.executable, "scripts/check_python.py", "--out", str(output / "python-parity.json")])
            report["wheel_sha256"] = sha(wheels[0])
            report["wasm_sha256"] = sha(ROOT / "target/wasm32-unknown-unknown/release/bpe_wasm.wasm")
        snapshot = source_snapshot()
        if snapshot["tree_sha256"] != report["provenance"]["source"]["tree_sha256"]:
            raise RuntimeError("source changed during validation; rerun the checks")
        binary = ROOT / ("target/release/bpe-cli.exe" if os.name == "nt" else "target/release/bpe-cli")
        write_json(ROOT / "docs/runs/cli-build.json", {"binary_sha256": sha(binary), "source_sha256": snapshot["tree_sha256"],
                   "validation": str(output), "provenance": report["provenance"]})
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(output / "checks.json", report)
        print(f"Retained validation evidence: {output}", flush=True)
    print(json.dumps({"passed": True, "scope": report["scope"], "checks": len(report["checks"])}))


if __name__ == "__main__": main()
