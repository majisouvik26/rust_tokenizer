#!/usr/bin/env python3
"""Bounded smoke benchmark with frozen IDs and five fresh-process runs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fixture_io import read_jsonl
from protocol import source_snapshot
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "bench/configs/smoke.json")
    parser.add_argument("--cpu-ready", action="store_true", help="confirm a quiet CPU for timing")
    args = parser.parse_args()
    if not args.cpu_ready:
        parser.error("timing requires --cpu-ready after choosing a quiet CPU")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["schema_version"] != 1 or min(config["process_runs"], config["iterations"], config["max_documents"]) < 1:
        raise ValueError("invalid benchmark configuration")
    native = None
    if "rust_reference_native" in config["engines"]:
        build_command = ["cargo", "bench", "--locked", "-p", "bpe", "--bench", "reference", "--no-run", "--message-format=json"]
        for line in command(build_command).splitlines():
            event = json.loads(line)
            if event.get("reason") == "compiler-artifact" and event.get("target", {}).get("name") == "reference":
                native = event["executable"]
        if not native:
            raise RuntimeError("native benchmark executable missing")
    output = ROOT / config["output"]
    if not output.resolve().is_relative_to((ROOT / "docs").resolve()):
        raise ValueError("new benchmark outputs must be inside docs/")
    output.mkdir(parents=True, exist_ok=False)
    rows, seen = [], set()
    fixture = ROOT / config["fixtures"]
    for row in read_jsonl(fixture):
        if row["text"] and row["text"] not in seen:
            seen.add(row["text"])
            rows.append(row)
        if len(rows) == config["max_documents"]:
            break
    workload = output / "workload.jsonl"
    workload.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    model = ROOT / config["model"]
    env = dict(os.environ, RAYON_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    manifest = {"schema_version": 1, "source": source_snapshot(), "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": command(["git", "rev-parse", "HEAD"]), "source_dirty": bool(command(["git", "status", "--porcelain", "--untracked-files=no"])),
                "model_file_sha256": sha(model), "fixture_sha256": sha(fixture), "workload_sha256": sha(workload),
                "cargo_lock_sha256": sha(ROOT / "Cargo.lock"), "python_lock_sha256": sha(ROOT / "requirements.lock"),
                "config": config, "config_sha256": sha(args.config),
                "machine": {"os": platform.platform(), "architecture": platform.machine(), "cpu_count_logical": os.cpu_count(),
                            "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").exists() else platform.processor(),
                            "rustc": command(["rustc", "--version", "--verbose"]), "python": sys.version},
                "versions": {name: importlib.metadata.version(name) for name in ["tiktoken", "tokenizers", "regex", "maturin"]},
                "build": {"profile": "release", "opt_level": 3, "lto": False, "codegen_units": 1, "rustflags": os.environ.get("RUSTFLAGS", "")},
                "protocol": {"threads": 1, "cache_capacity": 0, "warmup": "full ID preflight", "mode": "warm model; distinct documents within each sweep; workload reused between repetitions", "unit": "MiB of UTF-8 input/s", "limitations": "synthetic smoke data, shared virtual machine, uncontrolled power/frequency; no final speed claims", "native_boundary": "Rust in-process ID-only encode; no Python or IO inside timer", "python_boundary": "per-document calls returning IDs; includes native/Python conversion; HF also builds Encoding metadata", "decode": "strict byte output for Rust/tiktoken; HF decode timing omitted"}}
    manifest["commands"] = []
    raw = []
    for engine in config["engines"]:
        for process_run in range(config["process_runs"]):
            if engine == "rust_reference_native":
                worker_command = [native]
                worker_env = dict(env, BPE_BENCH_MODEL=str(model), BPE_BENCH_INPUT=str(workload), BPE_BENCH_ITERATIONS=str(config["iterations"]))
            else:
                worker_command = [sys.executable, str(ROOT / "bench/worker.py"), "--engine", engine, "--model", str(model), "--input", str(workload), "--iterations", str(config["iterations"])]
                worker_env = env
            result = json.loads(command(worker_command, env=worker_env, timeout=300))
            result.update({"process_run": process_run, "model_file_sha256": sha(model), "workload_sha256": sha(workload), "source_commit": manifest["source_commit"]})
            raw.append(result)
            manifest["commands"].append({"engine": engine, "process_run": process_run, "argv": worker_command,
                                         "environment": {k: v for k, v in worker_env.items() if k.startswith("BPE_BENCH_") or k in ["RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS"]}})
    summary = []
    for engine in config["engines"]:
        for operation in ["encode", "decode"]:
            runs = [r for r in raw if r["engine"] == engine and r[operation + "_seconds"]]
            if not runs:
                continue
            rates = [r["input_bytes"] / statistics.median(r[operation + "_seconds"]) / 2**20 for r in runs]
            summary.append({"engine": engine, "operation": operation, "process_runs": len(runs), "median_mib_s": statistics.median(rates), "min_mib_s": min(rates), "max_mib_s": max(rates)})
    (output / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))

if __name__ == "__main__":
    main()
