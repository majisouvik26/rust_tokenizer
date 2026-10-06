#!/usr/bin/env python3
"""Native optimization measurements on frozen development inputs, one process at a time."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
from protocol import ROOT, benchmark_environment, build_worker, new_output, provenance, run_logged, sha, source_snapshot, write_json


def variants(group, config, max_threads):
    def variant(name, backend="heap", *, reuse=True, cache=0, state="empty_each_sweep", threads=1, batch=1, threshold=None):
        return {"name": name, "backend": backend, "runtime": {"reuse_buffers": reuse, "heap_threshold": threshold,
                "cache_capacity": cache, "cache_bytes": config["cache_bytes_per_worker"]},
                "cache_state": state, "threads": threads, "batch_size": batch}
    if group == "profile":
        return [variant("scan_reused", "reference"), variant("heap_reused")]
    if group == "ablation":
        return [variant("scan_fresh", "reference", reuse=False), variant("scan_reused", "reference"),
                variant("heap_fresh", reuse=False), variant("heap_reused"),
                variant("heap_cache_empty", cache=config["cache_capacity"]),
                variant("heap_cache_warm", cache=config["cache_capacity"], state="warm")]
    if group == "batching":
        choices = [variant("heap_request", batch=1)]
        choices += [variant(f"heap_batch_t{threads}", threads=threads, batch=128)
                    for threads in config["thread_candidates"] if threads <= max_threads]
        return choices
    return [variant("scan_reused", "reference"), variant("heap_reused")] + [
        variant(f"auto_{value}", "auto", threshold=value) for value in config["threshold_candidates"]]


def summarize(raw):
    summary = []
    keys = sorted({(row["workload"], row["variant"]) for row in raw})
    for workload, variant in keys:
        runs = [row for row in raw if (row["workload"], row["variant"]) == (workload, variant)]
        rates = [row["input_bytes"] / statistics.median(row["encode_seconds"]) / 2**20 for row in runs]
        latencies = sorted(value for row in runs for sweep in row["batch_seconds"] for value in sweep)
        reference = [row for row in raw if row["workload"] == workload and row["variant"] == "scan_reused"]
        reference_rate = statistics.median(row["input_bytes"] / statistics.median(row["encode_seconds"]) / 2**20 for row in reference) if reference else None
        summary.append({"workload": workload, "variant": variant, "process_runs": len(runs),
                        "median_mib_s": statistics.median(rates), "min_mib_s": min(rates), "max_mib_s": max(rates),
                        "throughput_ratio_to_scan_reused": statistics.median(rates) / reference_rate if reference_rate else None,
                        "batch_p50_seconds": statistics.median(latencies),
                        "batch_p95_seconds": latencies[math.ceil(len(latencies) * .95) - 1],
                        "batch_size": runs[0]["batch_size"], "threads": runs[0]["threads"],
                        "median_allocation_calls": statistics.median(row["allocation_calls"] for row in runs),
                        "median_allocated_bytes_requested": statistics.median(row["allocated_bytes_requested"] for row in runs),
                        "max_peak_rss_bytes": max((row["peak_rss_bytes"] for row in runs if row["peak_rss_bytes"] is not None), default=None)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="bench/configs/optimization.json")
    parser.add_argument("--group", choices=["profile", "ablation", "batching", "thresholds"], default="ablation")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--executable-manifest", default="docs/runs/build/executable.json")
    parser.add_argument("--cpu-ready", action="store_true", help="confirm a quiet CPU and accept the timing workload")
    parser.add_argument("--max-threads", type=int, default=1, help="explicit CPU allocation; do not substitute logical CPU count for physical cores")
    parser.add_argument("--machine-note", default="not supplied")
    parser.add_argument("--out")
    args = parser.parse_args()
    if not args.build_only and not args.cpu_ready:
        parser.error("timing requires --cpu-ready after you have chosen an idle laptop or a dedicated remote CPU")
    if args.max_threads < 1: parser.error("--max-threads must be positive")
    output = new_output("build" if args.build_only else args.group, ROOT / args.out if args.out else None)
    if args.build_only:
        worker = build_worker(output)
        write_json(output / "executable.json", {"path": worker, "sha256": sha(worker), "provenance": provenance()})
        print(f"Built only. Executable manifest: {output / 'executable.json'}")
        return
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != 1 or min(config["process_runs"], config["iterations"], config["worker_timeout_seconds"]) < 1:
        raise ValueError("invalid optimization configuration")
    executable_manifest = json.loads((ROOT / args.executable_manifest).read_text(encoding="utf-8"))
    executable = executable_manifest["path"]
    current = provenance()
    if sha(executable) != executable_manifest["sha256"] or current["source"]["tree_sha256"] != executable_manifest["provenance"]["source"]["tree_sha256"]:
        raise ValueError("source or binary changed; rebuild to a new output directory")
    if current["build"] != executable_manifest["provenance"]["build"]:
        raise ValueError("build flags changed; rebuild before measuring")
    dev_path = ROOT / config["development_manifest"]
    development = json.loads(dev_path.read_text(encoding="utf-8"))
    if development["split"] != "development" or development["production_id_mismatches"] != 0:
        raise ValueError("verified development workloads required")
    model = ROOT / config["model"]
    if sha(model) != development["model_sha256"]: raise ValueError("model does not match frozen development IDs")
    workloads = development["workloads"]
    for row in workloads:
        if sha(dev_path.parent / row["path"]) != row["sha256"]: raise ValueError("workload changed: " + row["name"])
    choices = variants(args.group, config, args.max_threads)
    manifest = {"schema_version": 2, "status": "running", "group": args.group, "config": config, "config_sha256": sha(config_path),
                "provenance": current, "machine_note": args.machine_note, "max_threads_authorized": args.max_threads,
                "executable_sha256": sha(executable), "model_sha256": sha(model), "development_manifest": development,
                "development_manifest_sha256": sha(dev_path), "variants": choices,
                "timer": "Rust batch calls plus ID allocations; per-batch timers included; model/pool/file IO excluded",
                "allocation_pass": "separate untimed sweep; allocation/reallocation request counts, not live or peak bytes",
                "cache": "empty_each_sweep resets outside timer; warm retains preflight and earlier sweeps; per-lane FIFO",
                "memory": "Linux process VmHWM includes model, preflight, timings, allocation pass and profiling; null if unsupported",
                "limits": "synthetic development measurements; no production claims or automatic threshold selection",
                "commands": []}
    write_json(output / "manifest.json", manifest)
    raw = []
    rng = random.Random(config["order_seed"])
    runs = 1 if args.group == "profile" else config["process_runs"]
    try:
        for process_run in range(runs):
            jobs = [(workload, variant) for workload in workloads for variant in choices]
            rng.shuffle(jobs)
            for workload, variant in jobs:
                label = f"{len(raw):04d}-{workload['name']}-{variant['name']}-r{process_run}"
                request = {"model": str(model), "input": str(dev_path.parent / workload["path"]),
                           **{key: value for key, value in variant.items() if key != "name"},
                           "iterations": 1 if args.group == "profile" else config["iterations"], "profile": args.group == "profile"}
                request_path = output / "requests" / (label + ".json")
                write_json(request_path, request)
                env = dict(benchmark_environment(), BPE_BENCH_REQUEST=str(request_path))
                event, stdout = run_logged([executable], output, label, config["worker_timeout_seconds"], env)
                event["environment"] = {key: env[key] for key in ["BPE_BENCH_REQUEST", "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}
                manifest["commands"].append(event)
                write_json(output / "manifest.json", manifest)
                if event["exit_code"] != 0: raise RuntimeError(f"worker {event['status']} / exit {event['exit_code']}: {label}; inspect logs")
                row = json.loads(stdout)
                if row["mismatches"] != 0 or not all(value > 0 for value in row["encode_seconds"]): raise ValueError("invalid worker measurement")
                row.update({"variant": variant["name"], "workload": workload["name"], "process_run": process_run,
                            "workload_sha256": workload["sha256"], "source_sha256": current["source"]["tree_sha256"]})
                raw.append(row)
                with (output / "raw.jsonl").open("a", encoding="utf-8", newline="\n") as file:
                    file.write(json.dumps(row) + "\n")
                write_json(output / "summary.json", summarize(raw))
                print(f"Completed {label}", flush=True)
        if source_snapshot()["tree_sha256"] != current["source"]["tree_sha256"]:
            raise RuntimeError("source changed during measurements; discard this run")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    else:
        manifest["status"] = "completed"
    finally:
        write_json(output / "manifest.json", manifest)
        print(f"Outputs: {output}", flush=True)


if __name__ == "__main__": main()
