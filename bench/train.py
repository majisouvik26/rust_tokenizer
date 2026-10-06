#!/usr/bin/env python3
"""Compare serial reference and incremental training in bounded fresh processes."""
import argparse
import json
from pathlib import Path
import random
import statistics
from protocol import ROOT, benchmark_environment, new_output, provenance, run_logged, sha, source_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/runs/development/training.jsonl")
    parser.add_argument("--binary", default="target/release/bpe-cli.exe" if __import__("os").name == "nt" else "target/release/bpe-cli")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--pretokenizer", choices=["raw", "gpt2"], default="gpt2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cpu-ready", action="store_true")
    parser.add_argument("--machine-note", default="not supplied")
    parser.add_argument("--out")
    args = parser.parse_args()
    if not args.cpu_ready: parser.error("timing requires --cpu-ready after choosing a quiet CPU")
    if min(args.runs, args.timeout, args.min_frequency) < 1 or args.vocab_size < 256: parser.error("invalid training limits")
    source, binary = (ROOT / args.input).resolve(), (ROOT / args.binary).resolve()
    if not binary.is_file(): parser.error("release CLI is missing; run validation first")
    output = new_output("training", ROOT / args.out if args.out else None)
    metadata = provenance()
    cli_build = ROOT / "docs/runs/cli-build.json"
    if not cli_build.is_file(): parser.error("missing docs/runs/cli-build.json; run validate_project.py first")
    built = json.loads(cli_build.read_text(encoding="utf-8"))
    if built["binary_sha256"] != sha(binary) or built["source_sha256"] != metadata["source"]["tree_sha256"]:
        parser.error("release CLI/source mismatch; run validation after your last code change")
    if built["provenance"]["build"] != metadata["build"]:
        parser.error("build flags changed; validate the intended build before timing")
    with source.open(encoding="utf-8") as file: records = [json.loads(line) for line in file]
    if not records or any(set(row) != {"text"} or not isinstance(row["text"], str) for row in records): parser.error("input must contain text-only JSONL records")
    manifest = {"schema_version": 1, "status": "running", "provenance": metadata, "machine_note": args.machine_note,
                "input": str(source), "input_sha256": sha(source), "binary_sha256": sha(binary),
                "documents": len(records), "utf8_bytes": sum(len(row["text"].encode()) for row in records),
                "vocab_size": args.vocab_size, "min_frequency": args.min_frequency, "pretokenizer": args.pretokenizer,
                "timeout_per_process": args.timeout, "threads": 1,
                "timer": "CLI training_seconds excludes file parsing and model save; wall_seconds includes entire fresh process",
                "memory": "CLI process VmHWM on Linux, including corpus parsing/model save; null on unsupported platforms", "commands": []}
    write_json(output / "manifest.json", manifest)
    raw, hashes = [], set()
    try:
        for index in range(args.runs):
            engines = ["reference", "incremental"]
            random.Random(1618 + index).shuffle(engines)
            for backend in engines:
                model = output / "models" / f"{backend}-{index}.json"
                model.parent.mkdir(parents=True, exist_ok=True)
                argv = [str(binary), "train", "--input", str(source), "--trainer", backend,
                        "--vocab-size", str(args.vocab_size), "--min-frequency", str(args.min_frequency),
                        "--pretokenizer", args.pretokenizer, "--out", str(model)]
                event, stdout = run_logged(argv, output, f"{backend}-{index}", args.timeout, benchmark_environment())
                manifest["commands"].append(event)
                write_json(output / "manifest.json", manifest)
                if event["status"] == "timeout":
                    row = {"backend": backend, "process_run": index, "status": "timeout", "timeout_seconds": args.timeout}
                elif event["exit_code"] != 0:
                    raise RuntimeError(f"training failed: {backend}; see {event['log']}")
                else:
                    result = json.loads(stdout)
                    digest = sha(model)
                    if digest != result["model_sha256"]: raise ValueError("saved model digest mismatch")
                    hashes.add(digest)
                    if len(hashes) != 1: raise ValueError("trainers or processes learned different canonical models")
                    row = {**result, "backend": backend, "process_run": index, "status": "completed",
                           "wall_seconds": event["wall_seconds"], "reached_vocab_cap": result["vocab_size"] == args.vocab_size}
                raw.append(row)
                with (output / "raw.jsonl").open("a", encoding="utf-8", newline="\n") as file: file.write(json.dumps(row) + "\n")
                print(f"{backend} run {index}: {row['status']}", flush=True)
        if source_snapshot()["tree_sha256"] != metadata["source"]["tree_sha256"]:
            raise RuntimeError("source changed during training measurements; discard this run")
        summary = []
        for backend in ["reference", "incremental"]:
            times = [row["training_seconds"] for row in raw if row["backend"] == backend and row["status"] == "completed"]
            summary.append({"backend": backend, "completed": len(times), "requested": args.runs,
                            "median_training_seconds": statistics.median(times) if times else None,
                            "min_training_seconds": min(times, default=None), "max_training_seconds": max(times, default=None),
                            "max_peak_rss_bytes": max((row["peak_rss_bytes"] for row in raw if row["backend"] == backend and row.get("peak_rss_bytes") is not None), default=None)})
        complete = all(row["status"] == "completed" for row in raw)
        write_json(output / "summary.json", {"results": summary, "complete_comparison": complete,
                   "unique_completed_models": len(hashes), "model_sha256": next(iter(hashes), None),
                   "speedup_reference_over_incremental": summary[0]["median_training_seconds"] / summary[1]["median_training_seconds"] if complete else None})
        manifest["status"] = "completed" if complete else "incomplete_timeouts"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
        print(f"Outputs: {output}")


if __name__ == "__main__": main()
