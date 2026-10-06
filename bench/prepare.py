#!/usr/bin/env python3
"""Freeze bounded synthetic development workloads and verify both production ID oracles."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from protocol import ROOT, new_output, sha, write_json
sys.path.insert(0, str(ROOT / "scripts"))
from fixture_io import read_jsonl
from baselines import engines


def prepare(output):
    fixture = ROOT / "fixtures/gpt2.jsonl"
    rows = read_jsonl(fixture)
    manifest = json.loads((ROOT / "fixtures/gpt2.manifest.json").read_text(encoding="utf-8"))
    if sha(fixture) != manifest["fixture_sha256"] or sha(ROOT / "models/gpt2.json") != manifest["model_file_sha256"]:
        raise ValueError("frozen model/fixture hash mismatch")
    tk, hf, _ = engines()
    groups = {}
    for name, predicate in [
        ("short", lambda row: 0 < len(row["text"].encode()) <= 96),
        ("code", lambda row: row.get("domain") == "code"),
        ("unicode", lambda row: any(ord(c) > 127 for c in row["text"])),
    ]:
        seen = set()
        selected = []
        for row in rows:
            if predicate(row) and row["text"] not in seen:
                seen.add(row["text"])
                selected.append(row)
            if len(selected) == 128: break
        if not selected: raise ValueError("empty workload " + name)
        groups[name] = selected
    rng = random.Random(271828)
    stress = []
    for size in [128, 512, 2048, 8192]:
        for kind, text in [
            ("repeat", "a" * size),
            ("alternating", ("abc" * (size // 3 + 1))[:size]),
            ("letters", "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(size))),
            ("spaces", " " * size),
        ]:
            stress.append({"case_id": f"stress-{kind}-{size}", "domain": kind, "text": text})
    groups["long_chunks"] = stress
    workloads = []
    for name, selected in groups.items():
        path = output / (name + ".jsonl")
        for row in selected:
            ids = tk.encode(row["text"], allowed_special=set(), disallowed_special=())
            hf_ids = hf.encode(row["text"], add_special_tokens=False).ids
            if ids != hf_ids or ("ids" in row and ids != row["ids"]):
                raise ValueError(f"oracle mismatch in {name}: {row['case_id']}")
            if tk.decode_bytes(ids) != row["text"].encode("utf-8"):
                raise ValueError("oracle byte mismatch")
            row["ids"] = ids
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8", newline="\n")
        workloads.append({"name": name, "path": path.name, "sha256": sha(path), "documents": len(selected),
                          "bytes": sum(len(row["text"].encode()) for row in selected)})
    training, seen, size = [], set(), 0
    for row in rows:
        text = row["text"]
        if not text or text in seen: continue
        count = len(text.encode())
        if size + count > 100 * 1024: continue
        training.append({"text": text})
        seen.add(text)
        size += count
        if size >= 100 * 1024 - 64: break
    training_path = output / "training.jsonl"
    training_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training), encoding="utf-8", newline="\n")
    result = {"schema_version": 1, "split": "development", "kind": "synthetic; no held-out production claim",
              "fixture_sha256": sha(fixture), "model_sha256": sha(ROOT / "models/gpt2.json"),
              "generator_sha256": sha(__file__), "seed": 271828, "workloads": workloads,
              "versions": {name: importlib.metadata.version(name) for name in ["tiktoken", "tokenizers", "regex"]},
              "training": {"path": training_path.name, "sha256": sha(training_path), "bytes": size, "documents": len(training)},
              "production_id_mismatches": 0, "selection": "distinct records per workload; no repeated padding to reach quota; workloads may overlap"}
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/runs/development")
    args = parser.parse_args()
    output = new_output("development", ROOT / args.out)
    try:
        result = prepare(output)
    except Exception as error:
        write_json(output / "failure.json", {"error": str(error), "status": "failed"})
        raise
    print(json.dumps({"output": str(output), "workloads": result["workloads"]}, indent=2))


if __name__ == "__main__": main()
