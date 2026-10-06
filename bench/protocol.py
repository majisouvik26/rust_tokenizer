"""Shared provenance and serial subprocess capture; no experiment starts on import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def new_output(label, supplied=None):
    output = Path(supplied) if supplied else ROOT / "docs/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + label)
    output = output.resolve()
    if not output.is_relative_to((ROOT / "docs").resolve()):
        raise ValueError("new experiment outputs must be inside docs/")
    output.mkdir(parents=True, exist_ok=False)
    return output


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", stderr=subprocess.PIPE).strip()


def source_snapshot():
    files = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in {".git", "target", ".venv", "__pycache__", "results", "artifacts", "dist", "docs"}]
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(ROOT)
            if path.suffix in {".rs", ".py", ".toml", ".lock", ".json", ".yml", ".txt"} or path.name == ".gitattributes":
                files.append({"path": relative.as_posix(), "sha256": sha(path)})
    files.sort(key=lambda row: row["path"])
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = command(["git", "rev-parse", "HEAD"])
        dirty = bool(command(["git", "status", "--porcelain", "--untracked-files=normal"]))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "tree_sha256": digest, "files": files}


def provenance():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source_snapshot(),
            "machine": {"os": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(), "python": sys.version,
                        "rustc": command(["rustc", "--version", "--verbose"])},
            "locks": {name: sha(ROOT / name) for name in ["Cargo.lock", "requirements.lock"]},
            "build": {"profile": "release", "opt_level": 3, "lto": False, "codegen_units": 1,
                      "rustflags": os.environ.get("RUSTFLAGS", ""), "cargo_build_target": os.environ.get("CARGO_BUILD_TARGET", "host")}}


def run_logged(argv, output, label, timeout=300, env=None):
    """Wait for one worker, preserve its logs, and kill it on the declared cap."""
    start = time.perf_counter()
    try:
        result = subprocess.run(argv, cwd=ROOT, env=env, timeout=timeout, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        stdout, stderr, code, status = result.stdout, result.stderr, result.returncode, "completed"
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        if isinstance(stdout, bytes): stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes): stderr = stderr.decode("utf-8", errors="replace")
        code, status = None, "timeout"
    elapsed = time.perf_counter() - start
    log = output / "logs" / (label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    return {"argv": [str(arg) for arg in argv], "status": status, "exit_code": code,
            "wall_seconds": elapsed, "log": str(log.relative_to(output)), "log_sha256": sha(log)}, stdout


def build_worker(output, bench="optimization"):
    argv = ["cargo", "bench", "--locked", "-p", "bpe", "--features", "parallel", "--bench", bench, "--no-run", "--message-format=json"]
    event, stdout = run_logged(argv, output, "build", timeout=1800)
    write_json(output / "build.json", event)
    if event["exit_code"] != 0:
        raise RuntimeError("build failed; inspect " + str(output / event["log"]))
    executables = [row["executable"] for line in stdout.splitlines() if line.startswith("{")
                   for row in [json.loads(line)] if row.get("reason") == "compiler-artifact" and row.get("target", {}).get("name") == bench and row.get("executable")]
    if not executables: raise RuntimeError("benchmark executable was not reported by Cargo")
    return executables[-1]


def benchmark_environment():
    return dict(os.environ, TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
