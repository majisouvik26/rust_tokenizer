#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from protocol import source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    args = parser.parse_args()
    output = (ROOT / args.out).resolve() if args.out else ROOT / "docs/exports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-review.zip")
    if not output.is_relative_to((ROOT / "docs/exports").resolve()): parser.error("--out must be inside docs/exports/")
    if output.exists(): parser.error("output already exists; choose a new filename")
    runs = ROOT / "docs/runs"
    if not runs.is_dir(): parser.error("no docs/runs/ evidence exists yet")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = source_snapshot()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(runs.rglob("*")):
            if path.is_file() and not path.is_symlink(): archive.write(path, path.relative_to(ROOT))
        archive.writestr("source-snapshot.json", json.dumps(snapshot, indent=2) + "\n")
        for entry in snapshot["files"]:
            archive.write(ROOT / entry["path"], "source/" + entry["path"])
        for path in sorted((ROOT / "docs").glob("*.md")):
            archive.write(path, path.relative_to(ROOT))
    print(f"Review archive: {output}\nUpload this archive, including failed-run logs. Nothing was pushed or uploaded.")


if __name__ == "__main__": main()
