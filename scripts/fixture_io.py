"""JSONL boundaries are literal newline bytes, not all Unicode line breaks."""
import json
from pathlib import Path

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]
